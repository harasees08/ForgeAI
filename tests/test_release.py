from dataclasses import replace
from importlib.util import find_spec
import json
import unittest
from unittest.mock import Mock

from forgeai.agent import run_engineering
from forgeai.demo import demo_pages
from forgeai.documents import passages_from_pages
from forgeai.ollama import ChatResponse, OllamaError
from forgeai.references import FLAGSHIP_QUESTION, identify_reference_pages, reference_pages
from forgeai.retrieval import Index


def response(message, done_reason="stop"):
    return ChatResponse(message, 0.3, None, None, None, None, done_reason)


VALID_PLAN = {"tool": "compare_materials"}


class ToolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.index = Index(passages_from_pages(demo_pages()))

    def test_native_missing_tool_calls_retries_with_structured_schema(self):
        client = Mock(model="a-native-tool-model")
        client.chat.side_effect = [
            response({"content": "Several unfinished thoughts"}, "length"),
            response({"content": json.dumps(VALID_PLAN)}),
        ]
        run = run_engineering(FLAGSHIP_QUESTION, self.index, client)
        self.assertEqual(run.selected_by, "Ollama")
        self.assertEqual(run.selection_method, "structured JSON retry")
        self.assertIn("output limit", run.selection_trace[0])
        self.assertIn("tools", client.chat.call_args_list[0].kwargs)
        self.assertIn("schema", client.chat.call_args_list[1].kwargs)

    def test_qwen_uses_one_structured_selection_when_successful(self):
        client = Mock(model="qwen3:4b")
        client.chat.return_value = response({"content": json.dumps(VALID_PLAN)})
        run = run_engineering(FLAGSHIP_QUESTION, self.index, client)
        self.assertEqual(run.selection_method, "structured JSON selection")
        client.chat.assert_called_once()
        self.assertIn("schema", client.chat.call_args.kwargs)

    def test_fallback_requires_opt_in_is_labeled_and_is_bounded(self):
        client = Mock(model="qwen3:4b")
        client.chat.return_value = response({"content": "{"}, "length")
        with self.assertRaisesRegex(OllamaError, "could not select"):
            run_engineering(FLAGSHIP_QUESTION, self.index, client)
        self.assertEqual(client.chat.call_count, 2)
        client.chat.reset_mock()
        run = run_engineering(FLAGSHIP_QUESTION, self.index, client, allow_fallback=True)
        self.assertEqual(run.selected_by, "deterministic fallback")
        self.assertEqual(len(run.selection_trace), 2)
        self.assertEqual(client.chat.call_count, 2)

    def test_invalid_units_fail_closed_even_with_fallback_enabled(self):
        client = Mock(model="a-native-tool-model")
        bad = {"name": "compare_materials", "arguments": {"force_n": 15000, "safety_factor": 2, "length_m": 120}}
        client.chat.return_value = response({"tool_calls": [{"function": bad}]})
        with self.assertRaisesRegex(OllamaError, "length_m does not match"):
            run_engineering(FLAGSHIP_QUESTION, self.index, client, allow_fallback=True)
        client.chat.assert_called_once()

    def test_structured_plan_cannot_override_python_arguments(self):
        client = Mock(model="qwen3:4b")
        client.chat.return_value = response({"content": json.dumps({**VALID_PLAN, "arguments": {"force_n": 1}})})
        with self.assertRaisesRegex(OllamaError, "select one tool only"):
            run_engineering(FLAGSHIP_QUESTION, self.index, client, allow_fallback=True)


@unittest.skipUnless(find_spec("pymupdf"), "PyMuPDF is not installed")
class ReferencePdfTests(unittest.TestCase):
    def test_real_reference_cards_preserve_attribution_and_calculation(self):
        pages = reference_pages()
        self.assertEqual(len(pages), 3)
        self.assertTrue(all(page.number == 1 and page.source and not page.synthetic for page in pages))
        self.assertEqual([(page.source.yield_page, page.source.density_page) for page in pages],
                         [(1, 2), (1, 2), (2, 1)])
        result = run_engineering(FLAGSHIP_QUESTION, Index(passages_from_pages(pages))).result
        self.assertEqual([row.material.name for row in result.candidates], [
            "Titanium Ti-6Al-4V, annealed", "Aluminum 7075-T6", "Aluminum 6061-T6",
        ])
        self.assertAlmostEqual(result.candidates[0].area_mm2, 36.231884, places=5)
        self.assertAlmostEqual(result.candidates[0].mass_kg, 0.01921739, places=6)

    def test_modified_card_cannot_inherit_publisher_identity(self):
        page = reference_pages()[0]
        modified = replace(page, source=None, text=page.text.replace("276 MPa", "999 MPa"))
        self.assertIsNone(identify_reference_pages([modified])[0].source)
