from dataclasses import asdict
from importlib.util import find_spec
import json
import unittest
from unittest.mock import Mock

from forgeai.agent import grounded_answer, request_arguments, run_engineering
from forgeai.demo import DEMO_QUESTION, demo_pages
from forgeai.documents import Page, passages_from_pages, pdf_pages
from forgeai.materials import extract_materials
from forgeai.ollama import ChatResponse, Ollama, OllamaError
from forgeai.retrieval import Index


def response(message):
    return ChatResponse(message, 0.1, None, None, None, None)


class IntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.pages = demo_pages()
        self.index = Index(passages_from_pages(self.pages))

    def test_page_boundaries_and_chunk_overlap(self):
        pages = [Page("Manual.pdf", 3, "one two three four five"), Page("Manual.pdf", 4, "six seven")]
        result = passages_from_pages(pages, max_words=3, overlap=1)
        self.assertEqual([(item.page.number, item.text) for item in result],
                         [(3, "one two three"), (3, "three four five"), (4, "six seven")])

    def test_rank_and_abstain(self):
        self.assertEqual(self.index.search("Synthetic Alloy B")[0].passage.page.document, self.pages[1].document)
        self.assertEqual(self.index.search("camera aperture"), [])
        client = Mock()
        self.assertTrue(grounded_answer("camera aperture", self.index, client).unsupported)
        client.chat.assert_not_called()

    def test_verify_citation_against_passage(self):
        index = Index(passages_from_pages([Page("Manual.pdf", 2, "Rated load is 15000 N.", "A-2")]))
        client = Mock()
        client.chat.return_value = response({"content": json.dumps({"claims": [
            {"text": "The rated load is 15 kN.", "passage_id": "P1", "quote": "Rated load is 15000 N."},
        ]})})
        result = grounded_answer("What is the rated load?", index, client)
        self.assertEqual((result.claims[0].citation.document, result.claims[0].citation.page,
                          result.claims[0].citation.label), ("Manual.pdf", 2, "A-2"))
        for invalid_id, quote in [("P7", "Rated load is 15000 N."), ("P1", "Rated load is 20000 N.")]:
            client.chat.return_value = response({"content": json.dumps({"claims": [
                {"text": "Bad evidence.", "passage_id": invalid_id, "quote": quote},
            ]})})
            with self.assertRaisesRegex(OllamaError, "citation validation"):
                grounded_answer("What is the rated load?", index, client)

    def test_flagship_minimum_area_and_mass_rank(self):
        tool = run_engineering(DEMO_QUESTION, self.index)
        report = tool.result
        self.assertEqual(tool.selected_by, "local rules")
        self.assertEqual((report.force_n, report.safety_factor, report.length_m), (15000, 2, None))
        self.assertEqual([row.material.name for row in report.candidates],
                         ["Synthetic Alloy B", "Synthetic Alloy A", "Synthetic Alloy C"])
        self.assertAlmostEqual(report.candidates[0].area_mm2, 60.0)
        self.assertAlmostEqual(report.candidates[0].kg_per_m, 0.27)
        self.assertIsNone(report.candidates[0].mass_kg)
        self.assertTrue(all(row.material.source.synthetic for row in report.candidates))
        self.assertTrue(all(row.material.source.page == 1 for row in report.candidates))
        json.dumps(asdict(tool), allow_nan=False)

    def test_flagship_with_length_compares_total_mass(self):
        report = run_engineering(DEMO_QUESTION + " Length: 120 mm.", self.index).result
        self.assertEqual(report.length_m, 0.12)
        self.assertAlmostEqual(report.candidates[0].mass_kg, 0.0324)

    def test_material_conversion_and_conflict(self):
        page = Page("Source.pdf", 4, "Material: Test\nYield strength: 40 ksi\nDensity: 2.7 g/cm³")
        found, issues = extract_materials([page])
        self.assertFalse(issues)
        self.assertAlmostEqual(found[0].yield_mpa, 275.79029172)
        self.assertAlmostEqual(found[0].density_kg_m3, 2700)
        other = Page("Other.pdf", 1, "Material: Test\nYield strength: 200 MPa\nDensity: 2700 kg/m3")
        missing = Page("Missing.pdf", 1, "Material: Example\nYield strength: 300 MPa")
        found, issues = extract_materials([page, other, missing])
        self.assertFalse(found)
        self.assertIn("sources disagree", issues[0])
        self.assertIn("unambiguous", issues[1])

    def test_model_may_select_only_a_matching_tool_call(self):
        client = Mock()
        call = {"name": "compare_materials", "arguments": {"force_n": 15000, "safety_factor": 2}}
        client.chat.return_value = response({"tool_calls": [{"function": call}]})
        self.assertEqual(run_engineering(DEMO_QUESTION, self.index, client).selected_by, "Ollama")
        for name, bad_args in [
            ("compare_materials", {"force_n": 15, "safety_factor": 2}),
            ("compare_materials", {"force_n": 15000, "safety_factor": 2, "length_m": 0.12}),
            ("shell", call["arguments"]),
        ]:
            client.chat.return_value = response({"tool_calls": [{"function": {"name": name, "arguments": bad_args}}]})
            with self.assertRaisesRegex(OllamaError, "Rejected"):
                run_engineering(DEMO_QUESTION, self.index, client)

    def test_tensile_stress_and_missing_factor(self):
        run = run_engineering("Find tensile stress for 15,000 N and a 120 mm² area.", self.index)
        self.assertEqual((run.name, run.result), ("tensile_stress", 125))
        with self.assertRaisesRegex(ValueError, "factor of safety"):
            request_arguments("Compare materials at 15 kN.")
        with self.assertRaisesRegex(ValueError, "length in"):
            request_arguments(DEMO_QUESTION + " Length: 10 inches.")

    @unittest.skipUnless(find_spec("pymupdf"), "PyMuPDF is not installed")
    def test_extract_pdf_text_preserves_page_and_empty_page(self):
        import pymupdf

        with pymupdf.open() as document:
            document.new_page()
            document.new_page().insert_text((72, 72), "Page two yield strength.")
            data = document.tobytes()
        pages = pdf_pages("Fixture.pdf", data)
        self.assertEqual([page.number for page in pages], [1, 2])
        self.assertFalse(pages[0].text.strip())
        self.assertEqual(Index(passages_from_pages(pages)).search("yield strength")[0].passage.page.number, 2)
        pdf_report = run_engineering(DEMO_QUESTION, Index(passages_from_pages(demo_pages(through_pdf=True)))).result
        self.assertEqual(len(pdf_report.candidates), 3)

    @unittest.skipUnless(find_spec("numpy"), "NumPy is not installed")
    def test_semantic_index_ranks_normalized_vectors(self):
        encoder = Mock()
        encoder.encode.side_effect = [[[10, 0], [0, 25]], [[0, 2]]]
        pages = [Page("A.pdf", 1, "Motor"), Page("B.pdf", 4, "Centrifugal pump")]
        matches = Index(passages_from_pages(pages), encoder).search("fluid")
        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].passage.page.document, matches[0].score), ("B.pdf", 1))


class OllamaContractTests(unittest.TestCase):
    def test_response_counters_and_model_placement(self):
        api = Ollama("qwen3:4b")
        api._request = Mock(side_effect=[
            {"details": {"family": "qwen3"}},
            {"done": True, "message": {"content": "text"}, "eval_count": 20, "eval_duration": 500_000_000},
            {"models": [{"name": "qwen3:4b", "size": 100, "size_vram": 80, "digest": "example"}]},
        ])
        self.assertEqual(api.chat([{"role": "user", "content": "test"}]).output_tokens_per_second, 40)
        self.assertEqual(api.placement()["status"], "partial GPU")
        self.assertEqual(api._request.call_args_list[0].args[0], "show")
        self.assertEqual(api._request.call_args_list[1].args[0], "chat")
        self.assertFalse(api._request.call_args_list[1].args[1]["stream"])

    def test_reject_remote_model_before_sending_documents(self):
        with self.assertRaisesRegex(ValueError, "local Ollama"):
            Ollama("example:cloud")
        api = Ollama("alias")
        api._request = Mock(return_value={"remote_host": "https://example.com"})
        with self.assertRaisesRegex(OllamaError, "remote model"):
            api.chat([{"role": "user", "content": "local document text"}])
        api._request.assert_called_once_with("show", {"model": "alias"})

    def test_rejects_an_incomplete_response(self):
        api = Ollama()
        api._request = Mock(return_value={"done": False})
        with self.assertRaisesRegex(OllamaError, "incomplete"):
            api.chat([])
