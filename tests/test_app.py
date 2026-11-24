from importlib.util import find_spec
from pathlib import Path
import unittest


@unittest.skipUnless(find_spec("streamlit"), "Streamlit is not installed")
class AppTests(unittest.TestCase):
    def test_synthetic_report_in_workspace(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value("Knowledge Base").run()
        next(button for button in app.button if button.label == "Load synthetic demo").click().run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value("Engineering Workspace").run()
        next(button for button in app.button if button.label == "Run").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.iloc[0]["Material"], "Synthetic Alloy B")
