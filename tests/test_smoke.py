import unittest

from scripts.smoke import parse_telemetry


class TelemetryParsingTests(unittest.TestCase):
    def test_numeric_fields_and_unsupported_metrics(self):
        rows = parse_telemetry('"Test, GPU", 576.76, 12.0, 8151, 7400, 25, 43, 800, [N/A]')
        self.assertEqual(rows, [{
            "gpu": "Test, GPU",
            "driver": "576.76",
            "compute_capability": "12.0",
            "vram_total_mib": 8151,
            "vram_free_mib": 7400,
            "gpu_utilization_pct": 25,
            "temperature_c": 43,
            "sm_clock_mhz": 800,
            "power_w": None,
        }])

    def test_empty_output_fails(self):
        with self.assertRaisesRegex(RuntimeError, "no GPU telemetry"):
            parse_telemetry("")

    def test_incomplete_row_fails(self):
        with self.assertRaises(ValueError):
            parse_telemetry("Test GPU, 576.76, 12.0, 8151")
