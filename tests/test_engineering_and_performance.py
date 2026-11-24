import unittest
from unittest.mock import Mock, patch

from forgeai.engineering import (
    component_mass_kg, component_volume_m3, factor_of_safety, required_area_mm2,
    specific_yield_strength_j_kg, tensile_stress_mpa,
)
from forgeai.performance import _embedding_trials, sample_telemetry, summarize


class EngineeringTests(unittest.TestCase):
    def test_tensile_stress_and_factor_of_safety(self):
        self.assertEqual(tensile_stress_mpa(15000, 120), 125)
        self.assertEqual(factor_of_safety(250, 125), 2)
        self.assertEqual(factor_of_safety(125, 250), 0.5)

    def test_required_area(self):
        self.assertEqual(required_area_mm2(15000, 2, 250), 120)

    def test_component_volume_and_mass(self):
        self.assertAlmostEqual(component_volume_m3(120, 0.1), 0.000012)
        self.assertAlmostEqual(component_mass_kg(2700, 0.000012), 0.0324)

    def test_specific_yield_strength(self):
        self.assertAlmostEqual(specific_yield_strength_j_kg(500, 4500), 111111.1111111111)

    def test_reject_invalid_input(self):
        for bad in (0, -1, float("nan"), float("inf"), True, "15000"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                tensile_stress_mpa(bad, 120)
        with self.assertRaises(ValueError):
            required_area_mm2(15000, 0.5, 250)
        with self.assertRaises(ValueError):
            component_volume_m3(120, 0)


class PerformanceTests(unittest.TestCase):
    def test_summary_and_timed_trials(self):
        self.assertEqual(summarize([1, 2, 3]), {"mean": 2, "sample_stddev": 1})
        encoder = Mock()
        synchronization = Mock()
        with patch("forgeai.performance.perf_counter", side_effect=[0, 1, 10, 11, 20, 21]):
            result, sample = _embedding_trials(encoder, ["one", "two"], 1, synchronization)
        self.assertEqual(result["texts_per_second"], [2, 2, 2])
        self.assertIs(sample, encoder.encode.return_value)
        self.assertEqual(encoder.encode.call_count, 4)
        self.assertEqual(synchronization.call_count, 6)

    def test_bounded_telemetry_samples(self):
        with patch("forgeai.performance.telemetry_check", return_value=[{"gpu": "test"}]) as telemetry:
            samples = sample_telemetry(2, 0)["samples"]
        self.assertEqual(telemetry.call_count, 2)
        self.assertEqual(samples[0]["gpus"], [{"gpu": "test"}])
        with self.assertRaises(ValueError):
            sample_telemetry(11)
