"""Scientific edge cases and input protection, using only the standard library."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import PredictionMetrics, read_seeds, make_configuration
import xml.etree.ElementTree as ET


class ReproductionTests(unittest.TestCase):
    def test_missing_predictions_do_not_inflate_accuracy(self):
        metric = PredictionMetrics()
        metric.add((1, "lane", 10, None, 30, 70))
        self.assertEqual(metric.result()["n_valid_pairs"], 0)
        self.assertIsNone(metric.result()["RMSE"])

    def test_errors_and_confusion_with_known_observations(self):
        metric = PredictionMetrics()
        metric.add((1, "lane", 0, 1, 1, 3))
        metric.add((2, "lane", 2, 2, 1, 3))
        metric.add((3, "lane", 4, 3, 1, 3))
        result = metric.result()
        self.assertAlmostEqual(result["RMSE"], math.sqrt(2 / 3))
        self.assertAlmostEqual(result["MAE"], 2 / 3)
        self.assertAlmostEqual(result["R2"], 0.75)
        self.assertEqual(
            result["confusion_rows_true_columns_predicted"],
            [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
        )

    def test_constant_target_r2_is_undefined(self):
        metric = PredictionMetrics()
        for _ in range(2):
            metric.add((1, "lane", 0, 0, 30, 70))
        self.assertIsNone(metric.result()["R2"])

    def test_duplicate_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seeds.txt"
            path.write_text("179\n179\n")
            with self.assertRaises(ValueError):
                read_seeds(path)

    def test_all_output_paths_stay_inside_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder).resolve()
            config = ET.parse(make_configuration(out))
            for element in ET.parse(out / "detectors.add.xml").iter():
                if "file" in element.attrib:
                    self.assertTrue(Path(element.get("file")).is_relative_to(out))
            for tag in (
                "output/tripinfo-output",
                "output/save-state.prefix",
                "report/log",
            ):
                self.assertTrue(Path(config.find(tag).get("value")).is_relative_to(out))


if __name__ == "__main__":
    unittest.main()
