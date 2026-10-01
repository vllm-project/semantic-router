"""Package temperatures for own-1.0 teacher labels (skips without torch; run in the training image)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "teacher_label imports torch")
class TeacherTemperatureTest(unittest.TestCase):
    def setUp(self) -> None:
        from v2.dec.teacher_label import teacher_temperatures

        self.read = teacher_temperatures
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, doc: dict) -> None:
        (self.root / name).write_text(json.dumps(doc), encoding="utf-8")

    def test_per_type_file_wins(self) -> None:
        self.write(
            "temperature.json",
            {"temperatures": {"choice": 1.3, "noul": 1.2, "score": 0.9}},
        )
        self.write("config.json", {"calibration": {"temperature": 7.0}})
        self.assertEqual(
            self.read(self.root), {"choice": 1.3, "noul": 1.2, "score": 0.9}
        )

    def test_single_config_temperature_applies_to_every_type(self) -> None:
        self.write("config.json", {"calibration": {"temperature": 1.0389139156246665}})
        self.assertEqual(
            self.read(self.root),
            dict.fromkeys(("choice", "noul", "score"), 1.0389139156246665),
        )

    def test_missing_or_invalid_temperature_is_an_error(self) -> None:
        self.write("config.json", {"model_name": "x"})
        with self.assertRaises(ValueError):
            self.read(self.root)
        self.write("config.json", {"calibration": {"temperature": -1.0}})
        with self.assertRaises(ValueError):
            self.read(self.root)

    def test_dec_teacher_needs_exactly_one_temperature_source(self) -> None:
        from v2.dec.teacher_label import dec_temperatures

        self.assertEqual(
            dec_temperatures(None, True, "x"),
            dict.fromkeys(("choice", "noul", "score"), 1.0),
        )
        report = self.root / "cal.json"
        report.write_text(
            json.dumps(
                {
                    "model_sha256": "x",
                    "temperature_by_type": {"choice": 1.1, "noul": 0.9, "score": 1.2},
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(dec_temperatures(report, False, "x")["score"], 1.2)
        with self.assertRaises(ValueError):
            dec_temperatures(report, True, "x")
        with self.assertRaises(ValueError):
            dec_temperatures(None, False, "x")
        with self.assertRaises(ValueError):
            dec_temperatures(report, False, "y")


if __name__ == "__main__":
    unittest.main()
