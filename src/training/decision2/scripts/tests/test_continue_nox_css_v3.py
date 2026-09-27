"""Strict Nox over-budget continuation loop tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.continue_nox_css_v3 import continue_css


class ContinueNoxCssV3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.predictions = root / "nox.css.predictions.jsonl"
        self.predictions.write_text('{"id":"first"}\n', encoding="utf-8")
        self.log = root / "first-failure.log"
        self.log.write_text(
            "ValueError: label: 18198 tokens exceeds max_length=16384; no truncation allowed\n",
            encoding="utf-8",
        )
        self.plan = root / "plan.json"
        self.plan.write_text(
            json.dumps(
                {"inference": [{"key": "nox", "paths": {"css": str(self.predictions)}}]}
            ),
            encoding="utf-8",
        )
        self.root = root

    def test_one_exact_failure_and_native_completion(self):
        def fake_recover(_plan, _sha, _log, path):
            path.write_text("receipt", encoding="utf-8")
            path.with_name(path.name + ".intent").write_text("intent", encoding="utf-8")
            self.predictions.write_text(
                self.predictions.read_text() + '{"id":"invalid"}\n', encoding="utf-8"
            )
            return {"failed_prompt_id": "css/example/2"}

        def fake_run(_plan, _sha, _panel, _recovery, _log, receipt):
            receipt.write_text("native", encoding="utf-8")
            return {"exit_code": 0}

        with (
            patch("scripts.continue_nox_css_v3.recover", side_effect=fake_recover),
            patch("scripts.continue_nox_css_v3.run", side_effect=fake_run),
        ):
            result = continue_css(self.plan, "a" * 64, self.log, self.root, 2)
        self.assertEqual(result["status"], "css_complete")
        self.assertEqual(result["events"][0]["failed_prompt_index"], 2)

    def test_non_budget_error_stops_without_recovery(self):
        self.log.write_text("ValueError: unrelated\n", encoding="utf-8")
        with patch("scripts.continue_nox_css_v3.recover") as recovery:
            with self.assertRaisesRegex(ValueError, "not an exact over-budget"):
                continue_css(self.plan, "a" * 64, self.log, self.root, 2)
        recovery.assert_not_called()


if __name__ == "__main__":
    unittest.main()
