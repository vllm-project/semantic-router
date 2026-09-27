"""Gold-free native over-budget recovery contract tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from inference.run import digest
from scripts.recover_nox_overbudget_v3 import PLAN_VERSION, recover, sha


class RecoverNoxOverbudgetV3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.prompts = root / "css.prompts.jsonl"
        self.predictions = root / "nox.css.predictions.jsonl"
        self.plan_path = root / "plan.json"
        self.log = root / "failure.log"
        self.receipt = root / "receipt.json"
        self.questions = {"label": {"type": "choice", "options": ["a", "b"]}}
        self.rows = [
            {
                "id": f"css/example/{i}",
                "state": f"state-{i}",
                "questions": self.questions,
            }
            for i in range(3)
        ]
        self.prompts.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8"
        )
        first = {
            "id": self.rows[0]["id"],
            "answers": {"label": {"type": "choice", "choice": "a"}},
            "latency_ms": 5.0,
            "usage": None,
            "model": "Decision-1.0-Nox",
            "backend": "nox",
            "model_id": "llm-semantic-router/Decision-1.0-Nox-4B",
            "adapter_version": "native-published-v2",
            "model_revision": "fixed-revision",
            "revision_attested": True,
            "model_config_sha256": "a" * 64,
            "source_input_sha256": digest(
                {"state": self.rows[0]["state"], "questions": self.questions}
            ),
            "runtime_matches_validated": True,
        }
        self.predictions.write_text(json.dumps(first) + "\n", encoding="utf-8")
        plan = {
            "plan_version": PLAN_VERSION,
            "css_prompts": {"path": str(self.prompts), "sha256": sha(self.prompts)},
            "inference": [
                {
                    "key": "nox",
                    "group": "decision1",
                    "model_id": first["model_id"],
                    "revision": first["model_revision"],
                    "paths": {"css": str(self.predictions)},
                }
            ],
        }
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        self.plan_sha = sha(self.plan_path)
        self.log.write_text(
            "Traceback (most recent call last):\n"
            "ValueError: label: 18198 tokens exceeds max_length=16384; no truncation allowed\n",
            encoding="utf-8",
        )

    def test_appends_one_explicitly_invalid_answer(self):
        result = recover(self.plan_path, self.plan_sha, self.log, self.receipt)
        self.assertEqual(result["failed_prompt_id"], "css/example/1")
        lines = [json.loads(line) for line in self.predictions.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1]["answers"]["label"]["error"], "input_over_budget")
        self.assertNotIn("choice", lines[1]["answers"]["label"])
        self.assertEqual(result["recovered_predictions_sha256"], sha(self.predictions))
        with self.assertRaises(FileExistsError):
            recover(self.plan_path, self.plan_sha, self.log, self.receipt)

    def test_other_native_error_is_fatal_and_does_not_append(self):
        old = sha(self.predictions)
        self.log.write_text("ValueError: unrelated failure\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not the exact native over-budget"):
            recover(self.plan_path, self.plan_sha, self.log, self.receipt)
        self.assertEqual(sha(self.predictions), old)

    def test_changed_plan_is_fatal(self):
        old = sha(self.predictions)
        with self.assertRaisesRegex(ValueError, "frozen plan digest changed"):
            recover(
                self.plan_path,
                hashlib.sha256(b"wrong").hexdigest(),
                self.log,
                self.receipt,
            )
        self.assertEqual(sha(self.predictions), old)


if __name__ == "__main__":
    unittest.main()
