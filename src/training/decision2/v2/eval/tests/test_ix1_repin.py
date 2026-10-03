"""CPU-only checks for moving a published run's pin to a runtime-only successor (v2.eval.ix1.repin)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.ix1 import repin

FILES = {"backbone/model.safetensors": "aa" * 32, "tokenizer.json": "bb" * 32}


def _manifest(model_sha: str = "cc" * 32, loaded: int = 7) -> dict:
    return {
        "base": None,
        "identity": {"fingerprint_files": FILES, "model_sha256": model_sha},
        "max_input_tokens": 16384,
        "model_name": "Decision-2.0-Nox-4B",
        "parameters": {"loaded": loaded},
        "profile": "qwen-full",
        "repo_id": "vllm-sr/Decision-2.0-Nox-4B",
    }


CURRENT = {
    "schema": "weights-vs-release/1",
    "note": "n",
    "match": True,
    "checks": {},
    "release": {"revision": "0" * 40},
    "scored_package": {
        "base": None,
        "files_sha256_recomputed": FILES,
        "identity": {"model_sha256": "cc" * 32},
        "max_input_tokens": 16384,
        "parameters_loaded": 7,
        "profile": "qwen-full",
    },
}


def _answers(choice: str, top: float, second: float) -> dict:
    return {
        "q": {
            "type": "choice",
            "choice": choice,
            "probabilities": {"a": top, "b": second},
        }
    }


class WeightsTests(unittest.TestCase):
    def run_weights(self, manifest: dict) -> dict:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
            return repin.weights(CURRENT, Path(d), "1" * 40)

    def test_same_weights_match_and_name_the_new_revision(self) -> None:
        out = self.run_weights(_manifest())
        self.assertTrue(out["match"])
        self.assertEqual(out["release"]["revision"], "1" * 40)
        self.assertEqual(out["scored_package"], CURRENT["scored_package"])

    def test_other_weights_or_parameters_do_not_match(self) -> None:
        self.assertFalse(self.run_weights(_manifest(model_sha="dd" * 32))["match"])
        self.assertFalse(self.run_weights(_manifest(loaded=8))["match"])


class NearTieTests(unittest.TestCase):
    def run_near_ties(self, stored: dict, rerun: dict) -> dict:
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "stored.jsonl", Path(d) / "rerun.jsonl"
            a.write_text(json.dumps(stored) + "\n")
            b.write_text(json.dumps(rerun) + "\n")
            report = {"mismatched_run_ids": ["r"], "statuses": {rerun["status"]: 1}}
            return repin.near_ties(report, a, [b])

    def record(self, status: str, answers: dict | None) -> dict:
        out = {
            "run_id": "r",
            "status": status,
            "engine": "e",
            "payload_sha256": "ab" * 32,
        }
        if answers is not None:
            out["response"] = {"answers": answers}
        return out

    def test_a_flip_within_the_margin_is_a_near_tie(self) -> None:
        out = self.run_near_ties(
            self.record("ok", _answers("a", 0.51, 0.49)),
            self.record("ok", _answers("b", 0.40, 0.60)),
        )
        self.assertTrue(out["all_flips_near_ties"])
        self.assertEqual(out["flipped_questions"][0]["stored_top2_margin"], 0.02)

    def test_a_wide_flip_or_a_status_change_is_not(self) -> None:
        wide = self.run_near_ties(
            self.record("ok", _answers("a", 0.9, 0.1)),
            self.record("ok", _answers("b", 0.2, 0.8)),
        )
        self.assertFalse(wide["all_flips_near_ties"])
        status = self.run_near_ties(
            self.record("ok", _answers("a", 0.51, 0.49)), self.record("error", None)
        )
        self.assertFalse(status["all_flips_near_ties"])


if __name__ == "__main__":
    unittest.main()
