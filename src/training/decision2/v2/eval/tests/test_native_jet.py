from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmark.score import evaluate_answer
from v2.eval import native_jet
from v2.eval.same_panel import input_digest

FAKE_JET = """
from inference import MARKER

class Jet:
    def __init__(self, path):
        assert MARKER == "release"

    def decide(self, state, questions):
        answers = {}
        for name, q in questions.items():
            if q["type"] == "noul" and "criteria" in q:
                raise ValueError("noul questions take no criteria")
            if state == "long" and q["type"] == "score":
                raise ValueError(f"{name}: complete prompt exceeds 16384 tokens; no truncation applied")
            if state == "broken":
                raise ValueError("unknown question type")
            if q["type"] == "noul":
                assert "Yes means: allowed" in q["instructions"]
                answers[name] = {"type": "noul", "probability": 0.8, "confidence": 0.3}
            elif q["type"] == "score":
                answers[name] = {"type": "score", "score": 1.8, "level": "high",
                                 "probabilities": [0.1, 0.0, 0.9], "confidence": 0.5}
            else:
                answers[name] = {"type": "choice", "choice": "a",
                                 "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.1}
        return {"answers": answers}
"""

QUESTIONS = {
    "c": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}},
    "n": {
        "type": "noul",
        "instructions": "?",
        "criteria": {"true": "allowed", "false": "denied"},
    },
    "s": {"type": "score", "instructions": "?", "criteria": ["low", "mid", "high"]},
}


def release(root: Path, revision: str = native_jet.MODEL_REVISION) -> Path:
    files = {
        "jet.py": FAKE_JET,
        "inference.py": 'MARKER = "release"\n',
        "format.py": "",
        "runtime.py": "",
        "calibration.json": "{}",
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    manifest = {
        "version": "v6.2.0",
        "files": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in files.items()
        },
    }
    (root / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    meta = root / ".cache/huggingface/download"
    meta.mkdir(parents=True)
    (meta / "jet.py.metadata").write_text(f"{revision}\netag\n", encoding="utf-8")
    return root


class NativeJetTest(unittest.TestCase):
    def setUp(self) -> None:
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        patcher = mock.patch.dict(sys.modules, {"torch": fake_torch})
        patcher.start()
        self.addCleanup(patcher.stop)
        release_modules = ("jet", "inference", "format", "runtime")
        saved = {
            name: sys.modules.pop(name)
            for name in release_modules
            if name in sys.modules
        }
        path_before = list(sys.path)

        def restore() -> None:
            for name in release_modules:
                sys.modules.pop(name, None)
            sys.modules.update(saved)
            sys.path[:] = path_before

        self.addCleanup(restore)

    def run_collect(
        self, tmp: str, states: list[str]
    ) -> tuple[dict, list[dict], list[dict]]:
        root = Path(tmp) / "model"
        root.mkdir()
        release(root)
        rows = [
            {"id": str(i), "state": s, "questions": QUESTIONS}
            for i, s in enumerate(states)
        ]
        prompts = Path(tmp) / "prompts.jsonl"
        prompts.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        output = Path(tmp) / "out.jsonl"
        summary = native_jet.collect(
            model_path=root,
            revision=native_jet.MODEL_REVISION,
            prompts=prompts,
            output=output,
        )
        got = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        return summary, rows, got

    def test_maps_answers_onto_scorer_fields_and_rejects_per_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, got = self.run_collect(tmp, ["short", "long"])
        self.assertEqual(summary["native_rejections"], 1)
        self.assertEqual(summary["verified_files"], 5)
        first = got[0]["answers"]
        self.assertEqual(first["n"], {"type": "noul", "noul": 0.8, "confidence": 0.3})
        self.assertEqual(first["s"]["probabilities"], {"0": 0.1, "1": 0.0, "2": 0.9})
        for name, value in (("c", "a"), ("n", True), ("s", 2)):
            gold = {"value": value, "label_to_semantic": {"a": "a", "b": "b"}}
            verdict = evaluate_answer(QUESTIONS[name], gold, first[name])
            self.assertTrue(verdict.get("correct"), (name, verdict))
        second = got[1]["answers"]
        self.assertEqual(second["s"]["invalid_reason"], "native_rejection")
        self.assertEqual(second["c"]["choice"], "a")
        self.assertEqual(
            got[1]["source_input_sha256"], input_digest(rows[1]["state"], QUESTIONS)
        )

    def test_other_native_errors_stop_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown question type"):
                self.run_collect(tmp, ["broken"])

    def test_rejects_modified_release_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release(root)
            (root / "runtime.py").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                native_jet.verify_release(root, native_jet.MODEL_REVISION)

    def test_rejects_unpinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = release(Path(tmp), revision="0" * 40)
            with self.assertRaisesRegex(ValueError, "pinned"):
                native_jet.verify_release(root, native_jet.MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
