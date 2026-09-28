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
from v2.eval import native_nimble
from v2.eval.same_panel import input_digest

FAKE_INFERENCE = """
import json
from serving_schema import MARKER

class ParallelScorer:
    temperature = 2.179

    def __init__(self, model_dir):
        assert MARKER == "release"

    def score(self, context, schema, score_fields=()):
        assert isinstance(context, str)
        if context == "long":
            raise ValueError("Longest prompt has 9000 tokens; limit is 8192. Nothing was truncated.")
        if context == "broken":
            raise ValueError("Choice code A is not one ordinary token at the answer boundary.")
        fields = {}
        for name, field in schema.items():
            assert set(field) <= {"type", "choices", "description", "choice_descriptions"}
            if field["type"] == "boolean":
                fields[name] = {"prediction": True, "probabilities": {"false": 0.2, "true": 0.8}}
            elif name in score_fields:
                fields[name] = {"prediction": 2, "expected_score": 1.8,
                                "probabilities": {"0": 0.1, "1": 0.0, "2": 0.9}}
            else:
                fields[name] = {"prediction": field["choices"][0],
                                "probabilities": {c: (0.7 if i == 0 else 0.3) for i, c in enumerate(field["choices"])}}
        return {"fields": fields}
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


def release(root: Path, revision: str = native_nimble.MODEL_REVISION) -> Path:
    files = {
        "inference.py": FAKE_INFERENCE,
        "serving_schema.py": 'MARKER = "release"\n',
        "adapter_model.safetensors": "weights",
        "schema_config.json": json.dumps(
            {
                "model": native_nimble.BASE_ID,
                "revision": native_nimble.BASE_REVISION,
                "max_length": 8192,
            }
        ),
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(t.encode()).hexdigest()}  {n}\n" for n, t in files.items()
        ),
        encoding="utf-8",
    )
    meta = root / ".cache/huggingface/download"
    meta.mkdir(parents=True)
    (meta / "inference.py.metadata").write_text(f"{revision}\netag\n", encoding="utf-8")
    return root


class NativeNimbleTest(unittest.TestCase):
    def setUp(self) -> None:
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        patcher = mock.patch.dict(sys.modules, {"torch": fake_torch})
        patcher.start()
        self.addCleanup(patcher.stop)
        release_modules = ("inference", "serving_schema")
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
        self, tmp: str, states: list
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
        summary = native_nimble.collect(
            model_path=root,
            revision=native_nimble.MODEL_REVISION,
            prompts=prompts,
            output=output,
        )
        got = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        return summary, rows, got

    def test_native_schema_carries_all_meanings(self) -> None:
        schema, score_fields = native_nimble.native_schema(QUESTIONS)
        self.assertEqual(score_fields, ("s",))
        self.assertEqual(schema["c"]["choices"], ["a", "b"])
        self.assertEqual(
            schema["n"],
            {
                "description": "?",
                "type": "boolean",
                "choice_descriptions": {"true": "allowed", "false": "denied"},
            },
        )
        self.assertEqual(schema["s"]["choices"], ["0", "1", "2"])
        self.assertEqual(schema["s"]["choice_descriptions"]["2"], "high")

    def test_maps_answers_onto_scorer_fields_and_rejects_whole_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, got = self.run_collect(tmp, [{"ticket": "x"}, "long"])
        self.assertEqual(summary["native_rejections"], 1)
        first = got[0]["answers"]
        for name, value in (("c", "a"), ("n", True), ("s", 2)):
            gold = {"value": value, "label_to_semantic": {"a": "a", "b": "b"}}
            verdict = evaluate_answer(QUESTIONS[name], gold, first[name])
            self.assertTrue(verdict.get("correct"), (name, verdict))
        self.assertEqual(
            {a["invalid_reason"] for a in got[1]["answers"].values()},
            {"native_rejection"},
        )
        self.assertEqual(
            got[0]["source_input_sha256"], input_digest(rows[0]["state"], QUESTIONS)
        )

    def test_other_native_errors_stop_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ordinary token"):
                self.run_collect(tmp, ["broken"])

    def test_rejects_modified_release_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = release(Path(tmp))
            (root / "serving_schema.py").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from SHA256SUMS"):
                native_nimble.verify_release(root, native_nimble.MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
