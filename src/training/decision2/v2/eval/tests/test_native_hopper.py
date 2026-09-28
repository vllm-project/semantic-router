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
from v2.eval import native_hopper

FAKE_MODEL = """
class Decider:
    slow_kernels = ["causal_conv1d"]

    def __init__(self, adapter=None, device=None, allow_slow_kernels=False):
        assert allow_slow_kernels

    def decide(self, req):
        (name, q), = req["questions"].items()
        if q["type"] == "noul":
            answer = {"type": "noul", "noul": 0.7}
        elif q["type"] == "score":
            answer = {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}
        else:
            answer = {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}
        return {"answers": {name: answer}}
"""

QUESTIONS = {
    "c": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}},
    "n": {"type": "noul", "instructions": "?", "criteria": {"true": "y", "false": "n"}},
    "s": {"type": "score", "instructions": "?", "criteria": ["low", "mid", "high"]},
}


def fixture(root: Path, commit: str = native_hopper.SOURCE_COMMIT) -> tuple[Path, Path]:
    model = root / "model"
    model.mkdir()
    files = {"adapter_model.safetensors": "w", "hopper.json": '{"kind": "per_kind"}'}
    for name, text in files.items():
        (model / name).write_text(text, encoding="utf-8")
    (model / "CHECKSUMS.txt").write_text(
        "".join(
            f"{hashlib.sha256(t.encode()).hexdigest()}  {n}\n" for n, t in files.items()
        ),
        encoding="utf-8",
    )
    meta = model / ".cache/huggingface/download"
    meta.mkdir(parents=True)
    (meta / "hopper.json.metadata").write_text(
        f"{native_hopper.MODEL_REVISION}\netag\n", encoding="utf-8"
    )
    source = root / "hopper"
    (source / ".git").mkdir(parents=True)
    (source / ".git/HEAD").write_text(commit + "\n", encoding="utf-8")
    (source / "hopper_decisions/maps").mkdir(parents=True)
    (source / "hopper_decisions/__init__.py").write_text("", encoding="utf-8")
    (source / "hopper_decisions/model.py").write_text(FAKE_MODEL, encoding="utf-8")
    (source / "hopper_decisions/maps/hopper.json").write_text(
        files["hopper.json"], encoding="utf-8"
    )
    return model, source


class NativeHopperTest(unittest.TestCase):
    def setUp(self) -> None:
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        patcher = mock.patch.dict(sys.modules, {"torch": fake_torch})
        patcher.start()
        self.addCleanup(patcher.stop)
        path_before = list(sys.path)

        def restore() -> None:
            for name in [n for n in sys.modules if n.startswith("hopper_decisions")]:
                sys.modules.pop(name, None)
            sys.path[:] = path_before

        self.addCleanup(restore)

    def test_one_question_per_request_and_score_mean_for_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, source = fixture(Path(tmp))
            prompts = Path(tmp) / "prompts.jsonl"
            prompts.write_text(
                json.dumps({"id": "1", "state": {"t": "x"}, "questions": QUESTIONS})
                + "\n",
                encoding="utf-8",
            )
            output = Path(tmp) / "out.jsonl"
            native_hopper.collect(
                model_path=model,
                revision=native_hopper.MODEL_REVISION,
                source=source,
                prompts=prompts,
                output=output,
            )
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertAlmostEqual(row["answers"]["s"]["score"], 1.6)
        self.assertEqual(row["slow_kernels"], ["causal_conv1d"])
        for name, value in (("c", "a"), ("n", True), ("s", 2)):
            gold = {"value": value, "label_to_semantic": {"a": "a", "b": "b"}}
            verdict = evaluate_answer(QUESTIONS[name], gold, row["answers"][name])
            self.assertTrue(verdict.get("correct"), (name, verdict))

    def test_rejects_unpinned_source_and_changed_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, source = fixture(Path(tmp), commit="0" * 40)
            with self.assertRaisesRegex(ValueError, "pinned g-1.2.0"):
                native_hopper.verify_release(
                    model, native_hopper.MODEL_REVISION, source
                )
        with tempfile.TemporaryDirectory() as tmp:
            model, source = fixture(Path(tmp))
            (source / "hopper_decisions/maps/hopper.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "calibration map"):
                native_hopper.verify_release(
                    model, native_hopper.MODEL_REVISION, source
                )


if __name__ == "__main__":
    unittest.main()
