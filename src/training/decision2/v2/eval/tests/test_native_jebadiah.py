from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from v2.eval import native_jebadiah
from v2.eval.same_panel import input_digest

FAKE_MODEL = """
from types import SimpleNamespace

def load_tokenizer(path):
    return SimpleNamespace(chat_template="tpl")

def template_sha256(tok):
    return "tpl-sha"

def load_base(path, dtype=None, device=None):
    return object()

def read_temperatures(path):
    return {"choice": 1.2, "noul": 1.3, "score": 0.76}

class Scorer:
    def __init__(self, model, tok, temperatures=None, device=None):
        pass

    def render(self, state, q):
        if len(q.get("criteria", {})) > 3:
            raise ValueError("4 options exceed the 3 single-token labels")
        if state == "boom":
            raise ValueError("unknown question type")
        keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
        return SimpleNamespace(keys=keys, truncated=state == "long")

    def score_rendered(self, items):
        return [[0.25, 0.75] for _ in items]
"""

FAKE_PROMPT = """
def answer_from_probs(q, keys, probs):
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs[keys.index("true")]}
    return {"type": "choice", "choice": keys[1], "probabilities": dict(zip(keys, probs))}
"""

QUESTIONS = {
    "c": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}},
    "n": {"type": "noul", "instructions": "?", "criteria": {"true": "y", "false": "n"}},
    "wide": {"type": "choice", "instructions": "?", "criteria": {k: k for k in "wxyz"}},
}


def release(root: Path, revision: str = native_jebadiah.MODEL_REVISION) -> Path:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/jebadiah_model.py").write_text(FAKE_MODEL, encoding="utf-8")
    (root / "scripts/jebadiah_prompt.py").write_text(FAKE_PROMPT, encoding="utf-8")
    (root / "scripts/ainode_prompt_verbatim.py").write_text("", encoding="utf-8")
    (root / "temperatures.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("tpl", encoding="utf-8")
    (root / "prompt_contract.json").write_text(
        json.dumps({"chat_template_sha256": "tpl-sha"}), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-1.safetensors"}}), encoding="utf-8"
    )
    (root / "model-1.safetensors").write_text("weights", encoding="utf-8")
    meta = root / ".cache/huggingface/download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(f"{revision}\netag\n", encoding="utf-8")
    return root


class NativeJebadiahTest(unittest.TestCase):
    def setUp(self) -> None:
        fake_torch = types.SimpleNamespace(
            bfloat16="bf16", cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        patcher = mock.patch.dict(sys.modules, {"torch": fake_torch})
        patcher.start()
        self.addCleanup(patcher.stop)
        release_modules = ("jebadiah_model", "jebadiah_prompt")
        saved = {n: sys.modules.pop(n) for n in release_modules if n in sys.modules}
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
        root = release(Path(tmp) / "model")
        rows = [
            {"id": str(i), "state": s, "questions": QUESTIONS}
            for i, s in enumerate(states)
        ]
        prompts = Path(tmp) / "prompts.jsonl"
        prompts.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        output = Path(tmp) / "out.jsonl"
        summary = native_jebadiah.collect(
            model_path=root,
            revision=native_jebadiah.MODEL_REVISION,
            prompts=prompts,
            output=output,
        )
        got = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        return summary, rows, got

    def test_cut_prompts_and_label_overflow_are_invalid_not_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, got = self.run_collect(tmp, [{"ticket": "x"}, "long"])
        self.assertEqual(summary["invalid"], {"over_budget": 2, "native_rejection": 2})
        first = got[0]["answers"]
        self.assertEqual(first["c"]["choice"], "b")
        self.assertEqual(first["n"]["noul"], 0.75)
        self.assertEqual(first["wide"]["invalid_reason"], "native_rejection")
        self.assertEqual(got[1]["answers"]["c"]["invalid_reason"], "over_budget")
        self.assertEqual(list(got[0]["answers"]), list(QUESTIONS))
        self.assertEqual(
            got[0]["source_input_sha256"], input_digest(rows[0]["state"], QUESTIONS)
        )

    def test_other_native_errors_stop_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown question type"):
                self.run_collect(tmp, ["boom"])

    def test_requires_all_weight_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = release(Path(tmp))
            (root / "model-1.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "weights incomplete"):
                native_jebadiah.verify_release(root, native_jebadiah.MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
