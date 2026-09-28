from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from v2.eval import native_jet
from v2.eval.same_panel import input_digest

FAKE_JET = """
from inference import MARKER

class Jet:
    def __init__(self, path):
        assert MARKER == "release"

    def decide(self, state, questions):
        if state == "long":
            raise ValueError("q: complete prompt exceeds 16384 tokens; no truncation applied")
        return {"answers": {name: {"type": q["type"], "yes": 0.9} for name, q in questions.items()}}
"""


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
        for name in ("jet", "inference", "format", "runtime"):
            sys.modules.pop(name, None)
        self.addCleanup(
            lambda: [
                sys.modules.pop(n, None)
                for n in ("jet", "inference", "format", "runtime")
            ]
        )
        self.path_before = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.path_before))

    def test_collects_answers_and_records_native_rejection_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            release(root)
            rows = [
                {
                    "id": "a",
                    "state": "short",
                    "questions": {"q": {"type": "noul", "instructions": "?"}},
                },
                {
                    "id": "b",
                    "state": "long",
                    "questions": {"q": {"type": "noul", "instructions": "?"}},
                },
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
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(summary["native_rejections"], 1)
        self.assertEqual(summary["verified_files"], 5)
        self.assertEqual(got[0]["answers"]["q"]["yes"], 0.9)
        self.assertEqual(got[1]["answers"]["q"]["invalid_reason"], "native_rejection")
        self.assertEqual(
            got[1]["source_input_sha256"],
            input_digest(rows[1]["state"], rows[1]["questions"]),
        )

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
