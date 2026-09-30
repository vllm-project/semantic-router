"""The MoE training pipeline is BEST368 plus MoE backbones, with the dense Qwen path unchanged."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PINNED = HERE.parent / "pinned"
MOE_TAR = HERE / "pinned" / "moe-pipeline-2026-10-01.tar"
MOE_MANIFEST = HERE / "pinned" / "moe-pipeline-2026-10-01.manifest.json"
BEST_TAR = PINNED / "best368-pipeline-2026-09-26.tar"
BEST_MANIFEST = PINNED / "best368-pipeline-2026-09-26.manifest.json"
CHANGED = [
    "training/model/calibrate.py",
    "training/model/decision_model.py",
    "training/model/infer.py",
    "training/model/lora.py",
    "training/model/train.py",
]

PROBE = r"""
import json, sys
from torch import nn
from training.model.decision_model import encode
from training.model.lora import select_target_modules

class Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 + 3 for c in text]

class Config:
    layer_types = ["linear_attention", "full_attention"]
    num_hidden_layers = 2
    model_type = "qwen3_5_text"

def block(kind):
    b = nn.Module()
    names = ("q_proj", "k_proj", "v_proj", "o_proj") if kind == "full_attention" else (
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    attn = nn.Module()
    for n in names:
        setattr(attn, n, nn.Linear(2, 2))
    setattr(b, "self_attn" if kind == "full_attention" else "linear_attn", attn)
    b.mlp = nn.Module()
    for n in ("gate_proj", "up_proj", "down_proj"):
        setattr(b.mlp, n, nn.Linear(2, 2))
    return b

text = nn.Module()
text.config = Config()
text.layers = nn.ModuleList([block(k) for k in Config.layer_types])
row = {"id": "r", "state": {"s": 1}, "task_type": "noul", "instructions": "q",
       "options": [{"key": "true", "description": "yes"}, {"key": "false", "description": "no"}],
       "label": 0, "family": "f"}
print(json.dumps({"targets": select_target_modules(text), "encoded": encode(row, Tok(), 4096)}, sort_keys=True))
"""


def extract(tar: Path, root: Path) -> Path:
    with tarfile.open(tar) as archive:
        archive.extractall(root, filter="data")
    return root


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MoEPipeline(unittest.TestCase):
    def test_manifest_matches_tar_and_only_known_files_changed(self):
        manifest = json.loads(MOE_MANIFEST.read_text(encoding="utf-8"))
        best = json.loads(BEST_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["tar_sha256"], sha(MOE_TAR))
        self.assertEqual(manifest["derived_from"]["tar_sha256"], best["tar_sha256"])
        self.assertEqual(manifest["changed_files"], CHANGED)
        with tempfile.TemporaryDirectory() as tmp:
            root = extract(MOE_TAR, Path(tmp))
            actual = {
                str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))
            }
        self.assertEqual(actual, manifest["files_sha256"])
        self.assertEqual(set(actual), set(best["files_sha256"]))
        for name, digest in actual.items():
            if name not in CHANGED:
                self.assertEqual(digest, best["files_sha256"][name], name)

    def test_dense_qwen_targets_and_prompts_equal_best368(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is unavailable")
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            for tar in (BEST_TAR, MOE_TAR):
                root = extract(tar, Path(tmp) / tar.stem)
                result = subprocess.run(
                    [sys.executable, "-c", PROBE],
                    cwd=root,
                    env={"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0]["targets"]), 3 + 5 + 3 + 4)


if __name__ == "__main__":
    unittest.main()
