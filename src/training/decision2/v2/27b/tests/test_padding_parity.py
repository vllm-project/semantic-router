"""Padded vs unpadded parity for the pinned BEST368 decision model (CPU, FP32).

Runs inside the pinned image (Torch + Transformers with Qwen3.5); skipped where
they are missing. A tiny random Qwen3.5 text backbone (gated-delta and full
attention layers) with the pinned ``CandidateHead`` must give the same logits,
loss and gradients for a row alone and the same row right-padded next to a
longer one, which covers readout indices, padding side and attention masks.
"""

import hashlib
import importlib
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

PINNED = (
    Path(__file__).resolve().parents[1] / "pinned" / "best368-pipeline-2026-09-26.tar"
)


def _pinned_modules(root: Path):
    with tarfile.open(PINNED) as archive:
        archive.extractall(root, filter="data")
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    manifest = json.loads(
        PINNED.with_name("best368-pipeline-2026-09-26.manifest.json").read_text()
    )
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"pinned file differs: {name}")
    saved = {
        k: v
        for k, v in sys.modules.items()
        if k == "training" or k.startswith("training.")
    }
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        model = importlib.import_module("training.model.decision_model")
        loss = importlib.import_module("training.model.loss")
    finally:
        sys.path.remove(str(root))
        for key in [
            k for k in sys.modules if k == "training" or k.startswith("training.")
        ]:
            del sys.modules[key]
        sys.modules.update(saved)
    return model, loss


class PinnedPaddingParityTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5TextConfig,
            )
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        except ImportError as exc:
            self.skipTest(f"needs the pinned image runtime: {exc}")
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dm, self.loss = _pinned_modules(Path(self.tmp.name))
        torch.manual_seed(0)
        config = Qwen3_5TextConfig(
            vocab_size=97,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        config._attn_implementation = "sdpa"
        backbone = Qwen3_5TextModel(config).float().eval()
        self.model = self.dm.DecisionModel(
            backbone, self.dm.CandidateHead(64, 16), {}
        ).float()

    def tearDown(self):
        self.tmp.cleanup()

    def item(self, length, endpoints, label, kind="choice"):
        generator = self.torch.Generator().manual_seed(length)
        ids = self.torch.randint(1, 97, (length,), generator=generator).tolist()
        keys = (
            [str(i) for i in range(len(endpoints))]
            if kind == "score"
            else [f"k{i}" for i in range(len(endpoints))]
        )
        return {
            "id": f"r{length}",
            "ids": ids,
            "candidate_positions": endpoints,
            "query_position": length - 1,
            "label": label,
            "keys": keys,
            "task_type": kind,
            "family": "f",
            "teacher_probs": None,
        }

    def run_batch(self, items):
        torch = self.torch
        batch = self.dm.collate(items, 0)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(**batch)
        terms = self.loss.per_example_loss(
            logits,
            batch["labels"],
            batch["candidate_mask"],
            objective="ce_brier",
            brier_weight=0.5,
            teacher_probs=batch["teacher_probs"],
            replay_mask=batch["replay_mask"],
            replay_kl_weight=0.0,
        )
        return logits, terms["total"], torch

    def test_padded_row_matches_row_alone(self):
        short = self.item(13, [3, 7, 10], 1)
        long = self.item(29, [5, 9, 14, 20, 25], 3, kind="score")
        alone_logits, alone_loss, torch = self.run_batch([short])
        alone_loss.sum().backward()
        alone_grads = {
            n: p.grad.detach().clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }
        batch_logits, batch_loss, _ = self.run_batch([short, long])
        batch_loss[0].backward()
        torch.testing.assert_close(
            batch_logits[0, :3], alone_logits[0, :3], rtol=1e-5, atol=1e-5
        )
        self.assertTrue(torch.isinf(batch_logits[0, 3:]).all())
        torch.testing.assert_close(batch_loss[0], alone_loss[0], rtol=1e-5, atol=1e-6)
        for name, parameter in self.model.named_parameters():
            if name in alone_grads:
                torch.testing.assert_close(
                    parameter.grad, alone_grads[name], rtol=1e-4, atol=1e-6, msg=name
                )


if __name__ == "__main__":
    unittest.main()
