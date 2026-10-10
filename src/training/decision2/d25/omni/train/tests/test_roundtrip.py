from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from d25.omni.eval.engine import VisionCodeReadoutModel
from d25.omni.model import checkpoint, inputs, tiny
from d25.omni.train import arms
from d25.omni.train.collator import MultimodalCollator
from d25.omni.train.loss import mask_logits, row_losses, step_loss
from d25.omni.train.model import OmniDecisionModel, init_vision_digests, save_checkpoint
from d25.omni.train.rows import measure, read_rows


class RoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        cls.paths = tiny.fixtures()
        cls.init = cls.paths["init_vega"]
        cls.decision = checkpoint.read_decision_config(cls.init)
        processor = inputs.setup_processor(AutoProcessor.from_pretrained(str(cls.init)))
        cls.collator = MultimodalCollator(processor, cls.decision["codes"])
        rows, _ = read_rows([cls.paths["data"] / "mm.jsonl"], "main")
        for row, (tokens, _) in zip(
            rows, measure(rows, processor, cls.decision["codes"], "d25-vega", 16_384)
        ):
            row.tokens = tokens
        cls.rows = rows[:10]
        cls.tmp = Path(tempfile.mkdtemp(prefix="d25-omni-roundtrip-"))
        scales = arms.ARMS["O-graft-frozen"].scales()
        cls.model = OmniDecisionModel.from_checkpoint(cls.init)
        arms.apply(cls.model, scales)
        optimizer = torch.optim.AdamW(
            arms.param_groups(cls.model, scales, 0.01), lr=1e-2
        )
        batch = cls.collator(cls.rows[:6])
        loss = step_loss(
            row_losses(cls.model(**batch.inputs), batch.targets, batch.counts),
            batch.weights,
            6.0,
            1,
        )
        loss.backward()
        optimizer.step()
        cls.model.eval()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def probabilities(self, model) -> list[list[float]]:
        with torch.no_grad():
            batch = self.collator(self.rows)
            logits = mask_logits(model(**batch.inputs), batch.counts)
            values = (logits / self.decision["temperature"]).softmax(-1)
        return [
            row[:count].tolist() for row, count in zip(values, batch.counts.tolist())
        ]

    def engine_probabilities(self, directory: Path) -> list[list[float]]:
        engine = VisionCodeReadoutModel(directory, device="cpu")
        return engine.predict(
            [
                {"state": r.state, "question": r.question, "images": r.images}
                for r in self.rows
            ]
        )

    def assert_close(self, a, b, tol: float) -> None:
        self.assertLess(
            max(max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(a, b)), tol
        )

    def test_fp32_save_matches_training_model(self) -> None:
        out = self.tmp / "fp32"
        save_checkpoint(
            out,
            dict(self.model.state_dict()),
            self.init,
            {"attention_mode": "causal"},
            {"run": "roundtrip"},
            dtype=torch.float32,
        )
        self.assert_close(
            self.engine_probabilities(out), self.probabilities(self.model), 1e-5
        )

    def test_bf16_save_layout_provenance_and_frozen_vision(self) -> None:
        out = self.tmp / "bf16"
        decision, digests = save_checkpoint(
            out,
            dict(self.model.state_dict()),
            self.init,
            {"attention_mode": "causal"},
            {"run": "roundtrip", "step": 1},
        )
        reloaded = OmniDecisionModel.from_checkpoint(out).eval()
        self.assert_close(
            self.engine_probabilities(out), self.probabilities(reloaded), 1e-5
        )
        initial = init_vision_digests(self.init)
        encoder = [n for n in initial if checkpoint.component(n) == "vision_encoder"]
        merger = [n for n in initial if checkpoint.component(n) == "vision_merger"]
        self.assertTrue(all(digests[n] == initial[n] for n in encoder))
        self.assertTrue(any(digests[n] != initial[n] for n in merger))
        saved = checkpoint.read_decision_config(out)
        self.assertEqual(saved["provenance"]["run"], "roundtrip")
        self.assertEqual(saved["format_id"], "d25-omni-code-readout-v1")
        self.assertEqual(saved["codes"], self.decision["codes"])
        for name, sha in saved["provenance"]["shards_sha256"].items():
            self.assertEqual(checkpoint.file_sha256(out / name), sha)
        names = set(checkpoint.weight_files(out))
        self.assertTrue(any(n.startswith("visual.blocks.") for n in names))
        self.assertTrue(
            all(n.startswith(("visual.", "language_model.")) for n in names)
        )
        self.assertTrue((out / "processor_config.json").exists())
        self.assertEqual(
            decision["provenance"]["readout_sha256"],
            checkpoint.file_sha256(out / "readout.safetensors"),
        )
        with self.assertRaises(FileExistsError):
            save_checkpoint(out, dict(self.model.state_dict()), self.init, {}, {})


if __name__ == "__main__":
    unittest.main()
