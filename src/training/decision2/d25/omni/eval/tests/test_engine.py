from __future__ import annotations

import importlib.util
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from d25.omni.eval.engine import Unsupported, VisionCodeReadoutModel, make_batches
from d25.omni.model import checkpoint, tiny
from d25.omni.model.assemble import assemble
from d25.vega.common import decision_format as text_format


def reference_probabilities(ckpt: Path, rows: list[dict]) -> list[list[float]]:
    """Perplexity's DecisionModel forward (one request per pass) with the Vega prompt."""
    from transformers import AutoProcessor, Qwen3_5Model

    decision = checkpoint.read_decision_config(ckpt)
    processor = AutoProcessor.from_pretrained(str(ckpt))
    backbone = Qwen3_5Model.from_pretrained(
        str(ckpt), dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    readout = checkpoint.load_readout(ckpt).float()
    results = []
    for row in rows:
        text = text_format.render(
            processor.tokenizer, row["state"], row["question"], decision["codes"]
        )
        encoded = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden = backbone(**encoded, use_cache=False).last_hidden_state[:, -1]
        logits = (hidden.float() @ readout.T)[0]
        count = len(text_format.options(row["question"])[0])
        masked = logits.masked_fill(torch.arange(255) >= count, -1e9)
        results.append((masked / decision["temperature"]).softmax(-1)[:count].tolist())
    return results


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = tiny.fixtures()
        cls.engine = VisionCodeReadoutModel(cls.paths["init_vega"], device="cpu")
        data = cls.paths["data"]
        cls.images = sorted(str(p) for p in (data / "images").glob("*.png"))
        rng = random.Random(5)
        cls.rows = []
        for index in range(14):
            row = tiny.training_row(
                index,
                rng,
                images=rng.sample(cls.images, index % 4) if index % 2 else [],
            )
            cls.rows.append(
                {
                    "state": row["state"],
                    "question": row["question"],
                    "images": row["images"],
                }
            )
        cls.tmp = Path(tempfile.mkdtemp(prefix="d25-omni-engine-"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_distributions(self) -> None:
        probabilities = self.engine.predict(self.rows)
        self.assertEqual(len(probabilities), len(self.rows))
        for row, values in zip(self.rows, probabilities):
            self.assertEqual(len(values), len(text_format.options(row["question"])[0]))
            self.assertAlmostEqual(sum(values), 1.0, places=5)

    def test_batching_does_not_change_results(self) -> None:
        batched = self.engine.predict(self.rows)
        single = [self.engine.predict([row])[0] for row in self.rows]
        for a, b in zip(batched, single):
            self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-5)

    def test_batches_respect_budget_and_modality(self) -> None:
        plans = self.engine.plan(self.rows)
        for batch in make_batches(plans, token_budget=1500, max_batch_size=3):
            self.assertEqual(len({bool(p.images) for p in batch}), 1)
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(len(batch) == 1 or len(batch) * batch[0].tokens <= 1500)
            self.assertEqual(batch[0].tokens, max(p.tokens for p in batch))

    def test_images_are_used(self) -> None:
        base = {"state": "which picture", "question": {"type": "noul"}}
        none, one, two, swapped = self.engine.predict(
            [
                {**base, "images": []},
                {**base, "images": self.images[:1]},
                {**base, "images": self.images[:2]},
                {**base, "images": self.images[1::-1]},
            ]
        )
        self.assertNotEqual(none, one)
        self.assertNotEqual(two, swapped)

    def test_text_rows_match_reference_forward(self) -> None:
        text_rows = [row for row in self.rows if not row["images"]]
        ours = self.engine.predict(text_rows)
        reference = reference_probabilities(self.paths["init_vega"], text_rows)
        for a, b in zip(ours, reference):
            self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-6)

    def test_rejections_are_reported(self) -> None:
        many = {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}
        rows = [
            {"state": "x", "question": many, "images": []},
            {"state": "x", "question": {"type": "noul"}, "images": self.images[:5]},
            {
                "state": "<|image_pad|>",
                "question": {"type": "noul"},
                "images": self.images[:1],
            },
            {
                "state": "x",
                "question": {"type": "noul"},
                "images": [str(self.tmp / "missing.png")],
            },
            {"state": "x", "question": {"type": "score"}, "images": []},
            {"state": "x " * 3000, "question": {"type": "noul"}, "images": []},
            self.rows[0],
        ]
        outcomes = self.engine.score(rows)
        self.assertEqual(
            [o.status for o in outcomes],
            [
                "unsupported",
                "unsupported",
                "unsupported",
                "error",
                "unsupported",
                "ok",
                "ok",
            ],
        )
        small = VisionCodeReadoutModel(
            self.paths["init_vega"], device="cpu", max_length=512
        )
        self.assertEqual(small.score([rows[5]])[0].status, "unsupported")
        self.assertIn("max_length", small.score([rows[5]])[0].reason)
        with self.assertRaises(Unsupported):
            self.engine.predict(rows[:1])

    def test_relative_images_resolve_against_root(self) -> None:
        data = self.paths["data"]
        relative = {
            "state": "s",
            "question": {"type": "noul"},
            "images": ["images/0.png"],
        }
        absolute = {**relative, "images": [str(data / "images/0.png")]}
        self.assertEqual(
            self.engine.predict([relative], root=data), self.engine.predict([absolute])
        )

    def test_noncausal_checkpoint(self) -> None:
        stock = tiny.write_stock(self.tmp / "stock-8", layers=8)
        assemble(stock, self.tmp / "causal", stock_pin="skip")
        assemble(
            stock,
            self.tmp / "noncausal",
            attention_mode="noncausal_full_attention",
            stock_pin="skip",
        )
        noncausal = VisionCodeReadoutModel(self.tmp / "noncausal", device="cpu")
        causal = VisionCodeReadoutModel(self.tmp / "causal", device="cpu")
        self.assertEqual(noncausal.attention_mode, "noncausal_full_attention")
        a, b = noncausal.predict(self.rows[:4]), causal.predict(self.rows[:4])
        self.assertTrue(
            any(max(abs(x - y) for x, y in zip(p, q)) > 1e-5 for p, q in zip(a, b))
        )
        batched = noncausal.predict(self.rows)
        single = [noncausal.predict([row])[0] for row in self.rows]
        self.assertLess(
            max(max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(batched, single)),
            1e-5,
        )
        with self.assertRaises(ValueError):
            VisionCodeReadoutModel(
                self.tmp / "noncausal", device="cpu", attention_mode="causal"
            )

    def test_checkpoint_without_vision_tower(self) -> None:
        vega = self.paths["vega"]
        text_only = tiny.write_vega(
            self.tmp / "vega-no-vision", self.paths["stock"], vision="missing"
        )
        engine = VisionCodeReadoutModel(text_only, device="cpu")
        self.assertFalse(engine.has_vision)
        statuses = [o.status for o in engine.score(self.rows[:4])]
        self.assertEqual(statuses, ["ok", "unsupported", "ok", "unsupported"])
        self.assertTrue(VisionCodeReadoutModel(vega, device="cpu").has_vision)

    @unittest.skipUnless(
        importlib.util.find_spec("d25.vega.eval") is not None,
        "Vega engine not committed yet",
    )
    def test_text_parity_with_vega_engine(self) -> None:
        from d25.omni.eval.parity import compare
        from d25.vega.eval.engine import CodeReadoutModel

        text_rows = [row for row in self.rows if not row["images"]]
        report = compare(
            self.engine.predict,
            CodeReadoutModel(self.paths["init_vega"], "cpu").predict,
            text_rows,
            1e-5,
        )
        self.assertTrue(report["pass"], report)


if __name__ == "__main__":
    unittest.main()
