"""CPU contracts for the encoder family; skipped where torch is unavailable."""

import importlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        ModernBertConfig,
        ModernBertModel,
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3Model,
    )
except ImportError:  # pragma: no cover - local environments without torch
    torch = None

enc = importlib.import_module("v2.06b.encoder")
train = importlib.import_module("v2.06b.train")

WORDS = [
    "[PAD]",
    "[BOS]",
    "[SEP]",
    "[MASK]",
    "[UNK]",
    "choice",
    "noul",
    "score",
    "question:",
    "level",
    "0:",
    "1:",
    "2:",
    "pick",
    "red",
    "blue",
    "green",
    "the",
    "sky",
    "is",
    "No.",
    "Yes.",
    "statement",
    "or",
    "not",
    "satisfied.",
    "The",
]


def tokenizer():
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok), {
        "bos": 1,
        "sep": 2,
        "marker": 3,
        "pad": 0,
    }


def record(kind="choice"):
    q = {"id": "q", "type": kind.capitalize(), "text": "pick"}
    if kind == "choice":
        q["options"] = [
            {"id": "r", "text": "red"},
            {"id": "b", "text": "blue"},
            {"id": "g", "text": "green"},
        ]
    elif kind == "score":
        q["levels"] = [
            {"id": str(i), "value": i, "text": w}
            for i, w in enumerate(["red", "blue", "green"])
        ]
    return {"id": "x", "state_text": "the sky is blue", "question": q}


@unittest.skipIf(torch is None, "torch unavailable")
class PackerTest(unittest.TestCase):
    def test_kai_layout_markers_and_no_truncation(self):
        tok, ids = tokenizer()
        packer = enc.MarkerPacker(tok, ids, max_length=64)
        e = packer.encode(record("choice"))
        self.assertEqual(e["ids"][0], 1)
        self.assertEqual([e["ids"][p] for p in e["positions"]], [3, 3, 3])
        self.assertEqual(e["candidate_ids"], ["r", "b", "g"])
        self.assertEqual(e["ids"][-1], 2)
        with self.assertRaises(ValueError):
            enc.MarkerPacker(tok, ids, max_length=10).encode(record("choice"))

    def test_noul_defaults_and_score_levels(self):
        tok, ids = tokenizer()
        packer = enc.MarkerPacker(tok, ids)
        self.assertEqual(packer.encode(record("noul"))["candidate_ids"], ["no", "yes"])
        score = packer.encode(record("score"))
        self.assertEqual(score["values"], [0.0, 1.0, 2.0])
        self.assertEqual(score["kind"], "score")


@unittest.skipIf(torch is None, "torch unavailable")
class ModelTest(unittest.TestCase):
    def build(self):
        tok, ids = tokenizer()
        cfg = ModernBertConfig(
            vocab_size=len(WORDS),
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
            max_position_embeddings=128,
            global_attn_every_n_layers=1,
        )
        torch.manual_seed(0)
        # The ROCm image ships a Triton flash-attention build that cannot run on CPU tensors.
        backbone = ModernBertModel._from_config(cfg, attn_implementation="sdpa")
        model = enc._torch_module()(backbone, head_dim=8)
        return model.eval(), enc.MarkerPacker(tok, ids)

    def test_masked_candidates_and_padding_invariance(self):
        model, packer = self.build()
        short = packer.encode(record("noul"))
        long = packer.encode(record("choice"))
        with torch.no_grad():
            both = model(packer.collate([short, long]))
            alone = model(packer.collate([short]))
        self.assertEqual(both.shape, (2, 3))
        self.assertEqual(float(both[0, 2]), torch.finfo(torch.float32).min)
        self.assertTrue(
            torch.allclose(both[0, :2].softmax(-1), alone[0].softmax(-1), atol=1e-5)
        )

    def test_span_mean_pools_marker_and_description_only(self):
        _, packer = self.build()
        e = packer.encode(record("choice"))
        batch = packer.collate([e])
        hidden = torch.randn(1, len(e["ids"]), 4)
        pooled = enc.pool_candidates(hidden, batch, "span-mean")
        start, end = e["spans"][1]
        self.assertEqual(e["ids"][start], 3)
        self.assertTrue(torch.allclose(pooled[0, 1], hidden[0, start:end].mean(0)))
        marker = enc.pool_candidates(hidden, batch, "marker")
        self.assertTrue(torch.equal(marker[0, 1], hidden[0, e["positions"][1]]))

    def test_loss_terms(self):
        logits = torch.tensor([[2.0, 0.0, torch.finfo(torch.float32).min]])
        targets = torch.tensor([[1.0, 0.0, 0.0]])
        valid = torch.tensor([[1.0, 1.0, 0.0]])
        extra, brier, kl = train.extra_terms(
            logits, targets, valid, targets.clone(), 0.5, 1.0
        )
        p = torch.tensor([2.0, 0.0]).softmax(-1)
        self.assertAlmostEqual(
            float(brier[0]), float((p[0] - 1) ** 2 + p[1] ** 2), places=6
        )
        self.assertAlmostEqual(float(kl[0]), float(-torch.log(p[0])), places=6)
        self.assertAlmostEqual(
            float(extra[0]), 0.5 * float(brier[0]) + float(kl[0]), places=6
        )

    def test_micro_batches_are_type_homogeneous_and_budgeted(self):
        kinds = ["choice", "noul", "choice", "score", "choice"]
        lengths = [100, 50, 400, 30, 300]
        batches = train.micro_batches(
            list(range(5)), kinds, lengths, max_rows=8, budget=700
        )
        self.assertEqual(sorted(i for b in batches for i in b), list(range(5)))
        for b in batches:
            self.assertEqual(len({kinds[i] for i in b}), 1)
            self.assertLessEqual(len(b) * max(lengths[i] for i in b), 800)
        self.assertEqual(batches[0], [2])

    def test_target_vector(self):
        noul = {"question": {"type": "Noul"}, "target": {"probability": 1.0}}
        self.assertEqual(train.target_vector(noul, ["no", "yes"]), [0.0, 1.0])
        choice = {"question": {"type": "Choice"}, "target": {"choice_id": "b"}}
        self.assertEqual(train.target_vector(choice, ["a", "b"]), [0.0, 1.0])


@unittest.skipIf(torch is None, "torch unavailable")
class BidirectionalAndOrdinalTest(unittest.TestCase):
    def qwen(self, bidirectional, ordinal=False):
        cfg = Qwen3Config(
            vocab_size=len(WORDS),
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=128,
        )
        torch.manual_seed(0)
        backbone = Qwen3Model._from_config(cfg, attn_implementation="sdpa")
        torch.manual_seed(1)
        return enc._torch_module()(
            backbone, head_dim=8, bidirectional=bidirectional, ordinal_score=ordinal
        ).eval()

    def batches(self):
        tok, ids = tokenizer()
        packer = enc.MarkerPacker(tok, ids)
        a = record("choice")
        b = record("choice")
        b["state_text"] = "the sky is green"
        return packer.collate([packer.encode(a)]), packer.collate([packer.encode(b)])

    def test_state_after_markers_reaches_candidates_only_when_bidirectional(self):
        a, b = self.batches()
        with torch.no_grad():
            causal = self.qwen(False)
            self.assertTrue(torch.allclose(causal(a), causal(b)))
            bidi = self.qwen(True)
            self.assertFalse(torch.allclose(bidi(a), bidi(b)))

    def test_ordinal_readout_starts_as_the_plain_head(self):
        tok, ids = tokenizer()
        packer = enc.MarkerPacker(tok, ids)
        batch = packer.collate([packer.encode(record("score"))])
        with torch.no_grad():
            self.assertTrue(
                torch.equal(
                    self.qwen(True)(batch), self.qwen(True, ordinal=True)(batch)
                )
            )


if __name__ == "__main__":
    unittest.main()
