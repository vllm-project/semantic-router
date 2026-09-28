"""Padded and one-row micro-batches give the same loss and gradients (CPU, FP32).

Each trainer family is built around a tiny random backbone and its real
`loss` path; a micro-batch mixes lengths and candidate counts so padding
reaches the readout, attention, pooling and loss masks. The negative controls
break padding on purpose and must be caught.
"""

import importlib
import unittest

try:
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
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
parity = importlib.import_module("v2.06b.parity")
mixture = importlib.import_module("v2.06b.mixture")
encoder_tests = importlib.import_module("v2.06b.tests.test_encoder_cpu")

LOSS = {"ce": 1.0, "brier": 0.5, "score_rps": 0.0, "teacher_kl": 1.0}


def char_tokenizer():
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {"[PAD]": 0, "[UNK]": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]"), len(vocab)


def qwen3(vocab, *, seed=0):
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=2048,
    )
    torch.manual_seed(seed)
    return Qwen3Model._from_config(cfg, attn_implementation="sdpa")


def flat_row(index, kind, options, label, state):
    keys = {
        "choice": [f"K{i + 1}" for i in range(options)],
        "noul": ["false", "true"],
        "score": [str(i) for i in range(options)],
    }[kind]
    return {
        "id": f"r{index}",
        "state": state,
        "instructions": f"{kind} question {index}",
        "task_type": kind,
        "family": "unit",
        "label": label,
        "options": [
            {"key": k, "description": f"option {k} text {index}"} for k in keys
        ],
    }


def native_record(row):
    kind = row["task_type"]
    question = {"type": kind.capitalize()}
    if kind == "choice":
        question["options"] = [{"id": o["key"]} for o in row["options"]]
    elif kind == "score":
        question["levels"] = [{"id": o["key"]} for o in row["options"]]
    return {"source_row_id": row["id"], "question": question}


def causal_family(rows):
    from training.model.decision_model import CandidateHead, DecisionModel

    tok, vocab = char_tokenizer()
    family = train.CausalQwenFamily.__new__(train.CausalQwenFamily)
    family.spec = {"loss": LOSS}
    family.device = "cpu"
    family.rows = {row["id"]: row for row in rows}
    family.tokenizer = tok
    torch.manual_seed(1)
    family.model = DecisionModel(
        qwen3(vocab), CandidateHead(32, 8), {"head_variant": "shared"}
    )
    family.model.train()
    return family


def encoder_family(backbone, bidirectional):
    tok, ids = encoder_tests.tokenizer()
    family = train.EncoderFamily.__new__(train.EncoderFamily)
    family.spec = {"loss": LOSS}
    family.device = "cpu"
    torch.manual_seed(1)
    family.model = enc._torch_module()(
        backbone, head_dim=8, bidirectional=bidirectional
    )
    family.model.train()
    family.packer = enc.MarkerPacker(tok, ids)
    return family


def encoder_records():
    out = []
    for i, (kind, words) in enumerate(
        [("choice", 1), ("choice", 9), ("noul", 4), ("score", 12)]
    ):
        record = encoder_tests.record(kind)
        record["id"] = f"e{i}"
        record["state_text"] = " ".join(["the sky is blue"] * words)
        if kind == "choice":
            record["question"]["options"] = record["question"]["options"][: 2 + i % 2]
            record["target"] = {"choice_id": record["question"]["options"][0]["id"]}
        elif kind == "noul":
            record["target"] = {"probability": 1.0}
        else:
            record["target"] = {"probabilities": [0.0, 1.0, 0.0]}
        out.append(record)
    return out


@unittest.skipIf(torch is None, "torch unavailable")
class PaddingParityTest(unittest.TestCase):
    def assert_parity(self, report):
        self.assertLessEqual(
            report["fp32"]["loss_rel_diff"], parity.FP32_LOSS_TOLERANCE
        )
        self.assertLessEqual(
            report["fp32"]["grad_padded_vs_rows"]["rel_err"], parity.FP32_GRAD_TOLERANCE
        )
        self.assertTrue(report["passed"])

    def causal_case(self):
        rows = [
            flat_row(0, "choice", 2, 1, "short state"),
            flat_row(1, "choice", 4, 3, "a much longer state " * 12),
            flat_row(2, "choice", 3, 0, "medium state " * 4),
        ]
        teacher = [[0.3, 0.7], None, [0.2, 0.5, 0.3]]
        return causal_family(rows), [native_record(r) for r in rows], teacher

    def test_causal_endpoint_family(self):
        family, records, teacher = self.causal_case()
        self.assert_parity(
            parity.micro_batch_parity(family, records, teacher, "cpu", bf16=False)
        )

    def test_bidirectional_qwen_marker_family(self):
        family = encoder_family(qwen3(len(encoder_tests.WORDS)), bidirectional=True)
        records = encoder_records()[:2]
        self.assert_parity(
            parity.micro_batch_parity(
                family, records, [[0.6, 0.4], None], "cpu", bf16=False
            )
        )

    def test_modernbert_marker_family(self):
        cfg = ModernBertConfig(
            vocab_size=len(encoder_tests.WORDS),
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
            max_position_embeddings=256,
            global_attn_every_n_layers=1,
        )
        torch.manual_seed(0)
        family = encoder_family(
            ModernBertModel._from_config(cfg, attn_implementation="sdpa"),
            bidirectional=False,
        )
        records = encoder_records()[:2]
        self.assert_parity(
            parity.micro_batch_parity(family, records, [None, None], "cpu", bf16=False)
        )

    def test_dropout_is_zeroed_for_the_check_and_restored(self):
        cfg = ModernBertConfig(
            vocab_size=len(encoder_tests.WORDS),
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
            max_position_embeddings=256,
            global_attn_every_n_layers=1,
            embedding_dropout=0.3,
            mlp_dropout=0.3,
            attention_dropout=0.3,
        )
        torch.manual_seed(0)
        family = encoder_family(
            ModernBertModel._from_config(cfg, attn_implementation="sdpa"),
            bidirectional=False,
        )
        rates = [m.p for m in family.model.modules() if isinstance(m, torch.nn.Dropout)]
        report = parity.micro_batch_parity(
            family, encoder_records()[:2], [None, None], "cpu", bf16=False
        )
        self.assert_parity(report)
        self.assertGreater(report["dropout_modules_zeroed"], 0)
        self.assertEqual(
            rates,
            [m.p for m in family.model.modules() if isinstance(m, torch.nn.Dropout)],
        )
        self.assertTrue(family.model.training)

    def test_ignored_key_padding_mask_is_caught(self):
        family = encoder_family(qwen3(len(encoder_tests.WORDS)), bidirectional=True)
        collate = family.packer.collate

        def leaky(encoded, device="cpu"):
            batch = collate(encoded, device)
            return {**batch, "attention_mask": torch.ones_like(batch["attention_mask"])}

        family.packer.collate = leaky
        report = parity.micro_batch_parity(
            family, encoder_records()[:2], [None, None], "cpu", bf16=False
        )
        self.assertFalse(report["passed"])

    def test_left_padding_with_unshifted_readout_is_caught(self):
        family, records, teacher = self.causal_case()
        batch_of = family.batch

        def left_padded(rows):
            batch = batch_of(rows)
            ids, mask = batch["input_ids"], batch["attention_mask"]
            n = mask.sum(-1)
            width = ids.shape[1]
            order = torch.stack(
                [torch.roll(torch.arange(width), int(width - k)) for k in n]
            )
            return {
                **batch,
                "input_ids": ids.gather(1, order),
                "attention_mask": mask.gather(1, order),
            }

        family.batch = left_padded
        report = parity.micro_batch_parity(family, records, teacher, "cpu", bf16=False)
        self.assertFalse(report["passed"])

    def test_single_row_is_rejected(self):
        family, records, teacher = self.causal_case()
        with self.assertRaises(ValueError):
            parity.micro_batch_parity(family, records[:1], teacher[:1], "cpu")


class MixtureSelectionTest(unittest.TestCase):
    def test_whole_groups_within_share_and_deterministic(self):
        rows = [{"group_id": f"g{i // 3}", "id": f"x{i}"} for i in range(30)]
        tokens = [10 + (i % 7) for i in range(30)]
        chosen = mixture.select_groups(rows, tokens, 120, "seed", "A6g")
        self.assertEqual(
            chosen, mixture.select_groups(rows, tokens, 120, "seed", "A6g")
        )
        self.assertLessEqual(sum(tokens[i] for i in chosen), 120)
        groups = {rows[i]["group_id"] for i in chosen}
        self.assertEqual(
            sorted(chosen),
            sorted(i for i, r in enumerate(rows) if r["group_id"] in groups),
        )
        self.assertNotEqual(
            chosen, mixture.select_groups(rows, tokens, 120, "other", "A6g")
        )

    def test_stratified_resample_is_proportional_and_copies_are_distinct(self):
        rows = [
            {
                "id": f"x{i}",
                "group_id": f"g{i // 2}",
                "source": "s1" if i < 40 else "s2",
                "task_type": "choice",
                "language": "en",
            }
            for i in range(60)
        ]
        tokens = [10] * 60
        chosen = mixture.resample_groups(rows, tokens, 300, "seed")
        self.assertEqual(chosen, mixture.resample_groups(rows, tokens, 300, "seed"))
        by_source = {
            s: sum(rows[i]["source"] == s for i in chosen) for s in ("s1", "s2")
        }
        self.assertEqual(by_source, {"s1": 20, "s2": 10})
        copy = mixture.duplicate(rows[chosen[0]])
        self.assertNotEqual(copy["id"], rows[chosen[0]]["id"])
        self.assertNotEqual(copy["group_id"], rows[chosen[0]]["group_id"])
        self.assertEqual(copy["teacher_source_id"], rows[chosen[0]]["id"])


if __name__ == "__main__":
    unittest.main()
