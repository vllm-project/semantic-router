"""Forward token budget of the release runtime (``qwen.forward_token_budget`` / ``micro_batches``).

A request's questions run as one padded batch unless that batch would pass the
32-bit element offsets of the FLA gated-delta kernels; then they run as several
batches within the budget, with the same answers. Needs torch; run in the
pinned image. ``gpu_long_request`` checks a synthetic long request end to end
on a real package.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2.release import build
from v2.release.tests.test_bf16_resident import staged_qwen

try:
    import torch
except ImportError:
    torch = None

VENDOR = (
    "calibration.py",
    "decision_model.py",
    "infer.py",
    "lora.py",
    "score_bias.py",
    "source.py",
)


def qwen3_5(heads: int, nested: bool = False):
    text = SimpleNamespace(
        linear_num_value_heads=heads, linear_key_head_dim=128, linear_value_head_dim=128
    )
    return SimpleNamespace(text_config=text) if nested else text


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + ord(c) % 250 for c in text]


class RowwiseModel(torch.nn.Module if torch else object):
    """Logits from each row's own tokens only; records every forward's padded shape."""

    def __init__(self):
        super().__init__()
        self.shapes: list[tuple[int, int]] = []

    def forward(
        self, input_ids, attention_mask, candidate_positions, candidate_mask, **_
    ):
        self.shapes.append(tuple(input_ids.shape))
        values = input_ids.gather(1, candidate_positions).float() / 100
        values = values + attention_mask.sum(1, keepdim=True).float() / 1000
        return values.masked_fill(~candidate_mask, -float("inf"))


def request(long_chars: int, short: int) -> dict:
    questions = {}
    for index in range(short + 1):
        words = "basin cedar " * (long_chars // 12 if index == 0 else 3 + index)
        questions[f"q{index}"] = {
            "type": "choice" if index % 2 else "noul",
            "instructions": {"task": f"pick {index}", "candidate": words},
            "criteria": (
                {"a": f"left {index}", "b": "right", "c": None}
                if index % 2
                else {"false": "no", "true": "yes"}
            ),
        }
    return {"state": {"note": "harbor"}, "questions": questions}


class BudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.qwen = staged_qwen(Path(cls.scratch.name))

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]
        cls.scratch.cleanup()

    def test_budget_follows_value_heads(self):
        budget = self.qwen.forward_token_budget
        self.assertEqual(budget(qwen3_5(48)), 349_525)
        self.assertEqual(budget(qwen3_5(32)), 524_287)
        self.assertEqual(budget(qwen3_5(16, nested=True)), 1_048_575)
        self.assertIsNone(budget(SimpleNamespace(hidden_size=1024)))

    def test_batches_split_only_past_the_budget(self):
        split = self.qwen.micro_batches
        self.assertEqual(split([100, 7, 50], None), [[0, 1, 2]])
        self.assertEqual(split([100, 7, 50], 312), [[0, 1, 2]])
        lengths = [14_223] + [300 + i for i in range(31)]
        groups = split(lengths, 349_525)
        self.assertEqual(groups, [[0] + list(range(9, 32)), list(range(1, 9))])
        for group in split(lengths, 20_000):
            padded = -(-max(lengths[i] for i in group) // 8) * 8
            self.assertLessEqual(padded * len(group), 20_000)
            self.assertEqual(group, sorted(group))
        self.assertEqual(
            sorted(i for g in split(lengths, 20_000) for i in g), list(range(32))
        )
        with self.assertRaisesRegex(ValueError, "single question"):
            split([20_001, 3], 20_000)


@unittest.skipIf(torch is None, "needs torch")
class SystemOneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        scratch = Path(cls.scratch.name)
        cls.qwen = staged_qwen(scratch)
        vendor = scratch / "decision2/_vendor/dev2model"
        for name in VENDOR:
            shutil.copyfile(build.SOURCE_ROOT / "training/model" / name, vendor / name)

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]
        cls.scratch.cleanup()

    def backend(self, budget):
        return self.qwen.QwenDecision(
            RowwiseModel(),
            CharTokenizer(),
            torch.device("cpu"),
            {"choice": 1.0, "noul": 1.0, "score": 1.0},
            100_000,
            torch,
            batch_tokens=budget,
        )

    def test_split_request_gives_the_same_answers(self):
        row = request(long_chars=1_200, short=11)
        whole = self.backend(None)
        split = self.backend(4_000)
        expected, tokens = whole.system_one(row["state"], row["questions"])
        answers, split_tokens = split.system_one(row["state"], row["questions"])
        self.assertEqual(answers, expected)
        self.assertEqual(list(answers), list(row["questions"]))
        self.assertEqual(split_tokens, tokens)
        self.assertEqual(len(whole.model.shapes), 1)
        self.assertGreater(len(split.model.shapes), 1)
        self.assertTrue(
            all(rows * width <= 4_000 for rows, width in split.model.shapes)
        )

    def test_request_within_the_budget_is_one_batch(self):
        row = request(long_chars=120, short=3)
        backend = self.backend(10_000)
        backend.system_one(row["state"], row["questions"])
        self.assertEqual(len(backend.model.shapes), 1)
        self.assertEqual(backend.model.shapes[0][0], 4)


if __name__ == "__main__":
    unittest.main()
