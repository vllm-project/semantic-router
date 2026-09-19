"""Validate the warmup ratio-to-steps conversion used by the training scripts.

`transformers.TrainingArguments` dropped `warmup_ratio` in 5.15.0, so the ten
scripts that passed it raise `TypeError` on a fresh install. Only `warmup_steps`
survives, and it is an absolute count, so the ratio has to be resolved against
the number of optimizer steps the run will take.

These tests pin the conversion against the arithmetic the Trainer did
internally, because the point of the helper is that a converted script warms up
over the same span it used to — not merely that it stops raising.
"""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from typing import ClassVar

TRAINING_ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = TRAINING_ROOT / "model_classifier" / "common_lora_utils.py"


def _load_helper():
    """Import the helper without importing torch.

    `common_lora_utils` imports torch at module scope for its GPU utilities,
    which this test neither needs nor should require in CI, so only the
    module prologue up to the first unrelated function is executed.
    """
    source = HELPER_SOURCE.read_text(encoding="utf-8")
    prologue = source.split("def get_target_modules_for_model")[0].replace(
        "import torch", ""
    )
    namespace: dict = {}
    exec(compile(prologue, str(HELPER_SOURCE), "exec"), namespace)
    return namespace["warmup_steps_from_ratio"]


def trainer_reference(
    ratio: float,
    examples: int,
    batch_size: int,
    epochs: float,
    grad_accum: int = 1,
    world_size: int = 1,
) -> int:
    """What the Trainer computed for `warmup_ratio`, before it was removed.

    `get_warmup_steps` was `ceil(num_training_steps * warmup_ratio)`, and
    `num_training_steps` came from the dataloader length divided by gradient
    accumulation, times the epoch count.
    """
    steps_per_epoch = math.ceil(examples / (batch_size * world_size))
    updates_per_epoch = max(steps_per_epoch // grad_accum, 1)
    total_steps = math.ceil(epochs * updates_per_epoch)
    return math.ceil(total_steps * ratio)


class WarmupStepsFromRatioTest(unittest.TestCase):
    # Loaded per-class rather than at module scope so that the AST guard below
    # still runs — and still reports the real call sites — on a tree where the
    # helper does not exist yet.
    @classmethod
    def setUpClass(cls) -> None:
        cls.warmup_steps_from_ratio = staticmethod(_load_helper())

    def test_matches_the_trainer_arithmetic(self) -> None:
        """The converted scripts must warm up over the same span as before."""
        cases = [
            # ratio, examples, batch, epochs, grad_accum, world_size
            (0.06, 10_000, 16, 3, 2, 1),  # ft_linear_lora shape
            (0.1, 5_000, 32, 5, 1, 1),  # ft_linear_lora_consistency shape
            (0.06, 1_000, 8, 1, 1, 1),
            (0.1, 50_000, 64, 10, 8, 4),  # multi-device
            (0.06, 7, 4, 2, 1, 1),  # dataset smaller than one full batch
        ]
        for ratio, examples, batch, epochs, accum, world in cases:
            with self.subTest(ratio=ratio, examples=examples, batch=batch):
                self.assertEqual(
                    self.warmup_steps_from_ratio(
                        ratio, examples, batch, epochs, accum, world
                    ),
                    trainer_reference(ratio, examples, batch, epochs, accum, world),
                )

    def test_a_zero_ratio_means_no_warmup(self) -> None:
        self.assertEqual(self.warmup_steps_from_ratio(0.0, 1_000, 8, 3), 0)
        self.assertEqual(self.warmup_steps_from_ratio(-1.0, 1_000, 8, 3), 0)

    def test_a_positive_ratio_never_rounds_down_to_nothing(self) -> None:
        """A tiny ratio asking for zero steps would silently drop warmup."""
        self.assertEqual(self.warmup_steps_from_ratio(1e-9, 10, 8, 1), 1)

    def test_an_empty_dataset_is_rejected_rather_than_divided_by(self) -> None:
        with self.assertRaises(ValueError):
            self.warmup_steps_from_ratio(0.06, 0, 8, 3)

    def test_degenerate_sizes_are_clamped_not_divided_by_zero(self) -> None:
        for batch, accum, world in ((0, 1, 1), (8, 0, 1), (8, 1, 0)):
            with self.subTest(batch=batch, accum=accum, world=world):
                self.assertGreaterEqual(
                    self.warmup_steps_from_ratio(0.06, 1_000, batch, 3, accum, world), 1
                )

    def test_warmup_never_exceeds_the_total_steps_it_is_a_ratio_of(self) -> None:
        for ratio in (0.01, 0.06, 0.1, 0.5, 1.0):
            with self.subTest(ratio=ratio):
                total = trainer_reference(1.0, 10_000, 16, 3, 2, 1)
                self.assertLessEqual(
                    self.warmup_steps_from_ratio(ratio, 10_000, 16, 3, 2, 1), total
                )


class NoScriptStillPassesRemovedArgumentsTest(unittest.TestCase):
    """Pin all ten call sites, so a future edit cannot quietly reintroduce one.

    Matching source text would miss a rename or a reformat, so the check walks
    the AST and looks for the shape: a `TrainingArguments(...)` call carrying a
    keyword `transformers` no longer accepts.
    """

    REMOVED: ClassVar[set[str]] = {"warmup_ratio", "logging_dir"}

    def test_no_training_arguments_call_uses_a_removed_keyword(self) -> None:
        offenders = []
        for path in sorted(TRAINING_ROOT.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if name != "TrainingArguments":
                    continue
                used = {k.arg for k in node.keywords if k.arg in self.REMOVED}
                if used:
                    offenders.append(
                        f"{path.relative_to(TRAINING_ROOT)}:{node.lineno} "
                        f"passes {sorted(used)}"
                    )
        self.assertEqual(
            offenders,
            [],
            "transformers 5.15+ rejects these keywords:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
