"""Contract tests pinning the divisor the evaluation uses for per-sample latency.

Slicing a HuggingFace ``Dataset`` returns a dict of column name -> list of
values rather than a list of rows, so ``len(batch)`` counts columns. The
inference loop used it as the batch size, which made every reported latency
wrong by rows/columns: 5.33x too high for jailbreak at the default batch size,
8x for pii, and too *low* once the batch grew past the column count. Nothing
failed, because a plausible number still came out the other end.

See https://github.com/vllm-project/semantic-router/issues/3642.

``mom_collection_eval`` imports torch, datasets and transformers, so it cannot
be imported by the stdlib-only python3 that ``make test-training-contracts``
runs. ``batch_row_count`` is lifted out of the source with ``ast`` and executed
on its own instead, which needs no third-party module.
"""

from __future__ import annotations

import ast
import unittest
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EVAL_SOURCE = REPOSITORY_ROOT / "src/training/model_eval/mom_collection_eval.py"

# Column names the registry's datasets actually arrive with, so the fixtures
# below are the shapes the loop really sees rather than invented ones.
JAILBREAK_COLUMNS = ["text", "label", "label_text"]
PII_COLUMNS = ["tokens", "labels"]
INTENT_COLUMNS = [
    "question_id",
    "text",
    "options",
    "answer",
    "answer_index",
    "cot_content",
    "category",
    "src",
    "label",
]


def load_module() -> ast.Module:
    """Parse the evaluation script without importing it."""
    return ast.parse(EVAL_SOURCE.read_text(encoding="utf-8"))


def load_function(name: str) -> ast.FunctionDef:
    """Return the top-level function definition named `name`."""
    for node in load_module().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{EVAL_SOURCE} defines no function named {name!r}")


def compile_function(name: str) -> Callable:
    """Execute one function on its own, free of the script's imports."""
    namespace: dict = {}
    exec(
        compile(ast.Module([load_function(name)], []), str(EVAL_SOURCE), "exec"),
        namespace,
    )
    return namespace[name]


def make_batch(columns: list[str], rows: int) -> dict:
    """Build a slice shaped the way `Dataset.__getitem__` returns one."""
    return {column: list(range(rows)) for column in columns}


class BatchRowCountTest(unittest.TestCase):
    """`batch_row_count` has to report rows even when columns look like a count."""

    def setUp(self) -> None:
        self.batch_row_count = compile_function("batch_row_count")

    def test_counts_rows_not_columns(self):
        batch = make_batch(JAILBREAK_COLUMNS, rows=16)
        self.assertEqual(self.batch_row_count(batch), 16)
        self.assertNotEqual(self.batch_row_count(batch), len(batch))

    def test_token_classification_shape(self):
        # The pii path was the worst of them: 2 columns against 16 rows.
        self.assertEqual(self.batch_row_count(make_batch(PII_COLUMNS, rows=16)), 16)

    def test_intent_shape(self):
        self.assertEqual(self.batch_row_count(make_batch(INTENT_COLUMNS, rows=16)), 16)

    def test_short_final_batch(self):
        # A trailing batch is shorter than --batch_size and has to say so, or
        # avg_ms stops being total time over rows evaluated.
        self.assertEqual(self.batch_row_count(make_batch(JAILBREAK_COLUMNS, rows=5)), 5)

    def test_single_row_batch(self):
        self.assertEqual(self.batch_row_count(make_batch(INTENT_COLUMNS, rows=1)), 1)

    def test_batch_smaller_than_column_count(self):
        # Below the column count the old divisor made latency look too low
        # rather than too high, which is why a run cannot be corrected after
        # the fact without knowing the batch size it used.
        self.assertEqual(self.batch_row_count(make_batch(INTENT_COLUMNS, rows=4)), 4)


class LatencyDivisorTest(unittest.TestCase):
    """The loop has to divide by the row count, not by the slice's length."""

    def setUp(self) -> None:
        self.loop = load_function("evaluate_single_model")
        self.extend_call = self.find_latency_extend()

    def find_latency_extend(self) -> ast.Call:
        for node in ast.walk(self.loop):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "extend"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "lats"
            ):
                return node
        raise AssertionError("evaluate_single_model no longer collects latencies")

    def test_latency_does_not_divide_by_the_slice_length(self):
        for node in ast.walk(self.loop):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "batch"
            ):
                self.fail(
                    "len(batch) is the column count of a Dataset slice, not the "
                    "batch size; use batch_row_count(batch)"
                )

    def test_latency_is_spread_over_the_row_count(self):
        source = ast.unparse(self.extend_call)
        rows = {
            target.id
            for node in ast.walk(self.loop)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "batch_row_count"
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue(rows, "evaluate_single_model never calls batch_row_count")
        # Both the divisor and the number of entries appended come from it, so
        # lats holds one entry per row and avg_ms averages over samples.
        self.assertEqual(
            sum(name in rows for name in _names(self.extend_call)),
            2,
            f"expected the row count on both sides of the latency spread: {source}",
        )


def _names(node: ast.AST) -> list[str]:
    return [child.id for child in ast.walk(node) if isinstance(child, ast.Name)]


if __name__ == "__main__":
    unittest.main()
