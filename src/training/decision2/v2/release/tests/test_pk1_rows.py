"""Row-for-row pk1 check (stdlib, synthetic rows)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release import pk1_rows
from v2.release.layout import canonical, sha_file


def row(
    i: int, family: str, keys: tuple[str, ...] = ("a", "b"), label: str = "a"
) -> dict:
    options = [{"key": k, "description": f"option {k}"} for k in keys]
    return {
        "id": f"r{i}",
        "family": family,
        "input_sha256": f"{i:064x}",
        "options": options,
        "label": label,
        "state": f"state {i}",
    }


def lines(rows: list[dict]) -> list[tuple[bytes, dict]]:
    return [(canonical(r).encode(), json.loads(canonical(r))) for r in rows]


class Pk1RowsTest(unittest.TestCase):
    def setUp(self):
        self.original = [
            row(1, "x"),
            row(2, "x", keys=("result_2", "result_1"), label="result_1"),
            row(3, "excluded"),
            row(4, "y"),
        ]
        renumbered = row(2, "x", keys=("result_1", "result_2"), label="result_2")
        self.published = [
            self.original[0],
            renumbered,
            self.original[2],
            self.original[3],
        ]
        self.mixture = [self.original[0], renumbered, self.original[3], row(9, "other")]

    def check(self, mixture, original=True):
        return pk1_rows.compare(
            lines(mixture),
            lines(self.published),
            {"excluded"},
            lines(self.original) if original else None,
        )

    def test_identical_prefix_passes(self):
        result = self.check(self.mixture)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["published_kept_rows"], 3)
        self.assertEqual(result["excluded_family_rows"], {"excluded": 1})
        self.assertEqual(result["original"]["kept_rows_changed_by_renumbering"], 1)
        self.assertEqual(
            result["kept_lines_sha256"], result["mixture_prefix_lines_sha256"]
        )

    def test_changed_label_fails_and_names_the_field(self):
        mixture = list(self.mixture)
        mixture[2] = {**mixture[2], "label": "b"}
        result = self.check(mixture)
        self.assertFalse(result["passed"])
        self.assertEqual(result["differing_fields"], {"label": 1})

    def test_unrenumbered_row_fails(self):
        mixture = list(self.mixture)
        mixture[1] = self.original[1]
        result = self.check(mixture)
        self.assertFalse(result["passed"])
        self.assertEqual(result["original"]["mixture_rows_still_in_original_form"], 1)

    def test_order_and_excluded_rows_fail(self):
        swapped = [self.mixture[1], self.mixture[0], *self.mixture[2:]]
        self.assertFalse(self.check(swapped)["kept_rows_form_mixture_prefix_in_order"])
        with_excluded = [*self.mixture, self.original[2]]
        result = self.check(with_excluded, original=False)
        self.assertEqual(result["excluded_ids_in_mixture"], 1)
        self.assertFalse(result["passed"])

    def test_repeated_input_under_new_id_fails(self):
        copy = {**self.original[0], "id": "r1#copy"}
        result = self.check([*self.mixture, copy])
        self.assertEqual(result["other_mixture_rows_repeating_a_kept_input"], 1)
        self.assertFalse(result["passed"])

    def test_pinned_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "rows.jsonl"
            path.write_text("".join(canonical(r) + "\n" for r in self.published))
            self.assertEqual(len(pk1_rows.read_lines(path, sha_file(path))), 4)
            with self.assertRaises(ValueError):
                pk1_rows.read_lines(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
