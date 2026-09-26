from __future__ import annotations

import hashlib
import unittest
from unittest import mock

from transfer import build as transfer

from training.data import build_budget_subsets as budget
from training.data import build_pilot as pilot


def row(row_id: str, source: str, family: str, group_id: str, label: str = "A") -> dict:
    options = [
        {"key": "A", "description": "first"},
        {"key": "B", "description": "second"},
    ]
    if source == "css_flute_official_train":
        options = [
            {"key": "Idiom", "description": "idiom"},
            {"key": "Sarcasm", "description": "sarcasm"},
        ]
    return {
        "id": row_id,
        "source": source,
        "family": family,
        "group_id": group_id,
        "task_type": "choice",
        "language": "en",
        "options": options,
        "label": next(i for i, option in enumerate(options) if option["key"] == label),
        "state": f"context {row_id}",
        "input_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
    }


class BudgetSubsetTests(unittest.TestCase):
    def test_nested_stratified_subsets_preserve_complete_groups(self) -> None:
        legacy = [
            row(f"legacy-{i}", "legacy:one", "natural", f"g-{i}") for i in range(8)
        ]
        legacy += [
            row("legacy-8", "legacy:two", "synthetic", "paired"),
            row("legacy-9", "legacy:two", "synthetic", "paired"),
        ]
        program = [
            row(
                f"program-{i}",
                "decision2_programmatic_original_v1",
                "pilot_string_composition",
                f"p-{i}",
            )
            for i in range(2)
        ]
        flute = [
            row(
                "flute-0",
                "css_flute_official_train",
                "css_flute_figurative_type",
                "f-0",
                "Idiom",
            ),
            row(
                "flute-1",
                "css_flute_official_train",
                "css_flute_figurative_type",
                "f-1",
                "Sarcasm",
            ),
        ]
        rows = legacy + program + flute
        with mock.patch.object(
            budget, "SOURCE_COUNTS", {"legacy": 10, "programmatic": 2, "flute": 2}
        ):
            subsets, quotas = budget.nested_subsets(rows, "fixed", (7, 10))
            reordered, _ = budget.nested_subsets(list(reversed(rows)), "fixed", (7, 10))
        self.assertEqual(
            {size: sorted(x["id"] for x in part) for size, part in subsets.items()},
            {size: sorted(x["id"] for x in part) for size, part in reordered.items()},
        )
        self.assertEqual(
            {size: len(part) for size, part in subsets.items()}, {7: 7, 10: 10}
        )
        self.assertTrue({x["id"] for x in subsets[7]} < {x["id"] for x in subsets[10]})
        for part in subsets.values():
            paired = [x for x in part if x["group_id"] == "paired"]
            self.assertIn(len(paired), (0, 2))
        for size, quota in quotas.items():
            self.assertEqual(sum(quota.values()), size)
            self.assertEqual(
                {
                    name: sum(budget.bucket(x) == name for x in subsets[size])
                    for name in quota
                },
                quota,
            )

    def test_panel_hash_overlap_is_rejected_without_gold(self) -> None:
        sample = row("one", "legacy:one", "natural", "one")
        text = sample["state"]
        excluded = {
            "raw_context_sha256": {pilot.sha_bytes(text.encode())},
            "normalized_context_sha256": {transfer.normalized_context_sha256(text)},
            "panel_input_sha256": set(),
        }
        with self.assertRaisesRegex(ValueError, "overlaps CSS panel"):
            budget.panel_overlap([sample], excluded)

    def test_rejects_unexpected_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected frozen combined source"):
            budget.bucket({"source": "css_test"})


if __name__ == "__main__":
    unittest.main()
