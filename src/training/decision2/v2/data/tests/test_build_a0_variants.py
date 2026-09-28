from __future__ import annotations

import unittest

from training.model.data import INPUT_FIELDS, digest, validate_row
from v2.data.build_a0_variants import (
    FLUTE_SOURCE,
    a0p,
    a0s,
    derangement,
    native_prompt,
    replay_prompt_set,
)


def _row(
    index: int, kind: str, *, source: str = "s", group: str | None = None, **extra
):
    if kind == "noul":
        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
    elif kind == "score":
        options = [{"key": str(i), "description": f"level {i}"} for i in range(3)]
    else:
        options = [{"key": f"k{i}", "description": f"candidate {i}"} for i in range(4)]
    row = {
        "id": f"r{index:04d}",
        "state": f"state {index}",
        "instructions": extra.pop("instructions", "Pick the best candidate."),
        "options": options,
        "label": index % len(options),
        "task_type": kind,
        "family": "f",
        "group_id": group or f"g{index}",
        "language": "en",
        "split": "train",
        "source": source,
        "evaluation_role": "train",
        "render_template": "t",
        "audit_metadata": {},
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return validate_row(row, "train")


class A0VariantsTest(unittest.TestCase):
    def test_a0s_drops_only_flute(self) -> None:
        rows = [_row(0, "choice", source=FLUTE_SOURCE), _row(1, "choice")]
        self.assertEqual([row["id"] for row in a0s(rows)], ["r0001"])

    def test_derangement_has_no_fixed_point_and_is_deterministic(self) -> None:
        for size in range(2, 12):
            order = derangement(size, f"seed{size}")
            self.assertEqual(sorted(order), list(range(size)))
            self.assertTrue(all(order[i] != i for i in range(size)))
            self.assertEqual(order, derangement(size, f"seed{size}"))

    def test_a0p_moves_gold_with_its_description(self) -> None:
        rows = [
            _row(1, "choice"),
            _row(2, "choice", instructions="Pick the first option that fits."),
            _row(3, "choice", instructions="Answer a question about option B."),
            _row(4, "score"),
            _row(5, "noul"),
            _row(6, "choice", source=FLUTE_SOURCE),
        ]
        permuted, skipped = a0p(rows)
        self.assertEqual([row["id"] for row in permuted], ["r0001:p1"])
        self.assertEqual(skipped, {"flute_excluded": 1, "positional_reference": 2})
        original, new = rows[0], permuted[0]
        self.assertEqual(
            new["options"][new["label"]], original["options"][original["label"]]
        )
        self.assertNotEqual(new["label"], original["label"])
        self.assertEqual(new["group_id"], original["group_id"])
        validate_row(new, "train")

    def test_replay_quotas_whole_groups_and_no_flute(self) -> None:
        rows = []
        for index in range(60):
            rows.append(_row(2 * index, "choice", group=f"pair{index}"))
            rows.append(_row(2 * index + 1, "noul", group=f"pair{index}"))
        rows += [_row(200 + index, "score") for index in range(20)]
        rows += [
            _row(300 + index, "choice", source=FLUTE_SOURCE) for index in range(20)
        ]
        chosen = replay_prompt_set(rows, {"choice": 10, "noul": 10, "score": 5})
        kinds = [row["task_type"] for row in chosen]
        self.assertEqual(
            (kinds.count("choice"), kinds.count("noul"), kinds.count("score")),
            (10, 10, 5),
        )
        self.assertFalse(any(row["source"] == FLUTE_SOURCE for row in chosen))
        by_group = {}
        for row in chosen:
            by_group.setdefault(row["group_id"], []).append(row["task_type"])
        self.assertTrue(
            all(len(v) == 2 for g, v in by_group.items() if g.startswith("pair"))
        )

    def test_native_prompt_preserves_option_order(self) -> None:
        row = _row(7, "choice")
        row["options"] = list(reversed(row["options"]))
        prompt = native_prompt(row)
        self.assertEqual(
            list(prompt["questions"]["decision"]["criteria"]), ["k3", "k2", "k1", "k0"]
        )
        score = native_prompt(_row(8, "score"))
        self.assertEqual(
            score["questions"]["decision"]["criteria"],
            ["level 0", "level 1", "level 2"],
        )


if __name__ == "__main__":
    unittest.main()
