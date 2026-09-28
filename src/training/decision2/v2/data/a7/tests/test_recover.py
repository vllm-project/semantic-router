from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.data.a7 import build_a7, recover
from v2.data.freeze import canonical_jsonl

GENERATOR = {
    "type": "objective_generator",
    "generator": "objective_generator",
    "seed": 1,
}


def _noul(
    index: int, keys: list[str], descriptions: list[str], label: int, **extra
) -> dict:
    row = {
        "id": f"n{index}",
        "state": extra.pop(
            "state", f"signer {index} of team red signs with an active signature"
        ),
        "instructions": "Only members of team red are authorized.",
        "options": [{"key": k, "description": d} for k, d in zip(keys, descriptions)],
        "label": label,
        "task_type": "noul",
        "family": "authorization",
        "group_id": extra.pop("group", f"auth-{index}"),
        "language": "en",
        "source": GENERATOR,
        "split": "train",
    }
    row.update(extra)
    return row


def _score(
    index: int, state: dict, ranges: list[tuple[int, int]], order: list[int], label: int
) -> dict:
    options = [
        {
            "key": f"result_{level}",
            "description": f"Level {level}: between {ranges[level][0]} and {ranges[level][1]} conditions hold, inclusive.",
        }
        for level in order
    ]
    return {
        "id": f"s{index}",
        "state": json.dumps(state),
        "instructions": "Count conditions with value true and select the matching rubric level.",
        "options": options,
        "label": label,
        "task_type": "score",
        "family": "rubric",
        "group_id": f"rubric-{index}",
        "language": "en",
        "source": GENERATOR,
        "split": "train",
    }


class MapKeysTest(unittest.TestCase):
    def test_noul_keys_follow_descriptions_not_labels(self) -> None:
        for label in (0, 1):
            row = _noul(
                1,
                ["result_1", "result_0"],
                ["The action is not valid.", "The action is valid."],
                label,
            )
            mapped = recover.map_keys(row)
            self.assertEqual([o["key"] for o in mapped["options"]], ["false", "true"])
            self.assertEqual(mapped["label"], label)
        unknown = _noul(2, ["a", "b"], ["Yes", "No"], 0)
        with self.assertRaises(build_a7.Excluded):
            recover.map_keys(unknown)

    def test_score_is_reordered_by_level_and_checked_against_the_state(self) -> None:
        ranges = [(0, 0), (1, 1), (2, 3), (4, 6)]
        state = {"a": True, "b": True, "c": False, "d": True}
        row = _score(1, state, ranges, order=[2, 3, 1, 0], label=0)
        mapped = recover.map_keys(row)
        self.assertEqual([o["key"] for o in mapped["options"]], ["0", "1", "2", "3"])
        self.assertTrue(mapped["options"][2]["description"].startswith("Level 2:"))
        self.assertEqual(mapped["label"], 2)
        wrong = _score(2, state, ranges, order=[2, 3, 1, 0], label=1)
        with self.assertRaises(build_a7.Excluded) as caught:
            recover.map_keys(wrong)
        self.assertEqual(caught.exception.reason, "rule7e_oracle_disagrees")
        gap = _score(
            3, state, [(0, 0), (2, 2), (3, 3), (4, 6)], order=[0, 1, 2, 3], label=2
        )
        with self.assertRaises(build_a7.Excluded):
            recover.map_keys(gap)


class RecoverTest(unittest.TestCase):
    def test_recover_dedupes_and_inherits_frozen_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared_state = (
                "signer 7 of team red signs with an active signature (shared)"
            )
            stage1 = [
                _noul(
                    1,
                    ["result_0", "result_1"],
                    ["The action is valid.", "The action is not valid."],
                    0,
                ),
                _noul(
                    2,
                    ["cedar", "birch"],
                    ["The action is not valid.", "The action is valid."],
                    1,
                    state=shared_state,
                    group="auth-shared",
                ),
                _noul(3, ["x", "y"], ["Yes", "No"], 0),
                _noul(
                    4,
                    ["false", "true"],
                    ["The action is not valid.", "The action is valid."],
                    1,
                ),
            ]
            data = canonical_jsonl(stage1)
            (root / "stage1.jsonl").write_bytes(data)
            spec = {
                "sources": [
                    {
                        "name": "stage1",
                        "path": str(root / "stage1.jsonl"),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "rows": 4,
                    }
                ]
            }
            frozen = root / "final"
            frozen.mkdir()
            aho_row = build_a7.normalize_row(
                _noul(
                    9,
                    ["false", "true"],
                    ["The action is not valid.", "The action is valid."],
                    0,
                    state=shared_state,
                    group="auth-frozen",
                    instructions="Only members of team blue are authorized.",
                ),
                "stage3",
                "0" * 64,
            )
            aho_row = dict(aho_row, split="select", evaluation_role="select")
            (frozen / "A7o.aho.jsonl").write_bytes(canonical_jsonl([aho_row]))
            result = recover.recover(spec, [frozen])
            parts = result["parts"]
            ids = {
                part: sorted(r["audit_metadata"]["a7"]["original_id"] for r in rows)
                for part, rows in parts.items()
            }
            self.assertEqual(ids["aho"], ["n2"])
            self.assertEqual(ids["train"] + ids["aho"], sorted(ids["train"] + ["n2"]))
            self.assertNotIn("n4", ids["train"] + ids["aho"])
            self.assertNotIn("n3", ids["train"] + ids["aho"])
            recovered = parts["aho"][0]
            self.assertEqual(recovered["audit_metadata"]["a7"]["sub_arm"], "A7r")
            self.assertEqual(
                recovered["audit_metadata"]["a7"]["rule_7e"]["original_keys"],
                ["cedar", "birch"],
            )
            self.assertEqual(
                result["manifest"]["excluded"],
                {"stage1|rule7e_unmapped_noul|authorization|noul": 1},
            )


if __name__ == "__main__":
    unittest.main()
