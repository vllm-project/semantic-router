from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.sources.common import (
    cap_groups,
    choice_options,
    is_aho,
    make_row,
    noul_options,
    rotate,
    score_options,
    write_arm,
)


def _choice(index: int, group: str) -> dict:
    options, label = rotate(choice_options(["a", "b", "c", "d"]), 2, f"seed{index}")
    return make_row(
        arm="t1",
        source="src",
        family="fam",
        task_type="choice",
        language="en",
        group_key=group,
        local_id=str(index),
        state=f"state {index}",
        instructions="Pick one.",
        options=options,
        label=label,
        render_template="tmpl",
    )


class SourcesCommonTest(unittest.TestCase):
    def test_rotation_keeps_gold_description(self) -> None:
        for seed in ("a", "b", "c", "d", "e"):
            options, label = rotate(choice_options(["w", "x", "y", "z"]), 1, seed)
            self.assertEqual(options[label]["description"], "x")

    def test_option_helpers(self) -> None:
        self.assertEqual([o["key"] for o in noul_options("ko")], ["false", "true"])
        self.assertEqual(
            [o["key"] for o in score_options(["l", "m", "h"])], ["0", "1", "2"]
        )
        with self.assertRaises(ValueError):
            score_options(["only"])

    def test_cap_groups_keeps_whole_groups(self) -> None:
        rows = [_choice(i, f"g{i // 2}") for i in range(20)]
        kept = cap_groups(rows, 7, "s")
        groups = {}
        for row in kept:
            groups.setdefault(row["group_id"], 0)
            groups[row["group_id"]] += 1
        self.assertTrue(all(count == 2 for count in groups.values()))
        self.assertLessEqual(len(kept), 7)

    def test_write_arm_splits_aho_and_refuses_overwrite(self) -> None:
        rows = [_choice(i, f"g{i}") for i in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_arm(rows, Path(tmp), "t1", {"seed": "x"})
            aho = [json.loads(line) for line in (Path(tmp) / "t1.aho.jsonl").open()]
            self.assertTrue(
                all(row["split"] == "select" and is_aho(row["group_id"]) for row in aho)
            )
            self.assertEqual(manifest["train"]["rows"] + manifest["aho"]["rows"], 200)
            with self.assertRaises(FileExistsError):
                write_arm(rows, Path(tmp), "t1", {})


if __name__ == "__main__":
    unittest.main()
