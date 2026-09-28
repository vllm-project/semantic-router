import collections
import math
import tempfile
import unittest
from pathlib import Path

from training.model.data import check_partition_isolation, load_partition
from v2.data.verifiable import build, core
from v2.data.verifiable.tests import checks

SEED = "decision2-a4-v2"
GROUPS = 504  # 4,032 rows per file
CHANCE = 0.25
RANKED = {"a2_calendar", "a2_counting"}
EXACT = ("argmin", "argmax", "median")


class A4v2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        files, cls.manifest = build.generate("a4v2", SEED, GROUPS, 0.3)
        build.write(cls.out, "a4v2", files, cls.manifest)
        cls.parts = {
            name: (
                load_partition(cls.out / f"{name}.train.jsonl", "train"),
                load_partition(cls.out / f"{name}.aho.jsonl", "select"),
            )
            for name in ("a4v2h", "a4v2r")
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, name):
        train, aho = self.parts[name]
        return train + aho

    def cells(self, name, task):
        out = collections.defaultdict(list)
        for row in self.rows(name):
            if row["task_type"] == task:
                out[(row["family"], row["language"])].append(row)
        return out

    def test_size_contract_and_oracles(self):
        self.assertEqual(self.manifest["oracle_disagreements"], 0)
        self.assertEqual(self.manifest["dropped_rows"], 0)
        rows_per_scenario = (
            8 / 6
        )  # four single-row Choice groups and two Noul pairs per block
        self.assertEqual(
            build.a4v2_groups(None) * len(build.A2_FAMILIES) * rows_per_scenario, 2016
        )
        for train, aho in self.parts.values():
            self.assertEqual(
                len(train) + len(aho),
                GROUPS * len(build.A2_FAMILIES) * rows_per_scenario,
            )
            tasks = collections.Counter(r["task_type"] for r in train + aho)
            self.assertEqual(tasks["choice"], tasks["noul"])
            check_partition_isolation({"train": train, "select": aho})
            for row in train + aho:
                self.assertEqual(
                    row["split"] == "select", core.held_out(row["group_id"]), row["id"]
                )
                checks.assert_row(self, row)

    def test_identifiers_and_namespaces(self):
        for name in ("a4v2h", "a4v2r"):
            for row in self.rows(name):
                self.assertRegex(
                    row["id"], rf"^{name}-{row['family']}-(en|zh)-\d{{5}}-v[01]$"
                )
                if row["task_type"] == "choice":
                    self.assertTrue(row["id"].endswith("-v0"), row["id"])
                self.assertRegex(
                    row["group_id"],
                    rf"^a4v2:{row['family']}:{row['language']}:[0-9a-f]{{16}}$",
                )
                self.assertEqual(row["source"], f"decision2_verifiable_v2_{name}")
        v1 = build.generate("a4", SEED, 4, 0.3)[0]["a4h.train.jsonl"]
        self.assertFalse(
            {r["group_id"].split(":", 1)[1] for r in v1}
            & {r["group_id"].split(":", 1)[1] for r in self.rows("a4v2h")}
        )

    def test_files_share_states_gold_and_positions(self):
        hard = {r["id"].removeprefix("a4v2h-"): r for r in self.rows("a4v2h")}
        rand = {r["id"].removeprefix("a4v2r-"): r for r in self.rows("a4v2r")}
        self.assertEqual(set(hard), set(rand))
        for key, h in hard.items():
            r = rand[key]
            for field in ("state", "group_id", "label", "task_type", "language"):
                self.assertEqual(h[field], r[field], key)
            self.assertEqual(h["options"][h["label"]], r["options"][r["label"]], key)
            if h["task_type"] == "choice":
                self.assertEqual(h["instructions"], r["instructions"], key)
                near = {
                    o["description"]
                    for i, o in enumerate(h["options"])
                    if i != h["label"]
                }
                far = {
                    o["description"]
                    for i, o in enumerate(r["options"])
                    if i != r["label"]
                }
                self.assertNotEqual(near, far, key)
            elif h["label"] == 1:
                self.assertEqual(h["instructions"], r["instructions"], key)
            else:
                self.assertNotEqual(h["instructions"], r["instructions"], key)

    def test_noul_truth_is_exactly_balanced_per_family_and_language(self):
        for name in ("a4v2h", "a4v2r"):
            for cell, rows in self.cells(name, "noul").items():
                self.assertEqual(
                    2 * sum(r["label"] for r in rows), len(rows), (name, cell)
                )
            groups = collections.defaultdict(list)
            for row in self.rows(name):
                groups[row["group_id"]].append(row)
            for group_id, members in groups.items():
                tasks = sorted(r["task_type"] for r in members)
                self.assertIn(tasks, (["choice"], ["noul", "noul"]), group_id)
                if tasks == ["noul", "noul"]:
                    self.assertEqual(
                        sorted(r["label"] for r in members), [0, 1], group_id
                    )
                    self.assertEqual(members[0]["state"], members[1]["state"], group_id)

    def test_gold_positions_and_value_ranks_are_exactly_uniform(self):
        for name in ("a4v2h", "a4v2r"):
            for cell, rows in self.cells(name, "choice").items():
                positions = collections.Counter(r["label"] for r in rows)
                self.assertEqual(
                    set(positions.values()), {len(rows) // 4}, (name, cell, positions)
                )
                if cell[0] in RANKED:
                    ranks = collections.Counter()
                    for r in rows:
                        values = [
                            core.option_value(o["description"]) for o in r["options"]
                        ]
                        ranks[sorted(values).index(values[r["label"]])] += 1
                    self.assertEqual(
                        set(ranks.values()), {len(rows) // 4}, (name, cell, ranks)
                    )

    def test_random_distractors_straddle_the_gold(self):
        for cell, rows in self.cells("a4v2r", "choice").items():
            if cell[0] not in RANKED:
                continue
            signed = []
            for r in rows:
                values = [core.option_value(o["description"]) for o in r["options"]]
                signed.append(sum(v - values[r["label"]] for v in values))
            mean = sum(signed) / len(signed)
            spread = math.sqrt(sum((s - mean) ** 2 for s in signed) / (len(signed) - 1))
            self.assertLess(abs(mean) / (spread / math.sqrt(len(signed))), 3.5, cell)

    def test_option_only_heuristics_stay_at_chance(self):
        for name in ("a4v2h", "a4v2r"):
            pooled = collections.defaultdict(list)
            for cell, rows in self.cells(name, "choice").items():
                frequency = collections.Counter(
                    o["description"] for r in rows for o in r["options"]
                )
                credits = collections.defaultdict(list)
                for r in rows:
                    for heuristic, credit in core.heuristic_credits(
                        r["options"], r["label"], r["state"], frequency
                    ).items():
                        credits[heuristic].append(credit)
                for heuristic, values in credits.items():
                    excess = sum(values) / len(values) - CHANCE
                    exact = heuristic == "mentioned" or (
                        heuristic in EXACT and cell[0] in RANKED
                    )
                    # Design-controlled heuristics are exact; the rest carry binomial sampling noise
                    # (about 4 points per 100-row cell), so cells get a 4-SE allowance and the pooled
                    # file-level check below keeps the 3-point bound.
                    bound = (
                        0.03
                        if exact
                        else max(
                            0.03, 4 * math.sqrt(CHANCE * (1 - CHANCE) / len(values))
                        )
                    )
                    self.assertLessEqual(
                        excess, bound, (name, cell, heuristic, round(excess, 3))
                    )
                    pooled[heuristic] += values
            for heuristic, values in pooled.items():
                self.assertLessEqual(
                    sum(values) / len(values) - CHANCE, 0.03, (name, heuristic)
                )

    def test_build_is_deterministic(self):
        first = build.generate("a4v2", "unit-a4v2", 8, 0.3)[0]
        second = build.generate("a4v2", "unit-a4v2", 8, 0.3)[0]
        self.assertEqual(first, second)
        self.assertEqual(build.a4v2_groups(9), 12)


if __name__ == "__main__":
    unittest.main()
