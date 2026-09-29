"""CPU tests for ops/m7/m7_compose.py (synthetic rows, token lengths given directly)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "m7" / "m7_compose.py"
_spec = importlib.util.spec_from_file_location("m7_compose", SCRIPT)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


def row(
    i, group, source="src", task="noul", lang="en", family="fam", state="s", tag=""
):
    return {
        "id": f"{tag}r{i}",
        "group_id": f"{tag}g{group}",
        "input_sha256": f"{tag}h{i}",
        "source": source,
        "task_type": task,
        "language": lang,
        "family": family,
        "state": state,
    }


class ComposeTest(unittest.TestCase):
    def setUp(self):
        self.base = [row(i, i // 2) for i in range(300)]
        pool = list(self.base)
        for i in range(300, 460):
            long_state = "x" * (4000 if i % 5 == 0 else 100)
            pool.append(
                row(
                    i,
                    i // 2,
                    source=f"s{i % 3}",
                    task=("choice", "noul")[i % 2],
                    state=long_state,
                )
            )
        self.pool = pool
        hs1 = []
        fams = ("hs1_policy_packet", "hs1_unmet_condition", "hs1_quote_check")
        for i in range(60):
            state = "at least 8 nights consecutive nights" if i in (0, 1) else "policy"
            hs1.append(
                row(
                    i,
                    i // 2,
                    source="hs1",
                    family=fams[(i // 2) % 3],
                    state=state,
                    tag="h",
                )
            )
        self.hs1 = hs1
        self.pn1 = [row(i, i, source="tatoeba", lang="ja", tag="p") for i in range(10)]
        self.length = {
            r["id"]: 10 + (len(r["state"]) // 100)
            for r in self.base + pool + hs1 + self.pn1
        }
        self.pool_of = {
            r["id"]: ("H7" if r["id"] in ("r350", "r351") else "V") for r in pool
        }

    def run_compose(
        self,
        filler="pool",
        quarantine=frozenset({"g0"}),
        hs1=None,
        lp_budget=120,
        pn1=None,
    ):
        return mc.compose(
            "4b",
            self.base,
            self.pool,
            self.hs1 if hs1 is None else hs1,
            self.pn1 if pn1 is None else pn1,
            self.length,
            quarantine=set(quarantine),
            pool_of=self.pool_of,
            gold_pools={"H7"},
            filler=filler,
            hs1_whole=("hs1_policy_packet", "hs1_unmet_condition"),
            hs1_half="hs1_quote_check",
            hs1_drop=(" nights consecutive nights",),
            lp_min_chars=3000,
            lp_budget=lp_budget,
            pn1_repeat=3,
        )

    def test_matched_nested_and_blocks(self):
        arms, rep = self.run_compose()
        self.assertEqual(rep["quarantine_rows_dropped"], {"base": 2, "pool": 2})
        for arm in mc.ARMS:
            self.assertNotIn("g0", {r["group_id"] for r, _ in arms[arm]})
        largest_group = max(self.length.values()) * 2
        self.assertLessEqual(abs(rep["match"]["C_minus_H"]), largest_group)
        self.assertLessEqual(abs(rep["match"]["P_minus_H"]), largest_group)
        fill_c = {r["id"] for r, p in arms["C"] if p.startswith("fill")}
        fill_p = {r["id"] for r, p in arms["P"] if p.startswith("fill")}
        self.assertTrue(fill_p <= fill_c and fill_p)
        hs1_ids = {r["id"] for r, p in arms["H"] if p == "hs1"}
        self.assertNotIn("hr0", hs1_ids)
        self.assertEqual(rep["hs1"]["defect_groups_dropped"], 1)
        lp = [r for r, p in arms["H"] if p.startswith("lp")]
        self.assertTrue(lp)
        base_ids = {r["id"] for r in self.base}
        self.assertFalse(base_ids & {r["id"] for r in lp})
        pn1 = [r["id"] for r, p in arms["P"] if p == "pn1"]
        self.assertEqual(len(pn1), 30)
        self.assertIn("pr0~r3", pn1)
        pools = {r["id"]: p for r, p in arms["C"]}
        if "r350" in pools:
            self.assertEqual(pools["r350"], "fill-gold")
        self.assertTrue(all(p != "base-gold" for _, p in arms["C"]))

    def test_replay_filler_copies_base_rows(self):
        arms, rep = self.run_compose(
            filler="replay", hs1=self.hs1[:12], lp_budget=40, pn1=self.pn1[:3]
        )
        rep_rows = [r for r, p in arms["C"] if p == "replay"]
        self.assertTrue(rep_rows)
        self.assertTrue(all(r["id"].endswith("~r2") for r in rep_rows))
        base_ids = {r["id"] for r in self.base}
        self.assertTrue(all(r["id"].split("~r", 1)[0] in base_ids for r in rep_rows))
        self.assertEqual(rep["filler"]["kind"], "replay")

    def test_teacher_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "t.jsonl"
            src.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"r{i}",
                            "input_sha256": f"h{i}",
                            "teacher_probs": {"true": 0.5, "false": 0.5},
                        }
                    )
                    + "\n"
                    for i in range(5)
                )
            )
            out = tmp / "new.jsonl"
            self.assertEqual(mc.teacher_new(src, {"r1", "r3"}, out)["rows"], 2)
            with self.assertRaises(ValueError):
                mc.teacher_new(src, {"r9"}, tmp / "bad.jsonl")
            rep = mc.teacher_replay(src, {"r2": "r2~r2"}, tmp / "rep.jsonl")
            self.assertEqual(rep["rows"], 1)
            rec = json.loads((tmp / "rep.jsonl").read_text())
            self.assertEqual((rec["id"], rec["input_sha256"]), ("r2~r2", "h2"))

    def test_pick_matched_hits_the_target(self):
        groups = {f"g{i}": [row(i, i)] for i in range(50)}
        gtok = {g: 7 for g in groups}
        chosen, _ = mc.pick_matched(groups, gtok, 140, "seed")
        self.assertEqual(len(chosen), 20)
        with self.assertRaises(ValueError):
            mc.pick_matched(groups, gtok, 10_000, "seed")


if __name__ == "__main__":
    unittest.main()
