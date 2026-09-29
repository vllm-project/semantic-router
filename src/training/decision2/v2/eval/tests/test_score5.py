import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from v2.eval import score5
from v2.eval.htdev.build import sha_bytes

AXES = ("creativity", "helpfulness", "humor", "quality")


def synthetic_rows(per_cell: int = 40, groups: int = 780) -> list[dict]:
    rows = []
    for level in range(5):
        for axis in AXES:
            for i in range(per_cell):
                n = len(rows)
                options = [{"key": str(k), "description": f"L{k}"} for k in range(5)]
                rows.append(
                    {
                        "id": f"r{n}",
                        "group_id": f"g{n % groups}",
                        "label": level,
                        "split": "select",
                        "language": "en",
                        "state": f"Conversation {n}",
                        "instructions": f"Rate {axis}.",
                        "options": options[::-1] if n % 2 else options,
                        "input_sha256": f"{n:064x}",
                        "audit_metadata": {"a7": {"axis": axis}},
                    }
                )
    return rows


def answer(item_id: str, level) -> dict:
    return {"id": item_id, "answers": {"decision": {"type": "score", "score": level}}}


class SelectTest(unittest.TestCase):
    def test_balanced_levels_axes_and_one_row_per_group(self):
        chosen, info = score5.select(synthetic_rows())
        self.assertEqual(len(chosen), 500)
        self.assertEqual(
            Counter(r["label"] for r in chosen), {k: 100 for k in range(5)}
        )
        self.assertEqual(set(info["cells"].values()), {25})
        self.assertEqual(info["group_cap"], 1)
        self.assertEqual(max(Counter(r["group_id"] for r in chosen).values()), 1)
        again, _ = score5.select(list(reversed(synthetic_rows())))
        self.assertEqual({r["id"] for r in chosen}, {r["id"] for r in again})

    def test_short_cell_is_water_filled_within_its_level(self):
        rows = [
            r
            for r in synthetic_rows()
            if not (r["label"] == 2 and r["audit_metadata"]["a7"]["axis"] == "humor")
            or int(r["id"][1:]) % 40 < 10
        ]
        chosen, info = score5.select(rows)
        self.assertEqual(sum(r["label"] == 2 for r in chosen), 100)
        self.assertEqual(info["cells"]["2|humor"], 10)
        self.assertEqual(
            sorted(info["cells"][f"2|{a}"] for a in AXES)[1:], [30, 30, 30]
        )


class BuildTest(unittest.TestCase):
    def test_build_writes_gold_free_prompts_private_gold_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = "".join(json.dumps(r) + "\n" for r in synthetic_rows()).encode()
            (tmp / "aho.jsonl").write_bytes(data)
            out = tmp / "build"
            args = ["build", "--source", str(tmp / "aho.jsonl"), "--output-dir"]
            with self.assertRaises(SystemExit):
                score5.main(args + [str(out), "--source-sha256", "0" * 64])
            score5.main(args + [str(out), "--source-sha256", sha_bytes(data)])
            prompts = [json.loads(x) for x in open(out / "score5-dev.prompts.jsonl")]
            gold = [json.loads(x) for x in open(out / "score5-dev.gold.jsonl")]
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(len(prompts), 500)
            self.assertTrue(
                all(set(p) == {"id", "state", "questions"} for p in prompts)
            )
            criteria = prompts[0]["questions"]["decision"]["criteria"]
            self.assertEqual(criteria, [f"L{k}" for k in range(5)])
            self.assertEqual([p["id"] for p in prompts], [g["id"] for g in gold])
            self.assertEqual(
                os.stat(out / "score5-dev.gold.jsonl").st_mode & 0o777, 0o600
            )
            self.assertEqual(manifest["fit_pool_rows"], 300)
            self.assertEqual(manifest["by_level"], {str(k): 100 for k in range(5)})
            self.assertEqual(len(manifest["selected_row_id_sha256"]), 500)
            panel_rows = [json.loads(x) for x in open(out / "panel-rows.jsonl")]
            self.assertEqual(
                {r["panel_id"] for r in panel_rows}, {g["id"] for g in gold}
            )
            self.assertEqual(gold[0]["gold"]["decision"]["type"], "score")


class SummaryTest(unittest.TestCase):
    def gold(self) -> list[dict]:
        question = {"type": "score", "instructions": "Rate.", "criteria": list("abcde")}
        return [
            {
                "id": f"i{n}",
                "task": "score5/helpfulness",
                "source": score5.SOURCE,
                "group_id": f"g{n}",
                "language": "en",
                "long": False,
                "questions": {"decision": question},
                "gold": {"decision": {"type": "score", "value": n % 5}},
            }
            for n in range(100)
        ]

    def test_perfect_answers_are_not_flagged(self):
        gold = self.gold()
        preds = {g["id"]: answer(g["id"], n % 5) for n, g in enumerate(gold)}
        out = score5.summary(gold, preds, replicates=200)
        self.assertEqual(out["flags"], [])
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["modal_share"], 0.2)
        self.assertAlmostEqual(out["always_modal_accuracy"], 0.2)
        self.assertAlmostEqual(out["qwk_answered"], 1.0)

    def test_collapse_to_one_level_and_missing_answers(self):
        gold = self.gold()
        preds = {g["id"]: answer(g["id"], 4) for g in gold[:90]}
        out = score5.summary(gold, preds, replicates=200)
        self.assertEqual(out["invalid_or_missing"], 10)
        self.assertEqual(out["modal_share"], 1.0)
        self.assertEqual(out["rare_levels"], [0, 1, 2, 3])
        self.assertEqual(out["flags"], ["COLLAPSE", "NO-SIGNAL"])
        self.assertEqual(out["correct"], 18)

    def test_flag_thresholds(self):
        self.assertEqual(score5.flags(0.59, 1, 0.5), ["WARN"])
        self.assertEqual(score5.flags(0.60, 0, 0.5), ["COLLAPSE"])
        self.assertEqual(score5.flags(0.30, 2, 0.5), ["COLLAPSE"])
        self.assertEqual(score5.flags(0.40, 0, 0.21), ["WARN"])
        self.assertEqual(score5.flags(0.39, 0, 0.20), ["NO-SIGNAL"])
        self.assertEqual(score5.flags(None, 0, 0.0), ["COLLAPSE", "NO-SIGNAL"])

    def test_unknown_prediction_ids_are_refused(self):
        with self.assertRaises(ValueError):
            score5.summary(self.gold(), {"x": answer("x", 1)}, replicates=10)


if __name__ == "__main__":
    unittest.main()
