"""CPU tests for the decoder M8 tooling (ops/m8: data build helpers, A20r teacher conversion, development rules)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
V2 = HERE.parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m8" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


md = load("m8_data")
mt = load("m8_teacher")
mr = load("m8_rules")
RULE = json.loads(
    (HERE / "ops" / "m8" / "specs" / "m8-human-rated-rule.json").read_text()
)


def row(rid, kind="choice", source="src", group=None, options=None, label=0, lang="en"):
    if options is None:
        options = {
            "choice": [
                {"key": "K1", "description": "alpha"},
                {"key": "K2", "description": {"x": 1}},
            ],
            "noul": [
                {"key": "false", "description": "No"},
                {"key": "true", "description": "Yes"},
            ],
            "score": [{"key": str(i), "description": f"level {i}"} for i in range(3)],
        }[kind]
    return {
        "id": rid,
        "group_id": group or f"g-{rid}",
        "state": {"text": f"state {rid}", "n": 3},
        "instructions": "Decide.",
        "options": options,
        "task_type": kind,
        "label": label,
        "family": "fam",
        "source": source,
        "language": lang,
        "input_sha256": f"h-{rid}",
    }


class RuleTest(unittest.TestCase):
    def test_rule_is_the_9b_rule_verbatim(self):
        nine = json.loads(
            (V2 / "9b" / "lux9b" / "specs" / "m6-ka-x60-ajS.json").read_text()
        )
        self.assertEqual(RULE["soft_target_rule"], nine["soft_target_rule"])
        self.assertEqual(RULE["pool_alias"], {"A0s": "A0s-strict"})

    def test_human_rated(self):
        self.assertTrue(md.human_rated(row("a", source="x"), "A7q", RULE))
        self.assertTrue(
            md.human_rated(
                row("b", source="google_goemotions_official_train"), "A0s", RULE
            )
        )
        self.assertFalse(
            md.human_rated(
                row("c", source="decision2_programmatic_original_v1"), "A0s", RULE
            )
        )
        self.assertFalse(md.human_rated(row("d", source="musique"), "H3", RULE))
        replay = row("e", source="legacy:stage3_replay")
        self.assertFalse(md.human_rated(replay, "A0s", RULE))
        replay["upstream_label"] = 1
        self.assertTrue(md.human_rated(replay, "A0s", RULE))
        with self.assertRaises(ValueError):
            md.human_rated(row("f"), "Z9", RULE)


class PromptTest(unittest.TestCase):
    def test_round_trip_every_type(self):
        for kind in ("choice", "noul", "score"):
            r = row(f"r-{kind}", kind)
            prompt = md.to_prompt(r)
            self.assertEqual(set(prompt), {"id", "state", "questions"})
            self.assertEqual(prompt["id"], r["id"])
            self.assertNotIn("label", json.dumps(prompt))

    def test_noul_order_is_kept(self):
        r = row(
            "n2",
            "noul",
            options=[
                {"key": "true", "description": "Yes"},
                {"key": "false", "description": "No"},
            ],
        )
        prompt = md.to_prompt(r)
        self.assertEqual(list(prompt["questions"]["q"]["criteria"]), ["true", "false"])

    def test_score_keys_must_be_levels(self):
        r = row(
            "s2",
            "score",
            options=[
                {"key": "1", "description": "a"},
                {"key": "2", "description": "b"},
            ],
        )
        with self.assertRaises(ValueError):
            md.to_prompt(r)

    def test_build_slice_split_and_teacher(self):
        base = [
            row(
                f"r{i}",
                ["choice", "noul", "score"][i % 3],
                source="s",
                group=f"g{i // 2}",
            )
            for i in range(40)
        ]
        base.append(row("q1", group="quar"))
        length = {r["id"]: 100 for r in base}
        pools = {r["id"]: ("A7q" if i % 4 == 0 else "A7g") for i, r in enumerate(base)}
        lux = {
            r["id"]: {
                "id": r["id"],
                "input_sha256": r["input_sha256"],
                "teacher_probs": {
                    o["key"]: 1 / len(r["options"]) for o in r["options"]
                },
            }
            for r in base[:30]
        }
        out = md.build(
            base,
            length,
            pools,
            RULE,
            lux,
            quarantine={"quar"},
            budget=2000,
            seed="t",
            shards=3,
        )
        ids = {r["id"] for r in out["slice"]}
        self.assertNotIn("q1", ids)
        self.assertTrue(1800 <= out["stats"]["slice"]["tokens"] <= 2400)
        groups = {r["group_id"] for r in out["slice"]}
        self.assertEqual(ids, {r["id"] for r in base if r["group_id"] in groups})
        self.assertEqual(len(out["prompts"]), len(out["slice"]))
        self.assertEqual(sum(len(s) for s in out["shards"]), len(out["prompts"]))
        self.assertEqual({t["id"] for t in out["teacher_c"]}, ids & set(lux))
        for entry in out["ids"]:
            self.assertEqual(
                entry["class"], "human" if entry["pool"] == "A7q" else "typed"
            )


class TeacherTest(unittest.TestCase):
    def test_distribution(self):
        self.assertEqual(
            mt.distribution({"type": "noul", "noul": 0.25}, ["false", "true"]),
            {"false": 0.75, "true": 0.25},
        )
        d = mt.distribution(
            {"type": "choice", "choice": "B", "probabilities": {"A": 0.4, "B": 0.6}},
            ["A", "B"],
        )
        self.assertEqual(list(d), ["A", "B"])
        with self.assertRaises(ValueError):
            mt.distribution(
                {"type": "choice", "probabilities": {"A": 0.4, "C": 0.6}}, ["A", "B"]
            )
        with self.assertRaises(ValueError):
            mt.distribution(
                {"type": "noul", "error": "max_length_exceeded"}, ["false", "true"]
            )

    def test_smoke_and_convert(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            stored = [
                {
                    "id": f"p{i}",
                    "answers": {
                        "a": {
                            "type": "choice",
                            "choice": "A",
                            "probabilities": {"A": 0.7, "B": 0.3},
                        }
                    },
                }
                for i in range(3)
            ]
            (t / "stored.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in stored)
            )
            same = [json.loads(json.dumps(x)) for x in stored]
            rows = [row("r1", "noul", label=1), row("r2", "choice", label=0)]
            slice_preds = [
                {"id": "r1", "answers": {"q": {"type": "noul", "noul": 0.9}}},
                {
                    "id": "r2",
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "K2",
                            "probabilities": {"K1": 0.2, "K2": 0.8},
                        }
                    },
                },
            ]
            (t / "shard.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in same + slice_preds)
            )
            ok = mt.smoke(t / "stored.jsonl", t / "shard.jsonl", 3, 1e-4)
            self.assertEqual(ok["status"], "PASS")
            same[1]["answers"]["a"]["probabilities"] = {"A": 0.4, "B": 0.6}
            (t / "bad.jsonl").write_text("".join(json.dumps(x) + "\n" for x in same))
            self.assertEqual(
                mt.smoke(t / "stored.jsonl", t / "bad.jsonl", 3, 1e-4)["status"], "FAIL"
            )
            out = mt.convert(
                rows,
                {"r1": "human", "r2": "typed"},
                {p["id"]: p for p in slice_preds},
                {},
            )
            self.assertEqual([x["id"] for x in out["d1"]], ["r1", "r2"])
            self.assertEqual([x["id"] for x in out["d2"]], ["r1"])
            self.assertEqual(
                out["d1"][0]["teacher_probs"],
                {"false": 0.09999999999999998, "true": 0.9},
            )
            self.assertEqual(out["gold_agreement"]["noul"]["rate"], 1.0)
            self.assertEqual(out["gold_agreement"]["choice"]["rate"], 0.0)


def arm(
    c=501, n=264, s=362, rp=264, ag=341, tt=160, sr=362, t=0.704, h3=0.5625, proxy=61.5
):
    return {
        "by_type": {
            "choice": {"correct": c, "n": 800, "invalid": 0},
            "noul": {"correct": n, "n": 400, "invalid": 0},
            "score": {"correct": s, "n": 400, "invalid": 0},
        },
        "by_family": {
            "attribute_gate": {"correct": ag, "n": 400},
            "transition_table": {"correct": tt, "n": 400},
            "rule_precedence": {"correct": rp, "n": 400},
            "set_reconciliation": {"correct": sr, "n": 400},
        },
        "T": t,
        "H_mean": h3,
        "H": h3 - 0.02,
        "proxy": proxy,
    }


def ht(delta):
    return {
        "delta": delta,
        "ci95": [delta - 0.01, delta + 0.01],
        "verdict": "FLAG" if delta <= -0.02 else "GAIN" if delta >= 0.02 else "TIE",
    }


S5_OK = {"check": {"flags": []}}


class RulesTest(unittest.TestCase):
    def test_floors(self):
        ref = arm()
        self.assertTrue(mr.eligibility(arm(), ref, ht(0.0), S5_OK, S5_OK)["eligible"])
        self.assertTrue(
            mr.eligibility(arm(c=477, ag=319, tt=158), ref, ht(0.0), S5_OK, S5_OK)[
                "eligible"
            ]
        )
        self.assertFalse(
            mr.eligibility(arm(c=476, ag=318, tt=158), ref, ht(0.0), S5_OK, S5_OK)[
                "eligible"
            ]
        )
        self.assertTrue(
            mr.eligibility(arm(n=260, rp=260), ref, ht(0.0), S5_OK, S5_OK)["eligible"]
        )
        noul = mr.eligibility(arm(n=259, rp=259), ref, ht(0.0), S5_OK, S5_OK)
        self.assertFalse(noul["eligible"])
        self.assertTrue(any("Noul floor" in x for x in noul["reasons"]))
        self.assertFalse(
            mr.eligibility(arm(tt=119), ref, ht(0.0), S5_OK, S5_OK)["eligible"]
        )
        self.assertFalse(
            mr.eligibility(arm(), ref, ht(-0.02), S5_OK, S5_OK)["eligible"]
        )
        self.assertTrue(
            mr.eligibility(arm(), ref, ht(-0.019), S5_OK, S5_OK)["eligible"]
        )
        self.assertFalse(
            mr.eligibility(
                arm(), ref, ht(0.0), {"check": {"flags": ["COLLAPSE"]}}, S5_OK
            )["eligible"]
        )
        self.assertFalse(
            mr.eligibility(arm(), ref, ht(0.0), {"check": {"flags": ["WARN"]}}, S5_OK)[
                "eligible"
            ]
        )
        warn = {"check": {"flags": ["WARN"]}}
        self.assertTrue(mr.eligibility(arm(), ref, ht(0.0), warn, warn)["eligible"])
        self.assertFalse(mr.eligibility(arm(), ref, None, S5_OK, S5_OK)["eligible"])

    def test_pick_prefers_gain_then_largest_alpha(self):
        rows = [
            {"step": "1", "eligible": True, "htdev2": ht(0.0), "proxy": 60},
            {"step": "2/3", "eligible": True, "htdev2": ht(0.03), "proxy": 60},
            {"step": "1/3", "eligible": True, "htdev2": ht(0.025), "proxy": 60},
        ]
        self.assertEqual(mr.pick(rows)["step"], "2/3")
        rows[1]["eligible"] = rows[2]["eligible"] = False
        self.assertEqual(mr.pick(rows)["step"], "1")
        rows[0]["eligible"] = False
        self.assertIsNone(mr.pick(rows))

    def test_select_slots_and_proxy_drop(self):
        def r(point, proxy, ok=True):
            return {
                "step": "1",
                "point": point,
                "eligible": ok,
                "htdev2": ht(0.0),
                "proxy": proxy,
                "reasons": [] if ok else ["x"],
            }

        lines = {
            "L-D1": [r("d1", 62)],
            "L-D2": [r("d2", 53.9)],
            "L-C": [r("c", 61, ok=False)],
        }
        out = mr.select(lines, {})
        self.assertEqual(
            [(f["slot"], f["line"]) for f in out["finalists"]], [(1, "L-D1")]
        )
        self.assertEqual(out["proxy_dropped"], {"L-D2": "d2"})
        self.assertEqual(
            out["not_finalists"][-1]["reason"], "no point passes the gates"
        )

    def test_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for name, acc in (("x", 0.80), ("c", 0.81), ("y", 0.789)):
                d = t / name / "checkpoint-0000190"
                d.mkdir(parents=True)
                (t / name / "LATEST.json").write_text(
                    json.dumps({"checkpoint": "checkpoint-0000190"})
                )
                (t / name / "COMPLETE.json").write_text(json.dumps({"step": 190}))
                (d / "checkpoint.json").write_text(
                    json.dumps({"dev_metrics": {"family_macro_accuracy": acc}})
                )
            self.assertEqual(mr.early("D1", t / "x", t / "c")["decision"], "continue")
            self.assertEqual(mr.early("D2", t / "y", t / "c")["decision"], "stop")


if __name__ == "__main__":
    unittest.main()
