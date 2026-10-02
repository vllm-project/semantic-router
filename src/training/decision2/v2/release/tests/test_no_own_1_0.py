"""No-1.0 gate profile and card slot (stdlib, fixture reports)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release import card, gate, layout
from v2.release.tests.test_release import REPORTS, ROSTER, build_test_card, facts

OK_TYPES = {kind: {"verdict": "OK"} for kind in ("choice", "noul", "score")}


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class NoOwnGateTest(unittest.TestCase):
    """Item 1 at ~27B: >= 90% of the reference peer, H not below it, no type collapsed."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.work = Path(self.scratch.name)
        self.receipts = self.work / "receipts"
        run = self.work / "scored-run"
        candidate = write(
            run / "REPORT.json", {"v3": {"score": 67.209}, "model": {"label": "F1"}}
        )
        reference = write(
            self.work / "peer/REPORT.json",
            {"v3": {"score": 72.133}, "model": {"label": "AutoJev-27B"}},
        )
        self.paired = write(
            self.work / "paired.json",
            {
                "point": {
                    "left": {"score": 67.209},
                    "right": {"score": 72.133},
                    "delta": {"score": -4.924},
                },
                "ci95": {"low": -6.82, "high": -0.43},
                "axis_ci95": {"H": {"delta": {"low": -0.040, "high": 0.056}}},
            },
        )
        self.types = write(
            self.work / "types.json",
            {"schema": gate.TYPES_SCHEMA, "run": str(run), "types": OK_TYPES},
        )
        self.decision = self.work / "decision.json"
        self.spec = {
            "kind": "release",
            "model_name": "Decision-2.0-Vega-26B",
            "repo_id": "vllm-sr/Decision-2.0-Vega-26B",
            "expected_identity": {"model_sha256": "c" * 64},
            "scored": {"report_sha256": layout.sha_file(candidate)},
            "gate_receipt": str(self.decision),
            "gate_profile": {
                "name": "no-1.0",
                "reference": "autojev27",
                "v3_share": 0.9,
                "types": str(self.types),
            },
            "card": {
                "paired": str(self.paired),
                "reports": [
                    {"key": "candidate", "role": "candidate", "report": str(candidate)},
                    {
                        "key": "autojev27",
                        "role": "reference",
                        "report": str(reference),
                        "label": "AutoJev-27B",
                    },
                ],
            },
        }
        self.final = {
            "schema": gate.DECISION_SCHEMA,
            "status": "final",
            "decision": "release",
            "model_name": "Decision-2.0-Vega-26B",
            "repo_id": "vllm-sr/Decision-2.0-Vega-26B",
            "identity": {"model_sha256": "c" * 64},
            "report_sha256": self.spec["scored"]["report_sha256"],
            "paired_sha256": layout.sha_file(self.paired),
            "gate_profile": "no-1.0",
            "types_sha256": layout.sha_file(self.types),
            "decided_by": "coordinator",
            "rationale": "no-1.0 profile holds",
        }
        write(self.decision, self.final)
        revision = "a" * 40
        for name, value in {
            "build": {"parameters": {"loaded": 7}, "card": {"tradeoffs": ["x"]}},
            "repeat-pre": {"passed": True},
            "card-pre": {"passed": True},
            "upload": {"revision": revision, "manifest_sha256": "e" * 64},
            "download": {"passed": True, "revision": revision},
            "tree": {"passed": True, "files": 3},
            "post": {"passed": True, "loaded_parameters": 7},
            "repeat-post": {"passed": True},
            "card-post": {"passed": True},
            "readback": {"passed": True, "card_problems": [], "card_data": {}},
        }.items():
            write(self.receipts / f"{name}.json", value)
        self.write_spec()

    def tearDown(self):
        self.scratch.cleanup()

    def write_spec(self):
        write(self.receipts / "spec.json", self.spec)

    def item(self) -> dict:
        return gate.evaluate(self.work)["items"]["1_near_first_tier_no_1_0"]

    def test_profile_replaces_beat_own_1_0_and_seals(self):
        result = gate.evaluate(self.work)
        self.assertTrue(result["passed"], result["items"])
        self.assertNotIn("1_beats_own_1_0", result["items"])
        self.assertIn("64.920", self.item()["evidence"])
        self.assertIn(
            "results below AutoJev-27B (no Decision 1.0 at this size)",
            result["items"]["2_regressions_disclosed"]["evidence"],
        )
        self.assertEqual(gate.seal(self.work)["decided_by"], "coordinator")

    def test_without_the_profile_the_own_1_0_rule_applies(self):
        del self.spec["gate_profile"]
        self.write_spec()
        items = gate.evaluate(self.work)["items"]
        self.assertFalse(items["1_beats_own_1_0"]["passed"])
        self.assertIn(
            "results below own 1.0 listed in the build receipt",
            items["2_regressions_disclosed"]["evidence"],
        )

    def test_each_condition_fails_closed(self):
        self.spec["gate_profile"]["v3_share"] = 0.95
        self.write_spec()
        self.assertFalse(self.item()["passed"])
        self.spec["gate_profile"]["v3_share"] = 0.9
        self.write_spec()
        paired = json.loads(self.paired.read_text())
        paired["axis_ci95"]["H"]["delta"]["high"] = -0.001
        write(self.paired, paired)
        self.assertFalse(self.item()["passed"])
        paired["axis_ci95"]["H"]["delta"]["high"] = 0.056
        paired["point"]["right"]["score"] = 69.29
        write(self.paired, paired)
        self.assertIn("paired comparison is not", self.item()["evidence"])
        paired["point"]["right"]["score"] = 72.133
        write(self.paired, paired)
        self.assertTrue(self.item()["passed"])
        collapsed = {
            **OK_TYPES,
            "score": {"verdict": "COLLAPSED: one answer (4) takes 95%"},
        }
        types = json.loads(self.types.read_text())
        write(self.types, {**types, "types": collapsed})
        self.assertFalse(self.item()["passed"])
        write(self.types, {**types, "types": {"choice": {"verdict": "OK"}}})
        self.assertFalse(self.item()["passed"])
        write(self.types, {**types, "run": str(self.work / "peer")})
        self.assertIn("type check is not on the scored run", self.item()["evidence"])

    def test_decision_must_name_the_profile_and_type_check(self):
        self.assertEqual(gate.check(self.spec, self.decision)["gate_profile"], "no-1.0")
        for bad in (
            {k: v for k, v in self.final.items() if k != "gate_profile"},
            {**self.final, "types_sha256": "f" * 64},
        ):
            write(self.decision, bad)
            with self.assertRaises(ValueError):
                gate.check(self.spec, self.decision)

    def test_profile_shape_is_checked(self):
        for change in (
            {"name": "no-own"},
            {"reference": "candidate"},
            {"v3_share": 1.2},
            {"v3_share": 1},
            {"types": ""},
        ):
            spec = {
                **self.spec,
                "gate_profile": {**self.spec["gate_profile"], **change},
            }
            with self.assertRaises(ValueError):
                gate.gate_profile(spec)


def entries(reference: dict | None = None) -> list[dict]:
    return [
        {
            "key": "cand",
            "role": "candidate",
            "report": str(REPORTS / "lex.json"),
            "label": "Decision-2.0-Kai-0.6B",
        },
        reference
        or {
            "key": "bosun",
            "role": "reference",
            "report": str(REPORTS / "bosun.json"),
            "repo_id": "Hanno-Labs/bosun-v3.1-0.6b",
            "label": "Bosun",
        },
        {
            "key": "gliner",
            "role": "peer",
            "report": str(REPORTS / "gliner25.json"),
            "repo_id": "fastino/GLiNER2.5-Decide",
        },
    ]


VEGA = "Decision-2.0-Vega-27B"


class NoOwnCardTest(unittest.TestCase):
    def build(
        self,
        items: list[dict],
        paired: dict | None = None,
        peers: dict[str, dict] | None = None,
    ) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            return build_test_card(
                scratch,
                items,
                {**facts(), "model_name": VEGA, "comparison": "no-1.0"},
                paired=write(scratch / "paired.json", paired) if paired else None,
                paired_peers={
                    key: write(scratch / f"paired-{key}.json", value)
                    for key, value in (peers or {}).items()
                },
            )

    def test_index_compares_with_the_family_and_names_no_decision_1_0(self):
        result = self.build(entries())
        readme = result["readme"]
        self.assertEqual(result["problems"], [])
        # Synthetic Index: Vega 27B 60.5, Lux 9B 50.5; Lex is the weakest JevArena model here.
        self.assertIn(
            "**The strongest Decision 2.0 model:** +10.0 on the Jev Decision Index over "
            "Decision-2.0-Lux-9B.",
            readme,
        )
        self.assertIn("by area: Decision-2.0-Vega-27B and Decision-2.0-Lux-9B", readme)
        self.assertNotIn("Top JevArena", readme)
        self.assertNotIn("Ahead of", readme)
        self.assertTrue(result["tradeoffs"])

    def test_a_level_lead_over_the_reference_is_called_level(self):
        items = entries()
        items[0] = {**items[0], "report": str(REPORTS / "gliner25.json")}
        items[2] = {
            "key": "lex",
            "role": "peer",
            "report": str(REPORTS / "lex.json"),
            "label": "Decision 1.0 Lex",
        }
        level = {"point": {"delta": {"score": 4.0}}, "ci95": {"low": -0.5, "high": 8.0}}
        readme = self.build(items, level)["readme"]
        self.assertIn(
            "**Top JevArena score of its size:** 42.5 among the 2 same-size models compared, "
            "statistically level with Bosun (38.5).",
            readme,
        )
        above = {"point": {"delta": {"score": 4.0}}, "ci95": {"low": 1.0, "high": 8.0}}
        readme = self.build(items, above)["readme"]
        self.assertIn(
            "**Top JevArena score of its size:** 42.5, ahead of the other same-size model compared.",
            readme,
        )

    def test_peer_intervals_must_pair_the_candidate_with_that_peer(self):
        v3 = {
            name: json.loads((REPORTS / f"{name}.json").read_text())["v3"]["score"]
            for name in ("lex", "gliner25")
        }
        gliner = {
            "point": {
                "left": {"score": v3["lex"]},
                "right": {"score": v3["gliner25"]},
                "delta": {"score": v3["lex"] - v3["gliner25"]},
            },
            "ci95": {"low": -2.0, "high": 1.0},
        }
        self.assertEqual(
            self.build(entries(), peers={"gliner": gliner})["problems"], []
        )
        gliner["point"]["right"]["score"] += 1.0
        with self.assertRaises(ValueError):
            self.build(entries(), peers={"gliner": gliner})

    def test_reference_outside_the_licence_filter_is_not_named(self):
        jpt = {
            "key": "jpt",
            "role": "reference",
            "report": str(REPORTS / "jpt08b.json"),
            "repo_id": "kirp/jpt-0.8b",
            "label": "JPT",
        }
        result = self.build(entries(jpt))
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["tradeoffs"], [])
        self.assertNotIn("JPT", result["readme"])

    def test_roles_follow_the_comparison(self):
        own = {
            "key": "kai1",
            "role": "own-1.0",
            "report": str(REPORTS / "kai1.json"),
            "repo_id": "vllm-sr/Decision-1.0-Kai-0.6B",
        }
        with self.assertRaises(ValueError):
            card.select_reports([*entries(), own], ROSTER, "no-1.0")
        with self.assertRaises(ValueError):
            card.select_reports(
                [e for e in entries() if e["role"] != "reference"], ROSTER, "no-1.0"
            )
        with self.assertRaises(ValueError):
            card.select_reports(entries(), ROSTER)


if __name__ == "__main__":
    unittest.main()
