"""No-1.0 gate profile and card slot (stdlib, fixture reports)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release import card, gate, layout
from v2.release.tests.test_release import REPORTS, ROSTER, TEXT, facts

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
            "model_name": "DEV2.0-26B",
            "repo_id": "llm-semantic-router/DEV2.0-26B",
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
            "model_name": "DEV2.0-26B",
            "repo_id": "llm-semantic-router/DEV2.0-26B",
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
            "results below own 1.0 summarised in the card limitations",
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
            "label": "DEV2.0-0.6B",
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


class NoOwnCardTest(unittest.TestCase):
    def build(
        self,
        items: list[dict],
        paired: dict | None = None,
        peers: dict[str, dict] | None = None,
    ) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / "pkg"
            path = None
            if paired:
                path = write(Path(scratch) / "paired.json", paired)
            result = card.build_card(
                entries=items,
                roster=ROSTER,
                paired=path,
                facts={**facts(), "comparison": "no-1.0"},
                text=TEXT,
                work=Path(scratch) / "work",
                output=out,
                paired_peers={
                    key: write(Path(scratch) / f"paired-{key}.json", value)
                    for key, value in (peers or {}).items()
                },
            )
            files = {
                p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
            }
            files |= {"LICENSE", "NOTICE", "ATTRIBUTIONS.md"}
            readme = (out / "README.md").read_text()
            return {
                **result,
                "readme": readme,
                "evaluation": (out / "evaluation/EVALUATION.md").read_text(),
                "manifest": json.loads((out / "evaluation/manifest.json").read_text()),
                "problems": card.check_rendered(readme, files),
            }

    def test_reference_fills_the_slot_with_the_no_1_0_statement(self):
        paired = {
            "point": {"delta": {"score": -1.5}},
            "ci95": {"low": -3.0, "high": 0.2},
        }
        result = self.build(entries(), paired)
        readme = result["readme"]
        self.assertEqual(result["problems"], [])
        self.assertIn(
            "level with the reference same-size model, Bosun (38.52; difference -7.50, "
            "paired 95% CI -3.00 to +0.20). There is no Decision 1.0 model at this size.",
            readme,
        )
        self.assertIn("- **Below Bosun in places:** typed decisions", readme)
        self.assertIn(
            "There is no Decision 1.0 model at this size. The DEV2.0-0.6B minus Bosun "
            "JevArena difference is -1.50 (paired 95% interval [-3.00, +0.20]).",
            result["evaluation"],
        )
        self.assertIn("## Results below Bosun", result["evaluation"])
        self.assertIsNone(result["manifest"]["decision_1_0"])
        self.assertNotIn("paired_vs_own_1_0", result["manifest"])
        self.assertEqual(result["manifest"]["paired_vs_reference"]["delta"], -1.5)
        self.assertTrue(result["tradeoffs"])

    def test_strongest_reference_is_the_strongest_other_model(self):
        items = entries(
            {
                "key": "gliner",
                "role": "reference",
                "report": str(REPORTS / "gliner25.json"),
                "repo_id": "fastino/GLiNER2.5-Decide",
            }
        )
        items[2] = {
            "key": "bosun",
            "role": "peer",
            "report": str(REPORTS / "bosun.json"),
            "repo_id": "Hanno-Labs/bosun-v3.1-0.6b",
            "label": "Bosun",
        }
        readme = self.build(items)["readme"]
        self.assertIn("the strongest other same-size model, GLiNER2.5-Decide", readme)
        self.assertNotIn("the reference same-size model", readme)

    def test_paired_intervals_of_the_other_peers_on_the_evaluation_page(self):
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
        result = self.build(entries(), peers={"gliner": gliner})
        self.assertIn(
            "Against the other models shown: GLiNER2.5-Decide", result["evaluation"]
        )
        self.assertIn("[-2.00, +1.00].", result["evaluation"])
        self.assertIn("gliner", str(result["manifest"]["paired_vs_peers"]).lower())
        self.assertNotIn("paired_vs_peers", self.build(entries())["manifest"])
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
        self.assertIn("There is no Decision 1.0 model at this size.", result["readme"])
        self.assertNotIn("Below", result["readme"])
        self.assertIsNone(result["manifest"]["paired_vs_reference"])

    def test_roles_follow_the_comparison(self):
        own = {
            "key": "kai1",
            "role": "own-1.0",
            "report": str(REPORTS / "kai1.json"),
            "repo_id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
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
