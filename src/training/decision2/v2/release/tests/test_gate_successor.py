"""Successor gate profile: R1-R7 against the current revision's scored run (stdlib)."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.release import gate, layout

OK_TYPES = {kind: {"verdict": "OK"} for kind in ("choice", "noul", "score")}
REVISION = "9" * 40


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def paired(left: float, right: float, low: float, h_high: float, runs: tuple) -> dict:
    return {
        "schema_version": gate.PAIRED_SCHEMA,
        "point": {
            "left": {"score": left},
            "right": {"score": right},
            "delta": {"score": left - right},
        },
        "ci95": {"low": low, "high": low + 6},
        "axis_ci95": {"H": {"delta": {"low": h_high - 0.1, "high": h_high}}},
        "runs": {"left": str(runs[0]), "right": str(runs[1])},
        "predictions_sha256": {"left": {"typed": "t" * 64, "css": "c" * 64}},
    }


class SuccessorGateTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        root = self.root = Path(self.scratch.name)
        self.run, self.current_run = root / "m8", root / "m4"
        report = write(self.run / "REPORT.json", {"v3": {"score": 48.64}})
        seal = write(self.run / "SEAL.json", {"panels": {}})
        write(self.current_run / "REPORT.json", {"v3": {"score": 43.54}})
        kai = write(root / "kai/REPORT.json", {"v3": {"score": 35.94}})
        gliner = write(root / "gliner/REPORT.json", {"v3": {"score": 42.52}})
        mlx_current = write(root / "m4-mlx.jsonl", {"x": 1})
        self.current_decision = write(
            root / "current.decision.json",
            {"report_sha256": layout.sha_file(self.current_run / "REPORT.json")},
        )
        self.current_gate = write(
            root / "current.gate.json",
            {
                "schema": gate.GATE_SCHEMA,
                "repo_id": "llm-semantic-router/DEV2.0-0.6B",
                "revision": REVISION,
                "decision_sha256": layout.sha_file(self.current_decision),
            },
        )
        files = {
            "paired": write(
                root / "p-current.json",
                paired(48.64, 43.54, 2.65, 0.06, (self.run, self.current_run)),
            ),
            "types": write(
                root / "types.json",
                {"schema": gate.TYPES_SCHEMA, "run": str(self.run), "types": OK_TYPES},
            ),
            "mlx_paired": write(
                root / "mlx.json",
                {
                    "bootstrap": {"card_macro_ci95": [-0.006, 0.014]},
                    "delta": {"card_macro": 0.004},
                    "runs": {
                        "candidate": {"predictions_sha256": "m" * 64},
                        "released": {
                            "predictions_sha256": layout.sha_file(mlx_current)
                        },
                    },
                },
            ),
            "exposure": write(
                root / "exposure.json",
                {
                    "schema": gate.EXPOSURE_SCHEMA,
                    "groups": [],
                    "matched_rows": {},
                    "methods_agree": True,
                    "files": [1] * 6,
                },
            ),
            "public231": write(
                root / "public.json",
                {
                    "schema": gate.PUBLIC231_SCHEMA,
                    "left_correct": 152,
                    "right_correct": 142,
                    "mcnemar_exact_p": 0.087,
                    "verdict": "OK",
                    "runs": {"left": str(self.run), "right": str(self.current_run)},
                },
            ),
        }
        self.files = files
        own = write(
            root / "p-kai.json",
            paired(48.64, 35.94, 10.0, 0.19, (self.run, root / "kai")),
        )
        tier = write(
            root / "p-gliner.json",
            paired(48.64, 42.52, 4.24, 0.16, (self.run, root / "gliner")),
        )
        self.decision = root / "decision.json"
        self.spec = {
            "kind": "release",
            "model_name": "DEV2.0-0.6B",
            "repo_id": "llm-semantic-router/DEV2.0-0.6B",
            "expected_identity": {"model_sha256": "a" * 64},
            "scored": {
                "report_sha256": layout.sha_file(report),
                "seal_sha256": layout.sha_file(seal),
                "predictions_sha256": {
                    "typed-final": "t" * 64,
                    "css15": "c" * 64,
                    "mlx-diag": "m" * 64,
                },
            },
            "gate_receipt": str(self.decision),
            "card": {
                "paired": str(own),
                "reports": [
                    {"key": "m8", "role": "candidate", "report": str(report)},
                    {"key": "kai1", "role": "own-1.0", "report": str(kai)},
                    {
                        "key": "gliner25",
                        "role": "peer",
                        "report": str(gliner),
                        "label": "GLiNER2.5-Decide",
                    },
                ],
            },
            "gate_profile": {
                "name": "successor",
                "run": str(self.run),
                "current": {
                    "revision": REVISION,
                    "gate": str(self.current_gate),
                    "decision": str(self.current_decision),
                    "run": str(self.current_run),
                    "mlx_predictions": str(mlx_current),
                },
                **{name: str(path) for name, path in files.items()},
                "tier": {"reference": "gliner25", "v3_share": 0.9, "paired": str(tier)},
            },
        }

    def tearDown(self):
        self.scratch.cleanup()

    def items(self):
        return gate.successor_items(self.spec, gate.gate_profile(self.spec))

    def edit(self, name: str, **changes):
        path = Path(self.files[name]) if name in self.files else Path(name)
        value = json.loads(path.read_text())
        for key, item in changes.items():
            target = value
            *parents, last = key.split(".")
            for parent in parents:
                target = target[parent]
            target[last] = item
        path.write_text(json.dumps(value))

    def failing(self):
        return [name for name, item in self.items().items() if not item["passed"]]

    def test_all_pass(self):
        items = self.items()
        self.assertEqual(list(items), list(gate.SUCCESSOR_ITEMS))
        self.assertEqual(self.failing(), [], items)

    def test_each_rule_fails_on_its_criterion(self):
        cases = {
            "1_successor_R1_v3": ("paired", {"ci95.low": -0.1}),
            "1_successor_R3_no_type_collapsed": (
                "types",
                {"types.score": {"verdict": "COLLAPSED"}},
            ),
            "1_successor_R4_mlx_diag": (
                "mlx_paired",
                {"bootstrap.card_macro_ci95": [-0.03, -0.001]},
            ),
            "1_successor_R6_no_overlap_exposure": (
                "exposure",
                {"groups": [{"group": "g"}]},
            ),
            "1_successor_R7_public231": (
                "public231",
                {"left_correct": 130, "mcnemar_exact_p": 0.01, "verdict": "REGRESSION"},
            ),
        }
        for item, (name, changes) in cases.items():
            with self.subTest(item=item):
                self.setUp()
                self.edit(name, **changes)
                failing = self.failing()
                self.assertIn(item, failing)
                if item != "1_successor_R3_no_type_collapsed":
                    self.assertEqual(failing, [item])

    def test_human_transfer_below(self):
        self.edit("paired", **{"axis_ci95.H.delta.high": -0.01})
        self.assertEqual(self.failing(), ["1_successor_R2_human_transfer"])

    def test_tier_gates(self):
        for path, changes in (
            (lambda: self.spec["card"]["paired"], {"ci95.low": -1.0}),
            (
                lambda: self.spec["gate_profile"]["tier"]["paired"],
                {"axis_ci95.H.delta.high": -0.01},
            ),
        ):
            with self.subTest(changes=changes):
                self.setUp()
                self.edit(path(), **changes)
                self.assertEqual(self.failing(), ["1_successor_R5_tier_gates"])
        self.setUp()
        self.spec["gate_profile"]["tier"]["v3_share"] = 1.0
        write(Path(self.spec["card"]["reports"][2]["report"]), {"v3": {"score": 50.0}})
        self.edit(
            self.spec["gate_profile"]["tier"]["paired"], **{"point.right.score": 50.0}
        )
        self.assertEqual(self.failing(), ["1_successor_R5_tier_gates"])

    def test_paired_against_another_run_fails(self):
        self.edit("paired", **{"runs.right": str(self.root / "other")})
        self.assertIn("1_successor_R1_v3", self.failing())

    def test_contradicting_stored_verdict_fails(self):
        self.edit("public231", left_correct=130, mcnemar_exact_p=0.01)
        self.assertEqual(self.failing(), ["1_successor_R7_public231"])

    def test_current_revision_chain(self):
        for change in (
            lambda: self.edit(str(self.current_gate), revision="8" * 40),
            lambda: self.edit(str(self.current_decision), extra=1),
            lambda: self.edit(str(self.current_decision), report_sha256="0" * 64),
        ):
            with self.subTest():
                self.setUp()
                change()
                self.assertEqual(sorted(self.failing()), sorted(gate.SUCCESSOR_ITEMS))

    def decision_for(self, **changes) -> Path:
        profile = gate.gate_profile(self.spec)
        value = {
            "schema": gate.DECISION_SCHEMA,
            "decision": "release",
            "model_name": "DEV2.0-0.6B",
            "repo_id": "llm-semantic-router/DEV2.0-0.6B",
            "identity": {"model_sha256": "a" * 64},
            "report_sha256": self.spec["scored"]["report_sha256"],
            "paired_sha256": layout.sha_file(Path(self.spec["card"]["paired"])),
            "rationale": "successor rule",
            "decided_by": "coordinator",
            "gate_profile": "successor",
            "current_revision": REVISION,
            "evidence_sha256": gate.evidence_sha256(profile),
            **changes,
        }
        return write(self.decision, value)

    def test_decision_binds_profile_revision_and_evidence(self):
        gate.check(self.spec, self.decision_for(), final=True)
        evidence = gate.evidence_sha256(gate.gate_profile(self.spec))
        for changes in (
            {"gate_profile": None},
            {"current_revision": "8" * 40},
            {"evidence_sha256": {**evidence, "types": "0" * 64}},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    gate.check(self.spec, self.decision_for(**changes), final=True)

    def test_profile_validation(self):
        for change in (
            lambda s: s["gate_profile"].update(name="other"),
            lambda s: s["gate_profile"]["tier"].update(reference="missing"),
            lambda s: s["gate_profile"]["current"].update(revision="abc"),
        ):
            spec = json.loads(json.dumps(self.spec))
            change(spec)
            with self.subTest(), self.assertRaises(ValueError):
                gate.gate_profile(spec)

    def test_profile_cli(self):
        spec_path = write(self.root / "spec.json", self.spec)
        for expected in (0, 1):
            if expected:
                self.edit("paired", **{"ci95.low": -0.1})
            argv = ["gate", "profile", "--spec", str(spec_path)]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ) as out:
                with self.assertRaises(SystemExit) as stop:
                    gate.main()
            self.assertEqual(stop.exception.code, expected)
            self.assertEqual(json.loads(out.getvalue())["passed"], not expected)

    def test_own_1_0_profile_unchanged(self):
        del self.spec["gate_profile"]
        items, below = gate.first_items(self.spec)
        self.assertEqual(list(items), ["1_beats_own_1_0"])
        self.assertTrue(items["1_beats_own_1_0"]["passed"])
        self.assertEqual(below, "own 1.0")


class SuccessorNoOwnAndC1Test(SuccessorGateTest):
    """A tier without a Decision 1.0 model, and item 8 (C1 post-key) when named."""

    def setUp(self):
        super().setUp()
        profile = self.spec["gate_profile"]
        profile["tier"]["no_own_1_0"] = True
        self.spec["card"]["paired"] = profile["tier"]["paired"]
        self.summary = write(
            self.root / "c1" / "SUMMARY.json",
            {
                "schema": gate.C1_SUMMARY_SCHEMA,
                "role": "successor",
                "c1": 58.1,
                "item8": {
                    "verdict": "PASS",
                    "delta": 0.8,
                    "ci95": [-0.9, 2.4],
                    "name": "base",
                },
                "baseline_entry": {"identity": "a" * 64},
            },
        )
        profile["c1_postkey"] = str(self.summary)

    def test_all_pass(self):
        items = self.items()
        self.assertEqual(list(items), [*gate.SUCCESSOR_ITEMS, gate.SUCCESSOR_C1])
        self.assertEqual(self.failing(), [], items)
        self.assertIn("no Decision 1.0", items["1_successor_R5_tier_gates"]["evidence"])
        self.assertIn("c1_postkey", gate.evidence_sha256(gate.gate_profile(self.spec)))

    def test_c1_regression_or_other_weights_fail(self):
        self.edit(str(self.summary), **{"item8.verdict": "REGRESSION"})
        self.assertEqual(self.failing(), [gate.SUCCESSOR_C1])
        self.edit(
            str(self.summary),
            **{"item8.verdict": "PASS", "baseline_entry.identity": "b" * 64},
        )
        self.assertEqual(self.failing(), [gate.SUCCESSOR_C1])

    def test_card_must_compare_with_the_reference(self):
        own = write(
            self.root / "p-other.json",
            paired(48.64, 35.94, 10.0, 0.19, (self.run, self.root / "kai")),
        )
        self.spec["card"]["paired"] = str(own)
        self.assertEqual(self.failing(), ["1_successor_R5_tier_gates"])

    def test_mlx_pairing_in_the_9b_schema(self):
        mlx_current = self.spec["gate_profile"]["current"]["mlx_predictions"]
        write(
            Path(self.files["mlx_paired"]),
            {
                "schema": gate.MLX_PAIRED_9B,
                "types": ["choice", "noul"],
                "overall": {"ci95": {"low": -0.029, "high": 0.004}, "delta": -0.012},
                "left": {"predictions_sha256": "m" * 64},
                "right": {"predictions_sha256": layout.sha_file(Path(mlx_current))},
            },
        )
        self.assertEqual(self.failing(), [])
        self.edit("mlx_paired", **{"overall.ci95": {"low": -0.03, "high": -0.001}})
        self.assertEqual(self.failing(), ["1_successor_R4_mlx_diag"])

    # The own-1.0 variants of these are covered by SuccessorGateTest.
    def test_each_rule_fails_on_its_criterion(self):
        pass

    def test_current_revision_chain(self):
        pass

    def test_tier_gates(self):
        pass


if __name__ == "__main__":
    unittest.main()
