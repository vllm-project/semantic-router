"""Index-first gate profile (user decision 2026-10-02 09:55): I1 Index gain, I2 types, I3 audit (stdlib)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release import gate, layout

OK_TYPES = {kind: {"verdict": "OK"} for kind in ("choice", "noul", "score")}
REVISION = "9" * 40
NEW, OLD = "a" * 64, "b" * 64


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class IndexFirstGateTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        root = self.root = Path(self.scratch.name)
        self.run, self.current_run = root / "cand", root / "cur"
        report = write(self.run / "REPORT.json", {"v3": {"score": 53.9}})
        seal = write(self.run / "SEAL.json", {"panels": {}})
        write(self.current_run / "REPORT.json", {"v3": {"score": 50.2}})
        eos = write(root / "eos1/REPORT.json", {"v3": {"score": 40.0}})
        self.current_decision = write(
            root / "current.decision.json",
            {
                "report_sha256": layout.sha_file(self.current_run / "REPORT.json"),
                "identity": {"model_sha256": OLD},
            },
        )
        self.current_gate = write(
            root / "current.gate.json",
            {
                "schema": gate.GATE_SCHEMA,
                "repo_id": "llm-semantic-router/Decision-2.0-Eos-0.8B",
                "revision": REVISION,
                "decision_sha256": layout.sha_file(self.current_decision),
            },
        )
        receipt = {
            "schema": gate.INDEX_RECEIPT_SCHEMA,
            "results_sha256": "1" * 64,
            "model_source": {"model_sha256": NEW},
        }
        reference = {
            "schema": gate.INDEX_RECEIPT_SCHEMA,
            "results_sha256": "2" * 64,
            "model_source": {"model_sha256": OLD},
        }
        self.files = {
            "bootstrap": write(
                root / "boot.json",
                {
                    "schema": gate.INDEX_BOOT_SCHEMA,
                    "excluded": [],
                    "replicates": 2000,
                    "inputs_sha256": {"new": "1" * 64, "base": "2" * 64},
                    "headline": {"delta": 0.7, "ci95": [0.3, 1.1]},
                },
            ),
            "receipt": write(root / "receipt.json", receipt),
            "reference_receipt": write(root / "reference.json", reference),
            "types": write(
                root / "types.json",
                {"schema": gate.TYPES_SCHEMA, "run": str(self.run), "types": OK_TYPES},
            ),
            "audit": write(
                root / "audit.json",
                {
                    "schema": gate.CONTAMINATION_SCHEMA,
                    "planted_control": {"planted": 200, "found": 200, "missed": []},
                    "training_sets": {
                        "08bRA": {
                            "training_lines": 9,
                            "duplicate_rows": 1,
                            "item_rows": 0,
                        }
                    },
                },
            ),
            "reference_v3": write(root / "paired-v3.json", {"v3": "reference"}),
        }
        own = write(root / "p-eos1.json", {"own": 1})
        self.decision = root / "decision.json"
        self.spec = {
            "kind": "release",
            "model_name": "Decision-2.0-Eos-0.8B",
            "repo_id": "llm-semantic-router/Decision-2.0-Eos-0.8B",
            "expected_identity": {"model_sha256": NEW},
            "scored": {
                "report_sha256": layout.sha_file(report),
                "seal_sha256": layout.sha_file(seal),
            },
            "gate_receipt": str(self.decision),
            "card": {
                "paired": str(own),
                "reports": [
                    {"key": "candidate", "role": "candidate", "report": str(report)},
                    {"key": "eos1", "role": "own-1.0", "report": str(eos)},
                ],
            },
            "gate_profile": {
                "name": "index_first",
                "run": str(self.run),
                "current": {
                    "revision": REVISION,
                    "gate": str(self.current_gate),
                    "decision": str(self.current_decision),
                    "run": str(self.current_run),
                },
                "index": {
                    k: str(self.files[k])
                    for k in ("bootstrap", "receipt", "reference_receipt")
                },
                "types": str(self.files["types"]),
                "contamination": {
                    "audit": str(self.files["audit"]),
                    "training_set": "08bRA",
                },
                "references": {"v3": str(self.files["reference_v3"])},
            },
        }

    def tearDown(self):
        self.scratch.cleanup()

    def items(self):
        return gate.index_first_items(self.spec, gate.gate_profile(self.spec))

    def failing(self):
        return [name for name, item in self.items().items() if not item["passed"]]

    def edit(self, name: str, **changes):
        path = Path(self.files[name])
        value = json.loads(path.read_text())
        for key, item in changes.items():
            target = value
            *parents, last = key.split(".")
            for parent in parents:
                target = target[parent]
            target[last] = item
        path.write_text(json.dumps(value))

    def test_all_pass_and_print_no_index_value(self):
        items = self.items()
        self.assertEqual(list(items), list(gate.INDEX_FIRST_ITEMS))
        self.assertEqual(self.failing(), [], items)
        evidence = items["1_index_first_I1_index_gain"]["evidence"]
        self.assertIn("95% lower bound > 0: yes", evidence)
        for value in ("0.3", "0.7", "1.1"):
            self.assertNotIn(value, evidence)

    def test_each_item_fails_on_its_criterion(self):
        i1, i2, i3 = gate.INDEX_FIRST_ITEMS
        cases = [
            ("bootstrap", {"headline.ci95": [-0.1, 1.1]}, i1),
            ("bootstrap", {"excluded": ["HoVer"]}, i1),
            ("bootstrap", {"replicates": 1000}, i1),
            ("bootstrap", {"inputs_sha256.new": "3" * 64}, i1),
            ("receipt", {"model_source.model_sha256": "c" * 64}, i1),
            ("reference_receipt", {"model_source.model_sha256": "c" * 64}, i1),
            ("types", {"types.score.verdict": "COLLAPSED"}, i2),
            ("types", {"run": str(self.root / "elsewhere")}, i2),
            ("audit", {"planted_control.found": 199}, i3),
            ("audit", {"schema": "other"}, i3),
        ]
        for name, changes, item in cases:
            with self.subTest(name=name, changes=changes):
                original = Path(self.files[name]).read_text()
                self.edit(name, **changes)
                try:
                    self.assertEqual(self.failing(), [item])
                finally:
                    Path(self.files[name]).write_text(original)

    def test_missing_training_set_fails_the_audit(self):
        self.spec["gate_profile"]["contamination"]["training_set"] = "2bRA"
        self.assertEqual(self.failing(), [gate.INDEX_FIRST_ITEMS[2]])

    def test_broken_current_chain_fails_every_item(self):
        self.current_gate.write_text(
            json.dumps(
                {**json.loads(self.current_gate.read_text()), "revision": "8" * 40}
            )
        )
        self.assertEqual(sorted(self.failing()), sorted(gate.INDEX_FIRST_ITEMS))

    def test_decision_binds_profile_revision_and_evidence(self):
        profile = gate.gate_profile(self.spec)
        evidence = gate.index_first_evidence_sha256(profile)
        self.assertIn("reference_v3", evidence)
        self.assertIn("index_bootstrap", evidence)
        decision = {
            "schema": gate.DECISION_SCHEMA,
            "status": "final",
            "decision": "release",
            "model_name": self.spec["model_name"],
            "repo_id": self.spec["repo_id"],
            "identity": self.spec["expected_identity"],
            "report_sha256": self.spec["scored"]["report_sha256"],
            "paired_sha256": layout.sha_file(Path(self.spec["card"]["paired"])),
            "gate_profile": "index_first",
            "current_revision": REVISION,
            "evidence_sha256": evidence,
            "rationale": "Index-first rule",
            "decided_by": "coordinator",
        }
        write(self.decision, decision)
        gate.check(self.spec, self.decision, final=True)
        for change in (
            {"gate_profile": "successor"},
            {"current_revision": "8" * 40},
            {"evidence_sha256": {**evidence, "types": "0" * 64}},
        ):
            with self.subTest(change=change):
                write(self.decision, {**decision, **change})
                with self.assertRaises(ValueError):
                    gate.check(self.spec, self.decision, final=True)

    def test_profile_needs_every_input(self):
        for edit in (
            lambda p: p["index"].pop("reference_receipt"),
            lambda p: p["contamination"].pop("training_set"),
            lambda p: p["current"].update(revision="abc"),
            lambda p: p.update(references={"v3": None}),
            lambda p: p.pop("types"),
        ):
            spec = json.loads(json.dumps(self.spec))
            edit(spec["gate_profile"])
            with self.subTest(profile=spec["gate_profile"]), self.assertRaises(
                ValueError
            ):
                gate.gate_profile(spec)

    def test_first_items_dispatch(self):
        items, below = gate.first_items(self.spec)
        self.assertEqual(list(items), list(gate.INDEX_FIRST_ITEMS))
        self.assertEqual(below, "own 1.0")


if __name__ == "__main__":
    unittest.main()
