"""Offline regression tests for experiment identity and evidence integrity."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .__main__ import main
from .config import validate_config
from .fixture import fixture_records
from .plan import build_plan
from .records import validate_records
from .validation import ContractError, load_json

EXAMPLE = Path(__file__).parent / "testdata" / "synthetic.json"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = load_json(EXAMPLE)
        self.plan = build_plan(self.config, "fixture-revision", ["fixture"])

    def test_matrix_and_identity(self):
        self.assertEqual(len(self.plan["matrix"]), 8)
        reordered = dict(reversed(list(self.config.items())))
        self.assertEqual(
            self.plan, build_plan(reordered, "fixture-revision", ["fixture"])
        )
        self.assertEqual(
            self.plan["experiment_id"],
            build_plan(self.config, "fixture-revision", ["another-command"])[
                "experiment_id"
            ],
        )
        for field, value in (("id", "different"), ("seeds", [42])):
            changed = copy.deepcopy(self.config)
            changed[field] = value
            self.assertNotEqual(
                self.plan["experiment_id"],
                build_plan(changed, "fixture-revision", ["fixture"])["experiment_id"],
            )
        self.config["models"][0]["revision"] = "new-model"
        self.assertNotEqual(
            self.plan["experiment_id"],
            build_plan(self.config, "fixture-revision", ["fixture"])["experiment_id"],
        )

    def test_manifest_does_not_alias_input(self):
        self.config["models"][0]["revision"] = "changed"
        self.assertNotEqual(self.config, self.plan["config"])

    def test_identity_tracks_protocol_budget_and_code(self):
        mutations = [
            lambda c: c["budgets"][0].update(max_calls=9),
            lambda c: c["scorer"].update(revision="updated"),
            lambda c: c["models"][0]["pricing"].update(input_per_million=1),
        ]
        for mutation in mutations:
            config = copy.deepcopy(self.config)
            mutation(config)
            self.assertNotEqual(
                self.plan["experiment_id"],
                build_plan(config, "fixture-revision", ["fixture"])["experiment_id"],
            )
        self.assertNotEqual(
            self.plan["experiment_id"],
            build_plan(self.config, "another-revision", ["fixture"])["experiment_id"],
        )

    def test_module_cli_exit_codes(self):
        command = [sys.executable, "-m", __package__, "validate", "--config"]
        success = subprocess.run(
            [*command, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertTrue(json.loads(success.stdout)["valid"])
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            invalid.write_text('{"schema_version": "wrong"}', encoding="utf-8")
            failure = subprocess.run(
                [*command, str(invalid)], capture_output=True, text=True, check=False
            )
        self.assertEqual(failure.returncode, 2)
        self.assertIn("looper-tts:", failure.stderr)
        self.assertNotIn("Traceback", failure.stderr)

    def test_invalid_config(self):
        mutations = [
            lambda c: c.update(schema_version="v2"),
            lambda c: c.update(unexpected=True),
            lambda c: c["models"].append(copy.deepcopy(c["models"][0])),
            lambda c: c["arms"][0].update(model_ids=["missing"]),
            lambda c: c["arms"].pop(),
            lambda c: c["budgets"].pop(),
            lambda c: c["budgets"][0].update(max_calls=True),
            lambda c: c["budgets"][0].update(max_total_tokens=-1),
            lambda c: c["dataset"]["calibration_ids"].append("q1"),
            lambda c: c["dataset"]["items"][0].update(prompt="changed"),
            lambda c: c["arms"][1]["parameters"].update(threshold=1.1),
            lambda c: c["arms"][2]["parameters"].update(breadth=[0]),
            lambda c: c["models"][0]["pricing"].update(input_per_million=float("nan")),
            lambda c: c.update(seeds=[1, 1]),
        ]
        for mutation in mutations:
            config = copy.deepcopy(self.config)
            mutation(config)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_config(config)

    def test_fixture_roundtrip_preserves_unknown_and_zero(self):
        records = fixture_records(self.plan)
        roundtrip = json.loads(json.dumps(records, allow_nan=False))
        self.assertEqual(validate_records(roundtrip, self.plan), records)
        self.assertEqual(roundtrip["calls"][0]["usage"]["completion_tokens"], 0)
        self.assertIsNone(roundtrip["calls"][1]["usage"]["completion_tokens"])
        self.assertEqual(
            {r["status"] for r in records["results"]},
            {"success", "error", "budget_exhausted"},
        )
        self.assertTrue(any(c["status"] == "cached" for c in records["calls"]))

    def test_broken_evidence_links_and_states(self):
        mutations = [
            lambda b: b.update(experiment_id="wrong"),
            lambda b: b.update(evidence_kind="benchmark"),
            lambda b: b["results"].pop(),
            lambda b: b["calls"][0].update(item_id="missing"),
            lambda b: b["calls"][0].update(model_id="missing"),
            lambda b: b["calls"][0].update(attempt=0),
            lambda b: b["calls"][0]["usage"].update(total_tokens=-1),
            lambda b: b["calls"][3].update(cache_id=None),
            lambda b: b["calls"][3].update(latency_ms=0),
            lambda b: b["results"][0].update(call_ids=[b["calls"][1]["id"]]),
            lambda b: b["results"][0].update(call_ids=[]),
            lambda b: b["results"][0].update(score=2),
            lambda b: b["results"][1].update(score=1),
            lambda b: b["results"][2].update(budget_status="within"),
            lambda b: b["results"][0].update(scorer_id="different"),
        ]
        for mutation in mutations:
            records = fixture_records(self.plan)
            mutation(records)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_records(records, self.plan)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"id": "a", "id": "b"}', encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                load_json(path)

    def test_cli_offline_plan_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args = [
                "plan",
                "--config",
                str(EXAMPLE),
                "--output",
                directory,
                "--code-revision",
                "fixture-revision",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate", "--config", str(EXAMPLE)]), 0)
                self.assertEqual(main(args), 0)
            manifest = load_json(Path(directory) / "manifest.json")
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(len(manifest["matrix"]), 8)
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(args), 2)
            self.assertIn("refusing to overwrite", error.getvalue())
            self.assertEqual(manifest, load_json(Path(directory) / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
