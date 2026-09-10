"""Regression coverage for frozen identity, usage bounds and stage roles."""

import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from .fixture import fixture_records
from .plan import build_plan, validate_plan
from .records import validate_records
from .validation import ContractError, digest, load_json


class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_json(Path(__file__).parent / "testdata" / "synthetic.json")
        self.plan = build_plan(self.config, "fixture-revision", ["fixture"])
        self.records = fixture_records(self.plan)

    def test_rejects_config_tampering_even_with_updated_config_digest(self):
        mutations = [
            lambda c: c["budgets"][0].update(max_total_tokens=99999),
            lambda c: c["models"][0].update(revision="other-model"),
            lambda c: c["scorer"].update(revision="other-scorer"),
            lambda c: c["dataset"]["items"][0].update(id="other-item"),
            lambda c: c["dataset"]["items"][0].update(
                prompt="changed", prompt_sha256=hashlib.sha256(b"changed").hexdigest()
            ),
        ]
        for mutation in mutations:
            for update_digest in (False, True):
                plan = copy.deepcopy(self.plan)
                mutation(plan["config"])
                if update_digest:
                    plan["config_sha256"] = digest(plan["config"])
                with self.subTest(
                    mutation=mutation, update_digest=update_digest
                ), self.assertRaises(ContractError):
                    validate_records(self.records, plan)

    def test_rejects_invalid_config_with_fully_recomputed_identity(self):
        plan = copy.deepcopy(self.plan)
        plan["config"]["budgets"][0]["max_calls"] = -1
        plan["config_sha256"] = digest(plan["config"])
        plan["experiment_id"] = digest(
            {key: plan[key] for key in ("config", "code_revision", "planner_sha256")}
        )
        with self.assertRaisesRegex(ContractError, "max_calls"):
            validate_records(self.records, plan)

    def test_rejects_manifest_and_matrix_tampering(self):
        mutations = [
            lambda p: p.update(config_sha256="0" * 64),
            lambda p: p.update(experiment_id="0" * 64),
            lambda p: p.update(planner_sha256="0" * 64),
            lambda p: p.update(code_revision="other-revision"),
            lambda p: p.update(schema_version="unsupported"),
            lambda p: p.update(status="finished"),
            lambda p: p.update(command=[]),
            lambda p: p["matrix"].pop(),
            lambda p: p["matrix"].append(copy.deepcopy(p["matrix"][0])),
            lambda p: p["matrix"][0].update(id="0" * 64),
            lambda p: p["matrix"][0].update(arm_id="fusion"),
            lambda p: p["matrix"][0].update(budget_id="large"),
            lambda p: p["matrix"][0].update(seed=42),
            lambda p: p["matrix"][0].update(algorithm="fusion"),
            lambda p: p["matrix"][0].update(item_ids=[]),
            lambda p: p["matrix"][0].update(extra="ignored?"),
        ]
        for mutation in mutations:
            plan = copy.deepcopy(self.plan)
            mutation(plan)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_records(self.records, plan)

    def test_saved_plan_does_not_depend_on_current_source(self):
        plan = json.loads(json.dumps(self.plan))
        with patch.object(
            Path, "glob", side_effect=AssertionError("read local source")
        ):
            self.assertEqual(validate_plan(plan), self.plan)
            self.assertEqual(validate_records(self.records, plan), self.records)

    def test_known_token_total_lower_bound(self):
        for total, valid in ((1, False), (8191, False), (8192, True), (9000, True)):
            records = copy.deepcopy(self.records)
            records["calls"][0]["usage"] = {
                "prompt_tokens": 4096,
                "completion_tokens": 4096,
                "total_tokens": total,
            }
            with self.subTest(total=total):
                if valid:
                    validate_records(records, self.plan)
                else:
                    with self.assertRaisesRegex(ContractError, "total_tokens"):
                        validate_records(records, self.plan)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            records = copy.deepcopy(self.records)
            records["calls"][0]["usage"][key] = None
            validate_records(records, self.plan)

    def test_algorithm_stages_and_distinct_fusion_roles(self):
        for model_id in ("judge", "synthesis"):
            model = copy.deepcopy(self.config["models"][0])
            model["id"] = model_id
            self.config["models"].append(model)
        fusion = self.config["arms"][3]
        fusion["model_ids"] = ["model-a", "judge", "synthesis"]
        fusion["parameters"].update(
            judge_model_id="judge", synthesis_model_id="synthesis"
        )
        plan = build_plan(self.config, "fixture-revision", ["fixture"])
        records = fixture_records(plan)
        allowed = {
            "direct": {("generate", "model-a")},
            "confidence": {("generate", "model-a"), ("verify", "model-a")},
            "remom": {("generate", "model-a"), ("synthesize", "model-a")},
            "fusion": {
                ("generate", "model-a"),
                ("judge", "judge"),
                ("synthesize", "synthesis"),
            },
        }
        cells = {cell["id"]: cell for cell in plan["matrix"]}
        for index, call in enumerate(records["calls"]):
            algorithm = cells[call["cell_id"]]["algorithm"]
            for stage in ("generate", "verify", "select", "judge", "synthesize"):
                for model in ("model-a", "judge", "synthesis"):
                    changed = copy.deepcopy(records)
                    changed["calls"][index].update(stage=stage, model_id=model)
                    with self.subTest(algorithm=algorithm, stage=stage, model=model):
                        if (stage, model) in allowed[algorithm]:
                            validate_records(changed, plan)
                        else:
                            with self.assertRaises(ContractError):
                                validate_records(changed, plan)


if __name__ == "__main__":
    unittest.main()
