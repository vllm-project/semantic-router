"""Matched-budget enrollment checks use only sealed split metadata and one dev arm."""

import argparse
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from recipe_lab import loop, search_protocol
from recipe_lab.loop_evidence import digest
from tests import test_loop as fixtures
from tests.test_signature import encode_rows, receipts


class SearchProtocolTest(unittest.TestCase):
    def setUp(self):
        self.run_fixture = run = fixtures.LoopTest()
        run.setUp()
        self.addCleanup(run.doCleanups)
        cfg = run.root / "draft/config.yaml"
        doc = fixtures.get(cfg)
        r = doc["recipes"][1]
        r["signals"] = {
            "conversation": [
                {
                    "name": n,
                    "feature": {
                        "type": "count",
                        "source": {"type": "assistant_tool_cycle"},
                    },
                    "predicate": {"gte": g},
                }
                for n, g in (("seen", 1), ("late", 4))
            ]
        }
        r["decisions"][0]["rules"] = {
            "operator": "AND",
            "conditions": [
                {"type": "conversation", "name": "seen"},
                {
                    "operator": "NOT",
                    "conditions": [{"type": "conversation", "name": "late"}],
                },
            ],
        }
        fixtures.put(cfg, doc)
        binding = run.root / "draft/dsh/bindings/task/draft.json"
        fixtures.put(
            binding,
            {**fixtures.get(binding), "source_sha256": digest(cfg.read_bytes())},
        )
        folder = run.root / "fixed/dsh/runs/task/fixed/main"
        logs = receipts([1, 3, 4, 7, 8, 2])
        del logs[-2]["roles"]
        (folder / "requests.jsonl").write_bytes(encode_rows(logs))
        fixtures.put(
            folder / "result.json",
            {**fixtures.get(folder / "result.json"), "model_requests": 6},
        )
        self.protocol = {
            "schema": "recipe-lab-search-preflight-v1",
            "study": "demo",
            "based_on": run.record("draft"),
            "prefixes": run.record("fixed"),
            "space": "space.yaml",
            "seed": 42,
            "budget": {"episodes": 4, "proposal_model_tokens": 2000},
            "split_plans": {"dev": None, "holdout": None},
        }
        self.space = {
            "schema": "recipe-lab-window-space-v1",
            "action": "tool_result_window_upper",
            "base_config_sha256": digest(cfg.read_bytes()),
            "public_model": "public",
            "recipe": "main",
            "decision": "fast",
            "lower_signal": "seen",
            "upper_signal": "late",
            "allowed_upper": [3, 8],
        }

    def result(self, *target_prefixes, trials=None, manifest=None, manifest_sha=None):
        root = self.run_fixture.root
        fixtures.put(root / "space.yaml", self.space)
        self.protocol["space_sha256"] = digest((root / "space.yaml").read_bytes())
        path = root / "protocol.json"
        fixtures.put(path, self.protocol)
        return search_protocol.preflight(
            loop.Ledger(self.run_fixture.workspace),
            argparse.Namespace(
                protocol=path,
                target_prefixes=target_prefixes,
                candidate_trials_per_method=trials,
                target_dev_manifest=manifest,
                target_dev_manifest_sha256=manifest_sha,
            ),
        )

    def target_record(
        self,
        identity,
        counts,
        *,
        split="dev",
        repo=None,
        template_sha=None,
        template_file="prompt-template.txt"
    ):
        run = self.run_fixture
        src = run.root / ("prefix-" + identity[0] + str(len(counts)) + split)
        shutil.copytree(run.root / "fixed", src)
        shutil.copyfile(run.root / "draft/config.yaml", src / "config.yaml")
        deployment = src / "dsh/bindings/task/fixed.json"
        fixtures.put(
            deployment,
            dict(
                fixtures.get(deployment),
                source_sha256=digest((src / "config.yaml").read_bytes()),
            ),
        )
        frozen = src / "dsh/frozen.json"
        packet = fixtures.get(frozen)
        packet["cases"]["task"]["instruction_sha256"] = identity
        if template_sha:
            packet.update(split=split, solver_files={template_file: template_sha})
        if repo:
            packet["cases"]["task"]["repo"] = repo
        fixtures.put(frozen, packet)
        folder = src / "dsh/runs/task/fixed/main"
        for name in ("protocol.json", "result.json"):
            f = folder / name
            doc = fixtures.get(f)
            doc["prompt_sha256"] = identity
            if name == "protocol.json":
                doc["verified_task_packet_sha256"] = digest(frozen.read_bytes())
            else:
                doc["model_requests"] = len(counts)
            fixtures.put(f, doc)
        (folder / "requests.jsonl").write_bytes(encode_rows(receipts(counts)))
        return run.cli(
            "record",
            "--run",
            src,
            "--arm",
            "fixed",
            "--config",
            src / "config.yaml",
            "--public-model",
            "public",
            "--strong",
            "A",
            "--split",
            split,
            "--hypothesis",
            "target prefix collection",
        )["record_id"]

    def split(self, role, ids):
        path = self.run_fixture.root / (role + ".json")
        fixtures.put(
            path,
            {
                "schema": "recipe-lab-search-split-v1",
                "methods": search_protocol.METHODS,
                "study": "demo",
                "split": role,
                "instruction_sha256": ids,
            },
        )
        self.protocol["split_plans"][role] = {
            "file": path.name,
            "sha256": digest(path.read_bytes()),
        }

    def test_single_action_blocks_comparison_even_with_seed_and_budget(self):
        r = self.result()
        self.assertEqual(r["distinct_new_classes"], 1)
        self.assertEqual(r["coverage_by_task"]["task"]["unknown"], 1)
        self.assertEqual(
            r["blockers"],
            [
                "insufficient_distinguishable_new_actions",
                "needs_independent_dev_plan",
                "needs_independent_holdout_plan",
            ],
        )
        self.assertEqual(r["comparison"]["random_class_order"], [8])
        self.assertEqual(r, self.result())

    def test_shared_action_classes_random_order_and_blind_split(self):
        self.space["allowed_upper"] = [3, 4, 6, 8]
        self.split("dev", ["b" * 64])
        self.split("holdout", ["c" * 64])
        r = self.result()
        self.assertEqual(r["status"], "protocol_ready")
        self.assertEqual(r["target_development"]["status"], "not_checked")
        self.assertIsNone(r["paid_search_authorized"])
        self.assertFalse(r["search_executed"])
        self.assertEqual(r["distinct_new_classes"], 2)
        self.assertIn([4, 6], r["observed_equivalence_classes"].values())
        self.assertCountEqual(r["comparison"]["random_class_order"], [4, 8])
        self.assertEqual(
            r["comparison"]["random_class_order"],
            self.result()["comparison"]["random_class_order"],
        )
        self.split("holdout", ["b" * 64])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.result()

    def test_target_prefixes_recompute_classes_and_per_method_choice(self):
        self.space["allowed_upper"] = [1, 3, 4, 6, 8]
        dev = "b" * 64
        self.split("dev", [dev])
        self.split("holdout", ["c" * 64])
        flat = self.target_record(dev, [0])
        r = self.result(flat, trials=2)
        self.assertGreaterEqual(r["distinct_new_classes"], 2)
        self.assertEqual(r["target_development"]["distinct_new_classes"], 0)
        self.assertEqual(r["status"], "insufficient")
        self.assertEqual(r["comparison"]["shared_class_representatives"], [])
        target = self.target_record(dev, [1, 2, 4, 6, 8])
        r = self.result(target, trials=2)
        distinct = r["target_development"]["distinct_new_classes"]
        self.assertGreater(distinct, 2)
        self.assertEqual(r["status"], "target_actions_distinguished")
        self.assertEqual(
            r["comparison"]["target_new_classes_beyond_trials"], distinct - 2
        )
        self.assertIsNone(r["paid_search_authorized"])
        self.assertFalse(r["search_executed"])
        self.assertEqual(r["native_grader_qualification"], "not_assessed")
        exhausted = self.result(target, trials=distinct)
        self.assertIn(
            "candidate_trials_cover_all_distinguishable_new_actions",
            exhausted["blockers"],
        )
        self.assertEqual(
            exhausted["comparison"]["target_selection_headroom"], "exhaustive"
        )
        self.assertEqual(
            self.result(target)["comparison"]["target_selection_headroom"], "unknown"
        )
        self.split("dev", [dev, "d" * 64])
        self.assertIn(
            "missing_target_development_tasks", self.result(target)["blockers"]
        )
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            self.result(target, target)

    def test_same_allocated_dev_is_accepted_and_holdout_evidence_is_not_opened(self):
        self.space["allowed_upper"] = [3, 4, 6, 8]
        self.split("dev", ["a" * 64])
        self.split("holdout", ["c" * 64])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.result()
        with self.assertRaisesRegex(ValueError, "frozen parent configuration"):
            self.result(self.protocol["prefixes"], trials=2)
        parent_dev = self.target_record("a" * 64, [1, 3, 4, 7, 8, 2])
        r = self.result(parent_dev, trials=2)
        self.assertEqual(r["target_development"]["distinct_new_classes"], 2)
        self.assertEqual(r["comparison"]["target_new_classes_beyond_trials"], 0)
        self.assertEqual(r["status"], "insufficient")
        self.result(parent_dev, trials=1)
        hidden = self.target_record("c" * 64, [4], split="holdout")
        wrong = self.target_record("c" * 64, [4], split="dev")
        original = Path.read_bytes

        def guard(path):
            if any(p.startswith("prefix-c") for p in path.parts):
                self.fail("opened an unallocated or holdout episode")
            return original(path)

        with patch.object(Path, "read_bytes", guard):
            for ref in (hidden, wrong):
                with self.assertRaisesRegex(ValueError, "development"):
                    self.result(ref)
        self.split("holdout", ["a" * 64])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.result()

    def test_swe_bindings_recompute_the_frozen_prompt_and_verify_the_measured_template(
        self,
    ):
        root = self.run_fixture.root
        fmt = "Solve {repo}\n\n{problem_statement}."
        template = root / "prompt-template.txt"
        template.write_bytes(fmt.encode())
        statement = b"a task with trailing whitespace \r\n"
        raw_id = digest(statement)
        repo = "project/example"
        prompt_id = digest(
            fmt.format(repo=repo, problem_statement=statement.decode()).encode()
        )
        self.split("dev", [raw_id])
        self.split("holdout", ["c" * 64])
        self.space["allowed_upper"] = [1, 3, 4, 6, 8]
        sub = root / "D01"
        sub.mkdir()
        (sub / "instruction.txt").write_bytes(statement)
        data = {
            "schema": "recipe-lab-target-dev-manifest-v1",
            "split": "dev",
            "study": "demo",
            "template": {
                "file": template.name,
                "sha256": digest(template.read_bytes()),
            },
            "tasks": [
                {
                    "instruction_sha256": raw_id,
                    "repo": repo,
                    "statement_file": "D01/instruction.txt",
                }
            ],
        }
        path = root / "dev-manifest.json"
        fixtures.put(path, data)
        manifest_sha = digest(path.read_bytes())
        target = self.target_record(
            prompt_id,
            [1, 2, 4, 6, 8],
            repo=repo,
            template_sha=data["template"]["sha256"],
        )
        out = self.result(target, trials=2, manifest=path, manifest_sha=manifest_sha)
        self.assertEqual(
            out,
            self.run_fixture.cli(
                "search-preflight",
                "--protocol",
                root / "protocol.json",
                "--target-prefixes",
                target,
                "--candidate-trials-per-method",
                2,
                "--target-dev-manifest",
                path,
                "--target-dev-manifest-sha256",
                manifest_sha,
            ),
        )
        self.assertEqual(
            out["target_development"]["tasks_by_instruction"][raw_id][
                "observed_instruction_sha256"
            ],
            prompt_id,
        )
        self.assertEqual(
            out["target_development"]["identity_binding"]["template_sha256"],
            data["template"]["sha256"],
        )
        self.assertEqual(out["status"], "target_actions_distinguished")
        self.assertNotIn("trailing whitespace", json.dumps(out))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.result(target, manifest=path, manifest_sha="0" * 64)
        wrong = self.target_record(prompt_id, [0], repo=repo, template_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "solver template"):
            self.result(wrong, manifest=path, manifest_sha=manifest_sha)
        (sub / "instruction.txt").write_bytes(b"different task")
        with self.assertRaisesRegex(ValueError, "statement fingerprint"):
            self.result(target, manifest=path, manifest_sha=manifest_sha)
        data["tasks"][0].update(
            instruction_sha256="c" * 64, statement_file="sealed-do-not-open"
        )
        fixtures.put(path, data)
        with self.assertRaisesRegex(ValueError, "exactly match the frozen development"):
            self.result(target, manifest=path, manifest_sha=digest(path.read_bytes()))

    def test_non_code_instruction_template_is_bound_to_measured_bytes(self):
        root = self.run_fixture.root
        instruction = b"Summarize the quarterly inventory.\n"
        raw_id = digest(instruction)
        self.split("dev", [raw_id])
        self.split("holdout", ["c" * 64])
        template = root / "office-prompt.txt"
        template.write_bytes(b"Deliver a memo.\r\n{instruction}")
        template_sha = digest(template.read_bytes())
        observed = digest(b"Deliver a memo.\r\n" + instruction)
        (root / "quarter.txt").write_bytes(instruction)
        path = root / "dev-manifest.json"
        fixtures.put(
            path,
            {
                "schema": "recipe-lab-target-dev-manifest-v2",
                "study": "demo",
                "split": "dev",
                "template": {"file": template.name, "sha256": template_sha},
                "tasks": [
                    {"instruction_sha256": raw_id, "instruction_file": "quarter.txt"}
                ],
            },
        )
        options = {"manifest": path, "manifest_sha": digest(path.read_bytes())}
        target = self.target_record(
            observed, [1], template_sha=template_sha, template_file=template.name
        )
        result = self.result(target, **options)["target_development"]
        self.assertEqual(result["identity_binding"]["method"], "instruction_template")
        self.assertEqual(
            result["tasks_by_instruction"][raw_id]["observed_instruction_sha256"],
            observed,
        )
        wrong = self.target_record(
            observed, [1, 2], template_sha=template_sha, template_file="other.txt"
        )
        with self.assertRaisesRegex(ValueError, "solver template"):
            self.result(wrong, **options)

    def test_split_identity_and_metadata_only(self):
        self.split("holdout", ["d" * 64])
        self.protocol["split_plans"]["holdout"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.result()
        path = self.run_fixture.root / "holdout.json"
        data = fixtures.get(path)
        data["scores"] = []
        path.write_text(json.dumps(data), encoding="utf-8")
        self.protocol["split_plans"]["holdout"]["sha256"] = digest(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "metadata only"):
            self.result()

    def test_holdout_measurement_and_boolean_budget_rejected(self):
        self.protocol["budget"]["episodes"] = True
        with self.assertRaisesRegex(ValueError, "budget"):
            self.result()
        self.protocol["budget"]["episodes"] = 1
        self.protocol["prefixes"] = self.run_fixture.record(
            "draft", "--split", "holdout"
        )
        with self.assertRaisesRegex(ValueError, "development"):
            self.result()
