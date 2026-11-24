"""Exercise recipe planning through the CLI, measured arms and their original receipts."""

import copy
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import yaml

import test_loop as fixtures
from recipe_lab.loop_evidence import digest
from recipe_lab.planning_space import render_handoff
from test_planning_probe import sample as probe_sample
from test_signature import encode_rows, receipts


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.run_fixture = fixtures.LoopTest()
        self.run_fixture.setUp()
        self.addCleanup(self.run_fixture.doCleanups)
        self.root = self.run_fixture.root
        self.cli = self.run_fixture.cli
        self.config = self.root / "fixed/config.yaml"
        self.doc = fixtures.get(self.config)
        recipe = self.doc["recipes"][1]
        recipe["strategy"] = "priority"
        recipe["candidate_requirements"] = {
            "context": "known_limits",
            "capabilities": "declared",
        }
        recipe["decisions"][0].update(algorithm={"type": "static"}, priority=1)
        for card, name in zip(
            self.doc["providers"]["models"], ["GLM-5.3", "Qwen3.5-35B-A3B"]
        ):
            card.update(
                provider_model_id=name,
                capabilities=["text", "tool_use"],
                context_window_size=1024,
                max_output_tokens=64,
            )
        fixtures.put(self.config, self.doc)

    def measured(
        self,
        arm="fixed",
        *,
        config=None,
        counts=(1, 30, 31),
        output=32,
        correct=True,
        split="dev",
        based_on=None,
        assignment=None,
        task="task",
    ):
        folder = self.root / arm / "dsh/runs/task" / arm / "main"
        binding_path = self.root / arm / f"dsh/bindings/task/{arm}.json"
        config_path = self.root / arm / "config.yaml"
        config_path.write_bytes(
            config if config is not None else self.config.read_bytes()
        )
        binding = fixtures.get(binding_path)
        binding.update(source_sha256=digest(config_path.read_bytes()))
        if assignment:
            binding["assignment_id"] = assignment
        fixtures.put(binding_path, binding)
        rows = receipts(counts)
        for index, (req, reply) in enumerate(zip(rows[::2], rows[1::2])):
            req.update(params={"max_completion_tokens": output}, start=10 + index)
            reply.update(end=12 + index, usage={"prompt_tokens": 42})
        (folder / "requests.jsonl").write_bytes(encode_rows(rows))
        grade = {"correct": correct, "criteria": {"delivered": {"passed": correct}}}
        fixtures.put(folder / "grade/score.json", grade)
        result = folder / "result.json"
        fixtures.put(
            result,
            {**fixtures.get(result), "grade": grade, "model_requests": len(counts)},
        )
        if task != "task":
            dsh = self.root / arm / "dsh"
            frozen = fixtures.get(dsh / "frozen.json")
            spec = frozen["cases"].pop("task")
            spec["instruction_sha256"] = digest(task.encode())
            frozen["cases"][task] = spec
            fixtures.put(dsh / "frozen.json", frozen)
            for name in ("result.json", "protocol.json"):
                native = fixtures.get(folder / name)
                native["prompt_sha256"] = spec["instruction_sha256"]
                if name == "protocol.json":
                    native["verified_task_packet_sha256"] = digest(
                        (dsh / "frozen.json").read_bytes()
                    )
                fixtures.put(folder / name, native)
            for directory in ("runs", "bindings"):
                (dsh / directory / "task").rename(dsh / directory / task)
        extra = ["--based-on", based_on] if based_on else []
        return self.run_fixture.record(arm, "--split", split, *extra)

    def plan(self, ident, *args):
        return self.cli("plan-dev", "--base", ident, "--fast", "B", *args)

    def inventory(self, *args):
        return self.cli(
            "plan-space",
            "--config",
            self.config,
            "--public-model",
            "public",
            "--strong",
            "A",
            "--fast",
            "B",
            *args,
        )

    def test_inventory_without_dataset(self):
        space = self.inventory()
        self.assertEqual(space["base_config_sha256"], digest(self.config.read_bytes()))
        candidates = {c["id"]: c for c in space["candidates"]}
        self.assertEqual(candidates["parent"]["kind"], "fixed_strong")
        self.assertEqual(
            candidates["e7b-35b"]["predicate"],
            {"feature": "tool_results", "lower": 30, "upper": None},
        )
        self.assertNotIn("e7b-9b", candidates)
        self.assertFalse((self.run_fixture.workspace / "ledger.jsonl").exists())

    def test_provider_free_reference_and_mixed_library(self):
        entry = "vllm-sr/mom-v1-blend"
        source = (
            Path(__file__).resolve().parents[3]
            / "config/recipes/built-in/latest/mom-v1/config.yaml"
        )
        reference = self.inventory("--reference-entry", entry, source)["candidates"][-1]
        self.assertEqual(reference["reason"], "bind_reference_template_models")
        self.assertEqual(reference["reference_recipe"], "balance")
        self.assertIn(entry, reference["public_entries"])
        self.assertIn("long_context", reference["reference_decisions"])
        self.assertEqual(
            reference["reference_bindings"]["long_context"],
            {"models": [], "minimum_candidates": 1},
        )
        self.assertIsNone(reference["candidate_sha256"])

        library = self.root / "library"
        template = copy.deepcopy(self.doc)
        del template["providers"]
        templates = library / "template/config.yaml"
        fixtures.put(templates, template)
        fixtures.put(library / "deployed/config.yaml", self.doc)
        report = self.inventory("--reference-library", library)
        refs = {
            Path(c["source"]).as_posix(): c
            for c in report["candidates"]
            if c.get("reference_sha256")
        }
        self.assertEqual(refs["deployed/config.yaml"]["kind"], "fixed_strong")
        self.assertEqual(
            refs["template/config.yaml"]["reference_bindings"]["fast"]["models"],
            ["A"],
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.inventory("--anchor", templates, "--anchor-models", "A", "B")

    def test_invalid_reference_model_declarations_do_not_become_templates(self):
        path = self.root / "invalid.yaml"
        for models in (None, {}, [], [{}], [{"name": "A"}, {"name": "A"}]):
            with self.subTest(models=models):
                fixtures.put(path, {**self.doc, "providers": {"models": models}})
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.inventory("--reference", path)
        del self.doc["providers"]
        fixtures.put(self.config, self.doc)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.inventory()

    def test_plan_publish_anchor_and_duplicate_predicate(self):
        base = self.measured()
        out = self.root / "probe.yaml"
        plan = self.plan(base, "--out-candidate", out)
        next_step = plan["next"]
        self.assertFalse(plan["dispatch"])
        self.assertEqual(next_step["action"], "probe_public_route")
        self.assertEqual(next_step["affected_parent_calls"], [2, 3])
        self.assertEqual(next_step["observed_seconds"], 3)
        self.assertEqual(next_step["fraction_of_task"], 3 / 12.5)
        self.assertEqual(next_step["candidate_sha256"], digest(out.read_bytes()))
        actual = yaml.safe_load(out.read_bytes())
        self.assertEqual(actual["providers"], self.doc["providers"])
        self.assertEqual(actual["recipes"][0], self.doc["recipes"][0])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan(base, "--out-candidate", out)
        self.assertEqual(next_step["candidate_sha256"], digest(out.read_bytes()))
        # An independently supplied anchor with equivalent observed behavior is retained by name.
        anchored = self.plan(base, "--anchor", out)
        self.assertTrue(anchored["next"]["anchor"])
        self.assertEqual(len(anchored["ranked_probes"]), 1)
        self.assertIn(
            ["e7b-35b", anchored["next"]["candidate"]],
            anchored["same_observed_predicate"],
        )

    def test_refuses_holdout_wrong_config_and_pending_assignments(self):
        self.run_fixture.arm("held", "A", 12.5)
        held = self.measured("held", split="holdout")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan(held)
        base = self.measured()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan(base, "--prefixes", base)
        changed = copy.deepcopy(self.doc)
        changed["recipes"][1]["description"] = "separate configuration"
        another = self.measured("draft", config=yaml.safe_dump(changed).encode())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan(base, "--prefixes", another)
        assigned = self.measured("draft", assignment="pending")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.plan(assigned)

    def test_anchor_roles_transfer_a_reference_to_unrelated_models(self):
        reference = render_handoff(
            copy.deepcopy(self.doc),
            public_model="public",
            strong="A",
            fast="B",
            threshold=2,
        )
        source_names = {"A": "original-analyst", "B": "original-assistant"}
        for card in reference["providers"]["models"]:
            card["name"] = source_names[card["name"]]
        for recipe in reference["recipes"]:
            for decision in recipe.get("decisions", []):
                for ref in decision.get("modelRefs", []):
                    ref["model"] = source_names.get(ref["model"], ref["model"])
        source = self.root / "old.yaml"
        fixtures.put(source, reference)
        for card, name in zip(
            self.doc["providers"]["models"], ["slow-writer", "fast-editor"]
        ):
            card.update(provider_model_id=name)
        fixtures.put(self.config, self.doc)
        unresolved = next(
            c
            for c in self.inventory("--anchor", source)["candidates"]
            if c.get("anchor")
        )
        self.assertEqual(unresolved["reason"], "reference_model_mapping_required")
        dest = self.root / "draft-office.yaml"
        plan = self.plan(
            self.measured(),
            "--anchor",
            source,
            "--anchor-models",
            *source_names.values(),
            "--out-candidate",
            dest,
        )
        self.assertEqual(plan["next"]["action"], "probe_public_route")
        self.assertEqual(
            plan["replay"]["inputs"]["anchor_models"], list(source_names.values())
        )
        draft = yaml.safe_load(dest.read_bytes())
        self.assertEqual(draft["providers"], self.doc["providers"])
        self.assertEqual(
            draft["recipes"][1]["signals"]["conversation"][0]["predicate"]["gte"], 2
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.inventory("--anchor", source, "--anchor-models", "A", "A")

    def test_distinct_reference_competes_with_case_on_same_requests(self):
        base = self.measured()
        path = self.root / "reference.yaml"
        self.plan(base, "--out-candidate", path)
        reference = yaml.safe_load(path.read_bytes())
        reference["recipes"][1]["signals"]["conversation"][0]["predicate"]["gte"] = 31
        path.write_text(yaml.safe_dump(reference), encoding="utf-8")
        comparison = self.plan(base, "--reference", path)
        ranked = comparison["ranked_probes"]
        self.assertEqual([r["affected_parent_calls"] for r in ranked], [[2, 3], [3]])
        self.assertEqual([r["observed_seconds"] for r in ranked], [3, 2])
        self.assertEqual(comparison["same_observed_predicate"], [])
        anchored = self.plan(base, "--anchor", path)
        self.assertEqual(anchored["next"]["affected_parent_calls"], [3])

    def test_status_without_parent_success_exposure_or_capacity(self):
        for args, expected in [
            (dict(correct=False), "reproduce_passing_parent"),
            (dict(counts=(1, 2)), "collect_discriminating_prefix"),
            (dict(output=65), "inspect_backend_limits"),
        ]:
            with self.subTest(expected=expected):
                record = self.measured(**args)
                plan = self.plan(record)
                self.assertEqual(plan["ranked_probes"], [])
                self.assertEqual(plan["next"]["action"], expected)

    def test_development_feedback_retires_regression(self):
        base = self.measured()
        out = self.root / "first-candidate.yaml"
        self.plan(base, "--out-candidate", out)
        cand = self.measured(
            "draft", config=out.read_bytes(), correct=False, based_on=base
        )
        comp = self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            cand,
            "--disposition",
            "keep-baseline",
            "--reason",
            "lost strict pass",
        )
        auto = self.plan(base)
        self.assertEqual(auto["ranked_probes"], [])
        self.assertEqual(auto["next"]["action"], "protect_regressed_task")
        self.assertEqual(auto["prior_comparisons"][0]["comparison"], comp["record_id"])
        self.assertEqual(
            auto["prior_comparisons"][0]["next_action"], "protect_regressed_task"
        )
        self.assertIsNone(
            auto["prior_comparisons"][0]["tasks"]["task"]["paired_seconds_saved"]
        )
        # Rejecting the anchor must still allow a different, untested mechanism.
        alt = self.root / "alternative.yaml"
        alternate = yaml.safe_load(out.read_bytes())
        alternate["recipes"][1]["signals"]["conversation"][0]["predicate"]["gte"] = 31
        alt.write_text(yaml.safe_dump(alternate), encoding="utf-8")
        after = self.plan(base, "--anchor", out, "--reference", alt)
        self.assertEqual(after["next"]["action"], "probe_public_route")
        self.assertEqual(after["next"]["affected_parent_calls"], [3])
        self.assertFalse(after["next"]["anchor"])

    def test_measured_repeats_use_pooled_exposure_and_keep_failed_parent(self):
        base = self.measured()
        # Base has 3s exposure in 12.5s; the other pass has 2s in 25s.
        self.run_fixture.arm("repeat", "A", 25)
        second = self.measured("repeat", counts=(1, 2, 30))
        ranked = self.plan(base, "--prefixes", second)["ranked_probes"]
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["measurements"], [base, second])
        self.assertEqual(ranked[0]["observed_seconds"], 5)
        self.assertAlmostEqual(ranked[0]["fraction_of_task"], 5 / 37.5)
        self.run_fixture.arm("failed", "A", 8)
        failed = self.measured("failed", correct=False, counts=(0, 1, 31))
        self.assertEqual(self.plan(base, "--prefixes", failed)["ranked_probes"], [])

    def test_public_probe_evidence_controls_next_step(self):
        base = self.measured()
        ref = self.root / "threshold-two.yaml"
        doc = render_handoff(
            self.doc, public_model="public", strong="A", fast="B", threshold=2
        )
        ref.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        prepared = self.plan(base, "--anchor", ref)["next"]
        file = self.root / "gateway.json"
        for kind, expected in [
            ("fixture", "probe_real_backends"),
            ("real", "find_distinct_reproducible_guard"),
        ]:
            with self.subTest(kind=kind):
                receipt = probe_sample(kind)
                receipt.update(
                    source_sha=prepared["candidate_sha256"],
                    projected_sha=prepared["candidate_sha256"],
                    public_model="public",
                )
                for row in receipt["probes"]:
                    row["public_model"] = "public"
                    if row.get("decision"):
                        row["decision"] = {"fallback": "fast", "late": "lab_late_fast"}[
                            row["decision"]
                        ]
                        row["returned"] = row["selected"]
                fixtures.put(file, receipt)
                step = self.plan(base, "--anchor", ref, "--probe-receipt", file)["next"]
                self.assertEqual(step["action"], expected)
                self.assertEqual(step["public_probe"]["status"], f"{kind}_pass")
        isolated = self.root / "isolated.yaml"
        deployed = yaml.safe_load(ref.read_bytes())
        # This change to a capability limit is outside the allowed connection projection.
        deployed["providers"]["models"][0]["context_window_size"] += 1
        fixtures.put(isolated, deployed)
        receipt["projected_sha"] = digest(isolated.read_bytes())
        fixtures.put(file, receipt)
        self.assertEqual(
            self.plan(base, "--anchor", ref, "--probe-receipt", file)["next"]["action"],
            "complete_gateway_probe",
        )
        self.assertEqual(
            self.plan(
                base,
                "--anchor",
                ref,
                "--probe-receipt",
                file,
                "--probe-config",
                isolated,
            )["next"]["action"],
            "diagnose_gateway_probe",
        )
        receipt["source_sha"] = "0" * 64
        fixtures.put(file, receipt)
        step = self.plan(base, "--anchor", ref, "--probe-receipt", file)["next"]
        self.assertEqual(step["action"], "diagnose_gateway_probe")

    def test_promoted_candidate_is_confirmed_instead_of_reproposed(self):
        base = self.measured()
        out = self.root / "winner.yaml"
        self.plan(base, "--out-candidate", out)
        candidate = self.measured("draft", config=out.read_bytes(), based_on=base)
        self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            candidate,
            "--disposition",
            "advance-to-holdout",
            "--reason",
            "matched strict pass with time saved",
        )
        self.assertEqual(self.plan(base)["next"]["action"], "confirm_frozen_candidate")

    def test_conflicting_repetitions_do_not_promote_a_regression(self):
        base = self.measured()
        out = self.root / "candidate.yaml"
        self.plan(base, "--out-candidate", out)
        failed = self.measured(
            "draft", config=out.read_bytes(), based_on=base, correct=False
        )
        self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            failed,
            "--disposition",
            "keep-baseline",
            "--reason",
            "guard regression",
        )
        self.run_fixture.arm("passing", "B", 8)
        passed = self.measured("passing", config=out.read_bytes(), based_on=base)
        self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            passed,
            "--disposition",
            "advance-to-holdout",
            "--reason",
            "joint success and time gain",
        )
        self.assertEqual(
            self.plan(base)["next"]["action"], "repeat_conflicting_comparisons"
        )

    def test_ready_plan_enrolls_separately_measured_guard_and_freezes_feedback(self):
        base = self.measured()
        guard_ids = []
        for i in range(2):
            arm = f"guard-{i}"
            self.run_fixture.arm(arm, "A", 10)
            guard_ids.append(self.measured(arm, counts=(i,), task="guard"))
        candidate = self.root / "candidate.yaml"
        draft = render_handoff(
            self.doc, public_model="public", strong="A", fast="B", threshold=2
        )
        candidate.write_text(yaml.safe_dump(draft, sort_keys=False), encoding="utf-8")
        probe = probe_sample("real")
        probe.update(
            source_sha=digest(candidate.read_bytes()),
            projected_sha=digest(candidate.read_bytes()),
            public_model="public",
        )
        for row in probe["probes"]:
            row["public_model"] = "public"
            if row.get("decision"):
                row["decision"] = {"fallback": "fast", "late": "lab_late_fast"}[
                    row["decision"]
                ]
                row["returned"] = row["selected"]
        receipt = self.root / "public.json"
        fixtures.put(receipt, probe)
        plan = self.plan(
            base,
            "--anchor",
            candidate,
            "--probe-receipt",
            receipt,
            "--prefixes",
            guard_ids[0],
            "--prefixes",
            guard_ids[1],
        )
        self.assertEqual(plan["next"]["action"], "plan_paired_development")
        saved = self.root / "plan.json"
        fixtures.put(saved, plan)
        card = dict(
            schema="recipe-lab-schedule-v2",
            study="handoff",
            based_on=base,
            signature=saved.name,
            signature_sha256=digest(saved.read_bytes()),
            candidate=candidate.name,
            target_task="task",
            guard_task="guard",
            budget_episodes=4,
            repeats=1,
            hypotheses=["handoff saves time", "handoff adds rounds"],
            prediction="guard remains solved",
            decisions={
                "advance": "confirm",
                "revise": "measure",
                "drop": "keep parent",
            },
        )
        card_path = self.root / "study.json"
        fixtures.put(card_path, card)
        first = self.cli("schedule", "--card", card_path)
        self.assertEqual(first["next"]["task"], "guard")
        later = self.measured("draft", config=candidate.read_bytes(), based_on=base)
        self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            later,
            "--disposition",
            "advance-to-holdout",
            "--reason",
            "positive development gain",
        )
        self.assertEqual(
            self.cli("schedule", "--card", card_path)["next"], first["next"]
        )
        probe["passed"] = False
        fixtures.put(receipt, probe)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli("schedule", "--card", card_path)

    def test_two_failed_arms_trigger_parent_check(self):
        base = self.measured(correct=False)
        doc = render_handoff(
            self.doc, public_model="public", strong="A", fast="B", threshold=30
        )
        candidate = self.measured(
            "draft",
            config=yaml.safe_dump(doc, sort_keys=False).encode(),
            based_on=base,
            correct=False,
        )
        self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            candidate,
            "--disposition",
            "keep-baseline",
            "--reason",
            "absolute quality guard failed",
        )
        plan = self.plan(base)
        self.assertEqual(plan["next"]["action"], "reproduce_passing_parent")
        self.assertEqual(
            plan["prior_comparisons"][0]["next_action"], "reproduce_passing_parent"
        )

    def test_complex_reference_requires_probe_and_empty_library_is_rejected(self):
        ref = copy.deepcopy(self.doc)
        ref["recipes"][1]["decisions"][0]["algorithm"] = {
            "type": "some_other_algorithm"
        }
        file = self.root / "complex.yaml"
        fixtures.put(file, ref)
        record = self.measured(counts=(2,))
        report = self.plan(record, "--reference", file)
        self.assertEqual(report["next"]["action"], "inspect_reference_on_gateway")
        self.assertEqual(
            report["prefix_evidence"]["reference-1"][0]["matched_calls"], []
        )
        empty = self.root / "empty-library"
        empty.mkdir()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.inventory("--reference-library", empty)
