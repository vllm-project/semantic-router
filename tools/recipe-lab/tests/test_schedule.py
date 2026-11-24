"""Freeze a development card over real CLI test packets and resume its comparisons."""

import shutil
import unittest
from types import SimpleNamespace

import test_loop as fixtures
import test_signature
import test_window_action
from recipe_lab.loop import Ledger
from recipe_lab.loop_evidence import digest
from recipe_lab.schedule import canonical, decision, schedule
from recipe_lab.signature import signature


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        action = test_window_action.WindowTest()
        action.setUp()
        self.addCleanup(action.doCleanups)
        self.run = action.run_fixture
        self.root = action.root
        self.ledger = Ledger(self.run.workspace)
        self.card = self.root / "card.json"
        self.attempt = self.root / "attempt.json"
        self.space = self.root / "space.yaml"
        fixtures.put(self.space, action.space)
        for arm in ("fixed", "draft"):
            root = self.root / arm / "dsh"
            frozen = fixtures.get(root / "frozen.json")
            task_spec = frozen["cases"]["task"]
            frozen["cases"]["guard"] = dict(task_spec, instruction_sha256="f" * 64)
            fixtures.put(root / "frozen.json", frozen)
            shutil.copytree(root / "runs/task", root / "runs/guard")
            shutil.copytree(root / "bindings/task", root / "bindings/guard")
            for task in ("task", "guard"):
                folder = root / "runs" / task / arm / "main"
                cfg = fixtures.get(folder / "provider.json")
                cfg[1]["config"]["providers"]["source"]["api"] = "openai-completions"
                fixtures.put(folder / "provider.json", cfg)
                for file in ("protocol.json", "result.json"):
                    data = fixtures.get(folder / file)
                    data["prompt_sha256"] = frozen["cases"][task]["instruction_sha256"]
                    if file == "protocol.json":
                        data["verified_task_packet_sha256"] = digest(
                            (root / "frozen.json").read_bytes()
                        )
                    fixtures.put(folder / file, data)
                (folder / "requests.jsonl").write_bytes(
                    test_signature.encode_rows(test_signature.receipts([4]))
                )
        self.base = self.run.record("fixed")
        self.candidate = self.root / "draft/config.yaml"
        self.candidate.unlink()
        self.run.cli(
            "propose",
            "--base",
            self.base,
            "--space",
            self.space,
            "--upper",
            8,
            "--out",
            self.candidate,
        )
        for task in ("guard", "task"):
            binding = self.root / "draft/dsh/bindings" / task / "draft.json"
            fixtures.put(
                binding,
                dict(
                    fixtures.get(binding),
                    source_sha256=digest(self.candidate.read_bytes()),
                ),
            )
        self.measured = self.run.record("draft")
        self.report = signature(
            self.ledger,
            SimpleNamespace(
                base=self.base,
                prefixes=self.base,
                candidate=self.candidate,
                space=self.space,
                upper=8,
            ),
        )
        report = self.root / "signature.json"
        fixtures.put(report, self.report)
        self.protocol = dict(
            schema="recipe-lab-schedule-v1",
            study="pilot",
            based_on=self.base,
            signature=report.name,
            signature_sha256=digest(report.read_bytes()),
            space=self.space.name,
            candidate="draft/config.yaml",
            hypotheses=["more handoffs help", "more handoffs add work"],
            prediction="guard stays solved and target finishes quicker",
            target_task="task",
            guard_task="guard",
            decisions={
                "advance": "confirm on more dev cases",
                "revise": "check elapsed spans",
                "drop": "keep baseline",
            },
            budget_episodes=4,
            repeats=1,
        )
        fixtures.put(self.card, self.protocol)

    def poll(self, attempt=None):
        if attempt is not None:
            fixtures.put(self.attempt, attempt)
        return schedule(
            self.ledger,
            SimpleNamespace(
                card=self.card, attempt=self.attempt if attempt is not None else None
            ),
        )

    def fresh_run(
        self,
        slot,
        *,
        status="completed",
        seconds=8,
        early=False,
        rewrap=False,
        receipts=None,
        strict=None,
        termination=None,
    ):
        arm = "fixed" if slot["role"] == "baseline" else "draft"
        assignment = slot["assignment_id"]
        task = slot["task"]
        source = self.root / assignment
        shutil.copytree(self.root / arm, source)
        dsh = source / "dsh"
        binding = dsh / "bindings" / task / (arm + ".json")
        fixtures.put(binding, dict(fixtures.get(binding), assignment_id=assignment))
        folder = dsh / "runs" / task / arm / "main"
        if receipts is not None:
            (folder / "requests.jsonl").write_bytes(
                test_signature.encode_rows(receipts)
            )
        result = folder / "result.json"
        native = fixtures.get(result)
        if strict is not None:
            grade = {"correct": strict, "criteria": {"delivered": {"passed": strict}}}
            fixtures.put(folder / "grade/score.json", grade)
            native["grade"] = grade
        if termination is not None:
            native["termination"] = termination
        if not rewrap:
            native["execution_id"] = assignment
        if status == "model_endpoint_error":
            native["termination"] = status
        if status == "grader_error":
            native["judge_exit"] = 2
        fixtures.put(result, native)
        if status == "completed":
            measurement = self.run.cli(
                "record",
                "--run",
                source,
                "--arm",
                arm,
                "--config",
                source / "config.yaml",
                "--public-model",
                "public",
                "--strong",
                "A",
                "--hypothesis",
                "paired prospective case",
            )["record_id"]
            return dict(slot, measurement=measurement)
        controls = self.ledger.measure(self.base)["cases"][task]["controls"]
        common = dict(
            assignment_id=assignment, clock="CLOCK_BOOTTIME", boot_id="test-boot"
        )
        fixtures.put(
            folder / "execution-start.json",
            dict(
                common,
                monotonic_ns=5,
                controls_sha256=digest(canonical(controls).encode()),
            ),
        )
        fixtures.put(
            folder / "execution-end.json",
            dict(
                common,
                monotonic_ns=5 + seconds * 10**9,
                status=status,
                runner_exit=1 if status == "runner_error" else 0,
                judge_exit=2 if status == "grader_error" else None,
            ),
        )
        if early:
            for name in (
                "protocol.json",
                "result.json",
                "requests.jsonl",
                "provider.json",
            ):
                (folder / name).unlink()
        return dict(slot, invalid_run=str(source), arm=arm)

    def test_paired_schedule_and_idempotent_reimport(self):
        step = self.poll()
        self.assertEqual(step, self.run.cli("schedule", "--card", self.card))
        for task, role in (
            ("guard", "baseline"),
            ("guard", "candidate"),
            ("task", "candidate"),
            ("task", "baseline"),
        ):
            self.assertEqual((step["next"]["task"], step["next"]["role"]), (task, role))
            attempt = self.fresh_run(step["next"])
            step = self.poll(attempt)
            self.assertEqual(step, self.poll(attempt))
        self.assertEqual(step["status"], "advance")
        self.assertEqual(step["spent_episodes"], 4)
        self.assertEqual(step["seconds_all_attempts"], 45)
        self.assertEqual(step["paired_by_task"]["guard"]["joint_seconds_saved"], 2.5)

    def test_invalid_attempt_counts_and_retry_gets_new_assignment(self):
        slot = self.poll()["next"]
        attempt = self.fresh_run(slot, status="model_endpoint_error")
        step = self.poll(attempt)
        self.assertEqual(
            (
                step["next"]["role"],
                step["invalid_attempts"],
                step["invalid_attempt_wall_seconds"],
            ),
            ("baseline", 1, 8),
        )
        self.assertNotEqual(slot["assignment_id"], step["next"]["assignment_id"])
        self.assertEqual(step, self.poll(attempt))
        step = self.poll(self.fresh_run(step["next"]))
        self.assertEqual(step["spent_episodes"], 2)
        self.assertEqual(step["next"]["role"], "candidate")
        step = self.poll(self.fresh_run(step["next"]))
        step = self.poll(self.fresh_run(step["next"]))
        self.assertEqual(step["status"], "defer")
        self.assertEqual(step["seconds_all_attempts"], 40.5)

    def test_opposite_quality_pairs_require_repeat_before_latency_promotion(self):
        card = dict(self.protocol, budget_episodes=8, repeats=2)
        fixtures.put(self.card, card)
        step = self.poll()
        for _ in range(8):
            slot = step["next"]
            expected = (slot["repeat"] == 1) == (slot["role"] == "baseline")
            strict = expected if slot["task"] == "guard" else True
            step = self.poll(self.fresh_run(slot, strict=strict))
        self.assertEqual(step["status"], "defer")
        self.assertGreater(step["paired_by_task"]["task"]["joint_seconds_saved"], 0)
        summary = self.run.cli("decision-summary", "--card", self.card)
        self.assertEqual(summary["repeat_tasks"], ["guard"])

    def test_runner_and_judge_failures_have_measured_cost_and_no_grade(self):
        slot = self.poll()["next"]
        attempt = self.fresh_run(slot, status="runner_error", early=True)
        step = self.poll(attempt)
        self.assertEqual(step["invalid_attempt_wall_seconds"], 8)
        next_attempt = self.fresh_run(step["next"], status="grader_error", seconds=4)
        folder = self.root / next_attempt["assignment_id"] / "dsh/runs/guard/fixed/main"
        end = folder / "execution-end.json"
        genuine = fixtures.get(end)
        fixtures.put(end, {**genuine, "boot_id": "after-reboot"})
        with self.assertRaisesRegex(ValueError, "same-boot"):
            self.poll(next_attempt)
        fixtures.put(end, {**genuine, "judge_exit": 0})
        with self.assertRaisesRegex(ValueError, "grader failure"):
            self.poll(next_attempt)
        fixtures.put(end, genuine)
        step = self.poll(next_attempt)
        self.assertEqual(step["seconds_all_attempts"], 12)
        self.assertEqual(step["invalid_attempts"], 2)
        self.assertEqual(step["paired_by_task"]["guard"]["joint_successes"], 0)

    def test_wrong_identity_or_changed_card_fails_closed(self):
        slot = self.poll()["next"]
        with self.assertRaisesRegex(ValueError, "scheduled development configuration"):
            self.poll(dict(slot, measurement=self.measured))
        self.assertEqual(self.poll()["spent_episodes"], 0)
        fixtures.put(self.card, dict(self.protocol, budget_episodes=10))
        with self.assertRaisesRegex(ValueError, "changed after enrollment"):
            self.poll()

    def test_historical_parent_and_rewrapped_execution_are_rejected(self):
        slot = self.poll()["next"]
        with self.assertRaisesRegex(ValueError, "matching prospective assignment"):
            self.poll(dict(slot, measurement=self.base))
        with self.assertRaisesRegex(ValueError, "predates"):
            self.poll(self.fresh_run(slot, rewrap=True))
        self.assertEqual(self.poll()["spent_episodes"], 0)

    def test_foreign_assignment_rejected_even_when_slot_and_config_match(self):
        slot = self.poll()["next"]
        foreign = {**slot, "assignment_id": "c" * 64}
        attempt = self.fresh_run(foreign)
        with self.assertRaisesRegex(ValueError, "next scheduled slot"):
            self.poll(attempt)
        with self.assertRaisesRegex(ValueError, "matching prospective assignment"):
            self.poll({**attempt, "assignment_id": slot["assignment_id"]})
        self.assertEqual(slot, self.poll()["next"])

    def test_repeat_cannot_relabel_an_already_charged_physical_execution(self):
        fixtures.put(self.card, dict(self.protocol, repeats=2, budget_episodes=8))
        step = self.poll()
        first_candidate = None
        for _ in range(4):
            attempt = self.fresh_run(step["next"])
            if attempt["role"] == "candidate" and attempt["task"] == "guard":
                first_candidate = attempt
            step = self.poll(attempt)
        self.assertEqual(step["next"]["repeat"], 2)
        new = self.fresh_run(step["next"])
        task, arm = "guard", "draft"
        src = (
            self.root
            / first_candidate["assignment_id"]
            / "dsh/runs"
            / task
            / arm
            / "main"
        )
        clone = self.root / new["assignment_id"]
        dst = clone / "dsh/runs" / task / arm / "main"
        for name in ("requests.jsonl", "result.json"):
            shutil.copyfile(src / name, dst / name)
        new["measurement"] = self.run.cli(
            "record",
            "--run",
            clone,
            "--arm",
            arm,
            "--config",
            clone / "config.yaml",
            "--public-model",
            "public",
            "--strong",
            "A",
            "--hypothesis",
            "same episode relabeled",
        )["record_id"]
        with self.assertRaisesRegex(ValueError, "physical episode reused"):
            self.poll(new)

    def test_legacy_empty_state_upgrades_but_nonempty_history_is_preserved(self):
        state = self.root / "ledger/schedules/pilot/state.json"
        old = {"card_sha256": digest(self.card.read_bytes()), "attempts": []}
        fixtures.put(state, old)
        plan = self.poll()
        self.assertEqual(plan, self.poll())
        self.assertIn("enrollment", fixtures.get(state))
        old["attempts"] = [{"historical": True}]
        fixtures.put(state, old)
        with self.assertRaisesRegex(ValueError, "retain as historical"):
            self.poll()
        self.assertEqual(fixtures.get(state), old)

    def test_guard_loss_and_uninformative_signature(self):
        pair = [
            dict(task="guard", role=role, repeat=1, valid=True, strict=ok, seconds=7)
            for role, ok in (("baseline", True), ("candidate", False))
        ]
        self.assertEqual(decision(self.protocol, self.report, pair)["status"], "drop")
        self.report["by_task"]["task"]["changed"] = 0
        self.assertEqual(decision(self.protocol, self.report, [])["status"], "defer")
