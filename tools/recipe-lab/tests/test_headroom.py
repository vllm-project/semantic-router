"""Run the headroom command against frozen score matrices."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path
from statistics import stdev

from recipe_lab import loop
from recipe_lab.loop_evidence import digest


class HeadroomTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work = self.root / "ledger"
        self.manifest = {
            "study": "pilot",
            "split": "dev",
            "tasks": ["A", "B"],
            "arms": ["speed", "glm"],
            "repeats": [1, 2],
            "scores": "scores.json",
        }

    def rows(self, outcomes=None):
        rows = []
        for task, arm, repeat in product(
            *(self.manifest[k] for k in ("tasks", "arms", "repeats"))
        ):
            strict, seconds = (outcomes or {}).get((task, arm, repeat), (True, 10))
            rows.append(
                {
                    "study": "pilot",
                    "split": self.manifest["split"],
                    "task": task,
                    "arm": arm,
                    "repeat": repeat,
                    "controls_sha256": digest(task.encode()),
                    "policy_sha256": digest(arm.encode()),
                    "episode_sha256": digest(str((task, arm, repeat)).encode()),
                    "strict": strict,
                    "seconds": seconds,
                    "assessment": "valid",
                }
            )
        return rows

    def save(self, rows):
        source = self.root / "scores.json"
        source.write_text(json.dumps(rows), encoding="utf-8")
        self.manifest["scores_sha256"] = digest(source.read_bytes())
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def cli(self, *opts, scope="dev", command="headroom"):
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            loop.main(
                [
                    "--workspace",
                    str(self.work),
                    command,
                    "--manifest",
                    str(self.root / "manifest.json"),
                    *(["--scope", scope] if command == "headroom" else []),
                    *opts,
                ]
            )
        return json.loads(stdout.getvalue())

    def test_screen_keeps_long_single_policy_runs_for_diagnosis(self):
        self.manifest["arms"] = ["glm"]
        rows = self.rows({("A", "glm", 1): (False, 600), ("B", "glm", 2): (False, 601)})
        self.save([dict(row, termination="natural_completion") for row in rows])
        result = self.cli(command="screen-dev")
        self.assertEqual(result["quick_controls"], ["A"])
        self.assertEqual(result["by_task"]["A"]["strict_passes"], 1)
        self.assertEqual(result["by_task"]["B"]["pool"], "investigate_slow")
        self.assertEqual(result["excluded_tasks"], [])
        self.assertEqual(
            self.cli("--max-seconds", "601", command="screen-dev")["quick_controls"],
            ["A", "B"],
        )
        self.assertFalse(self.work.exists())

    def test_screen_requires_natural_completion_and_a_development_split(self):
        self.save(self.rows())
        result = self.cli(command="screen-dev")
        self.assertEqual(result["quick_controls"], [])
        self.assertEqual(result["by_task"]["A"]["pool"], "needs_evidence")
        for invalid in ("0", "nan", "inf"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                self.cli("--max-seconds", invalid, command="screen-dev")
        self.manifest["split"] = "retired-holdout"
        self.save(self.rows())
        with self.assertRaises(SystemExit):
            self.cli(command="screen-dev")

    def test_screen_preserves_route_contrasts_on_each_side_of_the_reference(self):
        self.manifest["repeats"] = [1]
        rows = self.rows(
            {
                ("A", "speed", 1): (True, 277),
                ("A", "glm", 1): (True, 1124),
                ("B", "speed", 1): (True, 820),
                ("B", "glm", 1): (True, 1500),
            }
        )
        self.save(
            [
                dict(
                    r,
                    termination="natural_completion",
                    model_requests=40,
                    model_request_union_seconds=72,
                )
                for r in rows
            ]
        )
        result = self.cli(command="screen-dev")
        self.assertEqual(result["routing_candidates"], ["A", "B"])
        self.assertEqual(
            result["by_task"]["A"]["paired_latency_contrasts"][0]["gap_seconds"], 847
        )
        self.assertEqual(
            result["by_task"]["A"]["request_profiles"][0]["outside_model_seconds"], 205
        )
        rows[0]["strict"] = False
        self.save([dict(r, termination="natural_completion") for r in rows])
        a = self.cli(command="screen-dev")["by_task"]["A"]
        self.assertEqual(a["paired_latency_contrasts"], [])
        self.assertEqual(a["quality_contrasts"][0]["failed"], "speed")

    def test_only_audited_mandatory_cost_excludes_a_task(self):
        rows = [dict(r, termination="natural_completion") for r in self.rows()]
        p = self.root / "fixed.json"
        receipt = {
            "study": "pilot",
            "split": "dev",
            "task": "A",
            "controls_sha256": rows[0]["controls_sha256"],
            "required_for_every_attempt": True,
            "requirement": "fixed official suite",
            "stage": "official_grading",
            "protocol_sha256": "f" * 64,
            "observations": [{"run_sha256": "a" * 64, "seconds": 900}],
        }

        def pin():
            p.write_text(json.dumps(receipt), encoding="utf-8")
            self.manifest["required_cost_receipts"] = {
                "A": {"file": p.name, "sha256": digest(p.read_bytes())}
            }
            self.save(rows)

        pin()
        self.assertEqual(self.cli(command="screen-dev")["excluded_tasks"], ["A"])
        original = self.manifest.copy()
        self.manifest.update(
            tasks=[],
            arms=[],
            repeats=[],
            unmeasured_controls={"A": rows[0]["controls_sha256"]},
        )
        for field in ("scores", "scores_sha256"):
            self.manifest.pop(field)
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        without_solver = self.cli(command="screen-dev")
        self.assertEqual(without_solver["excluded_tasks"], ["A"])
        self.assertEqual(without_solver["by_task"]["A"]["observations"], 0)
        self.manifest = original
        receipt["stage"] = "required_solver_work"
        pin()
        result = self.cli(command="screen-dev")
        self.assertEqual(result["excluded_tasks"], [])
        self.assertTrue(result["by_task"]["A"]["required_solver_cost_contradicted"])
        p.write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.cli(command="screen-dev")

    def test_select_once_per_task_and_preserve_repeat_pairs(self):
        rows = self.rows(
            {
                ("A", "speed", 1): (True, 1),
                ("A", "speed", 2): (True, 100),
                ("A", "glm", 1): (True, 60),
                ("A", "glm", 2): (True, 30),
                ("B", "glm", 1): (False, 1),
                ("B", "glm", 2): (False, 1),
            }
        )
        self.save(rows)
        result = self.cli("--mde-pair", "speed", "glm")
        self.assertEqual(result["best_fixed"], "speed")
        l0 = result["l0"]
        self.assertEqual(l0["chosen_arm_by_task"], {"A": "glm", "B": "speed"})
        self.assertEqual(l0["strict_passes"], 4)
        # Switching again after seeing each repeat could yield 51, which is infeasible.
        self.assertEqual(l0["successful_seconds_sum"], 110)
        delta = l0["versus_best_fixed"]
        self.assertEqual(
            (delta["joint_successes"], delta["joint_seconds_saved"]), (4, 11)
        )
        self.assertAlmostEqual(delta["joint_percent_saved"], 1100 / 121)
        self.assertEqual(result["mde"]["independent_tasks"], 1)
        self.assertIsNone(result["mde"]["estimate_seconds"])
        self.assertIsNotNone(result["by_task"]["A"]["speed"]["successful_wall_cv"])
        self.assertIsNone(result["by_task"]["B"]["glm"]["successful_wall_cv"])
        self.assertEqual(result["l2"]["status"], "unidentified")
        self.assertFalse(self.work.exists())

    def test_mixed_passes_have_distinct_guarded_oracle_and_paired_latency(self):
        self.manifest["repeats"] = [1, 2, 3]
        rows = self.rows(
            {
                ("A", "speed", 2): (False, 500),
                ("A", "speed", 3): (False, 500),
                ("A", "glm", 1): (False, 1),
                ("B", "glm", 1): (False, 1),
                ("B", "glm", 2): (False, 1),
            }
        )
        self.save(rows)
        result = self.cli()
        self.assertEqual(result["best_fixed"], "speed")
        self.assertEqual(result["l0"]["strict_passes"], 5)
        delta = result["l0"]["versus_best_fixed"]
        self.assertEqual(delta["lost_strict_cells"], [["A", 1]])
        self.assertEqual(delta["gained_strict_cells"], [["A", 2], ["A", 3]])
        self.assertEqual(delta["joint_successes"], 3)
        self.assertEqual(delta["joint_seconds_saved"], 0)
        guarded = result["l0_preserving_baseline_passes"]
        self.assertEqual(guarded["strict_passes"], 4)
        self.assertEqual(guarded["versus_best_fixed"]["lost_strict_cells"], [])
        self.assertTrue(result["by_task"]["A"]["glm"]["mixed_strict_over_repeats"])

    def test_mde_uses_task_means_and_requires_repeat_information(self):
        self.manifest["tasks"] = [str(i) for i in range(8)]
        rows = self.rows()
        for row in rows:
            if row["arm"] == "speed":
                row["seconds"] += int(row["task"]) + row["repeat"]
        self.save(rows)
        report = self.cli("--mde-pair", "speed", "glm")["mde"]
        self.assertEqual(report["independent_tasks"], 8)
        self.assertAlmostEqual(
            report["estimate_seconds"], 2.8 * stdev(range(8)) / math.sqrt(8)
        )
        rows[0]["strict"] = False
        self.save(rows)
        self.assertIsNone(
            self.cli("--mde-pair", "speed", "glm")["mde"]["estimate_seconds"]
        )

    def test_matrix_integrity(self):
        for field, value in (
            ("strict", 1),
            ("seconds", True),
            ("seconds", float("nan")),
            ("seconds", 0),
            ("task", "unseen"),
            ("split", "holdout"),
            ("study", "other"),
            ("assessment", "infra_error"),
            ("assessment", None),
            ("repeat", True),
            ("controls_sha256", "a" * 64),
            ("policy_sha256", "b" * 64),
        ):
            with self.subTest(field=field, value=value):
                rows = self.rows()
                rows[0][field] = value
                self.save(rows)
                with self.assertRaises(SystemExit):
                    self.cli()

        # The two labels refer to independent episodes of the very same policy.
        # Their different outcomes cannot be exploited by choosing a policy.
        self.save([dict(row, policy_sha256="a" * 64) for row in self.rows()])
        with self.assertRaises(SystemExit):
            self.cli()
        rows = self.rows()
        for invalid in (
            [*rows, rows[0]],
            rows[:-1],
            [dict(rows[0], episode_sha256=rows[1]["episode_sha256"]), *rows[1:]],
        ):
            self.save(invalid)
            with self.assertRaises(SystemExit):
                self.cli()
        self.save(rows)
        (self.root / "scores.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.cli()

    def objective(self):
        return {
            "schema": "recipe-lab-objective-v1",
            "study": "pilot",
            "priorities": [
                "strict_success",
                "joint_success_latency",
                "total_attempt_cost",
            ],
            "baseline": "glm",
            "arms": ["speed", "glm"],
            "repeats": [1, 2],
            "mde_pair": ["glm", "speed"],
            "dev_targets": ["A"],
            "dev_guards": ["B"],
            "policy_sha256_by_arm": {
                a: digest(a.encode()) for a in self.manifest["arms"]
            },
            "controls_sha256_by_dev_task": {
                t: digest(t.encode()) for t in self.manifest["tasks"]
            },
            "holdout_task_id_sha256": [digest(b"sealed-task")],
            "thresholds": {
                "max_lost_strict_cells": 0,
                "min_joint_tasks": 2,
                "min_joint_seconds_saved": 1,
                "max_solver_wall_ratio": 1,
            },
            "external_checks": ["Review monetary cost from provider invoices"],
        }

    def save_objective(self, spec, rows):
        path = self.root / "objective.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        self.manifest["objective"] = {
            "file": path.name,
            "sha256": digest(path.read_bytes()),
        }
        self.save(rows)
        return path

    def test_objective_scores_declared_thresholds_using_the_named_baseline(self):
        rows = self.rows({("A", "speed", 1): (True, 5), ("B", "speed", 2): (True, 5)})
        self.save(rows)
        self.assertEqual(self.cli()["objective"]["status"], "not_declared")
        self.save_objective(self.objective(), rows)
        obj = self.cli()["objective"]
        self.assertTrue(obj["by_arm"]["speed"]["observed_dev_thresholds_met"])
        self.assertEqual(obj["by_arm"]["speed"]["joint_seconds_saved"], 10)
        self.assertFalse(obj["independent_holdout_verified"])
        self.assertNotIn(digest(b"sealed-task"), json.dumps(obj))
        self.assertTrue(obj["declared_external_checks"])
        self.assertEqual(self.cli()["mde"]["fixed_pair"], ["glm", "speed"])
        with self.assertRaises(SystemExit):
            self.cli("--mde-pair", "speed", "glm")
        rows[0]["strict"] = False
        self.save(rows)
        by_arm = self.cli()["objective"]["by_arm"]["speed"]
        self.assertFalse(by_arm["observed_dev_thresholds_met"])
        self.assertFalse(by_arm["checks"]["preserves_strict_passes"])

    def test_objective_rejects_drift_and_overlap(self):
        for change in [
            {"holdout_task_id_sha256": [digest(b"B")]},
            {"dev_guards": ["A"]},
            {"dev_targets": ["other"]},
            {"repeats": [1]},
            {"arms": ["new", "glm"]},
            {"priorities": ["latency"]},
            {"study": "another"},
            {"policy_sha256_by_arm": {}},
            {"controls_sha256_by_dev_task": {"B": digest(b"A")}},
        ]:
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.save_objective({**self.objective(), **change}, self.rows())
                self.cli()
        spec = self.objective()
        path = self.save_objective(spec, self.rows())
        path.write_text(json.dumps({**spec, "baseline": "speed"}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.cli()

    def test_retired_holdout_must_be_explicit_and_stays_outside_next(self):
        self.manifest["split"] = "retired-holdout"
        self.save(self.rows())
        with self.assertRaises(SystemExit):
            self.cli()
        self.assertEqual(self.cli(scope="retired-holdout")["scope"], "retired-holdout")
        self.assertFalse(self.work.exists())
        data = io.StringIO()
        with redirect_stdout(data):
            loop.main(["--workspace", str(self.work), "next"])
        self.assertEqual(json.loads(data.getvalue())["measurements"], [])
        self.manifest["split"] = "holdout"
        self.save(self.rows())
        with self.assertRaises(SystemExit):
            self.cli(scope="retired-holdout")


if __name__ == "__main__":
    unittest.main()
