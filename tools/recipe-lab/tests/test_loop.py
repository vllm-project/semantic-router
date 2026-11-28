"""Exercise the ledger primitives and the development loop CLI surface."""

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from recipe_lab import loop
from recipe_lab.loop_evidence import digest


class LedgerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workspace = Path(tmp.name) / "ledger"
        self.ledger = loop.Ledger(self.workspace)

    def measurement(self, arm, seconds):
        return {
            "kind": "measurement",
            "split": "dev",
            "arm": arm,
            "recipe": "main",
            "public_model": "public",
            "strong_model": "A",
            "config": f"{arm}/config.yaml",
            "config_sha256": digest(f"{arm}".encode()),
            "agent_seconds": seconds,
            "hypothesis": "Probe tool use",
            "evidence": {},
            "cases": {},
        }

    def test_round_trip(self):
        first = self.ledger.append(self.measurement("fixed", 12.5))
        self.assertTrue(first["record_id"].startswith("rec-"))
        # Re-appending the identical measurement is a no-op, not a new row.
        again = self.ledger.append({k: v for k, v in first.items() if k != "record_id"})
        self.assertEqual(again, first)
        candidate = self.ledger.append(self.measurement("draft", 10))
        self.assertNotEqual(candidate["record_id"], first["record_id"])
        rows = self.ledger.rows()
        self.assertEqual(
            [r["record_id"] for r in rows],
            [first["record_id"], candidate["record_id"]],
        )
        # measure() resolves measurements by record id and rejects unknown ones.
        self.assertEqual(self.ledger.measure(first["record_id"]), first)
        with self.assertRaises(ValueError):
            self.ledger.measure("rec-unknown000")

    def test_tampered_ledger_is_rejected(self):
        self.ledger.append(self.measurement("fixed", 12.5))
        # Any rewrite without recomputing the fingerprint is refused.
        rows = [json.loads(line) for line in self.ledger.path.read_bytes().splitlines()]
        rows[0]["arm"] = "draft"
        self.ledger.path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            self.ledger.rows()

    def _graded_fixtures(self):
        """A real config file plus one physical episode (result + requests)."""
        run = self.workspace.parent / "run"
        folder = run / "dsh" / "runs" / "task" / "fixed" / "main"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(
            json.dumps({"termination": "natural_completion"}), encoding="utf-8"
        )
        (folder / "requests.jsonl").write_text(
            json.dumps({"event": "dispatch", "call": 1}) + "\n", encoding="utf-8"
        )
        config = run / "config.yaml"
        config.write_text(
            "version: v0.5\n"
            "entrypoints:\n"
            "  - model_names: [public]\n"
            "    recipe: main\n"
            "recipes:\n"
            "  - name: sibling\n"
            "  - name: main\n"
            "providers:\n"
            "  models:\n"
            "    - name: A\n",
            encoding="utf-8",
        )
        evidence = {
            str(folder / "result.json"): digest((folder / "result.json").read_bytes()),
            str(folder / "requests.jsonl"): digest(
                (folder / "requests.jsonl").read_bytes()
            ),
        }
        return config, evidence

    def test_relabelled_measurement_is_not_an_independent_comparison(self):
        config, evidence = self._graded_fixtures()

        def measurement(hypothesis):
            return {
                "kind": "measurement",
                "split": "dev",
                "arm": "fixed",
                "recipe": "main",
                "public_model": "public",
                "strong_model": "A",
                "config": str(config),
                "config_sha256": digest(config.read_bytes()),
                "hypothesis": hypothesis,
                "evidence": evidence,
                "cases": {},
            }

        first = self.ledger.append(measurement("first import"))
        second = self.ledger.append(measurement("Relabel the same physical run"))
        self.assertNotEqual(first["record_id"], second["record_id"])
        # Same physical episode (identical result/request fingerprints): a
        # relabelled re-import must not become an independent comparison.
        with self.assertRaises(ValueError):
            loop.compare(
                self.ledger,
                argparse.Namespace(
                    baseline=first["record_id"],
                    candidate=second["record_id"],
                    disposition=None,
                    reason=None,
                ),
            )

    def test_failed_publication_preserves_previous_ledger(self):
        self.ledger.append(self.measurement("fixed", 12.5))
        before = self.ledger.path.read_bytes()
        # A killed writer must leave the previous JSONL intact, including its
        # last line: publish the unchanged prefix and the new line atomically.
        with (
            patch.object(
                loop.os, "replace", side_effect=OSError("interrupted publish")
            ),
            self.assertRaises(OSError),
        ):
            self.ledger.append(self.measurement("draft", 10))
        self.assertEqual(before, self.ledger.path.read_bytes())
        leftovers = [
            p.name
            for p in self.workspace.iterdir()
            if p.name not in ("ledger.jsonl", "ledger.lock")
        ]
        self.assertEqual(leftovers, [])


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def get(path):
    return json.loads(path.read_bytes())


class LoopTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.workspace = self.root / "ledger"
        self.arm("fixed", "A", 12.5)
        self.arm("draft", "B", 10)

    def arm(self, arm, model, seconds):
        run = self.root / arm
        config = {
            "version": "v0.5",
            "entrypoints": [{"model_names": ["public"], "recipe": "main"}],
            "recipes": [
                {"name": "sibling"},
                {
                    "name": "main",
                    "decisions": [{"name": "fast", "modelRefs": [{"model": model}]}],
                },
            ],
            "providers": {
                "models": [
                    {"name": "A"},
                    {"name": "B", "provider_model_id": "B-upstream"},
                ]
            },
        }
        put(run / "config.yaml", config)
        spec = {
            "expected_criteria": ["delivered"],
            "instruction_sha256": "a" * 64,
            "judge": ["python", "judge.py"],
            "pinned_files": {"judge.py": "c" * 64},
            "image": "pinned-image",
            "author_agent_timeout_s": 99,
            "author_verifier_timeout_s": 10,
        }
        frozen = run / "dsh/frozen.json"
        put(
            frozen,
            {"replicates": 1, "cases": {"task": spec}, "runner_sha256": "d" * 64},
        )
        folder = run / "dsh/runs/task" / arm / "main"
        observed = {
            "model": "public",
            "arm": arm,
            "prompt_sha256": "a" * 64,
            "official_agent_timeout_s": 99,
            "official_verifier_timeout_s": 10,
        }
        put(
            folder / "protocol.json",
            dict(
                **observed,
                runner_sha256="d" * 64,
                verified_task_packet_sha256=digest(frozen.read_bytes()),
                image="pinned-image",
                dsh_version="stock",
                provider_impl_sha256="1" * 64,
                request_limit=None,
                configured_output_cap=None,
                manual_retries=False,
                prompt_mutations=False,
                docker_network="none",
                framework="verifier",
                declared_environment={},
                applied_cpus=2,
                applied_memory_mb=1024,
                storage="overlay",
            ),
        )
        put(
            folder / "provider.json",
            [
                {
                    "id": "agent-default-model",
                    "config": {"provider": "source", "model": "public"},
                },
                {
                    "id": "llm-pi-ai",
                    "config": {
                        "providers": {
                            "source": {
                                "models": [{"id": "public", "contextWindow": 131072}]
                            }
                        }
                    },
                },
            ],
        )
        grade = {"correct": True, "criteria": {"delivered": {"passed": True}}}
        put(folder / "grade/score.json", grade)
        put(
            folder / "result.json",
            dict(
                **observed,
                grade=grade,
                judge_exit=0,
                judge_sha256="c" * 64,
                agent_seconds=seconds,
                model_requests=1,
                termination="natural_completion",
            ),
        )
        requests = [
            {"event": "dispatch", "call": 1, "model": "public", "last_role": "tool"},
            {
                "event": "response",
                "call": 1,
                "http": 200,
                "done": True,
                "selected_model": model,
                "returned_model": "B-upstream" if model == "B" else "A",
                "elapsed_s": 1.23,
            },
        ]
        (folder / "requests.jsonl").write_text(
            "\n".join(map(json.dumps, requests)) + "\n", encoding="utf-8"
        )
        put(
            run / f"dsh/bindings/task/{arm}.json",
            {
                "model": "public",
                "client_window": 131072,
                "active_runtime_hash": "e" * 64,
                "source_sha256": digest((run / "config.yaml").read_bytes()),
            },
        )

    def cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            loop.main(["--workspace", str(self.workspace), *map(str, args)])
        return json.loads(output.getvalue())

    def record(self, arm, *extra):
        run = self.root / arm
        return self.cli(
            "record",
            "--run",
            run,
            "--arm",
            arm,
            "--config",
            run / "config.yaml",
            "--public-model",
            "public",
            "--strong",
            "A",
            "--hypothesis",
            "Probe tool use",
            *extra,
        )["record_id"]

    def test_round_trip(self):
        base = self.record("fixed")
        self.assertEqual(base, self.record("fixed"))
        candidate = self.record("draft", "--based-on", base)
        comparison = self.cli(
            "compare",
            "--baseline",
            base,
            "--candidate",
            candidate,
            "--disposition",
            "advance-to-holdout",
            "--reason",
            "Matched pass, shorter on dev.",
        )
        self.assertEqual(
            comparison["jointly_correct"]["task"],
            {"baseline_seconds": 12.5, "candidate_seconds": 10},
        )
        self.assertEqual(len(loop.Ledger(self.workspace).rows()), 3)
        context = self.cli("next")
        self.assertEqual(len(context["measurements"]), 2)
        self.assertIn(
            "e7b-tool-progress", [case["id"] for case in context["references"]]
        )
        self.assertEqual(self.cli("next"), context)
        self.assertEqual(
            context["measurements"][1]["cases"]["task"]["returned_models"],
            {"B-upstream": 1},
        )

    def test_working_proposal_targets_only_public_recipe(self):
        base = self.record("fixed")
        parent = get(self.root / "fixed/config.yaml")
        replacement = get(self.root / "draft/config.yaml")["recipes"][1]
        prop, out = self.root / "recipe.yaml", self.root / "candidate.yaml"
        put(prop, replacement)
        receipt = self.cli("propose", "--base", base, "--recipe", prop, "--out", out)
        self.assertEqual(receipt["config_sha256"], digest(out.read_bytes()))
        parent["recipes"][1] = replacement
        self.assertEqual(yaml.safe_load(out.read_bytes()), parent)
        with self.assertRaises(SystemExit):
            self.cli("propose", "--base", base, "--recipe", prop, "--out", out)
        put(prop, {"name": "sibling"})
        with self.assertRaises(SystemExit):
            self.cli(
                "propose",
                "--base",
                base,
                "--recipe",
                prop,
                "--out",
                self.root / "bad.yaml",
            )

    def test_native_output_limit_uses_strict_grade_and_the_final_client_call(self):
        folder = self.root / "fixed/dsh/runs/task/fixed/main"
        req = folder / "requests.jsonl"
        first, reply = [json.loads(line) for line in req.read_bytes().splitlines()]
        # The first reply finishes last in the log. Only call 2 is the final dispatch.
        rows = [
            first,
            {**first, "call": 2},
            dict(reply, call=2, finish="length"),
            dict(reply, finish="stop", usage={"completion_tokens": 999}),
        ]
        req.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        score = {"correct": False, "criteria": {"delivered": {"passed": False}}}
        put(folder / "grade/score.json", score)
        put(
            folder / "result.json",
            dict(
                get(folder / "result.json"),
                grade=score,
                termination="native_output_limit",
                model_requests=2,
            ),
        )
        self.record("draft")
        self.assertNotIn(
            "strict_failure_native_output_limit", self.cli("next")["symptoms"]
        )
        self.record("fixed")
        ctx = self.cli("next")
        self.assertIn("strict_failure_native_output_limit", ctx["symptoms"])
        last = ctx["measurements"][-1]["cases"]["task"]["terminal"]
        self.assertEqual((last["last_call"], last["finish_reason"]), (2, "length"))
        self.assertEqual(last["explicit_request_limits"], {})
        for key in ("completion_tokens", "reasoning_tokens", "upstream_request_limits"):
            self.assertIsNone(last[key])

    def test_corrupt_or_incomplete_receipts_rejected(self):
        f = self.root / "fixed/dsh/runs/task/fixed/main"
        for name, value in [
            ("grade/score.json", {"correct": False, "criteria": {}}),
            ("requests.jsonl", {"event": "dispatch", "call": 1}),
        ]:
            with self.subTest(name=name):
                path = f / name
                original = path.read_bytes()
                put(path, value)
                with self.assertRaises(SystemExit):
                    self.record("fixed")
                path.write_bytes(original)
        (self.root / "fixed/config.yaml").write_bytes(
            (self.root / "draft/config.yaml").read_bytes()
        )
        with self.assertRaises(SystemExit):
            self.record("fixed")

    def test_no_holdout_content_in_next_and_no_tuning_from_holdout(self):
        hidden = self.record("draft", "--split", "holdout")
        self.assertEqual(self.cli("next")["measurements"], [])
        with self.assertRaises(SystemExit):
            self.record("fixed", "--based-on", hidden)

    def test_frozen_split_binds_import_and_historical_readback(self):
        src = self.root / "draft/dsh"
        frozen = src / "frozen.json"
        put(frozen, {**get(frozen), "split": "holdout"})
        protocol = src / "runs/task/draft/main/protocol.json"
        put(
            protocol,
            {
                **get(protocol),
                "verified_task_packet_sha256": digest(frozen.read_bytes()),
            },
        )
        with (
            patch(
                "recipe_lab.loop_evidence.episode",
                side_effect=AssertionError("read holdout"),
            ),
            self.assertRaises(SystemExit),
        ):
            self.record("draft", "--split", "dev")
        hidden = self.record("draft")
        self.assertEqual(self.record("draft", "--split", "holdout"), hidden)
        ledger = loop.Ledger(self.workspace)
        sealed = ledger.measure(hidden)
        self.assertEqual(sealed["split"], "holdout")
        # Simulate an older importer accepting the same sealed packet as development.
        stale = ledger.append({**sealed, "split": "dev"})["record_id"]
        base = self.record("fixed")
        ledger.append(
            {"kind": "comparison", "split": "dev", "baseline": base, "candidate": stale}
        )
        original = ledger.path.read_bytes()
        context = self.cli("next")
        self.assertEqual([r["record_id"] for r in context["measurements"]], [base])
        self.assertIn("frozen task packet", context["excluded_measurements"][stale])
        self.assertEqual(context["comparisons"], [])
        with self.assertRaises(SystemExit):
            self.cli("compare", "--baseline", base, "--candidate", stale)
        self.assertEqual(ledger.path.read_bytes(), original)

    def test_unrecognized_frozen_split_is_refused_before_accessing_cases(self):
        frozen = self.root / "draft/dsh/frozen.json"
        for value in ("test", "", None):
            with self.subTest(value=value):
                put(frozen, {**get(frozen), "split": value})
                with (
                    patch(
                        "recipe_lab.loop_evidence.episode",
                        side_effect=AssertionError("read cases"),
                    ),
                    self.assertRaises(SystemExit),
                ):
                    self.record("draft")
        self.assertFalse(loop.Ledger(self.workspace).path.exists())

    def test_relabelled_measurement_is_not_an_independent_comparison(self):
        first = self.record("fixed")
        second = self.record("fixed", "--hypothesis", "Relabel the same physical run")
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.cli("compare", "--baseline", first, "--candidate", second)

    def test_runner_failure_cannot_become_a_strict_success(self):
        result = self.root / "fixed/dsh/runs/task/fixed/main/result.json"
        initial = get(result)
        for failure in (
            {"runner_error": "workspace mount failed"},
            {"termination": "grader_error"},
        ):
            put(result, {**initial, **failure})
            with self.subTest(failure=failure), self.assertRaises(
                SystemExit
            ), redirect_stderr(io.StringIO()):
                self.record("fixed")
        self.assertEqual(loop.Ledger(self.workspace).rows(), [])

    def test_model_endpoint_abort_never_records_even_if_grader_completes(self):
        self.record("fixed")
        ledger = loop.Ledger(self.workspace)
        before = ledger.path.read_bytes()
        folder = self.root / "draft/dsh/runs/task/draft/main"
        for passed in (False, True):
            with self.subTest(passed=passed):
                grade = {
                    "correct": passed,
                    "criteria": {"delivered": {"passed": passed}},
                }
                put(folder / "grade/score.json", grade)
                result = get(folder / "result.json")
                put(
                    folder / "result.json",
                    {**result, "grade": grade, "termination": "model_endpoint_error"},
                )
                errors = io.StringIO()
                with self.assertRaises(SystemExit), redirect_stderr(errors):
                    self.record("draft")
                self.assertIn("model_endpoint_error", errors.getvalue())
                self.assertEqual(ledger.path.read_bytes(), before)

    def test_legacy_endpoint_abort_is_excluded_from_proposal_and_comparison(self):
        base = self.record("fixed")
        result = self.root / "draft/dsh/runs/task/draft/main/result.json"
        put(result, {**get(result), "termination": "model_endpoint_error"})
        # Previous importers accepted a valid grade regardless of the stop reason.
        with patch("recipe_lab.loop_evidence.check_termination"):
            stale = self.record("draft")
        ledger = loop.Ledger(self.workspace)
        ledger.append(
            {
                "kind": "comparison",
                "split": "dev",
                "baseline": base,
                "candidate": stale,
                "wins": ["task"],
                "losses": [],
                "jointly_correct": {},
                "disposition": None,
                "reason": None,
            }
        )
        before = ledger.path.read_bytes()
        context = self.cli("next")
        self.assertEqual([row["record_id"] for row in context["measurements"]], [base])
        self.assertIn("model_endpoint_error", context["excluded_measurements"][stale])
        self.assertEqual(context["comparisons"], [])
        for command in (
            ("compare", "--baseline", stale, "--candidate", base),
            ("compare", "--baseline", base, "--candidate", stale),
            (
                "propose",
                "--base",
                stale,
                "--recipe",
                result,
                "--out",
                self.root / "new.yaml",
            ),
        ):
            with self.subTest(command=command):
                errors = io.StringIO()
                with self.assertRaises(SystemExit), redirect_stderr(errors):
                    self.cli(*command)
                self.assertIn("model_endpoint_error", errors.getvalue())
        self.assertEqual(ledger.path.read_bytes(), before)
        self.assertFalse((self.root / "new.yaml").exists())

    def test_recovered_transport_error_and_author_deadline_remain_measurements(self):
        folder = self.root / "draft/dsh/runs/task/draft/main"
        receipts = folder / "requests.jsonl"
        first, reply = [json.loads(line) for line in receipts.read_bytes().splitlines()]
        receipts.write_text(
            "\n".join(
                map(
                    json.dumps,
                    [
                        first,
                        {"event": "response", "call": 1, "http": 429, "done": False},
                        {**first, "call": 2},
                        {**reply, "call": 2},
                    ],
                )
            )
            + "\n",
            encoding="utf-8",
        )
        result = get(folder / "result.json")
        put(folder / "result.json", {**result, "model_requests": 2})
        recovered = self.record("draft")
        fixed = self.root / "fixed/dsh/runs/task/fixed/main/result.json"
        put(fixed, {**get(fixed), "termination": "official_agent_timeout"})
        deadline = self.record("fixed")
        outcomes = self.cli("next")["measurements"]
        self.assertEqual([row["record_id"] for row in outcomes], [recovered, deadline])
        self.assertEqual(outcomes[0]["cases"]["task"]["incomplete_responses"], 1)

    def test_mismatched_window_and_changed_evidence_block_comparison(self):
        base = self.record("fixed")
        binding = self.root / "draft/dsh/bindings/task/draft.json"
        put(binding, {**get(binding), "client_window": 1000})
        with self.assertRaises(SystemExit):
            self.record("draft")
        provider = self.root / "draft/dsh/runs/task/draft/main/provider.json"
        config = get(provider)
        config[1]["config"]["providers"]["source"]["models"][0]["contextWindow"] = 1000
        put(provider, config)
        candidate = self.record("draft")
        with self.assertRaises(SystemExit):
            self.cli("compare", "--baseline", base, "--candidate", candidate)
        put(binding, {})
        with self.assertRaises(SystemExit):
            self.cli("next")

    def test_incomplete_stream_retained_but_complete_stream_requires_model_identity(
        self,
    ):
        path = self.root / "draft/dsh/runs/task/draft/main/requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1].pop("returned_model")

        def write():
            path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")

        write()
        with self.assertRaises(SystemExit):
            self.record("draft")
        rows[-1].update(done=False, selected_model=None, elapsed_s=300.0)
        write()
        self.record("draft")
        self.assertIn("active_stream_cutoff", self.cli("next")["symptoms"])

    def test_failed_publication_preserves_previous_ledger(self):
        self.record("fixed")
        before = (self.workspace / "ledger.jsonl").read_bytes()
        with patch.object(
            loop.os, "replace", side_effect=OSError("interrupted publish")
        ), self.assertRaises(SystemExit):
            self.record("draft")
        self.assertEqual(before, (self.workspace / "ledger.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
