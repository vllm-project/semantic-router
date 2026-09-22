"""Kubernetes shards retain independent profile evidence and teardown boundaries."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
import run_e2e_batch as runner  # noqa: E402
from ci_plan import make_plan  # noqa: E402
from execution_batches import e2e_batches, validate_execution_batch  # noqa: E402

SHA = "a" * 40
PROFILES = ("category-remote-backend", "complexity-remote-backend")

# Actual child processes exercise environment, report isolation and cleanup,
# without requiring a Kubernetes daemon for the deterministic harness tests.
CHILD = r"""
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
profile = next(value.split("=", 1)[1] for value in args if value.startswith("-profile="))
cluster = next(value.split("=", 1)[1] for value in args if value.startswith("-cluster="))
raw = Path(os.environ["E2E_REPORT_DIR"])
state = Path(os.environ["TMPDIR"]).parent
clusters = Path(os.environ["FAKE_CLUSTERS"])
(clusters / cluster).touch()
outcome = json.loads(os.environ.get("FAKE_OUTCOMES", "{}" )).get(profile, "passed")
with Path(os.environ["FAKE_CALLS"]).open("a") as stream:
    stream.write(json.dumps({"kind": "profile", "profile": profile, "cluster": cluster,
        "state": str(state), "raw": str(raw), "args": args,
        "kubeconfig": os.environ["KUBECONFIG"], "prebuilt": os.environ["PREBUILT_RUNTIME_IMAGES"],
        "baseline": os.environ["E2E_BASELINE_SUITE"]}) + "\n")
assert not list(state.rglob("left-by-sibling"))
(state / "left-by-sibling").touch()
if outcome == "timeout":
    time.sleep(30)
if outcome != "missing":
    failed = outcome == "failed"
    cases = [{"Name": "required-case", "Passed": not failed}]
    if outcome == "incomplete":
        cases = []
    report = {"profile": "wrong-profile" if outcome == "wrong-profile" else profile,
        "cluster_name": "wrong-cluster" if outcome == "wrong-cluster" else cluster,
        "expected_cases": ["required-case"], "test_results": cases,
        "status": "FAILED" if failed else "PASSED", "exit_code": 1 if failed else 0,
        "total_tests": len(cases), "passed_tests": 0 if failed else len(cases),
        "failed_tests": 1 if failed else 0}
    (raw / "test-report.json").write_text(json.dumps(report))
    (raw / "test-report.md").write_text("Profile " + profile)
print("executed " + profile, flush=True)
raise SystemExit(7 if outcome == "exit" else 0)
"""
KIND = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
clusters = Path(os.environ["FAKE_CLUSTERS"])
with Path(os.environ["FAKE_CALLS"]).open("a") as stream:
    stream.write(json.dumps({"kind": "kind", "args": args}) + "\n")
if args[:2] == ["get", "clusters"]:
    print("\n".join(path.name for path in clusters.iterdir()))
elif args[:2] == ["delete", "cluster"]:
    cluster = args[-1]
    if os.environ.get("FAKE_LEAK") not in ("1", cluster):
        (clusters / cluster).unlink(missing_ok=True)
elif args[:2] == ["export", "logs"]:
    Path(args[2]).mkdir(exist_ok=True)
else:
    raise SystemExit("unexpected kind command: " + repr(args))
"""


class E2EBatchTests(unittest.TestCase):
    def test_real_profiles_reuse_prebuilt_inputs_and_keep_individual_receipts(self):
        result = self.execute()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["receipts"]), len(PROFILES))
        calls = [row for row in result["calls"] if row["kind"] == "profile"]
        self.assertEqual([row["profile"] for row in calls], list(PROFILES))
        for field in ("cluster", "state", "raw", "kubeconfig"):
            self.assertEqual(len({row[field] for row in calls}), len(PROFILES))
        self.assertTrue(all(row["prebuilt"] == "1" for row in calls))
        self.assertTrue(all(row["baseline"] == "standard" for row in calls))
        self.assertTrue(
            all("-use-existing-cluster=false" in row["args"] for row in calls)
        )
        self.assertEqual(result["remaining_clusters"], [])
        self.assertEqual(result["remaining_states"], [])
        for receipt in result["receipts"]:
            self.assertEqual(receipt["source_sha"], SHA)
            self.assertEqual(receipt["evidence"]["expected_cases"], ["required-case"])
            self.assertEqual(
                {row["id"] for row in receipt["artifacts"]},
                {"image:extproc", "image:provider-mocker"},
            )
            self.assertIn(receipt["id"] + "/test-report.json", result["raw"])
        # Every cluster was deleted and absence verified before the next profile.
        first = next(
            index
            for index, row in enumerate(result["calls"])
            if row.get("profile") == PROFILES[0]
        )
        second = next(
            index
            for index, row in enumerate(result["calls"])
            if row.get("profile") == PROFILES[1]
        )
        between = [row["args"][:2] for row in result["calls"][first + 1 : second]]
        self.assertIn(["delete", "cluster"], between)
        self.assertIn(["get", "clusters"], between)

    def test_failed_or_missing_profile_does_not_skip_independent_sibling(self):
        for outcome in (
            "exit",
            "timeout",
            "failed",
            "missing",
            "incomplete",
            "wrong-profile",
            "wrong-cluster",
        ):
            with self.subTest(outcome=outcome):
                result = self.execute(outcomes={PROFILES[0]: outcome})
                self.assertFalse(result["passed"])
                self.assertEqual(
                    [row["id"] for row in result["receipts"]], ["e2e." + PROFILES[1]]
                )
                self.assertEqual(
                    len([row for row in result["calls"] if row["kind"] == "profile"]),
                    len(PROFILES),
                )
                self.assertIn("e2e." + PROFILES[0] + "/failure.txt", result["raw"])
                self.assertEqual(result["summary"]["failed"], ["e2e." + PROFILES[0]])
                self.assertEqual(result["remaining_clusters"], [])

    def test_missing_artifact_or_wrong_source_fails_before_profile_execution(self):
        for condition in ("missing-image", "missing-artifact", "wrong-source"):
            with self.subTest(condition=condition):
                result = self.execute(condition=condition)
                self.assertFalse(result["passed"])
                self.assertEqual(result["receipts"], [])
                self.assertFalse(
                    any(row["kind"] == "profile" for row in result["calls"])
                )
                self.assertEqual(len(result["summary"]["failed"]), len(PROFILES))

    def test_teardown_failure_cannot_qualify_or_contaminate_later_profile(self):
        result = self.execute(condition="leaked-cluster")
        self.assertFalse(result["passed"])
        self.assertEqual(result["receipts"], [])
        self.assertEqual(
            len([row for row in result["calls"] if row["kind"] == "profile"]), 1
        )
        self.assertEqual(len(result["summary"]["failed"]), len(PROFILES))
        self.assertIn(
            "worker isolation unavailable",
            result["raw"]["e2e." + PROFILES[1] + "/failure.txt"],
        )
        self.assertEqual(result["after_retry_clusters"], [])
        self.assertEqual(result["after_retry_states"], [])

    def test_ort_profile_keeps_its_declared_runtime_and_full_baseline(self):
        result = self.execute(profiles=("vela-omni",), full=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["receipts"][0]["runtime"], "ort")
        calls = [row for row in result["calls"] if row["kind"] == "profile"]
        self.assertEqual(calls[0]["baseline"], "full")

    def test_duplicates_tampered_profile_and_unplanned_members_are_rejected(self):
        batch = self.plan()["e2e_batches"][0]
        for condition in ("duplicate", "profile", "images", "foreign-executor"):
            changed = copy.deepcopy(batch)
            if condition == "duplicate":
                changed["verifications"] *= 2
            elif condition == "profile":
                changed["verifications"][0]["profile"] = "missing-profile"
            elif condition == "images":
                changed["images"] = ["extproc"]
            else:
                changed["verifications"][0]["executor"] = "native"
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                validate_execution_batch(changed, "e2e")

    def test_timeout_kills_descendant_listener_and_retains_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            port_file = root / "port"
            listener = (
                "import socket,time; from pathlib import Path; "
                "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
                f"Path({str(port_file)!r}).write_text(str(s.getsockname()[1])); "
                "time.sleep(30)"
            )
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{listener!r}]); "
                "print('started',flush=True); time.sleep(30)"
            )
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                subprocess.TimeoutExpired
            ):
                runner.run_command(
                    [sys.executable, "-c", parent], dict(os.environ), root / "log", 1
                )
            self.assertIn("started", (root / "log").read_text())
            with socket.socket() as connection:
                self.assertNotEqual(
                    connection.connect_ex(("127.0.0.1", int(port_file.read_text()))), 0
                )

    def test_workflow_prepares_once_and_uploads_all_receipts_after_failure(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/integration-test-k8s.yml").read_text()
        )
        job = workflow["jobs"]["integration-test"]
        self.assertNotIn("strategy", job)
        self.assertEqual(job["name"], "Execute Contracts")
        steps = job["steps"]
        self.assertEqual(
            len(
                [
                    step
                    for step in steps
                    if step.get("uses") == "./.github/actions/load-ci-images"
                ]
            ),
            1,
        )
        self.assertEqual(
            len([step for step in steps if step.get("run") == "make build-e2e"]), 1
        )
        self.assertFalse(any("make e2e-test" in step.get("run", "") for step in steps))
        uploads = [
            step for step in steps if step.get("uses") == "actions/upload-artifact@v4"
        ]
        self.assertTrue(all(step["if"] == "always()" for step in uploads))
        self.assertTrue(uploads[0]["with"]["path"].endswith("/results/*.json"))
        self.assertIn("/raw/", uploads[1]["with"]["path"])
        self.assertNotIn("/state/", uploads[1]["with"]["path"])
        self.assertIn("job_timeout_minutes", job["timeout-minutes"])

    def test_cluster_state_can_use_large_worker_disk_without_archiving_pvcs(self):
        result = self.execute(condition="external-state")
        self.assertTrue(result["passed"])
        calls = [row for row in result["calls"] if row["kind"] == "profile"]
        self.assertTrue(all("worker-volume" in row["state"] for row in calls))
        self.assertFalse(any("worker-volume" in name for name in result["raw"]))

    @staticmethod
    def plan(profiles=PROFILES, *, full=False):
        if full:
            plan = make_plan([], source_sha=SHA, full=True)
            records = [
                row for row in plan["verifications"] if row.get("profile") in profiles
            ]
            return {**plan, "e2e_batches": e2e_batches(records)}
        return make_plan(
            [],
            source_sha=SHA,
            requested=tuple("e2e." + profile for profile in profiles),
        )

    def execute(self, *, outcomes=None, condition="", profiles=PROFILES, full=False):
        plan = self.plan(profiles, full=full)
        batch = plan["e2e_batches"][0]
        self.assertEqual(len(batch["verifications"]), len(profiles))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            clusters = root / "clusters"
            clusters.mkdir()
            for name, script in (("e2e", CHILD), ("kind", KIND)):
                path = root / "bin" / name
                path.write_text(f"#!{sys.executable}\n" + script)
                path.chmod(0o755)
            artifacts = [
                {"id": "image:" + image, "sha256": "b" * 64}
                for image in batch["images"]
            ]
            if condition == "missing-artifact":
                artifacts = []
            artifact_file = root / "artifacts.json"
            artifact_file.write_text(json.dumps(artifacts))
            env = {
                "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                "FAKE_CLUSTERS": str(clusters),
                "FAKE_CALLS": str(root / "calls"),
                "FAKE_OUTCOMES": json.dumps(outcomes or {}),
                "FAKE_LEAK": "1" if condition == "leaked-cluster" else "",
                "CI_IMAGE_RECEIPTS": str(artifact_file),
                "CI_NATIVE_RECEIPTS": "",
                "E2E_PREBUILT_EXT_PROC_IMAGE": "verified:extproc",
                "E2E_PREBUILT_PROVIDER_MOCKER_IMAGE": (
                    "" if condition == "missing-image" else "verified:provider-mocker"
                ),
            }
            if condition == "external-state":
                env["E2E_BATCH_STATE_ROOT"] = str(root / "worker-volume")
            original_check_output = subprocess.check_output

            def checked(command, **kwargs):
                if command[:2] == ["git", "rev-parse"]:
                    return "c" * 40 if condition == "wrong-source" else SHA
                return original_check_output(command, **kwargs)

            with patch.object(runner, "ROOT", root), patch.object(
                runner, "actual_platform", return_value="linux/amd64"
            ), patch.object(
                runner,
                "PROFILE_TIMEOUT_MINUTES",
                1 / 60 if "timeout" in (outcomes or {}).values() else 90,
            ), patch.object(
                runner.subprocess, "check_output", side_effect=checked
            ), patch.dict(
                os.environ, env
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                output = root / "output"
                passed = runner.run_batch(batch, output)
                with self.assertRaises(FileExistsError):
                    runner.run_batch(batch, output)
                remaining_clusters = list(clusters.iterdir())
                remaining_states = [
                    path for path in (output / "state").glob("*") if path.exists()
                ]
                # The workflow's always-running cleanup can recover interrupted teardown.
                with patch.dict(os.environ, {"FAKE_LEAK": ""}):
                    runner.cleanup_batch(output)
            return {
                "passed": passed,
                "receipts": [
                    json.loads(path.read_text())
                    for path in sorted((output / "results").glob("*.json"))
                ],
                "raw": {
                    str(path.relative_to(output / "raw")): path.read_text()
                    for path in (output / "raw").rglob("*")
                    if path.is_file()
                },
                "calls": (
                    [
                        json.loads(line)
                        for line in (root / "calls").read_text().splitlines()
                    ]
                    if (root / "calls").exists()
                    else []
                ),
                "summary": json.loads((output / "summary.json").read_text()),
                "remaining_clusters": remaining_clusters,
                "remaining_states": remaining_states,
                "after_retry_clusters": list(clusters.iterdir()),
                "after_retry_states": list((output / "state").glob("*")),
            }


if __name__ == "__main__":
    unittest.main()
