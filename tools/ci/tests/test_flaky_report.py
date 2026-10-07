from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flaky_report


def raw_report(profile: str, results: list[tuple[str, bool, int]]) -> dict:
    """One `e2e-evidence-*` framework report: name, passed, attempts."""
    return {
        "profile": profile,
        "test_results": [
            {"Name": name, "Passed": passed, "Attempts": attempts}
            for name, passed, attempts in results
        ],
    }


def receipt(profile: str, cases: list[tuple[str, bool]]) -> dict:
    """One `ci-result-*` receipt: `evidence.cases` is the gated inventory."""
    return {
        "schema_version": 1,
        "id": "e2e.envoy-ai-gateway",
        "evidence": {
            "profile": profile,
            "cases": [
                {"id": name, "status": "passed" if passed else "failed"}
                for name, passed in cases
            ],
        },
    }


def write_evidence(root: Path, run: str, report: dict) -> None:
    path = (
        root
        / run
        / "e2e-evidence-batch"
        / "raw"
        / report["profile"]
        / "test-report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))


def write_receipt(root: Path, run: str, value: dict) -> None:
    path = root / run / "ci-result-e2e.envoy-ai-gateway" / "envoy-ai-gateway.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sample(
    profile: str, results: list[tuple[str, bool, int]], **kwargs
) -> flaky_report.Sample:
    return flaky_report.Sample(
        run=kwargs.pop("run", "1"),
        profile=profile,
        cases={
            name: flaky_report.CaseOutcome(passed=passed, attempts=attempts)
            for name, passed, attempts in results
        },
        **kwargs,
    )


class Completed:
    """Stand-in for the subprocess result of a `gh` call."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_entry(run_id: str, conclusion: str, hour: int, branch: str = "main") -> dict:
    """One entry of the workflow-runs payload."""
    return {
        "id": run_id,
        "conclusion": conclusion,
        "created_at": f"{date.today():%Y-%m-%d}T{hour:02d}:00:00Z",
        "head_branch": branch,
    }


class AggregateTests(unittest.TestCase):
    def test_failure_and_retry_rates_are_reported_per_case(self) -> None:
        rows = flaky_report.aggregate(
            [
                sample("streaming", [("cases/a", False, 2), ("cases/b", True, 1)]),
                sample("streaming", [("cases/a", True, 2), ("cases/b", True, 1)]),
                sample("streaming", [("cases/a", True, 3), ("cases/b", True, 1)]),
                sample("streaming", [("cases/a", True, 1), ("cases/b", True, 1)]),
            ]
        )

        # cases/b never failed or retried, so it is not a row. The first sample
        # failed both attempts, which is a break and not a retry.
        self.assertEqual([row["case"] for row in rows], ["cases/a"])
        self.assertEqual(rows[0]["merged"], 4)
        self.assertEqual(rows[0]["failures"], 1)
        self.assertEqual(rows[0]["retries"], 2)
        self.assertEqual(rows[0]["failure_rate"], 0.25)
        self.assertEqual(rows[0]["retry_rate"], 0.5)

    def test_a_case_that_failed_every_attempt_is_a_failure_not_a_retry(self) -> None:
        rows = flaky_report.aggregate([sample("streaming", [("cases/a", False, 2)])])

        self.assertEqual(rows[0]["failures"], 1)
        self.assertEqual(rows[0]["failure_rate"], 1.0)
        self.assertEqual(rows[0]["retries"], 0)
        self.assertEqual(rows[0]["retry_rate"], 0.0)

    def test_a_broken_pr_case_is_neither_a_failure_nor_a_retry(self) -> None:
        rows = flaky_report.aggregate(
            [sample("streaming", [("cases/a", False, 2)], retry_only=True)]
        )

        # The branch broke the case. Reporting it as flaky would hide that.
        self.assertEqual(rows, [])

    def test_a_case_that_passed_on_a_later_attempt_is_a_retry(self) -> None:
        rows = flaky_report.aggregate([sample("streaming", [("cases/a", True, 2)])])

        self.assertEqual(rows[0]["failures"], 0)
        self.assertEqual(rows[0]["retries"], 1)
        self.assertEqual(rows[0]["retry_rate"], 1.0)

    def test_pr_runs_count_towards_retries_but_never_towards_failures(self) -> None:
        rows = flaky_report.aggregate(
            [
                sample("streaming", [("cases/a", True, 1)]),
                sample("streaming", [("cases/a", False, 1)], retry_only=True),
                sample("streaming", [("cases/a", True, 2)], retry_only=True),
            ]
        )

        self.assertEqual(rows[0]["merged"], 1)
        self.assertEqual(rows[0]["samples"], 3)
        # The PR failure is not a failure rate, but the PR retry is a retry rate.
        self.assertEqual(rows[0]["failures"], 0)
        self.assertEqual(rows[0]["retries"], 1)
        self.assertEqual(rows[0]["retry_rate"], 1 / 3)

    def test_a_case_seen_only_in_pr_runs_is_reported(self) -> None:
        rows = flaky_report.aggregate(
            [sample("streaming", [("cases/a", True, 2)], retry_only=True)]
        )

        self.assertEqual(rows[0]["merged"], 0)
        self.assertEqual(rows[0]["failures"], 0)
        self.assertEqual(rows[0]["failure_rate"], 0.0)
        self.assertEqual(rows[0]["retries"], 1)

    def test_worst_failure_rate_comes_first_then_retry_rate(self) -> None:
        rows = flaky_report.aggregate(
            [
                sample(
                    "streaming",
                    [
                        ("cases/rare", False, 1),
                        ("cases/always", False, 1),
                        ("cases/wobbly", True, 2),
                    ],
                ),
                sample(
                    "streaming",
                    [
                        ("cases/rare", True, 1),
                        ("cases/always", False, 1),
                        ("cases/wobbly", True, 1),
                    ],
                ),
            ]
        )

        self.assertEqual(
            [row["case"] for row in rows],
            ["cases/always", "cases/rare", "cases/wobbly"],
        )

    def test_the_same_case_name_in_two_profiles_stays_separate(self) -> None:
        rows = flaky_report.aggregate(
            [
                sample("alpha", [("cases/shared", False, 1)]),
                sample("beta", [("cases/shared", False, 1)]),
            ]
        )

        self.assertEqual({row["profile"] for row in rows}, {"alpha", "beta"})
        self.assertEqual(len(rows), 2)

    def test_a_clean_window_reports_nothing(self) -> None:
        self.assertEqual(
            flaky_report.aggregate([sample("streaming", [("cases/a", True, 1)])]), []
        )


class RenderTests(unittest.TestCase):
    def test_rendering_states_when_nothing_failed_or_retried(self) -> None:
        text = flaky_report.render(
            [], [sample("streaming", [("cases/a", True, 1)])], [], []
        )

        self.assertIn("No case failed and no case needed a retry in this window.", text)

    def test_rendering_tabulates_rates(self) -> None:
        rows = [
            {
                "profile": "streaming",
                "case": "cases/a",
                "merged": 1,
                "samples": 3,
                "failures": 0,
                "retries": 1,
                "failure_rate": 0.0,
                "retry_rate": 1 / 3,
            }
        ]

        text = flaky_report.render(
            rows, [sample("streaming", [("cases/a", True, 2)])], [], []
        )

        self.assertIn(
            "| `streaming` | `cases/a` | 1 | 0 | 0.0% | 3 | 1 | 33.3% |", text
        )

    def test_rendering_lists_the_runs_it_read(self) -> None:
        runs = [
            flaky_report.RunRef("main.yml", "11", "2026-09-30T02:00:00Z", "success"),
            flaky_report.RunRef("main.yml", "12", "2026-10-01T02:00:00Z", "failure"),
            flaky_report.RunRef(
                "pr.yml", "13", "2026-10-01T03:00:00Z", "success", retry_workflow=True
            ),
        ]

        text = flaky_report.render([], [], runs, [])

        self.assertIn(
            "Read 3 run(s); 0 profile report(s), 0 with raw e2e evidence.", text
        )
        self.assertIn(
            "`main.yml`: `11` (2026-09-30, success), no e2e artifacts, "
            "`12` (2026-10-01, failure), no e2e artifacts",
            text,
        )
        self.assertIn(
            "`pr.yml` (retries only): `13` (2026-10-01, success), no e2e artifacts",
            text,
        )

    def test_rendering_counts_the_reports_each_run_contributed(self) -> None:
        runs = [
            flaky_report.RunRef("main.yml", "11", "2026-09-30T02:00:00Z", "success")
        ]
        samples = [
            sample("streaming", [("cases/a", True, 1)], run="main-11"),
            sample("routing", [("cases/b", True, 1)], run="main-11"),
        ]

        text = flaky_report.render([], samples, runs, [])

        self.assertIn(
            "`main.yml`: `11` (2026-09-30, success), 2 profile report(s)", text
        )

    def test_rendering_says_why_a_run_had_no_artifacts(self) -> None:
        runs = [
            flaky_report.RunRef(
                "pr.yml",
                "13",
                "2026-10-01T03:00:00Z",
                "success",
                retry_workflow=True,
                note="no download: no valid artifacts found to download",
            )
        ]

        text = flaky_report.render([], [], runs, [])

        self.assertIn("no download: no valid artifacts found to download", text)
        self.assertNotIn("no e2e artifacts", text)

    def test_rendering_names_the_branch_of_an_unmerged_run(self) -> None:
        runs = [
            flaky_report.RunRef(
                "nightly-build.yml",
                "7",
                "2026-10-01T02:00:00Z",
                "failure",
                head_branch="feature/x",
            )
        ]

        text = flaky_report.render([], [], runs, [])

        self.assertIn("`7` (2026-10-01, failure, `feature/x`)", text)

    def test_rendering_lists_the_workflows_it_could_not_read(self) -> None:
        text = flaky_report.render(
            [], [], [], ["`pr.yml`: could not list runs: HTTP 503"]
        )

        self.assertIn("<summary>Collection problems</summary>", text)
        self.assertIn("- `pr.yml`: could not list runs: HTTP 503", text)


class CollectionTests(unittest.TestCase):
    def list_runs(self, entries: list[dict], **kwargs) -> tuple[list, str]:
        completed = Completed(
            returncode=kwargs.pop("returncode", 0),
            stdout=json.dumps({"workflow_runs": entries}),
            stderr=kwargs.pop("stderr", ""),
        )
        workflow = kwargs.pop("workflow", "pr.yml")
        with mock.patch.object(flaky_report.subprocess, "run", return_value=completed):
            return flaky_report.recent_runs(
                workflow,
                kwargs.pop("limit", 5),
                7,
                "o/r",
                retry_only=kwargs.pop("retry_only", True),
            )

    def test_a_nightly_run_on_a_branch_counts_towards_retries_only(self) -> None:
        runs, _ = self.list_runs(
            [run_entry("7", "failure", 7, branch="feature/x")],
            workflow="nightly-build.yml",
            retry_only=False,
        )

        self.assertFalse(runs[0].ran_merged_code)
        self.assertTrue(runs[0].retry_only)

    def test_a_nightly_run_on_main_counts_towards_failures(self) -> None:
        runs, _ = self.list_runs(
            [run_entry("7", "failure", 7)],
            workflow="nightly-build.yml",
            retry_only=False,
        )

        self.assertTrue(runs[0].ran_merged_code)
        self.assertFalse(runs[0].retry_only)

    def test_cancelled_runs_are_left_out(self) -> None:
        runs, problem = self.list_runs(
            [
                run_entry("9", "cancelled", 9),
                run_entry("8", "success", 8),
                run_entry("7", "failure", 7),
            ]
        )

        self.assertEqual(problem, "")
        self.assertEqual([run.run_id for run in runs], ["8", "7"])

    def test_the_limit_counts_only_runs_that_can_be_read(self) -> None:
        runs, _ = self.list_runs(
            [
                run_entry("9", "cancelled", 9),
                run_entry("8", "cancelled", 8),
                run_entry("7", "success", 7),
                run_entry("6", "failure", 6),
            ],
            limit=2,
        )

        self.assertEqual([run.run_id for run in runs], ["7", "6"])

    def test_a_listing_failure_is_told_apart_from_an_empty_window(self) -> None:
        runs, problem = self.list_runs(
            [], returncode=1, stderr="HTTP 503: service unavailable\nmore text"
        )

        self.assertEqual(runs, [])
        self.assertEqual(problem, "could not list runs: HTTP 503: service unavailable")

    def test_a_window_of_only_cancelled_runs_is_not_a_problem(self) -> None:
        runs, problem = self.list_runs([run_entry("9", "cancelled", 9)])

        self.assertEqual(runs, [])
        self.assertEqual(problem, "")

    def test_a_stale_page_is_reported_rather_than_read_as_an_empty_window(self) -> None:
        stale = {
            "id": "1",
            "conclusion": "success",
            "created_at": "2020-01-01T00:00:00Z",
        }

        runs, problem = self.list_runs([stale])

        cutoff = f"{date.today() - timedelta(days=7):%Y-%m-%d}"
        self.assertEqual(runs, [])
        self.assertEqual(
            problem, f"no run newer than {cutoff} came back, newest was 2020-01-01"
        )

    def test_collect_runs_names_the_workflow_it_could_not_list(self) -> None:
        args = SimpleNamespace(runs=5, days=7, download_dir=Path("unused"))

        with (
            mock.patch.object(flaky_report, "repo_slug", return_value="o/r"),
            mock.patch.object(flaky_report, "download_runs"),
            mock.patch.object(
                flaky_report,
                "recent_runs",
                side_effect=[([], ""), ([], "could not list runs: HTTP 503")],
            ),
        ):
            runs, problems = flaky_report.collect_runs(["main.yml"], ["pr.yml"], args)

        self.assertEqual(runs, [])
        self.assertEqual(problems, ["`pr.yml`: could not list runs: HTTP 503"])

    def test_a_failed_download_is_recorded_on_the_run(self) -> None:
        run = flaky_report.RunRef("pr.yml", "1", retry_workflow=True)
        completed = Completed(
            returncode=1, stderr="no valid artifacts found to download\n"
        )

        with mock.patch.object(flaky_report.subprocess, "run", return_value=completed):
            flaky_report.download_runs([run], Path("unused"))

        self.assertEqual(run.note, "no download: no valid artifacts found to download")


class LoadingTests(unittest.TestCase):
    def test_receipts_and_raw_reports_merge_into_one_sample_per_run_and_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_receipt(root, "main-1", receipt("streaming", [("cases/a", False)]))
            write_evidence(
                root, "main-1", raw_report("streaming", [("cases/a", True, 2)])
            )

            found = flaky_report.load_samples(root, [])

            # One execution, so one sample, and the raw report wins because it
            # knows the attempts.
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].run, "main-1")
            self.assertTrue(found[0].has_evidence)
            self.assertEqual(found[0].cases["cases/a"].attempts, 2)
            self.assertTrue(found[0].cases["cases/a"].passed)

    def test_a_receipt_without_raw_evidence_still_reports_the_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_receipt(root, "main-1", receipt("streaming", [("cases/a", False)]))

            found = flaky_report.load_samples(root, [])
            rows = flaky_report.aggregate(found)

            self.assertEqual(rows[0]["failures"], 1)
            # Nothing recorded the attempts, so the retry rate stays at zero.
            self.assertEqual(rows[0]["retries"], 0)
            self.assertFalse(found[0].has_evidence)

    def test_the_run_list_decides_which_runs_count_towards_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_evidence(
                root, "pr-7", raw_report("streaming", [("cases/a", False, 1)])
            )
            runs = [flaky_report.RunRef("pr.yml", "7", retry_workflow=True)]

            rows = flaky_report.aggregate(flaky_report.load_samples(root, runs))

            self.assertEqual(rows, [])

    def test_a_manual_nightly_run_on_a_branch_is_not_a_merged_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # The branch run failed the case, the main run passed it. Reading the
            # branch run as merged code reported main as failing the case.
            write_evidence(
                root,
                "nightly-build-7",
                raw_report("streaming", [("cases/a", False, 1)]),
            )
            write_evidence(
                root, "nightly-build-8", raw_report("streaming", [("cases/a", True, 1)])
            )
            runs = [
                flaky_report.RunRef("nightly-build.yml", "7", head_branch="feature/x"),
                flaky_report.RunRef("nightly-build.yml", "8", head_branch="main"),
            ]

            rows = flaky_report.aggregate(flaky_report.load_samples(root, runs))

            self.assertEqual(rows, [])

    def test_two_runs_of_the_same_profile_are_two_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_evidence(
                root, "main-1", raw_report("streaming", [("cases/a", False, 1)])
            )
            write_evidence(
                root, "main-2", raw_report("streaming", [("cases/a", True, 1)])
            )

            rows = flaky_report.aggregate(flaky_report.load_samples(root, []))

            self.assertEqual(rows[0]["samples"], 2)
            self.assertEqual(rows[0]["failure_rate"], 0.5)

    def test_non_e2e_receipts_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # An operator receipt has cases but no profile, so it is not an e2e
            # profile report and must not appear as one.
            write_receipt(
                root,
                "main-1",
                {
                    "id": "operator",
                    "evidence": {
                        "cases": [{"id": "variant/redis", "status": "failed"}]
                    },
                },
            )

            self.assertEqual(flaky_report.load_samples(root, []), [])

    def test_unreadable_and_unrelated_json_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_evidence(
                root, "main-1", raw_report("streaming", [("cases/a", False, 1)])
            )
            broken = (
                root
                / "main-2"
                / "e2e-evidence-batch"
                / "raw"
                / "streaming"
                / "test-report.json"
            )
            broken.parent.mkdir(parents=True)
            broken.write_text("{not json")
            (root / "unrelated.json").write_text(json.dumps({"other": True}))

            found = flaky_report.load_samples(root, [])

            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].run, "main-1")

    def test_a_report_without_attempts_counts_as_one_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_evidence(
                root,
                "main-1",
                {
                    "profile": "streaming",
                    "test_results": [{"Name": "cases/a", "Passed": True}],
                },
            )

            self.assertEqual(
                flaky_report.aggregate(flaky_report.load_samples(root, [])), []
            )


class CliTests(unittest.TestCase):
    def test_the_cli_writes_a_summary_from_a_directory_of_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_evidence(
                root, "main-1", raw_report("streaming", [("cases/a", False, 1)])
            )
            write_evidence(
                root, "main-2", raw_report("streaming", [("cases/a", True, 2)])
            )
            output = root / "summary.md"

            subprocess.run(
                [
                    "python3",
                    str(Path(flaky_report.__file__)),
                    "--from-dir",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            text = output.read_text()
            self.assertIn(
                "| `streaming` | `cases/a` | 2 | 1 | 50.0% | 2 | 1 | 50.0% |", text
            )

    def test_an_empty_window_still_writes_the_summary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "empty"
            root.mkdir()
            output = Path(temp_dir) / "out" / "summary.md"

            completed = subprocess.run(
                [
                    "python3",
                    str(Path(flaky_report.__file__)),
                    "--from-dir",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists(), "the workflow cats the output file")
            self.assertIn(
                "No case failed and no case needed a retry", output.read_text()
            )

    def test_the_cli_rejects_a_window_shorter_than_one_run(self) -> None:
        for arguments, message in (
            (["--runs", "0"], "--runs must be 1 or greater"),
            (["--days", "0"], "--days must be 1 or greater"),
        ):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    ["python3", str(Path(flaky_report.__file__)), *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)


if __name__ == "__main__":
    unittest.main()
