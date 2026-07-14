from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import select_e2e_profiles as selector  # noqa: E402


class SelectE2EProfilesTests(unittest.TestCase):
    def assert_selection(
        self,
        expected: list[str],
        *,
        event_name: str = "pull_request",
        **changes: bool,
    ) -> None:
        profiles = selector.select_e2e_profiles(event_name, changes)
        self.assertEqual(profiles, expected)
        expected_json = ",".join(f'"{profile}"' for profile in expected)
        expected_run = "true" if expected else "false"
        self.assertEqual(
            selector.github_output(profiles),
            f"profiles=[{expected_json}]\nshould_run={expected_run}",
        )

    def test_core_only_selects_baseline(self) -> None:
        self.assert_selection(["kubernetes", "dashboard"], core=True)

    def test_affected_only_selects_affected_profile(self) -> None:
        self.assert_selection(
            ["ml-model-selection"],
            e2e_ml_model_selection=True,
        )

    def test_core_and_ml_union_baseline_with_affected(self) -> None:
        self.assert_selection(
            ["kubernetes", "dashboard", "ml-model-selection"],
            core=True,
            e2e_ml_model_selection=True,
        )

    def test_common_change_unions_multiple_affected_profiles(self) -> None:
        self.assert_selection(
            [
                "kubernetes",
                "dashboard",
                "multimodal-routing",
                "ml-model-selection",
            ],
            e2e_common=True,
            e2e_multimodal_routing=True,
            e2e_ml_model_selection=True,
        )

    def test_baseline_and_dashboard_affected_are_stably_deduplicated(self) -> None:
        self.assert_selection(
            ["kubernetes", "dashboard"],
            core=True,
            e2e_dashboard=True,
        )

    def test_no_trigger_selects_nothing(self) -> None:
        self.assert_selection([])

    def test_manual_and_schedule_events_select_baseline(self) -> None:
        for event_name in ("workflow_dispatch", "schedule"):
            with self.subTest(event_name=event_name):
                self.assert_selection(
                    ["kubernetes", "dashboard"],
                    event_name=event_name,
                )

    def test_cli_emits_exact_github_outputs(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = selector.main(
                [
                    "--event-name",
                    "pull_request",
                    "--core",
                    "true",
                    "--e2e-ml-model-selection",
                    "true",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            stdout.getvalue(),
            'profiles=["kubernetes","dashboard","ml-model-selection"]\n'
            "should_run=true\n",
        )

    def test_cli_rejects_dynamic_profile_injection(self) -> None:
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            selector.main(
                [
                    "--event-name",
                    "pull_request",
                    "--profile",
                    "attacker-controlled",
                ]
            )


if __name__ == "__main__":
    unittest.main()
