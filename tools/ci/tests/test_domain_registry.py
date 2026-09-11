from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from classify_pr_changes import select_profiles  # noqa: E402
from domain_registry import (  # noqa: E402
    commands_for_domains,
    domain_records,
    job_records,
    matching_domains,
    path_matches,
    profile_paths,
    registry_schema_errors,
)


class DomainRegistryTests(unittest.TestCase):
    def test_registry_schema_is_complete(self) -> None:
        self.assertEqual(registry_schema_errors(), [])

    def test_registry_is_the_only_changed_path_source(self) -> None:
        self.assertTrue((REPO_ROOT / "tools/agent/domains.yaml").is_file())
        self.assertFalse((REPO_ROOT / "tools/agent/test-domain-registry.yaml").exists())
        self.assertFalse((REPO_ROOT / "tools/agent/task-matrix.yaml").exists())

    def test_path_matching_distinguishes_one_level_from_recursive(self) -> None:
        self.assertTrue(path_matches("e2e/root.go", "e2e/*.go"))
        self.assertFalse(path_matches("e2e/testcases/root.go", "e2e/*.go"))
        self.assertTrue(path_matches("e2e/testcases/root.go", "e2e/**/*.go"))
        self.assertTrue(path_matches("src/file.py", "src/**/*.py"))

    def test_pr_profile_paths_are_used_directly_by_classifier(self) -> None:
        paths = profile_paths("pr")
        for name, patterns in paths.items():
            with self.subTest(profile=name):
                concrete = patterns[0].replace("**", "case").replace("*", "case")
                self.assertIn(
                    name,
                    select_profiles((concrete,), full=False, suppress_expensive=False),
                )

    def test_domain_matching_can_report_overlapping_owners(self) -> None:
        self.assertEqual(
            matching_domains(("config/recipes/privacy/probes.yaml",)),
            ("router-core", "maintained-recipes"),
        )

    def test_domain_commands_are_deduplicated_in_registry_order(self) -> None:
        commands = commands_for_domains(
            ("router-core", "dashboard", "maintained-recipes"), "checks"
        )
        self.assertEqual(
            commands,
            (
                "make test-semantic-router",
                "make config-schema-check",
                "make dashboard-check",
                "make recipe-conformance-static",
            ),
        )

    def test_every_domain_job_is_declared_once(self) -> None:
        jobs = job_records()
        for name, domain in domain_records().items():
            with self.subTest(domain=name):
                self.assertTrue(set(domain["ci_jobs"]).issubset(jobs))


if __name__ == "__main__":
    unittest.main()
