"""Training lock discovery and failure text, with a fake uv and no network."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.ci import training_lock_contracts  # noqa: E402
from tools.ci.training_lock_check import (  # noqa: E402
    TrainingLockError,
    check_project,
    check_training_locks,
    discover_projects,
)

# Named so a reviewer can see the projects this check already covers. This is
# not a closed set: a new pyproject is discovered without editing the equality.
CURRENT_PROJECTS = (
    "src/training/model_eval",
    "src/training/model_experiment/mmlu_pro_solver_lora",
    "src/training/model_classifier/classifier_model_fine_tuning_lora",
    "src/training/model_classifier/escalation_risk",
    "src/training/model_classifier/fact_check_fine_tuning_lora",
    "src/training/model_classifier/modality_routing_classifier",
    "src/training/model_classifier/pii_model_fine_tuning_lora",
)


def _folders(training_root: Path, repo_root: Path) -> set[str]:
    return {
        path.resolve().relative_to(repo_root.resolve()).as_posix()
        for path in discover_projects(training_root)
    }


class TrainingLockCheckTests(unittest.TestCase):
    def test_discovery_includes_current_projects_without_a_closed_set(self):
        found = _folders(ROOT / "src" / "training", ROOT)
        present = [
            project
            for project in CURRENT_PROJECTS
            if (ROOT / project / "pyproject.toml").is_file()
        ]
        self.assertGreaterEqual(len(present), 6)
        for project in present:
            self.assertIn(project, found)
        self.assertGreaterEqual(len(found), len(present))
        self.assertFalse(any(".venv" in project for project in found))

    def test_discovery_skips_virtualenv_pyprojects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src" / "training" / "widget"
            vendored = project / ".venv" / "lib" / "site-packages" / "pandas"
            vendored.mkdir(parents=True)
            (project / "pyproject.toml").write_text("[project]\nname = 'widget'\n")
            (project / "uv.lock").write_text("version = 1\nrevision = 3\n")
            (vendored / "pyproject.toml").write_text("[project]\nname = 'pandas'\n")
            self.assertEqual(
                _folders(root / "src" / "training", root),
                {"src/training/widget"},
            )

    def test_missing_sibling_lock_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src" / "training" / "new_project"
            project.mkdir(parents=True)
            (project / "pyproject.toml").write_text("[project]\nname = 'new'\n")
            with self.assertRaises(TrainingLockError) as caught:
                check_project(project, root)
            message = str(caught.exception)
            self.assertIn("src/training/new_project", message)
            self.assertIn("uv lock", message)

    def test_failed_lock_check_names_the_folder_and_uv_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src" / "training" / "model_eval"
            project.mkdir(parents=True)
            (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
            (project / "uv.lock").write_text("version = 1\nrevision = 3\n")
            bindir = root / "bin"
            bindir.mkdir()
            marker = root / "uv-args"
            script = bindir / "uv"
            script.write_text(
                "#!/bin/sh\n"
                'printf \'%s\\n\' "$*" > "$UV_ARGS"\n'
                "echo 'lock would change' >&2\n"
                "exit 1\n"
            )
            script.chmod(0o755)
            with (
                patch.dict(os.environ, {"PATH": str(bindir), "UV_ARGS": str(marker)}),
                self.assertRaises(TrainingLockError) as caught,
            ):
                check_project(project, root)
            message = str(caught.exception)
            self.assertIn("src/training/model_eval", message)
            self.assertIn("uv lock", message)
            self.assertIn("lock would change", message)
            self.assertEqual(marker.read_text().strip(), "lock --check")

    def test_empty_discovery_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "training").mkdir(parents=True)
            with self.assertRaises(TrainingLockError) as caught:
                check_training_locks(root)
            self.assertIn("src/training", str(caught.exception))

    def test_contract_suite_has_one_case_per_project_and_fails_when_empty(self):
        projects = discover_projects(ROOT / "src" / "training")
        suite = training_lock_contracts.load_tests(
            unittest.defaultTestLoader, unittest.TestSuite(), None
        )
        self.assertEqual(suite.countTestCases(), len(projects))
        self.assertGreaterEqual(suite.countTestCases(), 6)
        with patch.object(
            training_lock_contracts, "discover_projects", return_value=[]
        ):
            empty = training_lock_contracts.load_tests(
                unittest.defaultTestLoader,
                unittest.defaultTestLoader.loadTestsFromTestCase(
                    training_lock_contracts.EmptyTrainingLockDiscovery
                ),
                None,
            )
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(empty)
        self.assertEqual(result.testsRun, 1)
        self.assertFalse(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
