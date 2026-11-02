"""One unittest case per training project, so a missing lock cannot pass quietly."""

from __future__ import annotations

import unittest
from pathlib import Path

from .training_lock_check import (
    REPO_ROOT,
    TRAINING_ROOT,
    TrainingLockError,
    check_project,
    discover_projects,
)


class EmptyTrainingLockDiscovery(unittest.TestCase):
    def test_training_projects_are_discovered(self) -> None:
        self.fail(
            "no training pyproject.toml files were found under src/training; "
            "refusing to pass with zero lockfile cases"
        )


def _case(directory: Path) -> type[unittest.TestCase]:
    folder = directory.resolve().relative_to(REPO_ROOT.resolve()).as_posix()

    def test_uv_lock_is_current(self: unittest.TestCase) -> None:
        try:
            check_project(directory, REPO_ROOT)
        except TrainingLockError as exc:
            self.fail(str(exc))

    test_uv_lock_is_current.__doc__ = f"{folder} has a current uv.lock."
    name = "Lock_" + folder.replace("/", "_").replace("-", "_").replace(".", "_")
    case = type(
        name,
        (unittest.TestCase,),
        {"test_uv_lock_is_current": test_uv_lock_is_current},
    )
    case.__module__ = __name__
    return case


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Replace the empty-discovery failure with one case per project."""
    del pattern
    projects = discover_projects(TRAINING_ROOT)
    if not projects:
        return standard_tests
    suite = unittest.TestSuite()
    for directory in projects:
        suite.addTests(loader.loadTestsFromTestCase(_case(directory)))
    return suite


if __name__ == "__main__":
    unittest.main()
