from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))

import check_version_contract as release_contract  # noqa: E402


class ReleaseRecipeContractTests(unittest.TestCase):
    def _validate_snapshot(
        self, present: bool, *, version: str = "9.8.7"
    ) -> tuple[str, list[str]]:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            recipe_root = Path(temporary) / "config" / "recipes" / "built-in"
            if present:
                snapshot = release_contract.recipe_snapshot_for_version(version)
                (recipe_root / snapshot).mkdir(parents=True)
            errors: list[str] = []
            with (
                mock.patch.object(
                    release_contract, "BUILT_IN_RECIPE_ROOT", recipe_root
                ),
                mock.patch.object(
                    release_contract, "release_snapshot_errors", return_value=[]
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                snapshot = release_contract.validate_release_recipes(errors, version)
            return snapshot, errors

    def test_release_semver_maps_to_minor_recipe_snapshot(self) -> None:
        self.assertEqual(release_contract.recipe_snapshot_for_version("9.8.7"), "v9.8")
        self.assertEqual(
            release_contract.recipe_snapshot_for_version("12.34.5-rc.1"),
            "v12.34",
        )

    def test_release_requires_aligned_recipe_snapshot(self) -> None:
        snapshot, errors = self._validate_snapshot(True)
        self.assertEqual(snapshot, "v9.8")
        self.assertEqual(errors, [])

    def test_release_rejects_missing_minor_recipe_snapshot(self) -> None:
        snapshot, errors = self._validate_snapshot(False)
        self.assertEqual(snapshot, "v9.8")
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "requires built-in Recipe snapshot config/recipes/built-in/v9.8",
            errors[0],
        )

    def test_release_surfaces_recipe_snapshot_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            recipe_root = Path(temporary) / "config" / "recipes" / "built-in"
            (recipe_root / "v9.8").mkdir(parents=True)
            errors: list[str] = []
            with (
                mock.patch.object(
                    release_contract, "BUILT_IN_RECIPE_ROOT", recipe_root
                ),
                mock.patch.object(
                    release_contract,
                    "release_snapshot_errors",
                    return_value=["Recipe snapshot drifted"],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                release_contract.validate_release_recipes(errors, "9.8.7")
            self.assertEqual(len(errors), 1)
            self.assertIn("Recipe snapshot drifted", errors[0])

    def test_github_outputs_expose_validated_recipe_snapshot(self) -> None:
        contract = release_contract.ReleaseContract(
            pyproject_version="9.8.7",
            sim_version="0.1.0",
            candle_version="9.8.7",
            candle_lock_version="9.8.7",
            helm_chart_version="9.8.7",
            helm_app_version="latest",
            release_images=("vllm-sr",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "github-output"
            release_contract.write_github_outputs(output, contract, "9.8.7")
            self.assertIn("recipe_snapshot=v9.8", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
