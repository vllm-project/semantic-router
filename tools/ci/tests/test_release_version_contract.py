"""Guard the version checker against release workflow contract drift."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))

import check_version_contract as release_contract  # noqa: E402


class ReleaseVersionContractTests(unittest.TestCase):
    def test_release_inventory_flows_to_build_and_publication(self) -> None:
        self.assertEqual(
            release_contract.parse_release_images(),
            (
                "dashboard",
                "operator",
                "operator-bundle",
                "vllm-sr",
                "vllm-sr-cuda",
                "vllm-sr-rocm",
            ),
        )
        errors: list[str] = []
        release_contract.validate_release_image_bridge(errors)
        self.assertEqual(errors, [])

    def test_release_inventory_rejects_duplicates_and_computed_values(self) -> None:
        for source in (
            'PRODUCTION_RELEASE_IMAGES = ("dashboard", "dashboard")',
            'PRODUCTION_RELEASE_IMAGES = tuple(["dashboard"])',
            'PRODUCTION_RELEASE_IMAGES = ("bad/name",)',
        ):
            with (
                self.subTest(source=source),
                mock.patch.object(release_contract, "read_text", return_value=source),
                self.assertRaisesRegex(
                    ValueError, "production release image inventory"
                ),
            ):
                release_contract.parse_release_images()

    def test_release_inventory_requires_build_definitions(self) -> None:
        original_read_text = release_contract.read_text

        def missing_build_definition(path: Path) -> str:
            if path == release_contract.CI_IMAGE_ARTIFACTS_PATH:
                return 'DEFINITIONS = {"dashboard": (".", "Dockerfile", [])}'
            return original_read_text(path)

        with (
            mock.patch.object(
                release_contract,
                "read_text",
                side_effect=missing_build_definition,
            ),
            self.assertRaisesRegex(ValueError, "no build definition"),
        ):
            release_contract.parse_release_images()

    def test_release_image_bridge_rejects_wrong_publication_inventory(self) -> None:
        original_read_text = release_contract.read_text

        def missing_ci_output(path: Path) -> str:
            content = original_read_text(path)
            if path == release_contract.RELEASE_WORKFLOW_PATH:
                return content.replace(
                    "images: ${{ needs.validate.outputs.images }}", "images: '[]'"
                )
            return content

        errors: list[str] = []
        with (
            mock.patch.object(
                release_contract, "read_text", side_effect=missing_ci_output
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            release_contract.validate_release_image_bridge(errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("release image input", errors[0])

    def test_helm_source_default_pins_a_stable_release(self) -> None:
        for app_version in ("v0.3.0", "v0.4.0"):
            with self.subTest(app_version=app_version):
                errors: list[str] = []
                release_contract.validate_source_helm_app_version(errors, app_version)
                self.assertEqual(errors, [])

        release_errors: list[str] = []
        release_contract.validate_source_helm_app_version(
            release_errors, "v0.4.0", "0.4.0"
        )
        self.assertEqual(release_errors, [])
        with contextlib.redirect_stdout(io.StringIO()):
            release_contract.validate_source_helm_app_version(
                release_errors, "v0.3.0", "0.4.0"
            )
        self.assertEqual(len(release_errors), 1)
        self.assertIn("expected 'v0.4.0'", release_errors[0])

        for app_version in ("latest", "v0.4", "v0.4.0-rc1"):
            with self.subTest(app_version=app_version):
                errors = []
                with contextlib.redirect_stdout(io.StringIO()):
                    release_contract.validate_source_helm_app_version(
                        errors, app_version
                    )
                self.assertEqual(len(errors), 1)
                self.assertIn("stable vMAJOR.MINOR.PATCH", errors[0])

    def _validate_with(self, pyproject: str, expected: str | None) -> list[str]:
        contract = release_contract.collect_contract()
        dev_cycle = release_contract.ReleaseContract(
            pyproject_version=pyproject,
            helm_chart_version=contract.helm_chart_version,
            helm_app_version="v0.4.0",
            release_images=contract.release_images,
        )
        with (
            mock.patch.object(
                release_contract, "collect_contract", return_value=dev_cycle
            ),
            mock.patch.object(
                release_contract,
                "UPGRADE_ROLLBACK_DOC_PATH",
                release_contract.UPGRADE_ROLLBACK_DOC_PATH,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return release_contract.validate(expected)[1]

    def test_main_carries_the_next_version_and_documents_the_pinned_release(
        self,
    ) -> None:
        # Between releases `main` is on the next minor, so dev builds sort
        # after the release; the docs keep the release the chart pins.
        with mock.patch.object(
            release_contract,
            "read_text",
            side_effect=self._docs_pinned_at("0.4.0"),
        ):
            self.assertEqual(self._validate_with("0.5.0", None), [])
            self.assertEqual(self._validate_with("0.4.0", None), [])
            behind = self._validate_with("0.3.0", None)
        self.assertEqual(len(behind), 1)
        self.assertIn("behind the released v0.4.0", behind[0])

    def test_a_release_check_still_requires_the_released_version(self) -> None:
        with mock.patch.object(
            release_contract,
            "read_text",
            side_effect=self._docs_pinned_at("0.4.0"),
        ):
            errors = self._validate_with("0.5.0", "0.4.0")
        self.assertTrue(
            any(
                "vllm-sr version has '0.5.0' but expected '0.4.0'" in e for e in errors
            ),
            errors,
        )

    @staticmethod
    def _docs_pinned_at(version: str):
        original = release_contract.read_text

        def read(path: Path) -> str:
            text = original(path)
            if path == release_contract.UPGRADE_ROLLBACK_DOC_PATH:
                return text.replace("0.4.0", version)
            return text

        return read


if __name__ == "__main__":
    unittest.main()
