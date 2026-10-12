"""Guard the version checker against release workflow contract drift."""

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

    @staticmethod
    def _contract(
        app_version: str,
        pyproject: str = "0.5.0",
        images: tuple[release_contract.ChartImage, ...] | None = None,
    ) -> release_contract.ReleaseContract:
        contract = release_contract.collect_contract()
        return release_contract.ReleaseContract(
            pyproject_version=pyproject,
            helm_chart_version=contract.helm_chart_version,
            helm_app_version=app_version,
            release_images=contract.release_images,
            chart_images=contract.chart_images if images is None else images,
        )

    @staticmethod
    def _chart_errors(
        contract: release_contract.ReleaseContract, release_version: str | None = None
    ) -> list[str]:
        errors: list[str] = []
        with contextlib.redirect_stdout(io.StringIO()):
            release_contract.validate_chart_images(errors, contract, release_version)
        return errors

    def test_the_chart_deploys_the_router_and_dashboard_on_its_app_version(
        self,
    ) -> None:
        prefix = release_contract.GHCR_IMAGE_PREFIX
        self.assertEqual(
            release_contract.parse_chart_images(),
            (
                release_contract.ChartImage("image", f"{prefix}/vllm-sr", ""),
                release_contract.ChartImage(
                    "dashboard.image", f"{prefix}/dashboard", ""
                ),
            ),
        )
        errors: list[str] = []
        release_contract.validate_chart_templates(errors)
        self.assertEqual(errors, [])

    def test_a_development_cycle_deploys_the_development_image(self) -> None:
        self.assertEqual(self._chart_errors(self._contract("latest")), [])
        for app_version in ("v0.4.0", "v0.5.0", "nightly-20261008"):
            with self.subTest(app_version=app_version):
                errors = self._chart_errors(self._contract(app_version))
                self.assertEqual(len(errors), 1)
                self.assertIn(f"has '{app_version}' but expected 'latest'", errors[0])

    def test_the_release_commit_pins_its_own_tag(self) -> None:
        self.assertEqual(self._chart_errors(self._contract("v0.5.0"), "0.5.0"), [])
        for app_version in ("latest", "v0.4.0"):
            with self.subTest(app_version=app_version):
                errors = self._chart_errors(self._contract(app_version), "0.5.0")
                self.assertEqual(len(errors), 1)
                self.assertIn("expected 'v0.5.0'", errors[0])

    def test_every_chart_image_follows_the_app_version_in_both_modes(self) -> None:
        router, dashboard = release_contract.parse_chart_images()
        pinned = (
            router,
            release_contract.ChartImage(
                dashboard.values_key, dashboard.repository, "v0.4.0"
            ),
        )
        for app_version, release_version in (("latest", None), ("v0.5.0", "0.5.0")):
            with self.subTest(release_version=release_version):
                errors = self._chart_errors(
                    self._contract(app_version, images=pinned), release_version
                )
                self.assertEqual(len(errors), 1)
                self.assertIn("dashboard.image.tag pins 'v0.4.0'", errors[0])

    def test_a_template_image_must_default_to_the_app_version(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            templates = Path(temporary)
            (templates / "sidecar.yaml").write_text(
                "containers:\n"
                '  - image: "{{ .Values.sidecar.image.repository }}:v0.4.0"\n',
                encoding="utf-8",
            )
            errors: list[str] = []
            with (
                mock.patch.object(release_contract, "HELM_TEMPLATES_DIR", templates),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                release_contract.validate_chart_templates(errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("sidecar.yaml: line 2 renders an image", errors[0])

    def test_main_is_in_a_development_cycle(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            contract, errors = release_contract.validate(None)
        self.assertEqual(errors, [])
        self.assertEqual(
            contract.helm_app_version, release_contract.DEVELOPMENT_IMAGE_TAG
        )

    def _validate_with(
        self, pyproject: str, expected: str | None, app_version: str = "latest"
    ) -> list[str]:
        with (
            mock.patch.object(
                release_contract,
                "collect_contract",
                return_value=self._contract(app_version, pyproject),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return release_contract.validate(expected)[1]

    def test_main_carries_the_next_version_and_documents_the_last_release(
        self,
    ) -> None:
        # Between releases `main` is on the next minor, so dev builds sort
        # after the release; the docs keep the release users install.
        with mock.patch.object(
            release_contract,
            "read_text",
            side_effect=self._docs_pinned_at("0.4.0"),
        ):
            self.assertEqual(self._validate_with("0.5.0", None), [])
            self.assertEqual(self._validate_with("0.4.0", None), [])
            behind = self._validate_with("0.3.0", None)
        self.assertEqual(len(behind), 1)
        self.assertIn("behind the released v0.4.0 that the docs pin", behind[0])

    def test_the_documented_release_comes_from_the_runbook(self) -> None:
        self.assertEqual(release_contract.runbook_release(), "0.4.0")
        original = release_contract.read_text

        def without_runbook_pin(path: Path) -> str:
            text = original(path)
            if path == release_contract.UPGRADE_ROLLBACK_DOC_PATH:
                return text.replace("helm show chart", "helm inspect chart")
            return text

        with mock.patch.object(
            release_contract, "read_text", side_effect=without_runbook_pin
        ):
            errors = self._validate_with("0.5.0", None)
        self.assertEqual(len(errors), 1)
        self.assertIn("upgrade runbook pins no release", errors[0])

    def test_a_release_check_accepts_its_release_commit(self) -> None:
        # main's latest catalog has moved on from v0.4; the snapshot drift
        # check is covered by the catalog contract tests.
        with (
            mock.patch.object(
                release_contract,
                "read_text",
                side_effect=self._docs_pinned_at("0.4.0"),
            ),
            mock.patch.object(
                release_contract, "release_snapshot_errors", return_value=[]
            ),
        ):
            self.assertEqual(self._validate_with("0.4.0", "0.4.0", "v0.4.0"), [])

    def test_a_release_check_still_requires_the_released_version(self) -> None:
        with mock.patch.object(
            release_contract,
            "read_text",
            side_effect=self._docs_pinned_at("0.4.0"),
        ):
            errors = self._validate_with("0.5.0", "0.4.0", "v0.4.0")
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
