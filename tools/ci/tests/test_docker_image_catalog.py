from __future__ import annotations

import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from docker_image_catalog import (  # noqa: E402
    CATALOG_PATH,
    load_image_catalog,
    platform_targets,
)
from domain_registry import image_records  # noqa: E402
from image_artifacts import DEFINITIONS  # noqa: E402
from validate_workflows import Workflow, load_workflows  # noqa: E402
from workflow_policy_validation import (  # noqa: E402
    CATALOG_WORKFLOW,
    validate_catalog_workflow,
    validate_docker_image_catalog,
)


class DockerImageCatalogTests(unittest.TestCase):
    def test_catalog_contains_the_ci_image_inventory(self) -> None:
        self.assertEqual(set(load_image_catalog()), set(image_records()))

    def test_build_definitions_come_from_the_catalog(self) -> None:
        catalog = load_image_catalog()
        self.assertEqual(
            DEFINITIONS,
            {
                image: (
                    definition.context,
                    definition.dockerfile,
                    list(definition.platforms),
                )
                for image, definition in catalog.items()
            },
        )
        self.assertIn("provider-mocker", DEFINITIONS)
        self.assertNotIn("anthropic-shim", DEFINITIONS)
        self.assertNotIn("llm-katan", DEFINITIONS)

    def test_resolver_emits_each_catalog_definition(self) -> None:
        resolver = REPO_ROOT / "tools" / "ci" / "docker_image_catalog.py"
        for image, definition in load_image_catalog().items():
            result = subprocess.run(
                [sys.executable, str(resolver), image],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(
                result.stdout,
                f"context={definition.context}\n"
                f"dockerfile={definition.dockerfile}\n"
                f"platforms={','.join(definition.platforms)}\n",
            )

    def test_platform_targets_trim_whitespace_and_reject_invalid_values(self) -> None:
        self.assertEqual(
            platform_targets("linux/amd64, linux/arm64"),
            ("linux/amd64", "linux/arm64"),
        )
        with self.assertRaisesRegex(ValueError, "invalid Linux platforms"):
            platform_targets("linux/amd64,")
        with self.assertRaisesRegex(ValueError, "invalid Linux platforms"):
            platform_targets("windows/amd64")

    def test_unknown_image_is_rejected(self) -> None:
        resolver = REPO_ROOT / "tools" / "ci" / "docker_image_catalog.py"
        result = subprocess.run(
            [sys.executable, str(resolver), "unknown-image"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Unknown image 'unknown-image'", result.stderr)

    def test_workflow_policy_accepts_shared_catalog_consumers(self) -> None:
        load_errors: list[str] = []
        workflows = load_workflows(load_errors)
        self.assertEqual(load_errors, [])
        errors: list[str] = []

        validate_docker_image_catalog(workflows, errors)

        self.assertEqual(errors, [])

    def test_workflow_policy_rejects_catalog_resolver_image_drift(self) -> None:
        load_errors: list[str] = []
        workflow = load_workflows(load_errors)[CATALOG_WORKFLOW]
        self.assertEqual(load_errors, [])
        workflow_data = deepcopy(workflow.data)
        definition = next(
            step
            for step in workflow_data["jobs"]["image"]["steps"]
            if step.get("id") == "definition"
        )
        definition["env"]["IMAGE"] = "vllm-sr"
        errors: list[str] = []

        validate_catalog_workflow(
            CATALOG_WORKFLOW,
            Workflow(path=workflow.path, data=workflow_data),
            errors,
        )

        self.assertEqual(
            errors,
            [
                f".github/workflows/{CATALOG_WORKFLOW}: catalog resolver must "
                "receive the matrix image"
            ],
        )

    def test_workflow_policy_rejects_platform_override(self) -> None:
        load_errors: list[str] = []
        workflow = load_workflows(load_errors)[CATALOG_WORKFLOW]
        self.assertEqual(load_errors, [])
        workflow_data = deepcopy(workflow.data)
        build = next(
            step
            for step in workflow_data["jobs"]["image"]["steps"]
            if str(step.get("uses", "")).startswith("docker/build-push-action@")
        )
        build["with"]["platforms"] = "linux/amd64"
        errors: list[str] = []

        validate_catalog_workflow(
            CATALOG_WORKFLOW,
            Workflow(path=workflow.path, data=workflow_data),
            errors,
        )

        self.assertEqual(
            errors,
            [
                f".github/workflows/{CATALOG_WORKFLOW}: build platforms must come "
                "from the shared Docker image catalog"
            ],
        )

    def test_catalog_documents_its_single_edit_contract(self) -> None:
        text = CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn("add or rename a mapping here only", text)


if __name__ == "__main__":
    unittest.main()
