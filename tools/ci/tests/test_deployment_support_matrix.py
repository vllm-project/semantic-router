import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "ci" / "check_deployment_support_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "check_deployment_support_matrix", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
matrix_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix_check)

TEST_LABELS = {
    "deploy/helm/": "Helm chart",
    "deploy/kubernetes/gateway/": "Gateway integration",
    "deploy/kubernetes/gateway-config/": "Gateway integration",
    "config/runtime/cache/": "Cache example",
}


class DeploymentSupportMatrixTests(unittest.TestCase):
    def make_repo(self, root: Path) -> Path:
        for path in (
            "deploy/helm",
            "deploy/kubernetes/gateway",
            "deploy/kubernetes/gateway-config",
            "config/runtime/cache",
        ):
            (root / path).mkdir(parents=True)
        return root

    def write_matrix(self, root: Path, rows: list[tuple[str, str]]) -> Path:
        path = root / "support-matrix.md"
        path.write_text(
            "\n".join(
                [
                    "## Maintained reference stacks",
                    "",
                    "| Option | Classification | Notes |",
                    "| --- | --- | --- |",
                    *[
                        f"| {label} | {classification} | Test row. |"
                        for label, classification in rows
                    ],
                    "",
                    "## Hardware overlays",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_complete_matrix_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            matrix = self.write_matrix(
                root,
                [
                    ("[Helm chart](helm)", "Maintained reference stack"),
                    ("Gateway integration", "Supported integration"),
                    ("Cache example", "Experimental example"),
                ],
            )

            self.assertEqual(matrix_check.validate(root, matrix, TEST_LABELS), [])

    def test_missing_option_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            matrix = self.write_matrix(
                root,
                [
                    ("Helm chart", "Maintained reference stack"),
                    ("Gateway integration", "Supported integration"),
                ],
            )

            self.assertEqual(
                matrix_check.validate(root, matrix, TEST_LABELS),
                ["unclassified options: Cache example"],
            )

    def test_duplicate_unknown_option_and_invalid_class_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            matrix = self.write_matrix(
                root,
                [
                    ("Helm chart", "Maintained"),
                    ("Helm chart", "Maintained"),
                    ("Gateway integration", "Supported integration"),
                    ("Cache example", "Experimental example"),
                    ("Removed option", "Deprecated"),
                ],
            )

            errors = matrix_check.validate(root, matrix, TEST_LABELS)
            self.assertIn("duplicate options: Helm chart", errors)
            self.assertIn("unexpected options: Removed option", errors)
            self.assertIn("invalid classifications: Helm chart (Maintained)", errors)

    def test_unregistered_assets_and_stale_mappings_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            labels = {
                "deploy/helm/": "Helm chart",
                "deploy/kubernetes/gateway/": "Gateway integration",
                "deploy/kubernetes/gateway-config/": "Gateway integration",
                "deploy/removed/": "Removed option",
            }
            matrix = self.write_matrix(
                root,
                [
                    ("Helm chart", "Maintained reference stack"),
                    ("Gateway integration", "Supported integration"),
                    ("Removed option", "Deprecated"),
                ],
            )

            errors = matrix_check.validate(root, matrix, labels)
            self.assertIn("unregistered assets: config/runtime/cache/", errors)
            self.assertIn("asset mappings without directories: deploy/removed/", errors)

    def test_public_matrix_rejects_repository_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            matrix = self.write_matrix(
                root,
                [
                    ("Helm chart", "Maintained reference stack"),
                    ("Gateway integration", "Supported integration"),
                    ("Cache example", "Experimental example"),
                ],
            )
            matrix.write_text(
                matrix.read_text(encoding="utf-8")
                + "\nDo not expose `deploy/kubernetes/gateway/` here.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                matrix_check.validate(root, matrix, TEST_LABELS),
                [
                    "public matrix exposes repository paths: "
                    "deploy/kubernetes/gateway/"
                ],
            )

    def test_repository_matrix_is_complete(self) -> None:
        self.assertEqual(matrix_check.validate(REPO_ROOT), [])


if __name__ == "__main__":
    unittest.main()
