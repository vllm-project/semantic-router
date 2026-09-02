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


class DeploymentSupportMatrixTests(unittest.TestCase):
    def make_repo(self, root: Path) -> Path:
        for path in (
            "deploy/helm",
            "deploy/kubernetes/gateway",
            "config/runtime/cache",
        ):
            (root / path).mkdir(parents=True)
        return root

    def write_matrix(self, root: Path, rows: list[tuple[str, str]]) -> Path:
        path = root / "support-matrix.md"
        path.write_text(
            "\n".join(
                [
                    "| Asset path | Classification | Notes |",
                    "| --- | --- | --- |",
                    *[
                        f"| `{asset}` | {classification} | Test row. |"
                        for asset, classification in rows
                    ],
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
                    ("deploy/helm/", "Maintained reference stack"),
                    ("deploy/kubernetes/gateway/", "Supported integration"),
                    ("config/runtime/cache/", "Experimental example"),
                ],
            )

            self.assertEqual(matrix_check.validate(root, matrix), [])

    def test_missing_asset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            matrix = self.write_matrix(
                root,
                [
                    ("deploy/helm/", "Maintained reference stack"),
                    ("deploy/kubernetes/gateway/", "Supported integration"),
                ],
            )

            self.assertEqual(
                matrix_check.validate(root, matrix),
                ["unclassified assets: config/runtime/cache/"],
            )

    def test_duplicate_unknown_path_and_invalid_class_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(Path(directory))
            matrix = self.write_matrix(
                root,
                [
                    ("deploy/helm/", "Maintained"),
                    ("deploy/helm/", "Maintained"),
                    ("deploy/kubernetes/gateway/", "Supported integration"),
                    ("config/runtime/cache/", "Experimental example"),
                    ("deploy/removed/", "Deprecated"),
                ],
            )

            errors = matrix_check.validate(root, matrix)
            self.assertIn("duplicate classifications: deploy/helm/", errors)
            self.assertIn(
                "matrix entries without asset directories: deploy/removed/", errors
            )
            self.assertIn("invalid classifications: deploy/helm/ (Maintained)", errors)

    def test_repository_matrix_is_complete(self) -> None:
        self.assertEqual(matrix_check.validate(REPO_ROOT), [])


if __name__ == "__main__":
    unittest.main()
