import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "ci" / "check_backend_target_compatibility.py"
SPEC = importlib.util.spec_from_file_location(
    "check_backend_target_compatibility", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
matrix_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix_check)


class BackendTargetCompatibilityTests(unittest.TestCase):
    def write_matrix(self, root: Path, rows: list[tuple[str, list[str]]]) -> Path:
        path = root / "matrix.md"
        path.write_text(
            "\n".join(
                [
                    matrix_check.BEGIN_MARKER,
                    "",
                    "| Target form | " + " | ".join(matrix_check.SURFACES) + " |",
                    "| --- | "
                    + " | ".join("---" for _ in matrix_check.SURFACES)
                    + " |",
                    *[
                        f"| {target} | " + " | ".join(cells) + " |"
                        for target, cells in rows
                    ],
                    "",
                    matrix_check.END_MARKER,
                ]
            ),
            encoding="utf-8",
        )
        return path

    def make_evidence(self, root: Path) -> None:
        for relative_path, markers in matrix_check.EVIDENCE.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(markers), encoding="utf-8")

    def complete_rows(self) -> list[tuple[str, list[str]]]:
        return [
            (target, ["Supported"] * len(matrix_check.SURFACES))
            for target in sorted(matrix_check.TARGET_FORMS)
        ]

    def test_complete_matrix_and_evidence_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_evidence(root)
            matrix = self.write_matrix(root, self.complete_rows())
            self.assertEqual(matrix_check.validate(root, matrix), [])

    def test_missing_target_and_evidence_marker_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_evidence(root)
            rows = self.complete_rows()[1:]
            matrix = self.write_matrix(root, rows)
            evidence_path = root / next(iter(matrix_check.EVIDENCE))
            evidence_path.write_text("removed", encoding="utf-8")

            errors = matrix_check.validate(root, matrix)
            self.assertTrue(
                any(error.startswith("missing target forms:") for error in errors)
            )
            self.assertTrue(
                any(error.startswith("evidence marker missing") for error in errors)
            )

    def test_duplicate_target_bad_status_and_column_count_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_evidence(root)
            rows = self.complete_rows()
            target = rows[0][0]
            rows.append((target, ["Future"] * (len(matrix_check.SURFACES) - 1)))
            matrix = self.write_matrix(root, rows)

            errors = matrix_check.validate(root, matrix)
            self.assertIn(f"duplicate target form: {target}", errors)
            self.assertIn(
                f"{target}: expected {len(matrix_check.SURFACES)} surface cells, "
                f"got {len(matrix_check.SURFACES) - 1}",
                errors,
            )

    def test_repository_matrix_is_complete(self) -> None:
        self.assertEqual(matrix_check.validate(REPO_ROOT), [])

    def test_rendered_helm_config_requires_valid_model_cross_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rendered = Path(directory) / "rendered.yaml"
            rendered.write_text(
                """apiVersion: v1
kind: ConfigMap
data:
  config.yaml: |
    providers:
      models:
        - name: local/custom
          backend_refs:
            - base_url: https://provider.example/v1
    routing:
      modelCards:
        - name: local/custom
      decisions:
        - name: route
          priority: 100
          modelRefs:
            - model: local/custom
""",
                encoding="utf-8",
            )
            self.assertEqual(matrix_check.validate_rendered_helm(rendered), [])

            rendered.write_text(
                rendered.read_text(encoding="utf-8").replace(
                    "          priority: 100\n", ""
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "rendered routing.decisions[0] must define positive priority",
                matrix_check.validate_rendered_helm(rendered),
            )


if __name__ == "__main__":
    unittest.main()
