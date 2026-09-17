"""Generated Kubernetes artifacts must fail checks without being repaired."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
GO_MAKEFILE = REPO_ROOT / "tools/make/golang.mk"
DOCS_MAKEFILE = REPO_ROOT / "tools/make/docs.mk"
WORKFLOW = REPO_ROOT / ".github/workflows/operator-ci.yml"
API_PACKAGE = Path("src/semantic-router/pkg/apis/vllm.ai/v1alpha1")
CRD_DIRECTORIES = (
    Path("deploy/kubernetes/crds"),
    Path("deploy/helm/semantic-router/crds"),
)


class KubernetesGeneratedContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = os.environ.copy()
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.environment.pop(name, None)

    def write(self, path: Path, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def prepare_generator(self) -> None:
        # Stand in for controller-gen to exercise the Make check's filesystem
        # boundary, including a newly generated CRD with no committed mirror.
        self.write(
            Path("bin/controller-gen"),
            f"#!{sys.executable}\n"
            "import sys\n"
            "from pathlib import Path\n"
            "arguments = dict(arg.split('=', 1) for arg in sys.argv[1:] if '=' in arg)\n"
            "crds = Path(arguments['output:crd:dir'])\n"
            "code = Path(arguments['output:object:dir'])\n"
            "crds.mkdir(parents=True)\n"
            "code.mkdir(parents=True)\n"
            "(code / 'zz_generated.deepcopy.go').write_text('fresh code\\n')\n"
            "(crds / 'vllm.ai_examples.yaml').write_text('fresh CRD\\n')\n"
            "if Path('new-type').exists():\n"
            "    (crds / 'vllm.ai_new.yaml').write_text('new CRD\\n')\n",
        )
        (self.root / "bin/controller-gen").chmod(0o755)
        self.environment["GOPATH"] = str(self.root)
        self.write(API_PACKAGE / "zz_generated.deepcopy.go", "fresh code\n")
        for directory in CRD_DIRECTORIES:
            self.write(directory / "vllm.ai_examples.yaml", "fresh CRD\n")

    def check_root_artifacts(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "make",
                "-f",
                str(GO_MAKEFILE),
                "-o",
                "install-controller-gen",
                "generate-api-check",
            ],
            cwd=self.root,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_root_gate_accepts_current_outputs_without_rewriting(self) -> None:
        self.prepare_generator()
        before = self.snapshot()
        result = self.check_root_artifacts()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_root_gate_rejects_drift_missing_and_new_outputs_without_rewriting(
        self,
    ) -> None:
        self.prepare_generator()
        cases = (
            (API_PACKAGE / "zz_generated.deepcopy.go", "stale code\n"),
            (CRD_DIRECTORIES[0] / "vllm.ai_examples.yaml", "stale CRD\n"),
            (CRD_DIRECTORIES[1] / "vllm.ai_examples.yaml", None),
            (Path("src/semantic-router/new-type"), "new source type\n"),
        )
        for path, content in cases:
            with self.subTest(path=path):
                target = self.root / path
                original = target.read_text() if target.exists() else None
                if content is None:
                    target.unlink()
                else:
                    self.write(path, content)
                before = self.snapshot()
                result = self.check_root_artifacts()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("make generate-api", result.stderr)
                self.assertEqual(self.snapshot(), before)
                if original is None:
                    target.unlink()
                else:
                    target.write_text(original)

    def test_operator_gate_rejects_new_untracked_generated_artifacts(self) -> None:
        workflow = yaml.safe_load(WORKFLOW.read_text())
        steps = workflow["jobs"]["manifests"]["steps"]
        generation = next(
            step["run"]
            for step in steps
            if step.get("name") == "Generate manifests and API code"
        )
        self.assertEqual(generation, "make manifests generate")
        verification = next(
            step["run"] for step in steps if step.get("name") == "Verify no changes"
        )
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=self.root,
            env=self.environment,
            check=True,
        )
        self.write(Path("new.generated.yaml"), "new: generated\n")
        result = subprocess.run(
            ["bash", "-e", "-c", verification],
            cwd=self.root,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new.generated.yaml", result.stdout)
        self.assertIn("out of date", result.stdout)

    def test_crd_reference_gate_propagates_failure_after_identical_output(self) -> None:
        self.write(Path("website/docs/api/crd-reference.md"), "current reference\n")
        generator = Path("tools/codegen/crd/generate-reference.sh")
        self.write(
            generator,
            f"#!{sys.executable}\n"
            "import shutil\n"
            "import sys\n"
            "shutil.copyfile('website/docs/api/crd-reference.md', sys.argv[1])\n"
            "sys.exit(23)\n",
        )
        (self.root / generator).chmod(0o755)
        before = self.snapshot()
        result = subprocess.run(
            [
                "make",
                "-f",
                str(DOCS_MAKEFILE),
                "-o",
                "install-crd-ref-docs",
                "docs-crd-check",
            ],
            cwd=self.root,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
