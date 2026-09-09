import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_MAKE = REPO_ROOT / "tools/make/agent.mk"
PRECOMMIT_MAKE = REPO_ROOT / "tools/make/pre-commit.mk"


class HarnessMakeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "main"
        self.repo.mkdir()
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Harness Test")
        self.git("config", "user.email", "harness@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", "/dev/null")
        self.git("commit", "--quiet", "--allow-empty", "-m", "initial")
        self.linked = self.root / "linked"
        self.git("worktree", "add", "--quiet", "-b", "other", str(self.linked))

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def test_worktrees_resolve_distinct_editable_environments(self) -> None:
        printer = self.root / "print.mk"
        printer.write_text("print-env:;@echo $(AGENT_VENV)\n", encoding="utf-8")

        def environment(cwd: Path) -> str:
            return subprocess.check_output(
                [
                    "make",
                    "-s",
                    "--no-print-directory",
                    "-f",
                    str(AGENT_MAKE),
                    "-f",
                    str(printer),
                    "print-env",
                ],
                cwd=cwd,
                text=True,
            ).strip()

        self.assertEqual(environment(self.repo), str(self.repo / ".venv-agent"))
        self.assertEqual(environment(self.linked), str(self.linked / ".venv-agent"))

    def test_container_check_mounts_linked_git_metadata_and_worktree(self) -> None:
        tools = self.root / "tools"
        tools.mkdir()
        log = self.root / "docker.json"
        selection = self.root / "selected-files.txt"
        selection.write_text("a file.py\n", encoding="utf-8")
        docker = tools / "docker"
        docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] == 'run':\n"
            "    Path(os.environ['DOCKER_TEST_LOG']).write_text(json.dumps(sys.argv[2:]))\n"
            "elif sys.argv[1:3] == ['image', 'inspect']:\n"
            "    print('sha256:test')\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_TEST_LOG": str(log),
        }
        subprocess.run(
            [
                "make",
                "-s",
                "-f",
                str(AGENT_MAKE),
                "-f",
                str(PRECOMMIT_MAKE),
                "precommit-local",
                f"CHANGED_FILES_PATH={selection}",
            ],
            cwd=self.linked,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        args = json.loads(log.read_text())
        self.assertIn(f"{self.linked}:{self.linked}", args)
        git_common = self.repo / ".git"
        self.assertIn(f"{git_common}:{git_common}:ro", args)
        self.assertEqual(args[args.index("-w") + 1], str(self.linked))
        self.assertIn(f"GIT_CONFIG_VALUE_0={self.linked}", args)
        self.assertIn(f"{selection}:/tmp/harness-changed-files.txt:ro", args)
        self.assertIn("CHANGED_FILES_PATH=/tmp/harness-changed-files.txt", args)

    def test_existing_ci_lint_binary_does_not_trigger_source_install(self) -> None:
        binaries = self.root / "ci-tools"
        binaries.mkdir()
        go = binaries / "go"
        go.write_text(
            f'#!/bin/sh\nif [ "$*" = "env GOPATH" ]; then echo "{self.root}/gopath"; '
            'else echo "unexpected Go install" >&2; exit 99; fi\n',
            encoding="utf-8",
        )
        linter = binaries / "golangci-lint"
        linter.write_text(
            "#!/bin/sh\necho 'golangci-lint has version 2.5.0 built with go1.25'\n",
            encoding="utf-8",
        )
        go.chmod(0o755)
        linter.chmod(0o755)
        subprocess.run(
            ["make", "-s", "-f", str(AGENT_MAKE), "harness-go-bootstrap"],
            cwd=self.repo,
            env={**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"},
            check=True,
        )

    def test_core_baseline_refuses_existing_service_before_build_or_cleanup(
        self,
    ) -> None:
        runtime = self.root / "runtime"
        log = self.root / "runtime-calls"
        data = self.root / "manual-data"
        data.mkdir()
        sentinel = data / "keep"
        sentinel.write_text("manual stack", encoding="utf-8")
        runtime.write_text(
            f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{log}'\necho redis-semantic-cache\n",
            encoding="utf-8",
        )
        runtime.chmod(0o755)
        result = subprocess.run(
            [
                "make",
                "-s",
                "-f",
                str(AGENT_MAKE),
                "test-and-build-local-run",
                f"CONTAINER_RUNTIME={runtime}",
                f"MILVUS_DATA_DIR={data}",
                "MILVUS_CONTAINER_NAME=milvus",
                "QDRANT_CONTAINER=qdrant",
            ],
            cwd=self.linked,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to replace existing datastore", result.stderr)
        self.assertEqual(sentinel.read_text(), "manual stack")
        self.assertEqual(log.read_text().strip(), "ps -a --format {{.Names}}")


if __name__ == "__main__":
    unittest.main()
