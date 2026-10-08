import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_MAKE = (REPO_ROOT / "tools/make/agent.mk").read_text(encoding="utf-8")
PRECOMMIT_MAKE = (REPO_ROOT / "tools/make/pre-commit.mk").read_text(encoding="utf-8")
DASHBOARD_MAKE = (REPO_ROOT / "tools/make/dashboard.mk").read_text(encoding="utf-8")
PRECOMMIT_CONFIG = yaml.safe_load(
    (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
)


def target_block(name: str, source: str = HARNESS_MAKE) -> str:
    lines = source.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith(f"{name}:")
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index] and not lines[index][0].isspace() and ":" in lines[index]
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def local_hook(hook_id: str) -> dict:
    for repo in PRECOMMIT_CONFIG["repos"]:
        if repo.get("repo") != "local":
            continue
        for hook in repo["hooks"]:
            if hook["id"] == hook_id:
                return hook
    raise AssertionError(f"missing hook {hook_id}")


def write_fake_python(path: Path, version: tuple[int, int, int], calls: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'echo "$0 $*" >> "{calls}"\n'
        'case "$1" in\n'
        f'-c) shift; exec "{sys.executable}" -c \'import sys; '
        f"sys.executable = sys.argv.pop(1); sys.version_info = {version}; "
        'exec(sys.argv.pop(1))\' "$0" "$@" ;;\n'
        '-m) if [ "$2" = venv ]; then for target; do :; done; '
        'mkdir -p "$target/bin"; cp "$0" "$target/bin/python"; fi ;;\n'
        "esac\n"
    )
    path.chmod(0o755)
    return path


class HarnessMakeContractTests(unittest.TestCase):
    def test_cli_unit_target_includes_upstream_embedding_and_runtime_image_contracts(
        self,
    ):
        source = (REPO_ROOT / "tools/make/docker.mk").read_text()
        unit = source.split("vllm-sr-test: vllm-sr-install-cli", 1)[1].split(
            "vllm-sr-test-integration:", 1
        )[0]
        # The whole directory runs, so no CLI test file can be left out of CI.
        self.assertIn("-m pytest -q src/vllm-sr/tests\n", unit)
        self.assertIn("run_cli_tests.py --verbose", unit)

    def test_router_image_platform_selects_the_runtime_accelerator(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "VLLM_SR_ACCELERATOR",
                "VLLM_SR_PLATFORM",
                "MAKEFLAGS",
                "MFLAGS",
                "MAKEOVERRIDES",
                "MAKEFILES",
            }
        }
        for arguments, accelerator in (
            ((), "cpu"),
            (("VLLM_SR_PLATFORM=amd",), "rocm"),
            (("VLLM_SR_PLATFORM=nvidia",), "cuda"),
            (("VLLM_SR_PLATFORM=amd", "VLLM_SR_ACCELERATOR=cpu"), "cpu"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["make", "-n", "docker-build-vllm-sr-router", *arguments],
                    cwd=REPO_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(
                    f"--target vllm-sr --build-arg ACCELERATOR={accelerator} "
                    "-f tools/docker/Dockerfile.extproc",
                    result.stdout,
                )

    def test_daily_interface_is_small_and_direct(self) -> None:
        for target in ("impact", "check", "verify", "ci-full", "harness-check"):
            self.assertIn(f"{target}:", HARNESS_MAKE)
        for obsolete in (
            "agent-report:",
            "agent-ci-gate:",
            "agent-feature-gate:",
            "agent-pr-gate:",
            "agent-lint:",
        ):
            self.assertNotIn(obsolete, HARNESS_MAKE)

    def test_check_has_one_direct_harness_entrypoint(self) -> None:
        check = target_block("check")

        self.assertIn("harness.py check", check)
        self.assertNotIn("mktemp", check)
        self.assertNotIn("run-python-lint", HARNESS_MAKE)
        self.assertNotIn("agent-changed-files-lint", HARNESS_MAKE)

    def test_ci_checks_all_generated_public_contracts(self) -> None:
        docs_make = (REPO_ROOT / "tools/make/docs.mk").read_text()
        check = target_block("generated-contract-check", docs_make)
        for dependency in (
            "config-schema-check",
            "api-docs-check",
            "agent-skill-check",
            "docs-generated-check",
            "docs-crd-check",
            "decision-runtime-catalog-check",
        ):
            self.assertIn(dependency, check)
        generate = target_block("generated-contract-generate", docs_make)
        self.assertIn("config-schema-generate", generate)
        self.assertIn("decision-runtime-catalog-generate", generate)
        self.assertLess(
            generate.index("api-docs-generate"), generate.index("agent-skill-sync")
        )
        quality = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/pre-commit.yml").read_text()
        )
        generated = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/check-generated.yml").read_text()
        )
        commands = "\n".join(
            step.get("run", "")
            for workflow in (quality, generated)
            for job in workflow["jobs"].values()
            for step in job["steps"]
        )
        for target in (
            "docs-generated-check",
            "config-schema-check",
            "api-docs-check",
            "docs-crd-check",
            "decision-runtime-catalog-check",
        ):
            self.assertIn(target, commands)
        self.assertTrue(
            any(
                line.startswith("docs-generated-check:") and "agent-skill-check" in line
                for line in docs_make.splitlines()
            )
        )
        core = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/test-and-build.yml").read_text()
        )
        self.assertFalse(
            any(
                "generated-contract-check" in step.get("run", "")
                for step in core["jobs"]["test-and-build"]["steps"]
            ),
            "generated contracts have one quality owner; core must not rerun them",
        )

    def test_decision_runtime_projection_checks_each_canonical_source(self) -> None:
        hook = local_hook("decision-runtime-catalog-generated")
        self.assertEqual(hook["entry"], "make decision-runtime-catalog-check")
        self.assertFalse(hook["pass_filenames"])
        for source in (
            "src/model-runtime/vllm_srun/registry/tables/decision1.py",
            "src/model-runtime/vllm_srun/registry/tables/decision2.py",
            "src/model-runtime/vllm_srun/registry/tables/common.py",
            "src/model-runtime/vllm_srun/systemone.py",
            "src/model-runtime/vllm_srun/families/decision1/family.py",
            "src/model-runtime/vllm_srun/families/decision1/questions.py",
            "src/model-runtime/vllm_srun/families/decision2/family.py",
            "dashboard/frontend/scripts/generate-decision-runtime-catalog.py",
            "dashboard/frontend/src/pages/decisionRuntimeCatalog.generated.json",
        ):
            with self.subTest(source=source):
                self.assertIsNotNone(re.search(hook["files"], source))
        self.assertIsNone(
            re.search(hook["files"], "src/model-runtime/vllm_srun/server.py")
        )
        result = subprocess.run(
            ["make", "-n", "decision-runtime-catalog-check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn(
            "python3 dashboard/frontend/scripts/generate-decision-runtime-catalog.py --check",
            result.stdout,
        )
        for dependency in ("npm ci", "pip install", "uv sync", "docker build"):
            self.assertNotIn(dependency, result.stdout)
        generated = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/check-generated.yml").read_text()
        )["jobs"]["generated"]
        checks = [step.get("run", "") for step in generated["steps"]]
        self.assertTrue(
            any(
                command.startswith("make ")
                and "decision-runtime-catalog-check" in command.split()
                for command in checks
            )
        )
        self.assertTrue(
            any(
                "--check decision-runtime-catalog-check" in command
                for command in checks
            )
        )

    def test_reference_drift_is_checked_even_for_docs_only_changes(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/pre-commit.yml").read_text()
        )
        job = workflow["jobs"]["quality"]
        self.assertNotIn("docs_only", job.get("if", ""))
        commands = [step.get("run", "") for step in job["steps"]]
        self.assertIn(
            "make docs-generated-check docs-cli-test docs-community-test docs-check-translation-coverage",
            commands,
        )
        self.assertFalse(any("pip install -e" in command for command in commands))

    def test_website_builds_reject_drift_before_generating_runtime_assets(self) -> None:
        scripts = json.loads((REPO_ROOT / "website/package.json").read_text())[
            "scripts"
        ]
        for target in ("build", "build:en", "build:zh", "deploy", "test"):
            with self.subTest(target=target):
                self.assertTrue(
                    scripts[target].startswith("npm run generated:check &&")
                )

    def test_verify_requires_explicit_domain_or_profile(self) -> None:
        verify = target_block("verify")

        self.assertIn('--domains "$(DOMAIN)"', verify)
        self.assertIn('--profiles "$(PROFILE)"', verify)

    def test_structure_hooks_are_direct_and_non_recursive(self) -> None:
        structure = local_hook("structure-check")
        architecture = local_hook("architecture-check")

        self.assertEqual(
            structure["entry"],
            ".venv-agent/bin/python tools/agent/scripts/structure_check.py",
        )
        self.assertEqual(
            architecture["entry"],
            ".venv-agent/bin/python tools/agent/scripts/architecture_check.py",
        )
        self.assertNotIn("agent-changed-files-lint", str(PRECOMMIT_CONFIG))

    def test_tool_environment_requires_python_3_10(self) -> None:
        # The quote catches a recipe that splices paths into shell or Python code.
        with tempfile.TemporaryDirectory(prefix="agent's-venv-") as directory:
            root = Path(directory)
            calls = root / "calls.log"
            system_python = write_fake_python(root / "bin/python3", (3, 9, 6), calls)
            newer_python = write_fake_python(root / "python3.12", (3, 12, 4), calls)
            venv = root / ".venv-agent"
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKEFILES"}
                and not key.startswith("AGENT_")
            }
            environment.update(PATH=f"{system_python.parent}:{os.environ['PATH']}")

            def install(*variables: str) -> subprocess.CompletedProcess[str]:
                calls.write_text("")
                return subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "-f",
                        "tools/make/agent.mk",
                        "harness-venv-install",
                        f"AGENT_VENV={venv}",
                        f"AGENT_WORKTREE_VENV={venv}",
                        *variables,
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            refused = install()
            self.assertNotEqual(refused.returncode, 0, refused.stdout)
            self.assertIn(f"python3 is Python 3.9.6 at {system_python}", refused.stderr)
            self.assertIn("Set AGENT_BOOTSTRAP_PYTHON", refused.stderr)
            self.assertFalse(venv.exists())

            write_fake_python(venv / "bin/python", (3, 9, 6), calls)
            rebuilt = install(f"AGENT_BOOTSTRAP_PYTHON={newer_python}")
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            rebuild_calls = calls.read_text().splitlines()
            self.assertIn(f"{newer_python} -m venv --clear {venv}", rebuild_calls)
            self.assertIn(
                f"{venv}/bin/python -m pip install -r tools/agent/requirements.txt",
                rebuild_calls,
            )
            self.assertEqual(
                (venv / "bin/python").read_text(), newer_python.read_text()
            )

            reused = install(f"AGENT_BOOTSTRAP_PYTHON={newer_python}")
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertNotIn(" -m venv ", calls.read_text())

    def test_linked_worktrees_share_one_tool_environment(self) -> None:
        install = target_block("harness-venv-install")

        self.assertIn(
            "git rev-parse --path-format=absolute --git-common-dir", HARNESS_MAKE
        )
        self.assertIn(
            "AGENT_VENV ?= $(AGENT_PRIMARY_WORKTREE)/.venv-agent", HARNESS_MAKE
        )
        self.assertIn('ln -sfn "$(AGENT_VENV)" "$(AGENT_WORKTREE_VENV)"', install)
        self.assertIn(
            "AGENT_PRE_COMMIT ?= $(AGENT_VENV)/bin/pre-commit", PRECOMMIT_MAKE
        )

    def test_go_bootstrap_rebuilds_a_linter_built_by_an_older_go(self) -> None:
        pin = (REPO_ROOT / "tools/linter/go/golangci-lint.version").read_text().strip()
        linter_package = "github.com/golangci/golangci-lint/v2/cmd/golangci-lint"
        for built_with, expected in (
            ("go1.26.8", [f"install {linter_package}@v{pin}"]),
            ("go1.27.0", []),
            ("go1.28.0", []),
        ):
            with (
                self.subTest(built_with=built_with),
                tempfile.TemporaryDirectory() as root,
            ):
                gopath = Path(root)
                installs = gopath / "installs.log"
                (gopath / "bin").mkdir()
                linter = gopath / "bin" / "golangci-lint"
                linter.write_text(
                    f"#!/bin/sh\necho 'golangci-lint has version {pin} "
                    f"built with {built_with}'\n"
                )
                go = gopath / "go"
                go.write_text(
                    "#!/bin/sh\n"
                    'case "$1 $2" in\n'
                    f'"env GOPATH") echo "{gopath}" ;;\n'
                    '"env GOVERSION") echo go1.27.1 ;;\n'
                    f'"version "*) echo "$2: {built_with}" ;;\n'
                    f'"install "*) echo "$*" >> "{installs}" ;;\n'
                    "*) exit 1 ;;\n"
                    "esac\n"
                )
                linter.chmod(0o755)
                go.chmod(0o755)
                subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "-f",
                        "tools/make/agent.mk",
                        "harness-go-bootstrap",
                    ],
                    cwd=REPO_ROOT,
                    env={**os.environ, "PATH": f"{gopath}:{os.environ['PATH']}"},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                recorded = (
                    installs.read_text().splitlines() if installs.exists() else []
                )
                self.assertEqual(recorded, expected)

    def test_dashboard_checks_keep_lockfiles_frozen(self) -> None:
        for target in (
            "dashboard-lint",
            "dashboard-lint-fix",
            "dashboard-type-check",
            "dashboard-test-frontend",
            "dashboard-test-e2e-evaluation",
        ):
            with self.subTest(target=target):
                block = target_block(target, DASHBOARD_MAKE)
                self.assertIn("dashboard-frontend-deps", block.splitlines()[0])
                self.assertNotIn("npm ci", block)
                self.assertNotIn("npm install", block)
                for line in block.splitlines():
                    if "npm " in line:
                        self.assertNotIn("2>/dev/null", line)

        block = target_block("dashboard-frontend-deps", DASHBOARD_MAKE)
        self.assertIn("npm ci", block)
        self.assertNotIn("npm install", block)
        combined = subprocess.run(
            [
                "make",
                "-n",
                "dashboard-check",
                "dashboard-test-e2e-evaluation",
                "dashboard-build",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(combined.count("npm ci"), 1)
        self.assertNotIn("npm install", combined)

    def test_dashboard_backend_tests_use_the_installed_cli_environment(self) -> None:
        backend = target_block("dashboard-test-backend", DASHBOARD_MAKE)
        self.assertIn("dashboard-test-backend: vllm-sr-install-cli", backend)
        self.assertIn("go test -json -count=1 ./...", backend)
        self.assertNotIn("VLLM_SR_EVALUATION_TEST_PYTHON", backend)

    def test_precommit_image_includes_the_ci_helm_toolchain(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/test-and-build.yml").read_text(
                encoding="utf-8"
            )
        )
        dockerfile = (REPO_ROOT / "tools/docker/Dockerfile.precommit").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"ARG HELM_VERSION={workflow['env']['HELM_VERSION']}", dockerfile)


if __name__ == "__main__":
    unittest.main()
