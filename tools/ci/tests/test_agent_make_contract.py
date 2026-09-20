import json
import os
import subprocess
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


class HarnessMakeContractTests(unittest.TestCase):
    def test_cli_unit_target_includes_upstream_embedding_and_runtime_image_contracts(
        self,
    ):
        source = (REPO_ROOT / "tools/make/docker.mk").read_text()
        unit = source.split("vllm-sr-test: vllm-sr-install-cli", 1)[1].split(
            "vllm-sr-test-integration:", 1
        )[0]
        for name in (
            "test_embedding_api_config.py",
            "test_model_binding_contract.py",
            "test_dashboard_dockerfile_surface.py",
        ):
            self.assertEqual(unit.count(f"src/vllm-sr/tests/{name}"), 1)
        self.assertIn("run_cli_tests.py --verbose", unit)

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
        ):
            self.assertIn(dependency, check)
        generate = target_block("generated-contract-generate", docs_make)
        self.assertIn("config-schema-generate", generate)
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

    def test_precommit_native_builds_do_not_replace_host_toolchain_outputs(
        self,
    ) -> None:
        for binding in ("candle-binding", "onnx-binding", "ml-binding", "nlp-binding"):
            self.assertIn(f"-v /app/{binding}/target \\", PRECOMMIT_MAKE)
        self.assertIn("$$CONTAINER_CMD run --rm", PRECOMMIT_MAKE)

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

        for target in ("dashboard-frontend-deps", "dashboard-wizmap-deps"):
            block = target_block(target, DASHBOARD_MAKE)
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
        self.assertEqual(combined.count("npm ci"), 2)
        self.assertNotIn("npm install", combined)

    def test_dashboard_backend_tests_use_the_installed_cli_environment(self) -> None:
        backend = target_block("dashboard-test-backend", DASHBOARD_MAKE)
        self.assertIn("dashboard-test-backend: vllm-sr-install-cli", backend)
        self.assertIn("go test -json -count=1 ./...", backend)
        self.assertNotIn("VLLM_SR_EVALUATION_TEST_PYTHON", backend)

    def test_native_environment_survives_subdirectory_and_vendor_overrides(
        self,
    ) -> None:
        makefile = """.PHONY: native-env-probe native-env-parent
native-env-parent:
	@$(NATIVE_ENV) $(MAKE) --no-print-directory -f tools/make/common.mk -f $(lastword $(MAKEFILE_LIST)) native-env-probe
native-env-probe:
	@cd src/semantic-router && $(NATIVE_ENV) python3 -c 'import json, os; print(json.dumps({k: os.environ[k] for k in ("LD_LIBRARY_PATH", "CGO_LDFLAGS")}))'
"""
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = "/opt/vendor runtime/lib:/opt/openvino/lib"
        environment["CGO_LDFLAGS"] = "-Wl,--as-needed -L/opt/vendor/lib"
        directories = [
            str(REPO_ROOT / binding / "target/release")
            for binding in (
                "candle-binding",
                "onnx-binding",
                "ml-binding",
                "nlp-binding",
            )
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mk") as fixture:
            fixture.write(makefile)
            fixture.flush()
            for target, depth in (("native-env-probe", 1), ("native-env-parent", 2)):
                with self.subTest(target=target):
                    result = subprocess.run(
                        [
                            "make",
                            "--no-print-directory",
                            "-f",
                            "tools/make/common.mk",
                            "-f",
                            fixture.name,
                            target,
                        ],
                        cwd=REPO_ROOT,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    actual = json.loads(result.stdout)
                    self.assertEqual(
                        actual["LD_LIBRARY_PATH"],
                        ":".join(
                            [*(directories * depth), environment["LD_LIBRARY_PATH"]]
                        ),
                    )
                    self.assertEqual(
                        actual["CGO_LDFLAGS"],
                        " ".join(
                            [
                                *(
                                    "-L" + directory
                                    for directory in directories * depth
                                ),
                                environment["CGO_LDFLAGS"],
                            ]
                        ),
                    )

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

    def test_native_search_paths_have_one_make_owner(self) -> None:
        for path in (REPO_ROOT / "tools/make").glob("*.mk"):
            if path.name == "common.mk":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if "LD_LIBRARY_PATH=" in line or "CGO_LDFLAGS=" in line:
                    self.assertNotIn("binding/target/release", line, str(path))


if __name__ == "__main__":
    unittest.main()
