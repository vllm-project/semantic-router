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

    def test_dashboard_workers_use_the_installed_cli_environment_by_default(
        self,
    ) -> None:
        backend = target_block("dashboard-test-backend", DASHBOARD_MAKE)
        self.assertIn("dashboard-test-backend: vllm-sr-install-cli", backend)
        self.assertIn(
            'VLLM_SR_EVALUATION_TEST_PYTHON="$${VLLM_SR_EVALUATION_TEST_PYTHON:-$(AGENT_PYTHON)}"',
            backend,
        )

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
