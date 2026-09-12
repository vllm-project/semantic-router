import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_MAKE = (REPO_ROOT / "tools/make/agent.mk").read_text(encoding="utf-8")
PRECOMMIT_MAKE = (REPO_ROOT / "tools/make/pre-commit.mk").read_text(encoding="utf-8")
PRECOMMIT_CONFIG = yaml.safe_load(
    (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
)


def target_block(name: str) -> str:
    lines = HARNESS_MAKE.splitlines()
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

    def test_precommit_native_builds_do_not_replace_host_toolchain_outputs(self) -> None:
        for binding in ("candle-binding", "onnx-binding", "ml-binding", "nlp-binding"):
            self.assertIn(f"-v /app/{binding}/target \\", PRECOMMIT_MAKE)
        self.assertIn("$$CONTAINER_CMD run --rm", PRECOMMIT_MAKE)


if __name__ == "__main__":
    unittest.main()
