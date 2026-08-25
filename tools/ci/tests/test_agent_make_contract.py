import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_MAKE = (REPO_ROOT / "tools/make/agent.mk").read_text(encoding="utf-8")
PRECOMMIT_CONFIG = yaml.safe_load(
    (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
)


def target_block(name: str) -> str:
    lines = AGENT_MAKE.splitlines()
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
    hooks = next(
        repo["hooks"]
        for repo in PRECOMMIT_CONFIG["repos"]
        if repo.get("repo") == "local"
        and any(hook["id"] == hook_id for hook in repo["hooks"])
    )
    return next(hook for hook in hooks if hook["id"] == hook_id)


class AgentMakeContractTests(unittest.TestCase):
    def test_python_requirements_use_a_content_stamp(self) -> None:
        install = target_block("agent-venv-install")

        self.assertIn("AGENT_REQUIREMENTS_STAMP ?=", AGENT_MAKE)
        self.assertIn("cmp -s tools/agent/requirements.txt", install)
        self.assertIn(
            'cp tools/agent/requirements.txt "$(AGENT_REQUIREMENTS_STAMP)"', install
        )

    def test_markdownlint_is_versioned_and_installed_lazily(self) -> None:
        markdown_bootstrap = target_block("agent-markdown-bootstrap")

        self.assertIn("AGENT_MARKDOWNLINT_VERSION ?= 0.43.0", AGENT_MAKE)
        self.assertIn("$(AGENT_MARKDOWNLINT) --version", markdown_bootstrap)
        self.assertIn(
            "markdownlint-cli@$(AGENT_MARKDOWNLINT_VERSION)", markdown_bootstrap
        )

    def test_node_fallback_is_repo_local_and_versioned(self) -> None:
        node_bootstrap = target_block("agent-node-bootstrap")

        self.assertIn("AGENT_NODE_VERSION ?= 22.17.0", AGENT_MAKE)
        self.assertIn('-m nodeenv --node="$(AGENT_NODE_VERSION)"', node_bootstrap)

    def test_composite_gates_reuse_the_prepared_python_environment(self) -> None:
        ci_gate = target_block("agent-ci-gate")
        fast_gate = target_block("agent-fast-gate")

        self.assertIn("agent-ci-gate: $(AGENT_BOOTSTRAP_DEPS)", ci_gate)
        self.assertGreaterEqual(ci_gate.count("AGENT_BOOTSTRAP_DONE=1"), 2)
        self.assertIn("agent-fast-gate: $(AGENT_BOOTSTRAP_DEPS)", fast_gate)
        self.assertIn("AGENT_BOOTSTRAP_DONE=1", fast_gate)
        self.assertNotIn("$(MAKE) agent-validate", fast_gate)

    def test_language_tooling_is_selected_from_changed_file_types(self) -> None:
        lint = target_block("agent-lint")

        self.assertIn("grep -Eq '\\.go$$'", lint)
        self.assertIn("$(MAKE) agent-go-bootstrap", lint)
        self.assertIn("grep -Eq '\\.rs$$'", lint)
        self.assertIn("$(MAKE) agent-rust-bootstrap", lint)

    def test_changed_file_gate_does_not_repeat_go_and_rust_lint(self) -> None:
        lint = target_block("agent-lint")

        self.assertIn(
            'PRECOMMIT_SKIP="agent-changed-files-lint,golang-lint,cargo-check"',
            lint,
        )
        self.assertIn("run-go-lint", lint)
        self.assertIn("run-rust-lint", lint)

    def test_markdown_and_yaml_hooks_accept_changed_files_directly(self) -> None:
        self.assertEqual(
            local_hook("md-fmt")["entry"],
            ".venv-agent/bin/python tools/agent/scripts/precommit_tool.py markdown",
        )
        self.assertEqual(
            local_hook("yaml-and-yml-fmt")["entry"],
            ".venv-agent/bin/yamllint --config-file=tools/linter/yaml/.yamllint",
        )
        self.assertEqual(
            local_hook("js-ts-lint")["entry"],
            ".venv-agent/bin/python tools/agent/scripts/precommit_tool.py website",
        )

    def test_security_scan_is_not_forced_for_docs_only_changes(self) -> None:
        self.assertNotIn("always_run", local_hook("supply-chain-security-scan"))


if __name__ == "__main__":
    unittest.main()
