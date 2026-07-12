from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agent_make_validation import collect_make_contract_errors  # noqa: E402


AGENT_MAKE = r'''
agent-serve-local:
	@VLLM_SR_CLI="$(AGENT_VENV)/bin/vllm-sr"; \
	"$$VLLM_SR_CLI" serve

agent-stop-local:
	@if [ ! -x "$(AGENT_VENV)/bin/vllm-sr" ]; then \
		exit 1; \
	fi; \
	"$(AGENT_VENV)/bin/vllm-sr" stop
'''

DOCKER_MAKE = r'''
vllm-sr-dev:
	@set -e; \
	if true; then \
		$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t router .; \
	fi
	@set -e; \
	if true; then \
		$(CONTAINER_RUNTIME) build $(VLLM_SR_DASHBOARD_BUILD_ARGS) -t dashboard .; \
	fi
'''


def test_accepts_fail_fast_builds_and_repo_owned_cli() -> None:
    assert collect_make_contract_errors(AGENT_MAKE, DOCKER_MAKE) == []


def test_rejects_path_resolved_cli() -> None:
    broken = AGENT_MAKE.replace('"$$VLLM_SR_CLI" serve', "vllm-sr serve")
    errors = collect_make_contract_errors(broken, DOCKER_MAKE)
    assert "agent-serve-local must execute its repo-owned VLLM_SR_CLI" in errors
    assert "agent-serve-local must not execute a PATH-resolved vllm-sr" in errors


def test_rejects_silent_or_ignored_stop_failures() -> None:
    silent = AGENT_MAKE.replace(
        'if [ ! -x "$(AGENT_VENV)/bin/vllm-sr" ]; then \\\n\t\texit 1; \\\n\tfi; \\\n\t"$(AGENT_VENV)/bin/vllm-sr" stop',
        'if [ -x "$(AGENT_VENV)/bin/vllm-sr" ]; then \\\n\t\t"$(AGENT_VENV)/bin/vllm-sr" stop || true; \\\n\tfi',
    )
    errors = collect_make_contract_errors(silent, DOCKER_MAKE)
    assert "agent-stop-local must fail when its repo-owned CLI is missing" in errors
    assert "agent-stop-local must not hide CLI stop failures" in errors


def test_rejects_non_fail_fast_dashboard_build_block() -> None:
    broken = DOCKER_MAKE.replace(
        "@set -e; \\\n\tif true; then \\\n\t\t$(CONTAINER_RUNTIME) build $(VLLM_SR_DASHBOARD_BUILD_ARGS)",
        "@if true; then \\\n\t\t$(CONTAINER_RUNTIME) build $(VLLM_SR_DASHBOARD_BUILD_ARGS)",
    )
    errors = collect_make_contract_errors(AGENT_MAKE, broken)
    assert "vllm-sr-dev dashboard build shell block must start with set -e" in errors
