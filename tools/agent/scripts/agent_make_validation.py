#!/usr/bin/env python3
"""Validate failure and executable-ownership contracts in agent Make targets."""

from __future__ import annotations

import re
from pathlib import Path


_TARGET_PATTERN = re.compile(r"^[A-Za-z0-9_./%+-]+(?:\s+[^:]*)?:")


def _target_recipes(makefile_text: str, target: str) -> list[str]:
    """Return every recipe body for a target, including split declarations."""
    lines = makefile_text.splitlines()
    recipes: list[str] = []
    declaration = re.compile(rf"^{re.escape(target)}(?:\s+[^:]*)?:")
    for index, line in enumerate(lines):
        if not declaration.match(line):
            continue
        body: list[str] = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith("\t"):
                body.append(candidate[1:])
                continue
            if not candidate.strip() or candidate.lstrip().startswith("#"):
                if body:
                    body.append(candidate)
                continue
            if _TARGET_PATTERN.match(candidate):
                break
            if body:
                break
        if body:
            recipes.append("\n".join(body))
    return recipes


def _logical_recipe_commands(recipe: str) -> list[str]:
    commands: list[str] = []
    current: list[str] = []
    for line in recipe.splitlines():
        current.append(line)
        if line.rstrip().endswith("\\"):
            continue
        commands.append("\n".join(current))
        current = []
    if current:
        commands.append("\n".join(current))
    return commands


def collect_make_contract_errors(
    agent_make_text: str,
    docker_make_text: str,
) -> list[str]:
    """Collect actionable errors without executing build or serve commands."""
    errors: list[str] = []

    serve_recipe = "\n".join(_target_recipes(agent_make_text, "agent-serve-local"))
    stop_recipe = "\n".join(_target_recipes(agent_make_text, "agent-stop-local"))
    if 'VLLM_SR_CLI="$(AGENT_VENV)/bin/vllm-sr"' not in serve_recipe:
        errors.append("agent-serve-local must resolve the CLI from $(AGENT_VENV)")
    if '"$$VLLM_SR_CLI" serve' not in serve_recipe:
        errors.append("agent-serve-local must execute its repo-owned VLLM_SR_CLI")
    if re.search(r"(^|[;&|]\s*)vllm-sr\s+serve\b", serve_recipe, re.MULTILINE):
        errors.append("agent-serve-local must not execute a PATH-resolved vllm-sr")
    if '"$(AGENT_VENV)/bin/vllm-sr" stop' not in stop_recipe:
        errors.append("agent-stop-local must use the same repo-owned CLI as serve")
    if '[ ! -x "$(AGENT_VENV)/bin/vllm-sr" ]' not in stop_recipe or "exit 1" not in stop_recipe:
        errors.append("agent-stop-local must fail when its repo-owned CLI is missing")
    if re.search(r'vllm-sr" stop\s*\|\|\s*true', stop_recipe):
        errors.append("agent-stop-local must not hide CLI stop failures")

    dev_recipe = "\n".join(_target_recipes(docker_make_text, "vllm-sr-dev"))
    critical_builds = {
        "router": "$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS)",
        "dashboard": "$(CONTAINER_RUNTIME) build $(VLLM_SR_DASHBOARD_BUILD_ARGS)",
    }
    logical_commands = _logical_recipe_commands(dev_recipe)
    for label, command_fragment in critical_builds.items():
        owners = [command for command in logical_commands if command_fragment in command]
        if len(owners) != 1:
            errors.append(
                f"vllm-sr-dev must contain exactly one {label} build command"
            )
            continue
        if not owners[0].lstrip().startswith("@set -e;"):
            errors.append(
                f"vllm-sr-dev {label} build shell block must start with set -e"
            )

    return errors


def validate_agent_make_contracts(repo_root: Path, errors: list[str]) -> None:
    """Validate canonical local build/serve recipes from repository files."""
    agent_make = (repo_root / "tools/make/agent.mk").read_text(encoding="utf-8")
    docker_make = (repo_root / "tools/make/docker.mk").read_text(encoding="utf-8")
    errors.extend(collect_make_contract_errors(agent_make, docker_make))
