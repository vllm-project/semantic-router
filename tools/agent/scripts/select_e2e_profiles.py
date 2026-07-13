#!/usr/bin/env python3
"""Select the allowlisted Kubernetes E2E profile matrix for GitHub Actions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence

BASELINE_PROFILES = ("kubernetes", "dashboard")
BASELINE_EVENTS = frozenset({"schedule", "workflow_dispatch"})
BASELINE_CHANGE_FLAGS = (
    "e2e_common",
    "core",
    "docker",
    "make",
    "ci",
    "agent_exec",
)

# Profile names are deliberately fixed here rather than accepted from workflow
# input. agent-validate keeps this allowlist aligned with e2e-profile-map.yaml.
AFFECTED_PROFILE_FLAGS = (
    ("e2e_istio", "istio"),
    ("e2e_agentgateway", "agentgateway"),
    ("e2e_kubernetes", "kubernetes"),
    ("e2e_aibrix", "aibrix"),
    ("e2e_dashboard", "dashboard"),
    ("e2e_llm_d", "llm-d"),
    ("e2e_routing_strategies", "routing-strategies"),
    ("e2e_production_stack", "production-stack"),
    ("e2e_dynamic_config", "dynamic-config"),
    ("e2e_multimodal_routing", "multimodal-routing"),
    ("e2e_ml_model_selection", "ml-model-selection"),
    ("e2e_multi_endpoint", "multi-endpoint"),
    ("e2e_authz_rbac", "authz-rbac"),
    ("e2e_streaming", "streaming"),
    ("e2e_anthropic_shim", "anthropic-shim"),
)

SUPPORTED_EVENTS = ("pull_request", "push", "schedule", "workflow_dispatch")
ALL_CHANGE_FLAGS = tuple(
    dict.fromkeys(
        (*BASELINE_CHANGE_FLAGS, *(flag for flag, _ in AFFECTED_PROFILE_FLAGS))
    )
)


def select_e2e_profiles(
    event_name: str,
    changes: Mapping[str, bool],
) -> list[str]:
    """Return baseline union affected profiles with stable de-duplication."""
    selected: list[str] = []
    seen: set[str] = set()

    needs_baseline = event_name in BASELINE_EVENTS or any(
        changes.get(flag, False) for flag in BASELINE_CHANGE_FLAGS
    )
    candidates: list[str] = []
    if needs_baseline:
        candidates.extend(BASELINE_PROFILES)
    candidates.extend(
        profile for flag, profile in AFFECTED_PROFILE_FLAGS if changes.get(flag, False)
    )

    for profile in candidates:
        if profile in seen:
            continue
        seen.add(profile)
        selected.append(profile)
    return selected


def github_output(profiles: Sequence[str]) -> str:
    """Render exact GitHub step outputs for a dynamic matrix."""
    matrix = json.dumps(list(profiles), separators=(",", ":"))
    should_run = "true" if profiles else "false"
    return f"profiles={matrix}\nshould_run={should_run}"


def parse_bool(value: str) -> bool:
    """Parse a GitHub boolean output without accepting arbitrary values."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected exactly 'true' or 'false'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True, choices=SUPPORTED_EVENTS)
    for flag in ALL_CHANGE_FLAGS:
        parser.add_argument(
            f"--{flag.replace('_', '-')}",
            dest=flag,
            type=parse_bool,
            default=False,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    changes = {flag: getattr(args, flag) for flag in ALL_CHANGE_FLAGS}
    profiles = select_e2e_profiles(args.event_name, changes)
    print(github_output(profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
