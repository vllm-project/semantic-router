#!/usr/bin/env python3
"""Verify deployment inventory coverage in the public support matrix."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MATRIX_PATH = Path("website/docs/installation/support-matrix.md")
CLASSIFICATIONS = {
    "Maintained reference stack",
    "Supported integration",
    "Experimental example",
    "Deprecated",
}

# Public documentation names user-facing choices rather than repository paths.
# This mapping is the contributor-side identity bridge; classifications remain
# canonical in the public matrix. Multiple directories may back one choice.
ASSET_LABELS = {
    "deploy/helm/": "Helm chart",
    "deploy/kserve/": "KServe example",
    "deploy/local/": "Local deployment",
    "deploy/openshift/": "OpenShift example",
    "deploy/operator/": "Kubernetes Operator",
    "deploy/kubernetes/agentgateway/": "agentgateway",
    "deploy/kubernetes/ai-gateway/": "Envoy AI Gateway",
    "deploy/kubernetes/aibrix/": "AIBrix",
    "deploy/kubernetes/anthropic-backend/": "Anthropic-compatible backend fixture",
    "deploy/kubernetes/crds/": "Kubernetes Operator",
    "deploy/kubernetes/dynamo/": "NVIDIA Dynamo",
    "deploy/kubernetes/hallucination/": "Hallucination policy demo",
    "deploy/kubernetes/istio/": "Istio Gateway",
    "deploy/kubernetes/jailbreak-onerror/": "Jailbreak error-handling demo",
    "deploy/kubernetes/llm-d/": "llm-d",
    "deploy/kubernetes/llm-katan/": "LLM Katan development backends",
    "deploy/kubernetes/llmd-base/": "llm-d",
    "deploy/kubernetes/observability/": "Observability demo",
    "deploy/kubernetes/response-api/": "Responses API Kubernetes demo",
    "deploy/kubernetes/route-action/": "Route action demo",
    "deploy/kubernetes/router-replay/": "Router replay recovery demo",
    "deploy/kubernetes/routing-strategies/": "Routing strategy demos",
    "deploy/kubernetes/streaming/": "Streaming with Envoy AI Gateway",
    "config/runtime/memory/": "Valkey agentic memory",
    "config/runtime/response-api/": "Responses API state with Redis",
    "config/runtime/response-cache/": "Response cache",
    "config/runtime/tools/": "Local tools database",
    "config/runtime/vector-store/": "Valkey vector store",
}

MATRIX_ROW = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<classification>[^|]+?)\s*\|"
)
MARKDOWN_LINK = re.compile(r"^\[(?P<label>[^]]+)\]\([^)]+\)$")
PUBLIC_REPO_PATH = re.compile(r"(?:deploy|config/runtime)/[^\s`|)]*")


def discover_assets(repo_root: Path) -> set[str]:
    """Return deployment and runtime-config directories requiring coverage."""
    assets = {
        f"deploy/{path.name}/"
        for path in (repo_root / "deploy").iterdir()
        if path.is_dir() and path.name != "kubernetes"
    }
    assets.update(
        f"deploy/kubernetes/{path.name}/"
        for path in (repo_root / "deploy" / "kubernetes").iterdir()
        if path.is_dir()
    )
    assets.update(
        f"config/runtime/{path.name}/"
        for path in (repo_root / "config" / "runtime").iterdir()
        if path.is_dir()
    )
    return assets


def normalize_label(cell: str) -> str:
    """Return the visible label from a plain-text or Markdown-link cell."""
    cell = cell.strip()
    match = MARKDOWN_LINK.fullmatch(cell)
    return match.group("label") if match else cell


def parse_matrix(matrix_path: Path) -> tuple[dict[str, str], list[str]]:
    """Return classified public options and duplicate option labels."""
    entries: dict[str, str] = {}
    duplicates: list[str] = []
    in_inventory = False

    for line in matrix_path.read_text(encoding="utf-8").splitlines():
        if line == "## Maintained reference stacks":
            in_inventory = True
            continue
        if in_inventory and line == "## Hardware overlays":
            break
        if not in_inventory:
            continue

        match = MATRIX_ROW.match(line)
        if match is None:
            continue
        label = normalize_label(match.group("label"))
        classification = match.group("classification").strip()
        if label.startswith("---") or classification == "Classification":
            continue
        if label in entries:
            duplicates.append(label)
        entries[label] = classification

    return entries, duplicates


def validate(
    repo_root: Path,
    matrix_path: Path | None = None,
    asset_labels: dict[str, str] | None = None,
) -> list[str]:
    """Return human-readable inventory and documentation validation errors."""
    matrix_path = matrix_path or repo_root / MATRIX_PATH
    if not matrix_path.is_file():
        return [f"support matrix is missing: {matrix_path}"]

    asset_labels = ASSET_LABELS if asset_labels is None else asset_labels
    discovered_assets = discover_assets(repo_root)
    registered_assets = set(asset_labels)
    expected_options = set(asset_labels.values())
    entries, duplicates = parse_matrix(matrix_path)
    actual_options = set(entries)
    errors: list[str] = []

    unregistered_assets = discovered_assets - registered_assets
    if unregistered_assets:
        errors.append("unregistered assets: " + ", ".join(sorted(unregistered_assets)))

    stale_mappings = registered_assets - discovered_assets
    if stale_mappings:
        errors.append(
            "asset mappings without directories: " + ", ".join(sorted(stale_mappings))
        )

    if duplicates:
        errors.append("duplicate options: " + ", ".join(sorted(duplicates)))

    missing_options = expected_options - actual_options
    if missing_options:
        errors.append("unclassified options: " + ", ".join(sorted(missing_options)))

    unexpected_options = actual_options - expected_options
    if unexpected_options:
        errors.append("unexpected options: " + ", ".join(sorted(unexpected_options)))

    invalid = {
        f"{label} ({classification})"
        for label, classification in entries.items()
        if classification not in CLASSIFICATIONS
    }
    if invalid:
        errors.append("invalid classifications: " + ", ".join(sorted(invalid)))

    exposed_paths = sorted(
        set(PUBLIC_REPO_PATH.findall(matrix_path.read_text(encoding="utf-8")))
    )
    if exposed_paths:
        errors.append(
            "public matrix exposes repository paths: " + ", ".join(exposed_paths)
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="support matrix path (default: repository public documentation)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    matrix_path = args.matrix.resolve() if args.matrix else None
    errors = validate(repo_root, matrix_path)
    if errors:
        for error in errors:
            print(f"deployment support matrix: {error}")
        return 1

    print(
        "deployment support matrix: "
        f"all {len(discover_assets(repo_root))} assets are classified across "
        f"{len(set(ASSET_LABELS.values()))} user-facing options"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
