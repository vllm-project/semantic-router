#!/usr/bin/env python3
"""
Sync vLLM Semantic Router Pipe versions across deployment locations.

This script helps maintain the three deployment-specific copies of the pipe
while preserving their configuration differences.

Usage:
    python sync_pipe_versions.py [--dry-run]
"""

import argparse
import difflib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sync pipe versions across deployments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    # Define paths
    repo_root = Path(__file__).parent.parent.parent
    canonical = repo_root / "tools/openwebui-pipe/vllm_semantic_router_pipe.py"
    docker_compose = (
        repo_root / "deploy/docker-compose/addons/vllm_semantic_router_pipe.py"
    )
    kubernetes = (
        repo_root
        / "deploy/kubernetes/observability/pipelines/vllm_semantic_router_pipe.py"
    )

    files = {
        "canonical": canonical,
        "docker-compose": docker_compose,
        "kubernetes": kubernetes,
    }

    # Check all files exist
    for name, path in files.items():
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            return 1

    # Read canonical version
    with open(canonical, "r") as f:
        canonical_content = f.read()

    print("=" * 80)
    print("vLLM Semantic Router Pipe - Version Sync Check")
    print("=" * 80)
    print(f"Canonical: {canonical}")
    print(f"Docker Compose: {docker_compose}")
    print(f"Kubernetes: {kubernetes}")
    print()

    # Known configuration differences that should be preserved
    # Format: (search_pattern, {deployment: replacement})
    config_patterns = [
        (
            'vsr_base_url: str = "',
            {
                "docker-compose": 'vsr_base_url: str = "http://envoy-proxy:8801"',
                "kubernetes": 'vsr_base_url: str = "http://localhost:8000"',
                "canonical": 'vsr_base_url: str = "http://localhost:8000"',
            },
        ),
        (
            'self.id = "',
            {
                "docker-compose": 'self.id = "auto"',
                "kubernetes": 'self.id = "vllm_semantic_router"',
                "canonical": 'self.id = "auto"',
            },
        ),
        (
            'self.name = "',
            {
                "docker-compose": 'self.name = "vsr/"',
                "kubernetes": 'self.name = "vllm-semantic-router/"',
                "canonical": 'self.name = "vllm-semantic-router/"',
            },
        ),
        (
            'request_body["model"] = "',
            {
                "docker-compose": 'request_body["model"] = "auto"',
                "kubernetes": 'request_body["model"] = "auto"',
                "canonical": 'request_body["model"] = "MoM"',
            },
        ),
    ]

    # Check differences
    differences_found = False
    for name, path in [
        ("docker-compose", docker_compose),
        ("kubernetes", kubernetes),
    ]:
        with open(path, "r") as f:
            content = f.read()

        # Compare line by line, ignoring known config differences
        canonical_lines = canonical_content.splitlines()
        content_lines = content.splitlines()

        diff = list(
            difflib.unified_diff(
                canonical_lines, content_lines, fromfile="canonical", tofile=name
            )
        )

        # Filter out known config differences
        significant_diffs = []
        for line in diff:
            is_config_diff = False
            for pattern, _ in config_patterns:
                if pattern in line:
                    is_config_diff = True
                    break
            if not is_config_diff and (line.startswith("+") or line.startswith("-")):
                significant_diffs.append(line)

        if significant_diffs:
            differences_found = True
            print(f"\n⚠️  Significant differences found in {name}:")
            print("-" * 80)
            for line in significant_diffs[:20]:  # Show first 20 differences
                print(line)
            if len(significant_diffs) > 20:
                print(f"... and {len(significant_diffs) - 20} more differences")
        else:
            print(f"✅ {name} is in sync with canonical (config differences OK)")

    if not differences_found:
        print("\n" + "=" * 80)
        print("✅ All versions are in sync!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("⚠️  Manual review required")
        print("=" * 80)
        print(
            "\nPlease review the differences above and update manually if needed."
        )
        print("Use 'git diff' to see detailed differences.")
        return 1


if __name__ == "__main__":
    exit(main())
