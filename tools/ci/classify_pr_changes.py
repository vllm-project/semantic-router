#!/usr/bin/env python3
"""Classify changed files into the smallest PR validation and image matrices."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from test_domain_registry import (
    domain_paths,
    domain_records,
    pr_job_order,
    profile_paths,
    profile_records,
)

FULL_E2E_PROFILES = tuple(
    name for name, data in profile_records().items() if data.get("full_ci")
)
BASELINE_E2E_PROFILE = "envoy-ai-gateway"
PRODUCTION_RELEASE_IMAGES = (
    "dashboard",
    "extproc",
    "extproc-rocm",
    "operator",
    "operator-bundle",
    "vllm-sr",
    "vllm-sr-cuda",
    "vllm-sr-rocm",
)
NIGHTLY_IMAGES = (
    "anthropic-shim",
    *PRODUCTION_RELEASE_IMAGES,
    "llm-katan",
    "vllm-sr-sim",
)

E2E_RULES = profile_paths("pr")
NON_PR_E2E_RULES = profile_paths("manual")
HARNESS_EXEC_PATTERNS = (
    "tools/agent/*.yaml",
    "tools/agent/requirements.txt",
    "tools/agent/scripts/**",
    "tools/ci/**",
    "tools/make/agent.mk",
    "tools/make/linter.mk",
    "tools/make/pre-commit.mk",
    "tools/docker/Dockerfile.precommit",
    ".pre-commit-config.yaml",
    ".github/workflows/pre-commit.yml",
)


@dataclass(frozen=True)
class Classification:
    signals: dict[str, bool]
    profiles: tuple[str, ...]
    pr_images: tuple[str, ...]
    publish_images: tuple[str, ...]
    selected_jobs: tuple[str, ...]

    def github_outputs(self) -> dict[str, str]:
        outputs = {name: str(value).lower() for name, value in self.signals.items()}
        outputs.update(
            {
                "e2e": str(bool(self.profiles)).lower(),
                "e2e_profiles": json.dumps(self.profiles, separators=(",", ":")),
                "full_e2e_profiles": json.dumps(
                    FULL_E2E_PROFILES, separators=(",", ":")
                ),
                "images": str(bool(self.pr_images)).lower(),
                "pr_images": json.dumps(self.pr_images, separators=(",", ":")),
                "publish_images": json.dumps(
                    self.publish_images, separators=(",", ":")
                ),
            }
        )
        return outputs


def path_matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-3])
    return fnmatch.fnmatchcase(path, pattern)


def any_matches(paths: tuple[str, ...], *patterns: str) -> bool:
    return any(path_matches(path, pattern) for path in paths for pattern in patterns)


def is_documentation_path(path: str) -> bool:
    return path.endswith((".md", ".mdx"))


def is_repository_ownership_path(path: str) -> bool:
    return path == ".github/CODEOWNERS" or Path(path).name == "OWNER"


def product_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if not is_documentation_path(path) and not is_repository_ownership_path(path)
    )


def is_harness_only_change(changed: tuple[str, ...]) -> bool:
    has_executable_change = any_matches(changed, *HARNESS_EXEC_PATTERNS)
    return has_executable_change and all(
        any_matches((path,), *HARNESS_EXEC_PATTERNS)
        or path.startswith("website/")
        or is_documentation_path(path)
        or is_agent_text(path)
        or is_repository_ownership_path(path)
        for path in changed
    )


def is_agent_text(path: str) -> bool:
    return (
        path == "AGENTS.md"
        or path.endswith("/AGENTS.md")
        or path.startswith(("tools/agent/docs/", "tools/agent/skills/"))
        or path
        in {
            ".github/copilot-instructions.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".prowlabels.yaml",
        }
        or path.startswith(".github/ISSUE_TEMPLATE/")
    )


def is_vllm_sr_product_source(path: str) -> bool:
    name = Path(path).name
    return (
        path.startswith("src/vllm-sr/")
        and not path.endswith((".md", ".mdx"))
        and name != "OWNER"
        and not name.startswith("Dockerfile")
        and name != ".dockerignore"
    )


def is_memory_change(changed: tuple[str, ...]) -> bool:
    return any_matches(changed, *domain_paths("memory"))


def is_router_learning_change(changed: tuple[str, ...]) -> bool:
    return any_matches(changed, *domain_paths("router-learning"))


def is_recipe_conformance_change(changed: tuple[str, ...]) -> bool:
    return any_matches(changed, *domain_paths("recipe-conformance"))


def detect_domains(changed: tuple[str, ...], *, full: bool) -> dict[str, bool]:
    product_changed = product_paths(changed)
    domains = _detect_direct_domains(changed, product_changed, full=full)
    harness_only = is_harness_only_change(changed) and not full
    domains["router_core"] = any_matches(
        product_changed,
        "src/semantic-router/**",
        "config/**",
        "go.mod",
        "go.sum",
        "Cargo.toml",
        "Cargo.lock",
    )
    domains["cli_core"] = domains["cli_source"] or any_matches(
        changed, "pyproject.toml"
    )
    domains["core"] = any(
        domains[key]
        for key in ("router_core", "cli_core", "fleet_sim", "native_binding")
    )
    domains["common_e2e"] = any_matches(
        product_changed,
        "e2e/pkg/**",
        "e2e/cmd/**",
        "e2e/testcases/**",
        "e2e/*.go",
        "e2e/go.mod",
        "e2e/go.sum",
    )
    domains["ci_infrastructure"] = (
        (domains["ci"] or domains["agent_exec"])
        and not domains["docs_only"]
        and not harness_only
    )
    domains["core_test"] = full or (
        not harness_only
        and (
            domains["core"]
            or domains["helm"]
            or domains["common_e2e"]
            or domains["classifier_contract"]
            or domains["api_docs_generated"]
            or (domains["make"] and not domains["docs_only"])
        )
    )
    domains["security"] = full or not domains["docs_only"]
    return domains


def _detect_direct_domains(
    changed: tuple[str, ...],
    product_changed: tuple[str, ...],
    *,
    full: bool,
) -> dict[str, bool]:
    domains = {
        "website": any(path.startswith("website/") for path in changed),
        "docs": any(is_documentation_path(path) for path in changed),
        "agent_text": any(is_agent_text(path) for path in changed),
        "docs_only": bool(changed)
        and all(
            path.startswith("website/")
            or path.endswith((".md", ".mdx"))
            or is_agent_text(path)
            or is_repository_ownership_path(path)
            for path in changed
        ),
        "ci": any_matches(
            changed,
            ".github/actions/**",
            ".github/workflows/**",
            ".mergify.yml",
        ),
        "dashboard": any_matches(product_changed, *domain_paths("dashboard")),
        "operator": any_matches(product_changed, *domain_paths("operator")),
        "openvino": any_matches(product_changed, *domain_paths("openvino")),
        "performance": any_matches(product_changed, *domain_paths("performance")),
        "paper": any_matches(product_changed, *domain_paths("paper")),
        "helm": any_matches(product_changed, *domain_paths("helm")),
        "make": any_matches(changed, "Makefile", "tools/make/**"),
        "fleet_sim": any_matches(product_changed, "src/fleet-sim/**"),
        "full": full,
    }
    domains["agent_exec"] = any_matches(
        changed,
        *HARNESS_EXEC_PATTERNS,
        ".github/actions/**",
        ".github/workflows/**",
        ".mergify.yml",
    )
    domains["classifier_contract"] = any_matches(
        changed,
        ".github/workflows/ci-changes.yml",
        ".github/workflows/pr.yml",
        ".github/workflows/main.yml",
        ".github/actions/**",
        "tools/ci/**",
    )
    # The apiserver API reference and OpenAPI JSON are generated from the route
    # catalog by `make api-docs-generate`. Editing either artifact by hand is
    # classified as docs-only, which would otherwise skip the `make
    # api-docs-check` drift gate that keeps them faithful to the catalog.
    domains["api_docs_generated"] = any_matches(
        changed,
        "website/docs/api/apiserver.md",
        "website/static/openapi/apiserver/**",
        "tools/openapi-gen/**",
    )
    domains["cli_source"] = any(
        is_vllm_sr_product_source(path) for path in product_changed
    )
    domains["cli"] = domains["cli_source"] or any_matches(
        product_changed, "tools/make/docker.mk", "e2e/testing/vllm-sr-cli/**"
    )
    domains["memory"] = is_memory_change(product_changed)
    domains["native_binding"] = any_matches(
        product_changed,
        "candle-binding/**",
        "ml-binding/**",
        "nlp-binding/**",
        "onnx-binding/**",
    )
    domains["router_learning"] = is_router_learning_change(product_changed)
    domains["recipe_conformance"] = full or is_recipe_conformance_change(
        product_changed
    )
    return domains


def select_profiles(
    changed: tuple[str, ...], domains: dict[str, bool], *, full: bool
) -> tuple[str, ...]:
    product_changed = product_paths(changed)
    profiles = [
        profile
        for profile, patterns in E2E_RULES.items()
        if any_matches(product_changed, *patterns)
    ]
    needs_baseline = any(
        domains[key]
        for key in (
            "router_core",
            "native_binding",
            "operator",
            "ci_infrastructure",
            "common_e2e",
        )
    )
    if needs_baseline and BASELINE_E2E_PROFILE not in profiles:
        profiles.insert(0, BASELINE_E2E_PROFILE)
    return FULL_E2E_PROFILES if full else tuple(profiles)


def append_unique(target: list[str], *images: str) -> None:
    target.extend(image for image in images if image not in target)


def select_base_images(
    changed: tuple[str, ...], domains: dict[str, bool]
) -> tuple[list[str], list[str]]:
    pr_images: list[str] = []
    publish_images: list[str] = []
    if domains["router_core"] or domains["native_binding"]:
        append_unique(pr_images, "extproc", "vllm-sr")
        append_unique(publish_images, "extproc", "vllm-sr")
    if domains["cli_core"] or "src/vllm-sr/Dockerfile" in changed:
        append_unique(pr_images, "vllm-sr")
        append_unique(publish_images, "vllm-sr")
    if "tools/docker/Dockerfile.extproc" in changed:
        append_unique(pr_images, "extproc")
        append_unique(publish_images, "extproc")
    if domains["dashboard"]:
        append_unique(pr_images, "dashboard")
        append_unique(publish_images, "dashboard")
    if domains["fleet_sim"]:
        append_unique(pr_images, "vllm-sr-sim")
        append_unique(publish_images, "vllm-sr-sim")
    if domains["operator"]:
        append_unique(pr_images, "operator", "operator-bundle")
        append_unique(publish_images, "operator", "operator-bundle")
    return pr_images, publish_images


def select_variant_images(
    changed: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    pr_images: list[str] = []
    publish_images: list[str] = []
    rules = (
        (("src/vllm-sr/Dockerfile.cuda", "tools/docker/*cuda*"), "vllm-sr-cuda"),
        (("src/vllm-sr/Dockerfile.rocm",), "vllm-sr-rocm"),
        (("tools/docker/Dockerfile.extproc-rocm",), "extproc-rocm"),
    )
    for patterns, image in rules:
        if any_matches(changed, *patterns):
            append_unique(pr_images, image)
            append_unique(publish_images, image)
    return pr_images, publish_images


def select_image_candidates(
    changed: tuple[str, ...], domains: dict[str, bool]
) -> tuple[list[str], list[str]]:
    product_changed = product_paths(changed)
    pr_images, publish_images = select_base_images(product_changed, domains)
    variant_pr, variant_publish = select_variant_images(product_changed)
    append_unique(pr_images, *variant_pr)
    append_unique(publish_images, *variant_publish)
    fixture_rules = (
        ("e2e/testing/llm-katan/**", "llm-katan"),
        ("e2e/testing/anthropic-shim/**", "anthropic-shim"),
        (".github/workflows/docker-validate.yml", "vllm-sr"),
    )
    for pattern, image in fixture_rules:
        if any_matches(product_changed, pattern):
            append_unique(pr_images, image)
    return pr_images, publish_images


def built_by_active_suites(
    domains: dict[str, bool], profiles: tuple[str, ...]
) -> set[str]:
    receipts: set[str] = set()
    for data in domain_records().values():
        selector = str(data.get("selector") or "")
        enabled = bool(profiles) if selector == "e2e" else domains.get(selector, False)
        if enabled:
            receipts.update(str(image) for image in data.get("image_receipts", []))
    selected_profiles = profile_records()
    for profile in profiles:
        receipts.update(
            str(image)
            for image in selected_profiles.get(profile, {}).get("image_receipts", [])
        )
    return receipts


def build_output_signals(
    changed: tuple[str, ...],
    domains: dict[str, bool],
    profiles: tuple[str, ...],
    *,
    has_images: bool,
) -> dict[str, bool]:
    signals = {
        name: domains[name]
        for name in (
            "core",
            "core_test",
            "dashboard",
            "website",
            "docs",
            "paper",
            "helm",
            "cli",
            "memory",
            "ci",
            "make",
            "operator",
            "openvino",
            "performance",
            "router_learning",
            "recipe_conformance",
            "agent_text",
            "agent_exec",
            "docs_only",
            "security",
            "full",
        )
    }
    signals["docs_paper"] = domains["website"] or domains["docs"] or domains["paper"]
    signals["docker"] = has_images
    signals.update(
        {
            f"e2e_{profile.replace('-', '_')}": profile in profiles
            for profile in E2E_RULES
        }
    )
    product_changed = product_paths(changed)
    signals.update(
        {
            f"e2e_{profile.replace('-', '_')}": any_matches(product_changed, *patterns)
            for profile, patterns in NON_PR_E2E_RULES.items()
        }
    )
    signals["e2e_common"] = domains["common_e2e"]
    return signals


def select_jobs(
    domains: dict[str, bool],
    profiles: tuple[str, ...],
    pr_images: tuple[str, ...],
) -> tuple[str, ...]:
    enabled_jobs: set[str] = set()
    for data in domain_records().values():
        selector = str(data.get("selector") or "")
        if selector == "always":
            enabled = True
        elif selector == "e2e":
            enabled = bool(profiles)
        elif selector == "images":
            enabled = bool(pr_images)
        else:
            enabled = domains.get(selector, False)
        if enabled:
            enabled_jobs.add(str(data["pr_job"]))
    return tuple(job for job in pr_job_order() if job in enabled_jobs)


def classify(
    paths: list[str] | tuple[str, ...], *, full: bool = False
) -> Classification:
    changed = tuple(sorted(set(paths)))
    domains = detect_domains(changed, full=full)
    profiles = select_profiles(changed, domains, full=full)
    candidates, publish_images = select_image_candidates(changed, domains)
    receipts = built_by_active_suites(domains, profiles)
    pr_images = tuple(image for image in candidates if image not in receipts)
    signals = build_output_signals(
        changed,
        domains,
        profiles,
        has_images=bool(candidates or publish_images),
    )
    return Classification(
        signals=signals,
        profiles=profiles,
        pr_images=pr_images,
        publish_images=tuple(publish_images),
        selected_jobs=select_jobs(domains, profiles, pr_images),
    )


def git_changed_files(base: str, head: str) -> list[str]:
    if base and set(base) != {"0"}:
        result = subprocess.run(
            ["git", "diff", "--name-only", "-z", base, head],
            check=True,
            capture_output=True,
        )
    else:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    return [path.decode() for path in result.stdout.split(b"\0") if path]


def write_github_outputs(path: Path, classification: Classification) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in classification.github_outputs().items():
            output.write(f"{name}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("paths", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or git_changed_files(args.base, args.head)
    result = classify(paths, full=args.full)
    if args.github_output:
        write_github_outputs(args.github_output, result)
    else:
        print(
            json.dumps(
                {
                    "paths": sorted(paths),
                    "jobs": result.selected_jobs,
                    "profiles": result.profiles,
                    "pr_images": result.pr_images,
                    "publish_images": result.publish_images,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
