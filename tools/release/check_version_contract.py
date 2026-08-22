#!/usr/bin/env python3
"""Validate repo release-version contracts without workflow-local parsers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from release_contract_markers import (
    candle_crate_workflow_markers,
    candle_release_notes_markers,
    sim_release_notes_markers,
    sim_release_workflow_markers,
    sim_upgrade_docs_markers,
    upgrade_runbook_fixture_markers,
)
from snapshot_model_catalog import release_snapshot_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "src/vllm-sr/pyproject.toml"
SIM_PYPROJECT_PATH = REPO_ROOT / "src/fleet-sim/pyproject.toml"
CANDLE_CARGO_PATH = REPO_ROOT / "candle-binding/Cargo.toml"
CANDLE_LOCK_PATH = REPO_ROOT / "candle-binding/Cargo.lock"
HELM_CHART_PATH = REPO_ROOT / "deploy/helm/semantic-router/Chart.yaml"
HELM_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/helm-publish.yml"
DOCKER_PUBLISH_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/docker-publish.yml"
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/release.yml"
SIM_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/pypi-publish-vllm-sr-sim.yml"
PUBLISH_CRATE_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/publish-crate.yml"
UPGRADE_ROLLBACK_DOC_PATH = REPO_ROOT / "website/docs/installation/upgrade-rollback.md"
BUILT_IN_CATALOG_ROOT = REPO_ROOT / "config/recipes/built-in"
GHCR_IMAGE_PREFIX = "ghcr.io/vllm-project/semantic-router"
HELM_CHART_REF = "oci://ghcr.io/vllm-project/charts/semantic-router"


SEMVER_RE = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
    r"(?:[-+][0-9A-Za-z.-]+)?$"
)
RELEASE_IMAGES_RE = re.compile(r"^\s+images:\s+'(\[[^']+\])'", re.MULTILINE)


@dataclass(frozen=True)
class ReleaseContract:
    pyproject_version: str
    sim_version: str
    candle_version: str
    candle_lock_version: str
    helm_chart_version: str
    helm_app_version: str
    release_images: tuple[str, ...]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def emit_github_error(path: Path, title: str, message: str) -> None:
    relpath = path.relative_to(REPO_ROOT)
    print(f"::error file={relpath},title={title}::{message}")


def parse_project_version(path: Path) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', read_text(path), re.MULTILINE)
    if not match:
        raise ValueError(
            f"could not find project version in {path.relative_to(REPO_ROOT)}"
        )
    return match.group(1)


def parse_cargo_package_version(path: Path) -> str:
    content = read_text(path)
    package_section = re.split(
        r"^\[(?!package\])", content, maxsplit=1, flags=re.MULTILINE
    )[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', package_section, re.MULTILINE)
    if not match:
        raise ValueError(
            f"could not find package version in {path.relative_to(REPO_ROOT)}"
        )
    return match.group(1)


def parse_cargo_lock_package_version(path: Path, package_name: str) -> str:
    content = read_text(path)
    pattern = re.compile(
        rf'(?ms)^\[\[package\]\]\nname = "{re.escape(package_name)}"\nversion = "([^"]+)"'
    )
    match = pattern.search(content)
    if not match:
        raise ValueError(
            f"could not find {package_name} package version in {path.relative_to(REPO_ROOT)}"
        )
    return match.group(1)


def parse_chart_key(path: Path, key: str) -> str:
    match = re.search(
        rf"^{re.escape(key)}:\s*\"?([^\"\n#]+)\"?", read_text(path), re.MULTILINE
    )
    if not match:
        raise ValueError(f"could not find {key} in {path.relative_to(REPO_ROOT)}")
    return match.group(1).strip()


def catalog_snapshot_for_version(version: str) -> str:
    match = SEMVER_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"invalid semantic release version: {version}")
    return f"v{match.group('major')}.{match.group('minor')}"


def parse_catalog_key(path: Path, key: str) -> str:
    match = re.search(
        rf"^{re.escape(key)}:\s*['\"]?([^'\"\s#]+)['\"]?\s*(?:#.*)?$",
        read_text(path),
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"could not find {key} in {path.relative_to(REPO_ROOT)}")
    return match.group(1)


def validate_release_catalog(errors: list[str], version: str) -> str:
    snapshot = catalog_snapshot_for_version(version)
    snapshot_dir = BUILT_IN_CATALOG_ROOT / snapshot
    catalog_path = snapshot_dir / "catalog.yaml"
    if not snapshot_dir.is_dir():
        message = (
            f"release v{version} requires built-in catalog snapshot "
            f"config/recipes/built-in/{snapshot}"
        )
        errors.append(f"{snapshot_dir.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(snapshot_dir, "Missing release catalog snapshot", message)
        return snapshot
    if not catalog_path.is_file():
        message = f"release catalog snapshot {snapshot} is missing catalog.yaml"
        errors.append(f"{catalog_path.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(catalog_path, "Missing release catalog manifest", message)
        return snapshot

    expected_values = {
        "channel": "release",
        "release": snapshot,
        "catalog_version": snapshot,
    }
    for key, expected in expected_values.items():
        try:
            actual = parse_catalog_key(catalog_path, key)
        except ValueError as error:
            message = str(error)
            errors.append(message)
            emit_github_error(catalog_path, "Release catalog mismatch", message)
            continue
        require_equal(
            errors,
            catalog_path,
            f"release catalog {key}",
            actual,
            expected,
        )
    try:
        drift_errors = release_snapshot_errors(BUILT_IN_CATALOG_ROOT, snapshot)
    except (OSError, ValueError) as error:
        drift_errors = [f"release snapshot validation failed: {error}"]
    for message in drift_errors:
        errors.append(f"{snapshot_dir.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(snapshot_dir, "Release catalog snapshot drift", message)
    return snapshot


def parse_release_images() -> tuple[str, ...]:
    match = RELEASE_IMAGES_RE.search(read_text(RELEASE_WORKFLOW_PATH))
    if match is None:
        raise ValueError(
            "could not find the release image inventory in release workflow"
        )
    images = sorted(json.loads(match.group(1)))
    docker_workflow = read_text(DOCKER_PUBLISH_WORKFLOW_PATH)
    missing = [image for image in images if f"{image})" not in docker_workflow]
    if missing:
        raise ValueError(
            "release images are missing from the canonical Docker publisher: "
            + ", ".join(missing)
        )
    return tuple(images)


def collect_contract() -> ReleaseContract:
    return ReleaseContract(
        pyproject_version=parse_project_version(PYPROJECT_PATH),
        sim_version=parse_project_version(SIM_PYPROJECT_PATH),
        candle_version=parse_cargo_package_version(CANDLE_CARGO_PATH),
        candle_lock_version=parse_cargo_lock_package_version(
            CANDLE_LOCK_PATH,
            "candle-semantic-router",
        ),
        helm_chart_version=parse_chart_key(HELM_CHART_PATH, "version"),
        helm_app_version=parse_chart_key(HELM_CHART_PATH, "appVersion"),
        release_images=parse_release_images(),
    )


def require_equal(
    errors: list[str], path: Path, label: str, actual: str, expected: str
) -> None:
    if actual == expected:
        return
    message = f"{label} has '{actual}' but expected '{expected}'"
    errors.append(f"{path.relative_to(REPO_ROOT)}: {message}")
    emit_github_error(path, "Version mismatch", message)


def require_contains(errors: list[str], path: Path, label: str, needle: str) -> None:
    if needle in read_text(path):
        return
    message = f"{label} missing required contract marker: {needle}"
    errors.append(f"{path.relative_to(REPO_ROOT)}: {message}")
    emit_github_error(path, "Release contract mismatch", message)


def require_markers(
    errors: list[str], path: Path, markers: tuple[tuple[str, str], ...]
) -> None:
    for label, marker in markers:
        require_contains(errors, path, label, marker)


def validate_helm_workflow(errors: list[str]) -> None:
    require_contains(
        errors,
        HELM_WORKFLOW_PATH,
        "Helm release chart version",
        'CHART_VERSION="$RELEASE_VERSION"',
    )
    require_contains(
        errors,
        HELM_WORKFLOW_PATH,
        "Helm release app version",
        'APP_VERSION="$RELEASE_TAG"',
    )
    require_contains(
        errors,
        HELM_WORKFLOW_PATH,
        "Helm package chart override",
        '--version "${{ steps.versions.outputs.chart_version }}"',
    )
    require_contains(
        errors,
        HELM_WORKFLOW_PATH,
        "Helm package app-version override",
        '--app-version "${{ steps.versions.outputs.app_version }}"',
    )


def validate_release_notes_images(
    errors: list[str], release_images: tuple[str, ...]
) -> None:
    release_notes = read_text(RELEASE_WORKFLOW_PATH)
    for image in release_images:
        if image in release_notes:
            continue
        message = f"release notes do not mention Docker release image '{image}'"
        errors.append(f"{RELEASE_WORKFLOW_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(
            RELEASE_WORKFLOW_PATH, "Release notes image mismatch", message
        )


def validate_upgrade_docs_images(
    errors: list[str], release_images: tuple[str, ...], version: str
) -> None:
    upgrade_docs = read_text(UPGRADE_ROLLBACK_DOC_PATH)
    for image in release_images:
        image_ref = f"{GHCR_IMAGE_PREFIX}/{image}:v{version}"
        if image_ref in upgrade_docs:
            continue
        message = (
            "upgrade and rollback docs do not include full tagged release image "
            f"'{image_ref}'"
        )
        errors.append(f"{UPGRADE_ROLLBACK_DOC_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(
            UPGRADE_ROLLBACK_DOC_PATH, "Upgrade docs image mismatch", message
        )


def validate_upgrade_runbook_fixtures(
    errors: list[str], contract: ReleaseContract, release_version: str
) -> None:
    require_markers(
        errors,
        UPGRADE_ROLLBACK_DOC_PATH,
        upgrade_runbook_fixture_markers(
            release_version=release_version,
            sim_version=contract.sim_version,
            helm_chart_ref=HELM_CHART_REF,
        ),
    )


def validate_sim_release_workflow(errors: list[str]) -> None:
    require_markers(errors, SIM_WORKFLOW_PATH, sim_release_workflow_markers())


def validate_sim_release_notes(errors: list[str]) -> None:
    require_markers(errors, RELEASE_WORKFLOW_PATH, sim_release_notes_markers())


def validate_sim_upgrade_docs(errors: list[str], sim_version: str) -> None:
    require_markers(
        errors, UPGRADE_ROLLBACK_DOC_PATH, sim_upgrade_docs_markers(sim_version)
    )


def validate_candle_crate_workflow(errors: list[str]) -> None:
    require_markers(
        errors, PUBLISH_CRATE_WORKFLOW_PATH, candle_crate_workflow_markers()
    )


def validate_candle_release_notes(errors: list[str]) -> None:
    require_markers(errors, RELEASE_WORKFLOW_PATH, candle_release_notes_markers())


def validate(expected_version: str | None) -> tuple[ReleaseContract, list[str]]:
    contract = collect_contract()
    errors: list[str] = []

    if not SEMVER_RE.match(contract.pyproject_version):
        errors.append(
            f"{PYPROJECT_PATH.relative_to(REPO_ROOT)}: invalid semantic version"
        )
        emit_github_error(PYPROJECT_PATH, "Invalid version", contract.pyproject_version)

    expected = expected_version or contract.pyproject_version
    require_equal(
        errors, PYPROJECT_PATH, "vllm-sr version", contract.pyproject_version, expected
    )
    require_equal(
        errors,
        CANDLE_CARGO_PATH,
        "candle-binding version",
        contract.candle_version,
        expected,
    )
    require_equal(
        errors,
        CANDLE_LOCK_PATH,
        "candle-binding lockfile version",
        contract.candle_lock_version,
        expected,
    )

    if contract.helm_app_version != "latest":
        message = (
            "source chart appVersion must stay 'latest'; release tags override it in CI"
        )
        errors.append(f"{HELM_CHART_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(HELM_CHART_PATH, "Helm appVersion contract mismatch", message)

    validate_helm_workflow(errors)
    validate_release_notes_images(errors, contract.release_images)
    validate_upgrade_docs_images(errors, contract.release_images, expected)
    validate_upgrade_runbook_fixtures(errors, contract, expected)
    validate_sim_release_workflow(errors)
    validate_sim_release_notes(errors)
    validate_sim_upgrade_docs(errors, contract.sim_version)
    validate_candle_crate_workflow(errors)
    validate_candle_release_notes(errors)
    # An explicit version is the publication boundary used by release.yml and
    # `make release-validate RELEASE_VERSION=...`. Source-only validation may
    # run before maintainers cut the next immutable minor snapshot.
    if expected_version is not None:
        validate_release_catalog(errors, expected)
    return contract, errors


def write_github_outputs(
    path: Path, contract: ReleaseContract, release_version: str
) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"pyproject_version={contract.pyproject_version}\n")
        output.write(f"candle_version={contract.candle_version}\n")
        output.write(f"candle_lock_version={contract.candle_lock_version}\n")
        output.write(f"helm_chart_version={contract.helm_chart_version}\n")
        output.write(f"helm_app_version={contract.helm_app_version}\n")
        output.write(f"sim_version={contract.sim_version}\n")
        output.write(f"release_images={','.join(contract.release_images)}\n")
        output.write(
            f"catalog_snapshot={catalog_snapshot_for_version(release_version)}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help="Expected stable release version without a leading v. Defaults to src/vllm-sr.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="Optional GITHUB_OUTPUT file to append workflow outputs to.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract, errors = validate(args.version)
    release_version = args.version or contract.pyproject_version

    if args.github_output:
        write_github_outputs(args.github_output, contract, release_version)

    print("Release version contract")
    print(f"  vllm-sr package:        {contract.pyproject_version}")
    print(f"  candle crate:           {contract.candle_version}")
    print(f"  candle lockfile:        {contract.candle_lock_version}")
    print(f"  helm chart source:      {contract.helm_chart_version}")
    print(f"  helm source appVersion: {contract.helm_app_version}")
    print(f"  vllm-sr-sim package:    {contract.sim_version} (independent tag stream)")
    print(f"  Docker release images:  {', '.join(contract.release_images)}")
    catalog_snapshot = catalog_snapshot_for_version(release_version)
    if args.version is not None:
        print(f"  Built-in catalog:       {catalog_snapshot} (release-bound)")
    else:
        print(
            f"  Built-in catalog target: {catalog_snapshot} "
            "(checked when --version is explicit)"
        )

    if not errors:
        print("  Status: pass")
        return 0

    print("  Status: fail", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
