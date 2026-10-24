#!/usr/bin/env python3
"""Validate repo release-version contracts without workflow-local parsers."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from release_contract_markers import (
    upgrade_runbook_fixture_markers,
)
from snapshot_model_catalog import release_snapshot_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "src/vllm-sr/pyproject.toml"
HELM_CHART_PATH = REPO_ROOT / "deploy/helm/semantic-router/Chart.yaml"
HELM_VALUES_PATH = REPO_ROOT / "deploy/helm/semantic-router/values.yaml"
HELM_TEMPLATES_DIR = REPO_ROOT / "deploy/helm/semantic-router/templates"
HELM_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/helm-publish.yml"
DOCKER_PUBLISH_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/docker-publish.yml"
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/release.yml"
CI_IMAGE_INVENTORY_PATH = REPO_ROOT / "tools/ci/classify_pr_changes.py"
CI_IMAGE_ARTIFACTS_PATH = REPO_ROOT / "tools/ci/image_artifacts.py"
UPGRADE_ROLLBACK_DOC_PATH = REPO_ROOT / "website/docs/installation/upgrade-rollback.md"
BUILT_IN_CATALOG_ROOT = REPO_ROOT / "config/recipes/built-in"
GHCR_IMAGE_PREFIX = "ghcr.io/vllm-project/semantic-router"
HELM_CHART_REF = "oci://ghcr.io/vllm-project/charts/semantic-router"
# The tag main publishes its images under. A development cycle's source chart
# deploys it; release.sh pins the release tag only on the release commit.
DEVELOPMENT_IMAGE_TAG = "latest"


SEMVER_RE = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
    r"(?:[-+][0-9A-Za-z.-]+)?$"
)
IMAGE_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RUNBOOK_RELEASE_RE = re.compile(
    rf"helm show chart {re.escape(HELM_CHART_REF)} --version "
    r"([0-9]+\.[0-9]+\.[0-9]+)\b"
)
TEMPLATE_IMAGE_RE = re.compile(r"^\s*(?:-\s+)?image:\s")


@dataclass(frozen=True)
class ChartImage:
    values_key: str
    repository: str
    tag: str


@dataclass(frozen=True)
class ReleaseContract:
    pyproject_version: str
    helm_chart_version: str
    helm_app_version: str
    release_images: tuple[str, ...]
    chart_images: tuple[ChartImage, ...]


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
    """Read the release inventory selected by the shared CI plan."""

    module = ast.parse(read_text(CI_IMAGE_INVENTORY_PATH))
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != "PRODUCTION_RELEASE_IMAGES":
            continue
        try:
            images = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError(
                "production release image inventory must be a literal tuple"
            ) from error
        if (
            not isinstance(images, tuple)
            or not images
            or any(
                not isinstance(image, str) or not IMAGE_NAME_RE.fullmatch(image)
                for image in images
            )
            or len(set(images)) != len(images)
        ):
            raise ValueError("production release image inventory is invalid")
        missing = sorted(set(images) - parse_defined_images())
        if missing:
            raise ValueError(
                "production release images have no build definition: "
                + ", ".join(missing)
            )
        return tuple(sorted(images))
    raise ValueError("could not find the production release image inventory")


def parse_defined_images() -> set[str]:
    """Find literal image keys in the canonical artifact builder."""

    module = ast.parse(read_text(CI_IMAGE_ARTIFACTS_PATH))
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != "DEFINITIONS":
            continue
        if not isinstance(statement.value, ast.Dict):
            break
        return {
            key.value
            for key in statement.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise ValueError("could not find canonical image build definitions")


def parse_chart_images() -> tuple[ChartImage, ...]:
    """Find every project image the chart's values deploy."""

    values = yaml.safe_load(read_text(HELM_VALUES_PATH))
    images: list[ChartImage] = []

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            repository = node.get("repository")
            if isinstance(repository, str) and repository.startswith(
                f"{GHCR_IMAGE_PREFIX}/"
            ):
                images.append(
                    ChartImage(".".join(path), repository, str(node.get("tag") or ""))
                )
            for key, value in node.items():
                walk(value, (*path, str(key)))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, (*path, str(index)))

    walk(values, ())
    if not images:
        raise ValueError(
            f"{HELM_VALUES_PATH.relative_to(REPO_ROOT)} deploys no {GHCR_IMAGE_PREFIX} image"
        )
    return tuple(images)


def collect_contract() -> ReleaseContract:
    return ReleaseContract(
        pyproject_version=parse_project_version(PYPROJECT_PATH),
        helm_chart_version=parse_chart_key(HELM_CHART_PATH, "version"),
        helm_app_version=parse_chart_key(HELM_CHART_PATH, "appVersion"),
        release_images=parse_release_images(),
        chart_images=parse_chart_images(),
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
        "Helm main-channel app version",
        f'APP_VERSION="{DEVELOPMENT_IMAGE_TAG}"',
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


def validate_chart_images(
    errors: list[str], contract: ReleaseContract, release_version: str | None = None
) -> None:
    """Every image the source chart deploys defaults to the mode's tag.

    A development cycle deploys the development image. The release commit,
    which release.sh writes, pins the release tag; the next commit restores the
    development image.
    """

    if release_version is None:
        expected = DEVELOPMENT_IMAGE_TAG
        reason = (
            "a development cycle deploys the development image; release.sh "
            "pins the release tag only on the release commit"
        )
    else:
        expected = f"v{release_version}"
        reason = "the release commit pins its own tag"
    if contract.helm_app_version != expected:
        message = (
            f"source chart appVersion has '{contract.helm_app_version}' but "
            f"expected '{expected}': {reason}"
        )
        errors.append(f"{HELM_CHART_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(HELM_CHART_PATH, "Helm appVersion contract mismatch", message)
    for image in contract.chart_images:
        if not image.tag:
            continue
        message = (
            f"{image.values_key}.tag pins '{image.tag}' for {image.repository}; "
            "leave it empty so the image follows the chart appVersion"
        )
        errors.append(f"{HELM_VALUES_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(HELM_VALUES_PATH, "Helm image tag contract mismatch", message)


def validate_chart_templates(errors: list[str]) -> None:
    """Every container image the chart renders defaults to the appVersion tag."""

    for path in sorted(HELM_TEMPLATES_DIR.glob("*")):
        if path.suffix not in {".yaml", ".tpl"}:
            continue
        for number, line in enumerate(read_text(path).splitlines(), start=1):
            if TEMPLATE_IMAGE_RE.match(line) and ".Chart.AppVersion" not in line:
                message = (
                    f"line {number} renders an image whose tag does not default "
                    "to .Chart.AppVersion"
                )
                errors.append(f"{path.relative_to(REPO_ROOT)}: {message}")
                emit_github_error(path, "Helm image tag contract mismatch", message)


def validate_release_image_bridge(errors: list[str]) -> None:
    """Require the canonical release image list to reach build and promotion."""

    markers = (
        (
            RELEASE_WORKFLOW_PATH,
            "release image input",
            "images: ${{ needs.validate.outputs.images }}",
        ),
        (
            RELEASE_WORKFLOW_PATH,
            "release image validation output",
            "images: ${{ steps.contract.outputs.release_images_json }}",
        ),
        (
            RELEASE_WORKFLOW_PATH,
            "release image builder",
            "uses: ./.github/workflows/build-artifacts.yml",
        ),
        (
            DOCKER_PUBLISH_WORKFLOW_PATH,
            "Docker promotion matrix",
            "image: ${{ fromJSON(inputs.images) }}",
        ),
        (
            DOCKER_PUBLISH_WORKFLOW_PATH,
            "qualified image promotion",
            "tools/ci/image_artifacts.py promote",
        ),
    )
    for path, label, marker in markers:
        require_contains(errors, path, label, marker)


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


def validate_upgrade_runbook_fixtures(errors: list[str], release_version: str) -> None:
    require_markers(
        errors,
        UPGRADE_ROLLBACK_DOC_PATH,
        upgrade_runbook_fixture_markers(
            release_version=release_version,
            helm_chart_ref=HELM_CHART_REF,
        ),
    )


def semver_core(version: str) -> tuple[int, int, int] | None:
    match = SEMVER_RE.fullmatch(version)
    if match is None:
        return None
    return (int(match["major"]), int(match["minor"]), int(match["patch"]))


def runbook_release() -> str | None:
    """The release whose chart the upgrade runbook installs."""

    match = RUNBOOK_RELEASE_RE.search(read_text(UPGRADE_ROLLBACK_DOC_PATH))
    return match.group(1) if match else None


def documented_release(
    errors: list[str], contract: ReleaseContract, expected_version: str | None
) -> str | None:
    """The release the docs and runbook pin.

    A release check documents the release itself. Between releases, `main`
    carries the next version (so dev builds sort after the last release) while
    the docs stay on the last release.
    """

    if expected_version is not None:
        require_equal(
            errors,
            PYPROJECT_PATH,
            "vllm-sr version",
            contract.pyproject_version,
            expected_version,
        )
        return expected_version
    released = runbook_release()
    if released is None:
        message = (
            "upgrade runbook pins no release: expected "
            f"'helm show chart {HELM_CHART_REF} --version X.Y.Z'"
        )
        errors.append(f"{UPGRADE_ROLLBACK_DOC_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(UPGRADE_ROLLBACK_DOC_PATH, "Documented release", message)
        return None
    current, pinned = semver_core(contract.pyproject_version), semver_core(released)
    if current is not None and pinned is not None and current < pinned:
        message = (
            f"vllm-sr version {contract.pyproject_version} is behind the released "
            f"v{released} that the docs pin"
        )
        errors.append(f"{PYPROJECT_PATH.relative_to(REPO_ROOT)}: {message}")
        emit_github_error(PYPROJECT_PATH, "Version behind release", message)
    return released


def validate(expected_version: str | None) -> tuple[ReleaseContract, list[str]]:
    contract = collect_contract()
    errors: list[str] = []

    if not SEMVER_RE.match(contract.pyproject_version):
        errors.append(
            f"{PYPROJECT_PATH.relative_to(REPO_ROOT)}: invalid semantic version"
        )
        emit_github_error(PYPROJECT_PATH, "Invalid version", contract.pyproject_version)

    documented = documented_release(errors, contract, expected_version)

    validate_chart_images(errors, contract, expected_version)
    validate_chart_templates(errors)
    validate_helm_workflow(errors)
    validate_release_image_bridge(errors)
    validate_release_notes_images(errors, contract.release_images)
    if documented is not None:
        validate_upgrade_docs_images(errors, contract.release_images, documented)
        validate_upgrade_runbook_fixtures(errors, documented)
    # An explicit version is the publication boundary used by release.yml and
    # `make release-check RELEASE_VERSION=...`. Source-only validation may
    # run before maintainers cut the next immutable minor snapshot.
    if expected_version is not None:
        validate_release_catalog(errors, expected_version)
    return contract, errors


def write_github_outputs(
    path: Path, contract: ReleaseContract, release_version: str
) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"pyproject_version={contract.pyproject_version}\n")
        output.write(f"helm_chart_version={contract.helm_chart_version}\n")
        output.write(f"helm_app_version={contract.helm_app_version}\n")
        output.write(f"release_images={','.join(contract.release_images)}\n")
        output.write(
            f"release_images_json={json.dumps(list(contract.release_images))}\n"
        )
        output.write(
            f"catalog_snapshot={catalog_snapshot_for_version(release_version)}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help=(
            "Check the release commit of this stable version (without a leading "
            "v). Without it, check the development cycle."
        ),
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
    if args.version is None:
        print("  Mode:                   development cycle")
    else:
        print(f"  Mode:                   release v{args.version}")
    print(f"  vllm-sr package:        {contract.pyproject_version}")
    print(f"  helm chart source:      {contract.helm_chart_version}")
    print(f"  helm source appVersion: {contract.helm_app_version}")
    print(
        "  Chart images:           "
        + ", ".join(image.repository for image in contract.chart_images)
    )
    if args.version is None:
        documented = runbook_release()
        print(
            f"  Documented release:     v{documented} (docs and runbook)"
            if documented
            else "  Documented release:     none"
        )
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
