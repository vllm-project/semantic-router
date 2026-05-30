"""Static release-contract marker sets used by check_version_contract."""

from __future__ import annotations

MarkerSet = tuple[tuple[str, str], ...]


def upgrade_runbook_fixture_markers(
    *, release_version: str, sim_version: str, helm_chart_ref: str
) -> MarkerSet:
    release_tag = f"v{release_version}"
    return (
        (
            "Helm chart existence check",
            f"helm show chart {helm_chart_ref} --version {release_version}",
        ),
        ("Helm upgrade command", "helm upgrade semantic-router"),
        ("Helm chart reference", helm_chart_ref),
        ("Helm upgrade version pin", f"--version {release_version}"),
        ("Helm safe value merge flag", "--reset-then-reuse-values"),
        (
            "Helm rollback command",
            "helm rollback semantic-router -n vllm-semantic-router-system --wait",
        ),
        (
            "Kubernetes rollout rollback command",
            "kubectl rollout undo deployment/semantic-router",
        ),
        (
            "Docker digest lookup",
            "DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}'",
        ),
        (
            "Make Docker release pull",
            f"make docker-pull-release DOCKER_TAG={release_tag}",
        ),
        (
            "Make Helm version upgrade",
            f"make helm-upgrade-version CHART_VERSION={release_version}",
        ),
        ("Helm values image pin", f'tag: "{release_tag}"'),
        ("Python CLI upgrade pin", f"pip install --upgrade vllm-sr=={release_version}"),
        (
            "Fleet simulator upgrade pin",
            f"pip install --upgrade vllm-sr-sim=={sim_version}",
        ),
    )


def sim_release_workflow_markers() -> MarkerSet:
    return (
        ("vllm-sr-sim tag trigger", '"vllm-sr-sim-v*"'),
        ("vllm-sr-sim tag extraction", 'TAG="${GITHUB_REF#refs/tags/}"'),
        ("vllm-sr-sim version extraction", 'VERSION="${TAG#vllm-sr-sim-v}"'),
        (
            "vllm-sr-sim package version guard",
            "PACKAGE_VERSION=$(grep '^version = ' pyproject.toml",
        ),
        (
            "vllm-sr-sim tag version guard",
            'TAG_VERSION="${{ steps.extract_version.outputs.version }}"',
        ),
        (
            "vllm-sr-sim publish command",
            "twine upload dist/* --skip-existing --verbose",
        ),
        (
            "vllm-sr-sim install snippet",
            "pip install vllm-sr-sim==${{ steps.extract_version.outputs.version }}",
        ),
    )


def sim_release_notes_markers() -> MarkerSet:
    return (
        ("unified release notes simulator tag stream", "vllm-sr-sim-v*"),
        (
            "unified release notes simulator version output",
            "${{ needs.validate.outputs.sim_version }}",
        ),
    )


def sim_upgrade_docs_markers(sim_version: str) -> MarkerSet:
    return (
        (
            "vllm-sr-sim upgrade command",
            f"pip install --upgrade vllm-sr-sim=={sim_version}",
        ),
        ("vllm-sr-sim independent release tag", "`vllm-sr-sim-v<version>`"),
        ("vllm-sr-sim publish workflow", "pypi-publish-vllm-sr-sim.yml"),
    )


def candle_crate_workflow_markers() -> MarkerSet:
    return (
        ("Candle crate tag trigger", "- 'v*'"),
        (
            "Candle crate version guard",
            "CRATE_VERSION=$(cargo metadata --no-deps --format-version 1",
        ),
        (
            "Candle crate tag version guard",
            'TAG_VERSION="${{ steps.extract_tag.outputs.version }}"',
        ),
        ("Candle crate CPU API smoke tests", "cargo test --no-default-features"),
        ("Candle crate CPU check", "cargo check --no-default-features --verbose"),
        (
            "Candle crate release build",
            "cargo build --release --no-default-features --verbose",
        ),
        (
            "Candle crate publish dry run",
            "cargo publish --dry-run --no-default-features --verbose",
        ),
        (
            "Candle crate publish command",
            "cargo publish --no-default-features --verbose",
        ),
        (
            "Candle static release artifact",
            "candle-binding/target/release/libcandle_semantic_router.a",
        ),
        (
            "Candle shared release artifact",
            "candle-binding/target/release/libcandle_semantic_router.so",
        ),
    )


def candle_release_notes_markers() -> MarkerSet:
    return (
        ("Candle crate release notes section", "### Rust Crate (crates.io)"),
        (
            "Candle crate release notes version",
            'candle-semantic-router = "${{ needs.validate.outputs.version }}"',
        ),
    )
