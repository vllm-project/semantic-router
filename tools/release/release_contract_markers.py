"""Static release-contract marker sets used by check_version_contract."""

from __future__ import annotations

MarkerSet = tuple[tuple[str, str], ...]


def upgrade_runbook_fixture_markers(
    *, release_version: str, helm_chart_ref: str
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
            "DIGEST=$(docker buildx imagetools inspect",
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
    )
