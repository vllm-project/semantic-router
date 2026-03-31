from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"
VLLM_SR_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile"
VLLM_SR_ROCM_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.rocm"


def test_dashboard_dockerfile_retries_backend_builder_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "apk_add_with_retry build-base" in content


def test_dashboard_dockerfile_retries_runtime_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "apk_add_with_retry ca-certificates curl docker-cli py3-pip python3 su-exec wget"
        in content
    )


def test_dashboard_dockerfile_copies_router_dsl_package_for_backend_builds() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "COPY src/semantic-router/pkg/dsl/ /app/src/semantic-router/pkg/dsl/" in content
    )
    assert (
        "COPY src/semantic-router/pkg/nlgen/ /app/src/semantic-router/pkg/nlgen/"
        in content
    )
    assert (
        "COPY src/semantic-router/internal/nlgen/ /app/src/semantic-router/internal/nlgen/"
        in content
    )


def test_vllm_sr_dockerfile_copies_router_dsl_package_for_dashboard_builder() -> None:
    content = VLLM_SR_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "COPY src/semantic-router/pkg/dsl/ /app/src/semantic-router/pkg/dsl/" in content
    )
    assert (
        "COPY src/semantic-router/pkg/nlgen/ /app/src/semantic-router/pkg/nlgen/"
        in content
    )
    assert (
        "COPY src/semantic-router/internal/nlgen/ /app/src/semantic-router/internal/nlgen/"
        in content
    )


def test_vllm_sr_rocm_dockerfile_copies_router_dsl_package_for_dashboard_builder() -> (
    None
):
    content = VLLM_SR_ROCM_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "COPY src/semantic-router/pkg/dsl/ /app/src/semantic-router/pkg/dsl/" in content
    )
    assert (
        "COPY src/semantic-router/pkg/nlgen/ /app/src/semantic-router/pkg/nlgen/"
        in content
    )
    assert (
        "COPY src/semantic-router/internal/nlgen/ /app/src/semantic-router/internal/nlgen/"
        in content
    )
