from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"
VLLM_SR_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile"
VLLM_SR_ROCM_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.rocm"


def test_dashboard_dockerfile_uses_glibc_builder_for_cgo_backend() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM golang:1.24-bookworm AS backend-builder" in content
    assert "apt_get_install_with_retry build-essential" in content


def test_dashboard_dockerfile_retries_runtime_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim-bookworm" in content
    assert (
        "apt_get_install_with_retry ca-certificates curl docker.io gosu wget" in content
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


def test_dashboard_dockerfile_copies_model_eval_scripts_and_requirements() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY src/training/model_eval/ /app/src/training/model_eval/" in content
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/src/training/model_eval/requirements.txt'
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
