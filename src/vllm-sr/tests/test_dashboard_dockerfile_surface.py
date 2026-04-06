from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"
VLLM_SR_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile"
VLLM_SR_ROCM_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.rocm"
EXTPROC_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc"
EXTPROC_ROCM_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc-rocm"


def assert_builtin_knowledge_base_copy_paths(content: str) -> None:
    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in content
    assert "COPY config/knowledge_bases/ /app/config-assets/knowledge_bases/" in content


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
        "COPY src/semantic-router/pkg/routerauthoring/ "
        "/app/src/semantic-router/pkg/routerauthoring/" in content
    )
    assert (
        "COPY src/semantic-router/pkg/routercontract/ "
        "/app/src/semantic-router/pkg/routercontract/" in content
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
        "COPY src/semantic-router/pkg/trainingartifacts/ "
        "/app/src/semantic-router/pkg/trainingartifacts/" in content
    )
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/src/training/model_eval/requirements.txt'
        in content
    )


def test_vllm_sr_dockerfile_stays_router_only() -> None:
    content = VLLM_SR_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG RUST_RUNTIME_COMPAT_IMAGE=rustlang/rust:nightly-bullseye" in content
    assert "ARG GO_RUNTIME_COMPAT_IMAGE=golang:1.24-bullseye" in content
    assert "GLIBC_2.39+" in content
    assert 'ENTRYPOINT ["/app/start-router.sh"]' in content
    assert_builtin_knowledge_base_copy_paths(content)
    assert "ENV VIRTUAL_ENV=/opt/vllm-sr-venv" in content
    assert "python3-yaml" in content
    assert "python3-venv" in content
    assert 'python3 -m venv "${VIRTUAL_ENV}"' in content
    assert "huggingface_hub[cli]==1.5.0" in content
    assert "COPY --from=dashboard-builder" not in content
    assert "COPY --from=frontend-builder" not in content
    assert "COPY --from=wizmap-builder" not in content
    assert "COPY src/vllm-sr/start-dashboard.sh" not in content
    assert "COPY dashboard/backend/config/openclaw-skills.json" not in content
    assert "COPY src/training/model_eval/" not in content


def test_vllm_sr_rocm_dockerfile_stays_router_only() -> None:
    content = VLLM_SR_ROCM_DOCKERFILE.read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["/app/start-router.sh"]' in content
    assert_builtin_knowledge_base_copy_paths(content)
    assert "ENV VIRTUAL_ENV=/opt/vllm-sr-venv" in content
    assert "python3-yaml" in content
    assert "python3-venv" in content
    assert 'python3 -m venv "${VIRTUAL_ENV}"' in content
    assert "huggingface_hub[cli]==1.5.0" in content
    assert "COPY candle-binding/go.mod candle-binding/semantic-router.go candle-binding/backend_contract.go ./" in content
    assert "COPY onnx-binding/go.mod onnx-binding/semantic-router.go onnx-binding/backend_contract.go ./" in content
    assert "COPY --from=dashboard-builder" not in content
    assert "COPY --from=frontend-builder" not in content
    assert "COPY --from=wizmap-builder" not in content
    assert "COPY src/vllm-sr/start-dashboard.sh" not in content
    assert "COPY dashboard/backend/config/openclaw-skills.json" not in content
    assert "COPY src/training/model_eval/" not in content


def test_extproc_dockerfile_copies_built_in_knowledge_bases() -> None:
    content = EXTPROC_DOCKERFILE.read_text(encoding="utf-8")

    assert_builtin_knowledge_base_copy_paths(content)
    assert "COPY config/kb/ /app/config/kb/" not in content


def test_extproc_rocm_dockerfile_copies_built_in_knowledge_bases() -> None:
    content = EXTPROC_ROCM_DOCKERFILE.read_text(encoding="utf-8")

    assert_builtin_knowledge_base_copy_paths(content)
    assert "COPY config/kb/ /app/config/kb/" not in content
