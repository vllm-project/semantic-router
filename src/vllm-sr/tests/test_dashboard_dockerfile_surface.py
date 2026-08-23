import importlib.util
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"
VLLM_SR_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile"
VLLM_SR_ROCM_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.rocm"
VLLM_SR_CUDA_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.cuda"
VLLM_SR_CUDA_DOCKERIGNORE = (
    REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.cuda.dockerignore"
)
EXTPROC_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc"
EXTPROC_ROCM_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc-rocm"
DASHBOARD_ENTRYPOINT = REPO_ROOT / "dashboard" / "backend" / "entrypoint.sh"
DASHBOARD_PERMISSION_HELPER = (
    REPO_ROOT / "dashboard" / "backend" / "entrypoint_permissions.py"
)
DASHBOARD_LOGS_HANDLER = REPO_ROOT / "dashboard" / "backend" / "handlers" / "logs.go"


def _load_dashboard_permission_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "dashboard_entrypoint_permissions",
        DASHBOARD_PERMISSION_HELPER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Dashboard permission helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DASHBOARD_PERMISSION_MODULE = _load_dashboard_permission_helper()


def test_dashboard_dockerfile_uses_glibc_builder_for_cgo_backend() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "FROM ${IMAGE_REGISTRY}library/golang:1.25-bookworm AS backend-builder"
        in content
    )
    assert "apt_get_install_with_retry build-essential" in content


def test_dashboard_dockerfile_retries_runtime_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM ${IMAGE_REGISTRY}library/python:3.11-slim-bookworm" in content
    assert (
        "apt_get_install_with_retry ca-certificates curl docker.io gosu wget" in content
    )
    assert (
        "COPY dashboard/backend/entrypoint_permissions.py /app/entrypoint_permissions.py"
        in content
    )
    assert "chown -R nonroot:nonroot /app" not in content
    assert "chown root:root" in content
    assert "/app/entrypoint.sh" in content
    assert "/app/entrypoint_permissions.py" in content
    assert "install -d -o nonroot -g root -m 0770 /app/data" in content


def test_dashboard_entrypoint_maps_runtime_socket_group_before_dropping_root() -> None:
    content = DASHBOARD_ENTRYPOINT.read_text(encoding="utf-8")

    assert "PERMISSION_HELPER=/app/entrypoint_permissions.py" in content
    assert "DASHBOARD_PERMISSION_HELPER" not in content
    assert (
        'if CONTAINER_SOCKET_GID=$(python3 "$PERMISSION_HELPER" socket-gid '
        '"$CONTAINER_SOCKET_PATH" 2>/dev/null); then' in content
    )
    assert 'add_nonroot_group_gid "$CONTAINER_SOCKET_GID"' in content
    assert "export OPENCLAW_CONTAINER_RUNTIME_DISABLED=false" in content
    assert "export OPENCLAW_CONTAINER_RUNTIME_DISABLED=true" in content
    assert "OPENCLAW_CONTAINER_RUNTIME_DISABLED=true" in content
    assert content.index("OPENCLAW_CONTAINER_RUNTIME_DISABLED=true") < content.index(
        'exec "$@"'
    )
    assert (
        'if [ "${OPENCLAW_CONTAINER_RUNTIME_DISABLED:-true}" != "false" ]; then'
        in content
    )
    assert content.index(
        "${OPENCLAW_CONTAINER_RUNTIME_DISABLED:-true}"
    ) < content.index("socket-gid")
    assert "continuing without socket access" in content
    assert "LOG_SPOOL_GID=${VLLM_SR_LOG_SPOOL_GID:-}" in content
    assert 'add_nonroot_group_gid "$LOG_SPOOL_GID"' in content
    assert "Invalid log spool group" in content
    assert 'groupadd --gid "$GROUP_GID" "$GROUP_NAME"' in content
    assert 'usermod -aG "$GROUP_NAME" nonroot' in content
    assert 'if [ "$GROUP_GID" -eq 0 ]' in content
    assert "Refusing to add Dashboard user to the root group" in content
    assert "safe_shared_path_gid" not in content
    assert "DATA_GID=65532" in content
    assert 'python3 "$PERMISSION_HELPER" prepare-tree' in content
    assert "DASHBOARD_RUNTIME_CONFIG_WRITABLE" not in content
    assert "CONFIG_FILE_PATH" not in content
    assert "RUNTIME_LOCK_DIR" not in content
    assert 'python3 "$PERMISSION_HELPER" prepare-bootstrap-token' in content
    assert '"$DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE" 65532 65532' in content
    assert "VLLM_SR_RECIPE_STORE_DIR" not in content
    assert "DASHBOARD_RECIPE_STORE_WRITABLE" not in content
    assert "router-management.token" not in content
    assert "DASHBOARD_READONLY=true" not in content
    assert 'if [ "$(id -u)" -ne 0 ]' in content
    assert content.index('if [ "$(id -u)" -ne 0 ]') < content.index(
        'usermod -aG "$GROUP_NAME" nonroot'
    )
    assert "chmod -R" not in content
    assert "chgrp -R" not in content
    assert "su-exec" not in content
    assert content.index('usermod -aG "$GROUP_NAME" nonroot') < content.index(
        'exec gosu nonroot "$@"'
    )


def test_dashboard_logs_handler_never_executes_a_container_runtime() -> None:
    content = DASHBOARD_LOGS_HANDLER.read_text(encoding="utf-8")

    assert '"os/exec"' not in content
    assert 'exec.Command("docker"' not in content
    assert 'exec.Command("podman"' not in content


def test_dashboard_permission_helper_pins_and_validates_runtime_socket(
    tmp_path: Path, monkeypatch
) -> None:
    # AF_UNIX paths are limited to roughly 108 bytes on Linux. Keep the leaf
    # relative so the contract also runs from deeply nested workspace runtimes.
    monkeypatch.chdir(tmp_path)
    socket_path = Path("s")
    runtime_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        runtime_socket.bind(str(socket_path))
        socket_gid = os.getgid() or 65534
        os.chown(socket_path, -1, socket_gid)
        socket_path.chmod(0o660)
        result = subprocess.run(
            [
                sys.executable,
                str(DASHBOARD_PERMISSION_HELPER),
                "socket-gid",
                str(socket_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.strip()) == socket_gid

        socket_path.chmod(0o600)
        result = subprocess.run(
            [
                sys.executable,
                str(DASHBOARD_PERMISSION_HELPER),
                "socket-gid",
                str(socket_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "group read/write" in result.stderr

        socket_path.chmod(0o666)
        result = subprocess.run(
            [
                sys.executable,
                str(DASHBOARD_PERMISSION_HELPER),
                "socket-gid",
                str(socket_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "must not grant other access" in result.stderr

        socket_path.chmod(0o660)
        socket_link = Path("socket-link")
        socket_link.symlink_to(socket_path)
        result = subprocess.run(
            [
                sys.executable,
                str(DASHBOARD_PERMISSION_HELPER),
                "socket-gid",
                str(socket_link),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "must be a Unix socket" in result.stderr
    finally:
        runtime_socket.close()

    ordinary = tmp_path / "ordinary-file"
    ordinary.write_text("not a socket", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "socket-gid",
            str(ordinary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must be a Unix socket" in result.stderr


def test_dashboard_permission_helper_normalizes_only_closed_var_run_alias() -> None:
    normalize = DASHBOARD_PERMISSION_MODULE._normalize_system_path

    assert normalize("/var/run") == "/run"
    assert normalize("/var/run/secrets/dashboard/key.pem") == (
        "/run/secrets/dashboard/key.pem"
    )
    assert normalize("/var/runfoo/secrets/key.pem") == "/var/runfoo/secrets/key.pem"


def test_dashboard_permission_helper_rejects_parent_traversal(tmp_path: Path) -> None:
    unsafe_path = str(tmp_path / "safe" / ".." / "secret")

    with pytest.raises(OSError, match="parent traversal"):
        DASHBOARD_PERMISSION_MODULE.open_path(unsafe_path)


def test_dashboard_permission_helper_rejects_symlinked_path_child(
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    secret = real_directory / "secret"
    secret.write_text("private-test-material", encoding="utf-8")
    secret.chmod(0o600)
    symlinked_child = tmp_path / "symlinked-child"
    symlinked_child.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(OSError):
        DASHBOARD_PERMISSION_MODULE.open_path(str(symlinked_child / "secret"))


def test_dashboard_permission_helper_rejects_symlinked_shared_tree_entry(
    tmp_path: Path,
) -> None:
    shared_tree = tmp_path / "shared-tree"
    shared_tree.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("sentinel", encoding="utf-8")
    outside.chmod(0o600)
    (shared_tree / "trap").symlink_to(outside)

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-tree",
            str(shared_tree),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


def test_dashboard_permission_helper_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    shared_tree = tmp_path / "shared-tree"
    shared_tree.mkdir()
    os.mkfifo(shared_tree / "trap")

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-tree",
            str(shared_tree),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )

    assert result.returncode != 0
    assert "unsafe file" in result.stderr


def test_dashboard_permission_helper_preserves_private_bootstrap_token(
    tmp_path: Path,
) -> None:
    token_directory = tmp_path / "bootstrap"
    token_directory.mkdir(mode=0o700)
    token = token_directory / "router-token"
    token.write_text("one-time-test-token", encoding="utf-8")
    token.chmod(0o600)

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-bootstrap-token",
            str(token),
            str(os.getuid()),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(token_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(token.stat().st_mode) == 0o600

    token.unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-bootstrap-token",
            str(token),
            str(os.getuid()),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_permission_helper_stages_private_secret_without_broadening_source(
    tmp_path: Path,
) -> None:
    source_directory = tmp_path / "router-secrets"
    source_directory.mkdir(mode=0o700)
    source = source_directory / "issuer-key.pem"
    source.write_text("private-test-material", encoding="utf-8")
    source.chmod(0o600)
    destination_directory = tmp_path / "dashboard-secrets"
    destination_directory.mkdir(mode=0o700)
    destination = destination_directory / "issuer-key.pem"

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "stage-private-file",
            str(source),
            str(destination),
            str(os.getuid()),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert destination.read_text(encoding="utf-8") == "private-test-material"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IMODE(source.stat().st_mode) == 0o600

    source.write_text("rotated-private-test-material", encoding="utf-8")
    source.chmod(0o600)
    restarted = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "stage-private-file",
            str(source),
            str(destination),
            str(os.getuid()),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert restarted.returncode == 0, restarted.stderr
    assert destination.read_text(encoding="utf-8") == "rotated-private-test-material"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_dashboard_dockerfile_copies_one_router_package_tree_for_backend_builds() -> (
    None
):
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY src/semantic-router/pkg/ /app/src/semantic-router/pkg/" in content
    assert "COPY src/semantic-router/pkg/dsl/" not in content
    assert "COPY src/semantic-router/pkg/nlgen/" not in content
    assert "COPY src/semantic-router/pkg/routerreplay/redaction/" not in content


def test_dashboard_dockerfile_copies_model_eval_scripts_and_requirements() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY src/training/model_eval/ /app/src/training/model_eval/" in content
    assert "ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu" in content
    assert "ARG TORCH_CPU_VERSION=" in content
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --index-url "${TORCH_CPU_INDEX_URL}" "torch==${TORCH_CPU_VERSION}"'
        in content
    )
    assert content.index('"torch==${TORCH_CPU_VERSION}"') < content.index(
        "-r /app/src/training/model_eval/requirements.txt"
    )
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/src/training/model_eval/requirements.txt'
        in content
    )


def test_vllm_sr_dockerfile_stays_router_only() -> None:
    content = VLLM_SR_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG RUST_RUNTIME_COMPAT_IMAGE=rustlang/rust:nightly-bullseye" in content
    assert "ARG GO_RUNTIME_COMPAT_IMAGE=library/golang:1.25-bookworm" in content
    assert (
        "FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${RUST_RUNTIME_COMPAT_IMAGE}"
        in content
    )
    assert "GLIBC_2.39+" in content
    assert 'ENTRYPOINT ["/app/start-router.sh"]' in content
    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in content
    assert "ENV VIRTUAL_ENV=/opt/vllm-sr-venv" in content
    assert "python3-yaml" in content
    assert "python3-venv" in content
    assert 'python3 -m venv "${VIRTUAL_ENV}"' in content
    assert "huggingface_hub==" in content
    assert "COPY --from=dashboard-builder" not in content
    assert "COPY --from=frontend-builder" not in content
    assert "COPY --from=wizmap-builder" not in content
    assert "COPY src/vllm-sr/start-dashboard.sh" not in content
    assert "COPY dashboard/backend/config/openclaw-skills.json" not in content
    assert "COPY src/training/model_eval/" not in content


def test_vllm_sr_rocm_dockerfile_stays_router_only() -> None:
    content = VLLM_SR_ROCM_DOCKERFILE.read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["/app/start-router.sh"]' in content
    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in content
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


def test_vllm_sr_rocm_dockerfile_uses_fully_qualified_base_images() -> None:
    """Podman with default short-name policy rejects unqualified base images;
    keep every FROM directive fully qualified so `make vllm-sr-dev
    VLLM_SR_PLATFORM=amd CONTAINER_RUNTIME=podman` works.
    """
    content = VLLM_SR_ROCM_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG RUST_RUNTIME_COMPAT_IMAGE=rustlang/rust:nightly-bullseye" in content
    assert "ARG ONNX_RUST_RUNTIME_COMPAT_IMAGE=library/rust:1.90-bullseye" in content
    assert "ARG GO_RUNTIME_COMPAT_IMAGE=library/golang:1.25-bookworm" in content
    assert (
        "FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${RUST_RUNTIME_COMPAT_IMAGE}"
        in content
    )
    assert "FROM ${IMAGE_REGISTRY}rocm/dev-ubuntu-22.04:7.0" in content


def test_vllm_sr_cuda_dockerfile_stays_router_only() -> None:
    content = VLLM_SR_CUDA_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04" in content
    assert "onnxruntime-gpu==1.22.0" in content
    assert "ENV AI_BINDING=onnx" in content
    assert 'ENTRYPOINT ["/app/start-router.sh"]' in content
    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in content
    assert (
        "COPY nlp-binding/go.mod nlp-binding/nlp_binding.go nlp-binding/nlp_binding_mock.go /build/../nlp-binding/"
        in content
    )
    assert (
        "COPY nlp-binding/go.mod nlp-binding/nlp_binding.go nlp-binding/nlp_binding_mock.go ./"
        not in content
    )
    assert "COPY --from=dashboard-builder" not in content
    assert "COPY --from=frontend-builder" not in content
    assert "COPY --from=wizmap-builder" not in content
    assert "COPY src/vllm-sr/start-dashboard.sh" not in content
    assert "COPY dashboard/backend/config/openclaw-skills.json" not in content
    assert "COPY src/training/model_eval/" not in content


def test_router_images_ship_the_management_api_runtime_sync_module() -> None:
    expected_copy = "COPY src/vllm-sr/cli/ /app/cli/"
    for dockerfile in (
        VLLM_SR_DOCKERFILE,
        VLLM_SR_ROCM_DOCKERFILE,
        VLLM_SR_CUDA_DOCKERFILE,
    ):
        content = dockerfile.read_text(encoding="utf-8")
        assert expected_copy in content, f"{dockerfile} omits runtime config sync"
        assert "pyyaml>=6.0.2" in content, f"{dockerfile} omits runtime YAML support"


def test_dashboard_runtime_image_binds_cli_version_metadata() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY src/vllm-sr/pyproject.toml /app/pyproject.toml" in content
    assert content.index("COPY src/vllm-sr/pyproject.toml /app/pyproject.toml") < (
        content.index("COPY src/vllm-sr/cli/ /app/cli/")
    )


def test_router_entrypoint_does_not_override_management_listener_config() -> None:
    content = (REPO_ROOT / "src" / "vllm-sr" / "start-router.sh").read_text()
    assert "-enable-api=true" in content
    assert "-api-port=" not in content
    assert "-api-bind=" not in content


def test_vllm_sr_cuda_dockerignore_excludes_runtime_state_and_large_unused_inputs() -> (
    None
):
    content = VLLM_SR_CUDA_DOCKERIGNORE.read_text(encoding="utf-8")

    assert "**/.vllm-sr/" in content
    assert "**/milvus-data/" in content
    assert "**/etcd/" in content
    assert "**/postgres-data/" in content
    assert "bench/" in content
    assert "slides/" in content


def test_extproc_dockerfile_copies_built_in_knowledge_bases() -> None:
    content = EXTPROC_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in content
    assert "COPY config/kb/ /app/config/kb/" not in content


def test_extproc_rocm_dockerfile_copies_built_in_knowledge_bases() -> None:
    content = EXTPROC_ROCM_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in content
    assert "COPY config/kb/ /app/config/kb/" not in content
