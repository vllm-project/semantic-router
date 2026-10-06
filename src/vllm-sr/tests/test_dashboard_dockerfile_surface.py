import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

LINUX_PERMISSION_HELPER = pytest.mark.skipif(
    sys.platform != "linux",
    reason="The deployed permission helper requires Linux O_PATH and /proc/self/fd.",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"
DASHBOARD_ENTRYPOINT = REPO_ROOT / "dashboard" / "backend" / "entrypoint.sh"
DASHBOARD_PERMISSION_HELPER = (
    REPO_ROOT / "dashboard" / "backend" / "entrypoint_permissions.py"
)
DASHBOARD_LOGS_HANDLER = REPO_ROOT / "dashboard" / "backend" / "handlers" / "logs.go"
BUILT_IN_MODEL_ASSETS = (
    Path("latest/catalog.yaml"),
    Path("latest/mom-v1/README.md"),
    Path("latest/mom-v1/config.yaml"),
    Path("latest/mom-v1/metadata.yaml"),
    Path("latest/mom-v1/probes.yaml"),
    Path("latest/mom-v1/recipe.dsl"),
)


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
        "apt_get_install_with_retry ca-certificates curl docker.io git gosu libseccomp2 wget"
        in content
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


def test_dashboard_dockerfile_exposes_an_immutable_source_revision_build_arg() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG VLLM_SR_SOURCE_REVISION=unavailable" in content
    assert "ARG VLLM_SR_SOURCE_REVISION\n" in content
    assert "ENV VLLM_SR_SOURCE_REVISION=${VLLM_SR_SOURCE_REVISION}" in content


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
    assert "continuing without socket access" in content
    assert "LOG_SPOOL_GID=${VLLM_SR_LOG_SPOOL_GID:-}" in content
    assert 'add_nonroot_group_gid "$LOG_SPOOL_GID"' in content
    assert "Invalid log spool group" in content
    assert 'groupadd --gid "$GROUP_GID" "$GROUP_NAME"' in content
    assert 'usermod -aG "$GROUP_NAME" nonroot' in content
    assert 'if [ "$GROUP_GID" -eq 0 ]' in content
    assert "Refusing to add Dashboard user to the root group" in content
    assert "safe_shared_path_gid" not in content
    assert "STATE_GID=65532" in content
    assert "ENVOY_STATE_GID=65532" in content
    assert "RECIPE_STORE_GID=65532" in content
    assert "DATA_GID=65532" in content
    assert 'python3 "$PERMISSION_HELPER" prepare-tree' in content
    assert "--credential-relative-path credentials/router-management.token" in content
    # Historical private evidence is not reopened or permission-normalized.
    assert "--exclude-path /app/data/evaluation" in content
    assert "EVALUATION_DATA_DIR" not in content
    assert "prepare-private-tree" not in content
    assert 'python3 "$PERMISSION_HELPER" prepare-file "$CONFIG_FILE_PATH"' in content
    assert (
        'python3 "$PERMISSION_HELPER" probe-config "$STATE_DIR" "$CONFIG_FILE_PATH"'
        in content
    )
    assert '[ ! -f "$CONFIG_FILE_PATH" ] || [ ! -d "$STATE_DIR" ]' in content
    assert (
        "RECIPE_STORE_DIR=${VLLM_SR_RECIPE_STORE_DIR:-${STATE_DIR}/.vllm-sr/recipe-store}"
        in content
    )
    assert (
        'python3 "$PERMISSION_HELPER" ensure-directory "$RECIPE_STORE_DIR"' in content
    )
    assert "DASHBOARD_RUNTIME_CONFIG_WRITABLE=$RUNTIME_CONFIG_WRITABLE" in content
    assert "DASHBOARD_RECIPE_STORE_WRITABLE=$RECIPE_STORE_WRITABLE" in content
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


@LINUX_PERMISSION_HELPER
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


@LINUX_PERMISSION_HELPER
def test_dashboard_permission_helper_creates_recipe_store_under_writable_parent(
    tmp_path: Path,
) -> None:
    store = tmp_path / "recipe-store"

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "ensure-directory",
            str(store),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert store.is_dir()


def test_dashboard_permission_helper_rejects_recipe_store_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    store = tmp_path / "recipe-store"
    store.symlink_to(target, target_is_directory=True)

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "ensure-directory",
            str(store),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


@LINUX_PERMISSION_HELPER
def test_dashboard_permission_helper_preserves_private_management_token(
    tmp_path: Path,
) -> None:
    store = tmp_path / "recipe-store"
    credentials = store / "credentials"
    credentials.mkdir(parents=True)
    token = credentials / "router-management.token"
    token.write_text("a" * 64, encoding="utf-8")
    token.chmod(0o600)
    record = store / "active.json"
    record.write_text("{}", encoding="utf-8")
    record.chmod(0o600)

    subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-tree",
            str(store),
            str(os.getgid()),
            "--credential-relative-path",
            "credentials/router-management.token",
            "--credential-uid",
            str(os.getuid()),
            "--credential-gid",
            str(os.getgid()),
        ],
        check=True,
    )

    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    assert stat.S_IMODE(record.stat().st_mode) == 0o660
    assert stat.S_IMODE(credentials.stat().st_mode) & stat.S_ISGID


@LINUX_PERMISSION_HELPER
def test_dashboard_permission_helper_probes_recipe_store_without_residue(
    tmp_path: Path,
) -> None:
    store = tmp_path / "recipe-store"
    store.mkdir()

    subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "probe-directory",
            str(store),
        ],
        check=True,
    )

    assert list(store.iterdir()) == []

    linked_store = tmp_path / "linked-store"
    linked_store.symlink_to(store, target_is_directory=True)
    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "probe-directory",
            str(linked_store),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert list(store.iterdir()) == []


def test_dashboard_permission_helper_rejects_symlinked_shared_tree_entry(
    tmp_path: Path,
) -> None:
    store = tmp_path / "recipe-store"
    store.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("sentinel", encoding="utf-8")
    outside.chmod(0o600)
    (store / "trap").symlink_to(outside)

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-tree",
            str(store),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


@LINUX_PERMISSION_HELPER
def test_dashboard_permission_helper_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    store = tmp_path / "recipe-store"
    store.mkdir()
    os.mkfifo(store / "trap")

    result = subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-tree",
            str(store),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )

    assert result.returncode != 0
    assert "unsafe file" in result.stderr


@LINUX_PERMISSION_HELPER
def test_dashboard_permission_helper_leaves_historical_evidence_untouched(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "data"
    evaluation = shared / "evaluation"
    ordinary = shared / "ordinary"
    evaluation.mkdir(parents=True)
    ordinary.mkdir()
    evidence = evaluation / "report.json"
    evidence.write_text("{}", encoding="utf-8")
    shared_file = ordinary / "state.json"
    shared_file.write_text("{}", encoding="utf-8")
    evaluation.chmod(0o700)
    evidence.chmod(0o600)
    ordinary.chmod(0o700)
    shared_file.chmod(0o600)

    subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_PERMISSION_HELPER),
            "prepare-tree",
            str(shared),
            str(os.getgid()),
            "--exclude-path",
            str(evaluation),
        ],
        check=True,
    )

    assert stat.S_IMODE(evaluation.stat().st_mode) == 0o700
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o600
    assert stat.S_IMODE(ordinary.stat().st_mode) & 0o070 == 0o070
    assert stat.S_IMODE(shared_file.stat().st_mode) & 0o060 == 0o060


def test_dashboard_dockerfile_copies_router_dsl_package_for_backend_builds() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "COPY src/semantic-router/pkg/dsl/ /app/src/semantic-router/pkg/dsl/" in content
    )
    assert (
        "COPY src/semantic-router/pkg/routerreplay/redaction/ "
        "/app/src/semantic-router/pkg/routerreplay/redaction/" in content
    )


def test_dashboard_dockerfile_ships_sr_bench_service_without_legacy_model_eval() -> (
    None
):
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY src/vllm-sr/cli/ /app/cli/" in content
    assert (REPO_ROOT / "src/vllm-sr/cli/sr_bench/service.py").is_file()
    assert not (REPO_ROOT / "src/vllm-sr/cli/evaluation").exists()
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/requirements.txt'
        in content
    )
    assert "COPY src/training/model_eval/" not in content
    assert "TORCH_CPU" not in content
    assert '"torch==' not in content


def test_dashboard_image_binds_the_generated_model_catalog() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY src/vllm-sr/cli/ /app/cli/" in content
    assert "COPY config/recipes/built-in/ /app/cli/model_assets/" in content

    source_root = REPO_ROOT / "config" / "recipes" / "built-in"
    source_assets = {
        path.relative_to(source_root)
        for path in (source_root / "latest").rglob("*")
        if path.is_file()
    }
    assert set(BUILT_IN_MODEL_ASSETS) <= source_assets
    tracked_package_assets = subprocess.run(
        ["git", "ls-files", "src/vllm-sr/cli/model_assets"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked_package_assets == ["src/vllm-sr/cli/model_assets/__init__.py"]


def test_dashboard_runtime_image_binds_cli_version_metadata() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    # Build contexts may have a restrictive umask and mode-0600 source files.
    # Explicit COPY permissions make version imports readable after gosu.
    version_copy = "COPY --chmod=0444 src/vllm-sr/pyproject.toml /app/pyproject.toml"
    assert version_copy in content
    assert content.index(version_copy) < (
        content.index("COPY src/vllm-sr/cli/ /app/cli/")
    )
    nonroot_catalog_check = (
        "RUN cd /tmp && gosu nonroot python3 -m cli.model_catalog_export >/dev/null"
    )
    assert nonroot_catalog_check in content
    assert content.index(nonroot_catalog_check) > content.index(
        "find /app/cli -type f -exec chmod 0444 {} +"
    )
