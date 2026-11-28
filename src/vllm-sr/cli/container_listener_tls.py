"""Docker mounts for the standalone Router's listener certificates and keys."""

import os
import posixpath
import re
import subprocess

from cli.utils import get_logger

log = get_logger(__name__)

LISTENER_TLS_PATHS_ENV = "VLLM_SR_LISTENER_TLS_PATHS"
_LISTENER_TLS_DIRS_ENV = "VLLM_SR_LISTENER_TLS_DIRS"
_MOUNT_ROOT = "/run/vllm-sr/listener-tls"
_IMAGE_LABEL = "io.vllm-sr.listener-tls-directory-mounts"
_IMAGE_ID = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")


def _listener_tls_file_paths(listeners, config_dir: str):
    for listener in listeners:
        tls = listener.get("tls") or {}
        for field in ("cert_file", "key_file"):
            path = str(tls.get(field) or "").strip()
            if not path:
                continue
            host_path = os.path.join(config_dir, path)
            if not os.path.isfile(host_path):
                raise ValueError(
                    f"listener '{listener.get('name', 'unknown')}': tls.{field} "
                    f"{host_path} is not a file"
                )
            yield path, os.path.abspath(host_path)


def _file_mount(path: str, host_path: str) -> str:
    container_path = os.path.join("/app", path) if not os.path.isabs(path) else path
    return f"{host_path}:{container_path}:ro,z"


def _listener_tls_mounts(listeners, config_dir: str) -> list[str]:
    """Validate file paths and produce the mounts understood by older images."""
    return [
        _file_mount(path, host_path)
        for path, host_path in _listener_tls_file_paths(listeners, config_dir)
    ]


def router_listener_tls_mounts(
    runtime: str, image: str, listeners, config_dir: str
) -> tuple[str, list[str], dict[str, str]]:
    files = list(_listener_tls_file_paths(listeners, config_dir))
    if not files:
        return image, [], {}

    allowed_dirs = {
        os.path.realpath(os.path.join(config_dir, os.path.expanduser(directory)))
        for directory in os.getenv(_LISTENER_TLS_DIRS_ENV, "").split(os.pathsep)
        if directory
    }
    if not allowed_dirs:
        return image, [_file_mount(path, host_path) for path, host_path in files], {}

    directories: dict[str, str] = {}
    mounts: list[str] = []
    paths: dict[str, str] = {}
    for path, host_path in files:
        parent = os.path.realpath(os.path.dirname(host_path))
        if parent not in allowed_dirs:
            mounts.append(_file_mount(path, host_path))
            continue
        # A file symlink can jump outside the bind mount. Preserve its old
        # mount behavior without adding more host directories to the container.
        if parent == os.path.dirname(parent) or os.path.islink(host_path):
            log.warning(
                "Listener TLS path %s uses a file mount. Restart the Docker "
                "Router after replacing it.",
                host_path,
            )
            mounts.append(_file_mount(path, host_path))
            continue
        target = directories.setdefault(parent, f"{_MOUNT_ROOT}/{len(directories)}")
        paths[path] = posixpath.join(target, os.path.basename(host_path))

    if not paths:
        return image, mounts, {}
    image_id = _directory_mount_image(runtime, image)
    if image_id is None:
        log.warning(
            "The Router image does not advertise listener TLS directory mounts. "
            "Restart the Docker Router after replacing a certificate or key."
        )
        return image, [_file_mount(path, host_path) for path, host_path in files], {}

    mounts.extend(f"{parent}:{target}:ro,z" for parent, target in directories.items())
    return image_id, mounts, paths


def _directory_mount_image(runtime: str, image: str) -> str | None:
    # Use the image ID that advertises support, so a concurrent tag update
    # cannot start an older binary with paths it does not recognize.
    capability = (
        '{{if eq (index .Config.Labels "' + _IMAGE_LABEL + '") "1"}}{{.Id}}{{end}}'
    )
    try:
        result = subprocess.run(
            [runtime, "image", "inspect", "--format", capability, image],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    image_id = result.stdout.strip()
    return (
        image_id if result.returncode == 0 and _IMAGE_ID.fullmatch(image_id) else None
    )
