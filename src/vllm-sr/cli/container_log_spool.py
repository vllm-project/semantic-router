"""Bounded, stack-scoped log spooling for the local split runtime."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

LOG_SPOOL_COMPONENTS = ("router", "envoy", "dashboard")
LOG_SPOOL_MAX_BYTES = 4 * 1024 * 1024
LOG_SPOOL_MAX_LINE_BYTES = 64 * 1024
LOG_SPOOL_READER_DIR = "/var/log/vllm-sr"
LOG_SPOOL_PRODUCER_FILE = "/var/log/vllm-sr-producer/current.log"
LOG_SPOOL_ROOT_ENV = "VLLM_SR_LOG_SPOOL_DIR"
LOG_SPOOL_GID_ENV = "VLLM_SR_LOG_SPOOL_GID"
LOG_SPOOL_REQUIRED_TOOLS = ("awk", "cat", "mkfifo", "rm", "tail", "wc")


# The producer wrapper keeps normal container stdout/stderr while retaining a
# bounded copy in the shared spool. It stores at most 64 KiB from any one log
# record and copy-truncates back to 75% whenever the spool reaches its limit.
# Paths and limits are supplied only by this module, never by dashboard input.
BOUNDED_LOG_SPOOL_SCRIPT = r"""
set -eu
spool_file=$1
max_bytes=$2
max_line_bytes=$3
shift 3
if [ "${1:-}" != "--" ]; then
    echo "bounded log spool: missing command separator" >&2
    exit 64
fi
shift
if [ "$#" -eq 0 ]; then
    echo "bounded log spool: missing service command" >&2
    exit 64
fi
case "$spool_file" in
    /var/log/vllm-sr-producer/*.log) ;;
    *) echo "bounded log spool: invalid producer path" >&2; exit 64 ;;
esac
case "$max_bytes:$max_line_bytes" in
    *[!0-9:]*|:*|*:) echo "bounded log spool: invalid size limit" >&2; exit 64 ;;
esac
if [ ! -f "$spool_file" ] || [ -L "$spool_file" ]; then
    echo "bounded log spool: producer file is unavailable or unsafe" >&2
    exit 73
fi
for required_tool in awk cat mkfifo rm tail wc; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        echo "bounded log spool: required runtime tool is unavailable: $required_tool" >&2
        exit 69
    fi
done

umask 077
keep_bytes=$((max_bytes * 3 / 4))
trim_file=/tmp/vllm-sr-log-spool-trim.$$
fifo=/tmp/vllm-sr-log-spool.$$.fifo
child_pid=
relay_pid=

cleanup() {
    if [ -n "$relay_pid" ]; then kill "$relay_pid" 2>/dev/null || true; fi
    if [ -n "$child_pid" ]; then kill "$child_pid" 2>/dev/null || true; fi
    rm -f "$fifo" "$trim_file"
}
forward_signal() {
    if [ -n "$child_pid" ]; then kill -TERM "$child_pid" 2>/dev/null || true; fi
}
trap forward_signal HUP INT TERM
trap cleanup EXIT

current_bytes=$(wc -c < "$spool_file")
if [ "$current_bytes" -gt "$max_bytes" ]; then
    tail -c "$keep_bytes" "$spool_file" > "$trim_file"
    cat "$trim_file" > "$spool_file"
    rm -f "$trim_file"
    current_bytes=$(wc -c < "$spool_file")
fi

mkfifo -m 600 "$fifo"
"$@" > "$fifo" 2>&1 &
child_pid=$!

LC_ALL=C awk \
    -v file="$spool_file" \
    -v max_bytes="$max_bytes" \
    -v keep_bytes="$keep_bytes" \
    -v max_line_bytes="$max_line_bytes" \
    -v current_bytes="$current_bytes" \
    -v trim_file="$trim_file" '
function rotate(    command) {
    close(file)
    command = "tail -c " keep_bytes " \"" file "\" > \"" trim_file "\" && cat \"" trim_file "\" > \"" file "\" && rm -f \"" trim_file "\""
    if (system(command) != 0) exit 74
    current_bytes = keep_bytes
}
{
    print $0
    fflush()
    stored = $0
    if (length(stored) > max_line_bytes) {
        marker = " ... [log record truncated]"
        prefix_bytes = max_line_bytes - length(marker)
        if (prefix_bytes < 0) prefix_bytes = 0
        stored = substr(stored, 1, prefix_bytes) marker
    }
    print stored >> file
    close(file)
    current_bytes += length(stored) + 1
    if (current_bytes > max_bytes) rotate()
}
' < "$fifo" &
relay_pid=$!

set +e
wait "$child_pid"
status=$?
child_pid=
wait "$relay_pid"
relay_status=$?
relay_pid=
set -e
if [ "$status" -eq 0 ] && [ "$relay_status" -ne 0 ]; then status=$relay_status; fi
exit "$status"
""".strip()


@dataclass(frozen=True)
class RuntimeLogSpool:
    """Host and container paths for one local runtime stack's log spool."""

    root: Path
    gid: int

    def host_file(self, component: str) -> Path:
        if component not in LOG_SPOOL_COMPONENTS:
            raise ValueError(f"unsupported log spool component: {component}")
        return self.root / f"{component}.log"

    def producer_mount(self, component: str) -> str:
        return f"{self.host_file(component)}:{LOG_SPOOL_PRODUCER_FILE}:z"

    def reader_mount(self) -> str:
        return f"{self.root}:{LOG_SPOOL_READER_DIR}:ro,z"


def prepare_runtime_log_spool(vllm_sr_dir: str, stack_name: str) -> RuntimeLogSpool:
    """Create fixed regular files without following pre-existing symlinks."""

    logs_root = Path(vllm_sr_dir) / "logs"
    root = logs_root / stack_name
    # GID 0 is deliberately never propagated into a container. A root-run CLI
    # can assign the dashboard's established nonroot group instead.
    gid = 65532 if os.getgid() == 0 else os.getgid()
    _ensure_directory(logs_root, 0o750, gid)
    _ensure_directory(root, 0o750, gid)

    for component in LOG_SPOOL_COMPONENTS:
        # Router and Dashboard wrappers start as root. Envoy may run as a
        # non-root image user, so only its producer file needs group write.
        mode = 0o660 if component == "envoy" else 0o640
        _ensure_regular_file(root / f"{component}.log", mode, gid)

    return RuntimeLogSpool(root=root, gid=gid)


def bounded_log_spool_entrypoint(
    service_entrypoint: str,
    service_args: list[str],
) -> tuple[str, list[str]]:
    """Wrap a service command with the portable bounded spool relay."""

    return (
        "/bin/sh",
        [
            "-c",
            BOUNDED_LOG_SPOOL_SCRIPT,
            "vllm-sr-log-spool",
            LOG_SPOOL_PRODUCER_FILE,
            str(LOG_SPOOL_MAX_BYTES),
            str(LOG_SPOOL_MAX_LINE_BYTES),
            "--",
            service_entrypoint,
            *service_args,
        ],
    )


def _ensure_directory(path: Path, mode: int, gid: int) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=mode)
        info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"log spool path is not a safe directory: {path}")
    os.chown(path, -1, gid)
    path.chmod(mode)


def _ensure_regular_file(path: Path, mode: int, gid: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise ValueError(f"log spool file is unavailable or unsafe: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"log spool path is not a private regular file: {path}")
        os.fchown(descriptor, -1, gid)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
