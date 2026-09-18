"""Independent sr-bench service wiring for the local runtime stack."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from cli.runtime_env_names import runtime_env_name_is_allowed
from cli.runtime_stack import RuntimeStackLayout

BENCH_CONFIG_ENV = ("SR_BENCH_URL", "SR_BENCH_TOKEN_ENV", "SR_BENCH_STORE")
BENCH_TOKEN_ENV = "SR_BENCH_TOKEN"
BENCH_IDENTITY_LABEL = "io.vllm-sr.sr-bench.identity"
MIN_SERVICE_TOKEN_CHARS = 32


@dataclass(frozen=True)
class BenchRuntime:
    origin: str
    token_env: str
    secrets: dict[str, str]
    store: Path | None = None

    @property
    def managed(self) -> bool:
        return self.store is not None


def _private_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise ValueError("sr-bench credential directory must not be a symlink")
    os.chmod(path.parent, 0o700)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("sr-bench credential must be a regular private file")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8", closefd=False) as stream:
            token = stream.read().strip()
            if not token:
                token = secrets.token_urlsafe(48)
                stream.write(token + "\n")
                stream.flush()
                os.fsync(fd)
            if len(token) < MIN_SERVICE_TOKEN_CHARS:
                raise ValueError(
                    "sr-bench service token must contain at least 32 characters"
                )
            return token
    finally:
        os.close(fd)


def _target_secret_refs(store: Path) -> set[str]:
    refs: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "api_key_env":
                    if not isinstance(item, str) or not runtime_env_name_is_allowed(
                        item
                    ):
                        raise ValueError(
                            "sr-bench credential references must be safe environment names"
                        )
                    refs.add(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for name in ("targets.json", "benchmark-options.json"):
        path = store / name
        if path.exists():
            visit(json.loads(path.read_text()))
    return refs


def prepare_bench_runtime(
    config_dir: str, stack: RuntimeStackLayout, host_env=None
) -> BenchRuntime:
    environment = os.environ if host_env is None else host_env
    origin = environment.get("SR_BENCH_URL", "")
    token_ref = environment.get("SR_BENCH_TOKEN_ENV", BENCH_TOKEN_ENV)
    if not runtime_env_name_is_allowed(token_ref) or token_ref in BENCH_CONFIG_ENV:
        raise ValueError("SR_BENCH_TOKEN_ENV must name a safe environment variable")
    if origin:
        parsed = urlparse(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "SR_BENCH_URL must be an HTTP(S) origin without credentials or a path"
            )
        token = environment.get(token_ref, "")
        if not token:
            raise ValueError("The configured sr-bench service token is missing")
        return BenchRuntime(origin.rstrip("/"), token_ref, {token_ref: token})

    root = Path(config_dir).resolve() / ".sr-bench" / stack.stack_name
    store = (
        Path(environment.get("SR_BENCH_STORE", str(root / "store")))
        .expanduser()
        .absolute()
    )
    if store.is_symlink():
        raise ValueError("SR_BENCH_STORE must not be a symlink")
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = store.resolve()
    os.chmod(store, 0o700)
    token = environment.get(token_ref) or _private_token(root / "service-token")
    values = {token_ref: token}
    for ref in _target_secret_refs(store):
        value = environment.get(ref)
        if not value:
            raise ValueError(
                f"Missing sr-bench target credential environment variable: {ref}"
            )
        values[ref] = value
    return BenchRuntime(
        f"http://{stack.sr_bench_container_name}:8090", token_ref, values, store
    )


def dashboard_bench_env(runtime: BenchRuntime) -> dict[str, str]:
    return {
        "SR_BENCH_URL": runtime.origin,
        "SR_BENCH_TOKEN_ENV": runtime.token_env,
        runtime.token_env: "",
    }


def bench_command_identity(command: list[str], credentials: dict[str, str]) -> str:
    # The hash binds reuse to the same command and private credential values;
    # neither the token nor credentials appear in Docker's command arguments.
    return hashlib.sha256(
        json.dumps(
            [command, credentials], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def reuse_bench_container(command: list[str], container_name: str) -> bool:
    """Reuse a matching running worker after Docker reports a name conflict.

    A configuration reload must not tear down the owner of in-flight requests.
    Unknown or stopped workers require explicit reconciliation, not a restart.
    """
    expected = next(
        (
            arg.split("=", 1)[1]
            for arg in command
            if arg.startswith(BENCH_IDENTITY_LABEL + "=")
        ),
        None,
    )
    if expected is None:
        return False
    result = subprocess.run(
        [
            command[0],
            "inspect",
            "--format",
            "{{json .State.Status}} {{json .Config.Labels}}",
            container_name,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return False
    decoder = json.JSONDecoder()
    try:
        status, end = decoder.raw_decode(result.stdout.strip())
        labels = json.loads(result.stdout.strip()[end:].strip())
    except (ValueError, TypeError):
        return False
    if status != "running":
        raise ValueError(
            "The sr-bench service is stopped; inspect its saved ledger before an explicit service restart"
        )
    if not isinstance(labels, dict) or labels.get(BENCH_IDENTITY_LABEL) != expected:
        raise ValueError(
            "The running sr-bench service has a different image, store or credential; reconcile it before replacing the service"
        )
    return True
