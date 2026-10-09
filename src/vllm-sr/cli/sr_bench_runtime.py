"""Independent sr-bench service wiring for the local runtime stack."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import sqlite3
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from cli.container_services import _is_port_in_use
from cli.runtime_env_names import runtime_env_name_is_allowed
from cli.runtime_stack import BENCH_PORT_ENV, PORT_OFFSET_ENV, RuntimeStackLayout

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
    # The service owns gated source downloads; never send this credential to the
    # browser or include its value in preparation requests or job receipts.
    if environment.get("HF_TOKEN"):
        values["HF_TOKEN"] = environment["HF_TOKEN"]
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


class BenchWorkerKeptError(ValueError):
    """The existing sr-bench worker stays as it is; the rest of the stack starts."""


def reconcile_bench_container(
    command: list[str], container_name: str, credentials: dict[str, str]
) -> str | None:
    """Reuse a matching worker, replace an owned idle one, or keep it running.

    Identity covers every launch argument and credential, so only an identical
    worker is reused. Any other change (image, host alias, port, credential or
    store) replaces the worker once its own journal shows no active work.
    A worker that cannot be replaced safely is kept and reported with
    ``BenchWorkerKeptError``: sr-bench is independent of the Router and Dashboard,
    and refusing it must not leave the stack down. A worker that never started
    is removed whatever its identity: it holds no ledger state, and left in
    place its name would fail every later serve.
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
        return None
    result = subprocess.run(
        [
            command[0],
            "inspect",
            "--format",
            '{"id":{{json .Id}},"status":{{json .State.Status}},'
            '"labels":{{json .Config.Labels}},"args":{{json .Args}}}',
            container_name,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    try:
        existing = json.loads(result.stdout)
        status = existing["status"]
        labels = existing["labels"]
    except (ValueError, KeyError, TypeError):
        return None
    if status == "created" and existing.get("id"):
        subprocess.run(
            [command[0], "rm", existing["id"]],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        host_port = int(command[command.index("-p") + 1].rsplit(":", 2)[-2])
        if _is_port_in_use(host_port):
            raise ValueError(
                f"sr-bench port {host_port} is already in use, so {container_name} "
                f"could not start. Stop the process using port {host_port}, set "
                f"{BENCH_PORT_ENV} for both serve and benchmark, or set a different "
                f"{PORT_OFFSET_ENV}."
            )
        return None
    if status != "running":
        raise BenchWorkerKeptError(
            f"The sr-bench service {container_name} is stopped; inspect its saved "
            "ledger before an explicit service restart. Benchmarks stay "
            "unavailable until then."
        )
    if isinstance(labels, dict) and labels.get(BENCH_IDENTITY_LABEL) == expected:
        return "reuse"
    store = _running_worker_store(existing)
    if store is None or not existing.get("id"):
        raise BenchWorkerKeptError(
            f"{container_name} is not an sr-bench worker this CLI started; it was "
            "left unchanged. Remove it to let serve start a new worker."
        )
    image = command[command.index("cli.sr_bench.service") - 2]
    subprocess.run(
        [command[0], "image", "inspect", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    try:
        _remove_idle_bench_container(command[0], existing["id"], store)
    except ValueError as exc:
        raise BenchWorkerKeptError(
            f"{exc}. The running sr-bench worker was kept with its previous "
            "settings; run serve again once it is idle to update it."
        ) from exc
    return "replace"


def _running_worker_store(existing: dict) -> Path | None:
    """The store a labelled worker was launched with, read from its arguments."""
    labels = existing.get("labels")
    args = existing.get("args")
    if not isinstance(labels, dict) or BENCH_IDENTITY_LABEL not in labels:
        return None
    if not isinstance(args, list) or "cli.sr_bench.service" not in args:
        return None
    try:
        store = args[args.index("--store") + 1]
    except (ValueError, IndexError):
        return None
    if not isinstance(store, str) or not os.path.isabs(store):
        return None
    return Path(store)


def _remove_idle_bench_container(runtime: str, container_id: str, store: Path) -> None:
    # Freeze admission before checking the durable journal. Checking /runs and
    # then stopping would race with a new request (and may miss other owners).
    try:
        subprocess.run(
            [runtime, "pause", container_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        # A client timeout does not establish whether the daemon paused it.
        subprocess.run(
            [runtime, "unpause", container_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        raise
    removed = False
    try:
        try:
            with sqlite3.connect(
                (store / "journal.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=1
            ) as db:
                busy = db.execute(
                    "SELECT 1 FROM runs WHERE status NOT IN "
                    "('completed','failed','cancelled','interrupted') LIMIT 1"
                ).fetchone()
        except (OSError, sqlite3.Error) as exc:
            raise ValueError(
                "Cannot verify the sr-bench journal; leaving its worker unchanged"
            ) from exc
        if busy:
            raise ValueError(
                "The sr-bench service has active runs; finish or cancel them before it is replaced"
            )
        # Preparation workers share this service lifecycle but have their own
        # durable journal. Inspect it while admission and its children are frozen.
        from cli.sr_bench.preparations import ACTIVE, JOB_ID  # noqa: PLC0415

        try:
            preparations = store / "dataset-preparations"
            if preparations.is_symlink() or (
                preparations.exists() and not preparations.is_dir()
            ):
                raise ValueError("Invalid preparation journal directory")
            preparing = False
            if preparations.exists():
                for path in preparations.iterdir():
                    if not path.name.startswith("prep-") or path.suffix != ".json":
                        continue
                    if path.is_symlink() or not JOB_ID.fullmatch(path.stem):
                        raise ValueError("Invalid preparation journal file")
                    job = json.loads(path.read_text())
                    if (
                        not isinstance(job, dict)
                        or job.get("id") != path.stem
                        or job.get("status") not in ACTIVE | {"completed", "failed"}
                    ):
                        raise ValueError("Invalid preparation journal record")
                    if job["status"] in ACTIVE:
                        preparing = True
                        break
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                "Cannot verify dataset preparations; leaving the sr-bench worker unchanged"
            ) from exc
        if preparing:
            raise ValueError(
                "The sr-bench service has active dataset preparations; wait for them to finish before it is replaced"
            )
        # No benchmark can be dispatched after the idle check while frozen.
        # SQLite's durable journal survives removal; only the owned ID is removed.
        subprocess.run(
            [runtime, "rm", "--force", container_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        removed = True
    finally:
        if not removed:
            subprocess.run(
                [runtime, "unpause", container_id],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
