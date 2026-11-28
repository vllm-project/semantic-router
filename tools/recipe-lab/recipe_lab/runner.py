"""Replay-tier runner: evaluate candidate configs against the answer bank.

Restart mode (the correct-but-slow v1): one candidate = one router
lifecycle (recipe-lab design, Sub-issue 2). The runner owns no evaluation
logic of its own:

1. start the mock/real backend and the router with a per-trial port-patched
   copy of the candidate config (-gateway standalone, so the OpenAI listener
   is served without Envoy);
2. wait for the decision model to report ready via the management API;
3. send every labeled case through the router (requests only trigger
   routing; correctness always comes from the answer bank);
4. pull the replay records, join with cases, grade from the bank,
   classify and summarize;
5. stop the processes (kill the process group so the spawned model runtime
   dies with the router) and return the summary.

Hot mode (H, `reload_mode="hot"`): one `HotReloadSession` keeps a single
router + backend alive and swaps candidate configs through the management
API, so a candidate costs seconds instead of a model-runtime restart.

Verified against the router source (H prerequisites):

- `PUT /api/v1/config` (JSON body `{"yaml": "<complete config YAML>"}`,
  pkg/apiserver/route_router_config_update.go `handleConfigPut`) validates
  hot-reload compatibility first and answers HTTP 409 RESTART_REQUIRED for
  incompatible documents. In standalone gateway mode those are listener
  changes, tracing changes, and routing-preview concurrency changes; the
  local-classifier check validates the *next* document itself
  (pkg/config/validator_classifier_signal.go), so threshold-only candidates
  are hot-reloadable.
- The server already waits (up to 20s, `waitForRuntimeConfigActivation`)
  for the runtime to publish the matching document hash and reports the
  result in the response's `activation_status`: "active" means the new
  router + classification service are atomically available; "pending" (HTTP
  202) means poll `GET /api/v1/config/hash` until `activation_status` is
  "active"; "failed" means runtime activation failed. No fixed sleep is
  used on the client side.
- A PUT persists the new document to the router's own `-config` source
  path, so the session boots the router from a scratch copy in the work
  directory and candidate files stay pristine.
- Every replay-tier request carries a candidate-scoped `x-request-id`
  (`rl-<tag>-<case_id>`), which the record's `request_id` keeps verbatim
  (see adapters/router_replay.py); the shared replay store has no
  delete/clear endpoint, so records are separated per candidate by that id.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .adapters.router_replay import (
    CaseJoin,
    fetch_replays,
    records_to_cases,
    records_to_cases_by_request_id,
)
from .attribution import CaseInput, attribute, summarize
from .bank import apply_bank
from .contract import ContractError, OraclePolicy
from .verdict import resolve_constraint


class RunnerError(RuntimeError):
    """Raised when a candidate evaluation could not complete."""


class RunnerTimeoutError(RunnerError):
    """The whole trial (readiness + requests + replay fetch) exceeded the
    budget's per-trial deadline. Recorded as a `status: timeout` ledger row
    instead of a metric so a hung stack can never rank as a worse recipe."""


class HotReloadRejectedError(RunnerError):
    """The management API refused to hot-reload a candidate (HTTP 409
    RESTART_REQUIRED or a failed activation). The caller falls back to the
    restart path for that candidate."""


REQUEST_ID_PREFIX = "rl-"


@dataclass
class RunnerEnv:
    """Process-level environment for one candidate evaluation.

    Gateway and management ports are allocated per trial and patched into a
    copy of the candidate config (see _patch_ports), so they can never collide
    with an unrelated process or a stale router on the defaults. The mock
    backend keeps an explicit port: configs that point at it must reference
    the same one.
    """

    router_bin: str
    mock_backend_script: str
    python_bin: str
    work_dir: Path
    backend_port: int = 8000
    ready_timeout_s: int = 300
    hf_home: str | None = None
    env_extra: dict[str, str] = field(default_factory=dict)
    mock_extra_args: tuple[str, ...] = field(default_factory=tuple)
    # H: "restart" (one router lifecycle per candidate) or "hot" (one
    # HotReloadSession swaps configs through PUT /api/v1/config).
    reload_mode: str = "restart"
    # H: env var name holding the management API bearer token; reused for the
    # config PUT and the replay fetch (same contract as collect-replays).
    api_key_env: str | None = None


def _http_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _free_port() -> int:
    """Reserve an ephemeral TCP port (bind-and-release race is negligible)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _patch_ports(candidate_config: Path, work_dir: Path) -> tuple[Path, int, int]:
    """Copy the candidate config into work_dir with per-trial ports patched in.

    The router binds its gateway listener(s) from the config, so the fixed
    default (8899) collides with anything already on that port - and worse, a
    stale router on the default ports answers the readiness probe for a router
    that never started. The management API port is passed to the router via
    -api-port, which also works on binaries whose config schema predates the
    management_api section. Returns (patched config path, primary gateway
    port, management port).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
    router_port = 0
    for listener in config.get("listeners") or []:
        listener["port"] = _free_port()
        if not router_port:
            router_port = listener["port"]
    if not router_port:
        raise RunnerError(
            f"candidate config has no gateway listeners: {candidate_config}"
        )
    management_port = _free_port()
    patched = work_dir / f"{candidate_config.name}.patched.yaml"
    patched.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return patched, router_port, management_port


def _apply_session_ports(yaml_text: str, listener_ports: list[int]) -> str:
    """Copy a candidate config onto the hot session's per-boot gateway ports.

    The session's router booted from a _patch_ports copy of the boot config,
    so a candidate PUT must carry the same listener ports or the router
    would rebind the gateway away from the session's router_url mid-run.
    """
    config = yaml.safe_load(yaml_text)
    if not isinstance(config, dict):
        raise RunnerError("candidate config is not a mapping")
    listeners = config.get("listeners") or []
    if len(listeners) != len(listener_ports):
        raise RunnerError(
            f"candidate has {len(listeners)} gateway listeners, boot config had {len(listener_ports)}; "
            "hot reload requires the same listener layout across candidates"
        )
    for listener, port in zip(listeners, listener_ports, strict=False):
        listener["port"] = port
    return yaml.safe_dump(config, sort_keys=False)


def _wait_model_runtime_ready(
    management_url: str,
    timeout_s: int,
    process: subprocess.Popen,
    status_file: Path | None = None,
) -> None:
    """Wait for the model runtime to report ready.

    Primary source: the router's own startup-status file (router-runtime.json
    next to the config; State.ready == true, pkg/startupstatus). Fallback: a
    loose '"ready":true' probe on the inventory endpoint (verify-models.sh
    style), which survives inventory payload-shape drift.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(f"router exited early with code {process.returncode}")
        if status_file is not None and status_file.is_file():
            try:
                state = json.loads(status_file.read_text(encoding="utf-8"))
                if state.get("ready") is True:
                    return
            except (json.JSONDecodeError, OSError):
                pass  # mid-write or partial state; keep polling
        try:
            with urllib.request.urlopen(
                management_url + "/api/v1/inventory/model-runtime", timeout=10.0
            ) as response:
                if '"ready":true' in response.read().decode("utf-8", errors="replace"):
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise RunnerError(f"model runtime not ready within {timeout_s}s")


def _send_prompt(
    router_url: str,
    model: str,
    prompt: str,
    timeout: float,
    request_id: str | None = None,
) -> int:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if request_id:
        # H: candidate-scoped id; the replay record keeps it verbatim so one
        # candidate's records can be pulled out of the shared store.
        headers["x-request-id"] = request_id
    request = urllib.request.Request(
        router_url + "/v1/chat/completions",
        data=body,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def evaluate_candidate(
    *,
    candidate_config: str | Path,
    cases: list[dict[str, Any]],
    oracle_policy: OraclePolicy,
    bank: dict[tuple[str, str], Any],
    env: RunnerEnv,
    router_model: str = "vllm-sr/auto",
    family: str = "d3",
    request_timeout: float = 180.0,
    trial_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate config; returns rows + summary + request log.

    trial_timeout_s (A5, budget.max_minutes_per_trial * 60) bounds the whole
    trial: readiness wait + request loop + replay fetch. Exceeding it raises
    RunnerTimeoutError; processes are always cleaned up in `finally`.
    """
    env.work_dir.mkdir(parents=True, exist_ok=True)
    patched_config, router_port, management_port = _patch_ports(
        Path(candidate_config), env.work_dir
    )
    management_url = f"http://127.0.0.1:{management_port}"
    router_url = f"http://127.0.0.1:{router_port}"
    deadline = (
        time.monotonic() + trial_timeout_s if trial_timeout_s is not None else None
    )

    def remaining_s() -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    def check_deadline(stage: str) -> None:
        left = remaining_s()
        if left is not None and left <= 0:
            raise RunnerTimeoutError(
                f"trial exceeded {trial_timeout_s:.0f}s budget before {stage}"
            )

    procs: list[subprocess.Popen] = []
    timings = {"switch_s": 0.0, "request_s": 0.0, "fetch_s": 0.0}
    try:
        # the router writes startup state (ready flag) next to its config file;
        # clear the previous candidate's state so we only see fresh readiness
        status_file = patched_config.parent / "router-runtime.json"
        if status_file.exists():
            status_file.unlink(missing_ok=True)

        # backend first: the router probes it during startup. Each child gets
        # its own session/process group, so the group-kill in `finally` cannot
        # reach the scheduler task this runner lives in.
        switch_started = time.monotonic()
        mock = subprocess.Popen(
            [
                env.python_bin,
                env.mock_backend_script,
                "--port",
                str(env.backend_port),
                *env.mock_extra_args,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        procs.append(mock)
        time.sleep(1.0)

        router_env = dict(os.environ)
        if env.hf_home:
            router_env["HF_HOME"] = env.hf_home
        router_env.update(env.env_extra)
        router = subprocess.Popen(
            [
                env.router_bin,
                "-config",
                str(patched_config),
                "-gateway",
                "standalone",
                "-api-port",
                str(management_port),
            ],
            # The handle is owned by the spawned router process and dies with its group.
            stdout=open(  # noqa: SIM115
                Path(env.work_dir) / "router.log", "w", encoding="utf-8"
            ),
            stderr=subprocess.STDOUT,
            env=router_env,
            start_new_session=True,  # the spawned model runtime dies with the group
        )
        procs.append(router)
        # readiness wait is bounded by the per-trial deadline too; a plain
        # not-ready error still applies when the deadline is not the cause
        ready_budget = env.ready_timeout_s
        left = remaining_s()
        if left is not None:
            if left <= 0:
                raise RunnerTimeoutError(
                    f"trial exceeded {trial_timeout_s:.0f}s budget before model-runtime readiness"
                )
            ready_budget = min(env.ready_timeout_s, int(left) + 1)
        try:
            _wait_model_runtime_ready(
                management_url, ready_budget, router, status_file=status_file
            )
        except RunnerError as exc:
            left = remaining_s()
            if "not ready within" in str(exc) and left is not None and left <= 0:
                raise RunnerTimeoutError(
                    f"trial exceeded {trial_timeout_s:.0f}s budget waiting for model-runtime readiness"
                ) from exc
            raise
        timings["switch_s"] = time.monotonic() - switch_started

        statuses: dict[str, int] = {}
        request_started = time.monotonic()
        for case in cases:
            check_deadline(f"case {case['case_id']}")
            statuses[case["case_id"]] = _send_prompt(
                router_url, router_model, case["prompt"], request_timeout
            )
        timings["request_s"] = time.monotonic() - request_started

        check_deadline("replay fetch")
        fetch_started = time.monotonic()
        records = fetch_replays(management_url, api_key_env=env.api_key_env)
        timings["fetch_s"] = time.monotonic() - fetch_started
        join_cases = [
            type(
                "CaseJoin",
                (),
                {"case_id": c["case_id"], "prompt": c["prompt"], "truth": c["truth"]},
            )
            for c in cases
        ]
        rows, join_stats = records_to_cases(records, join_cases, family=family)
        joined = {str(row.get("case_id") or "") for row in rows}
        if len(joined) != len(cases):
            # H: the capture must hold every case's replay; anything less
            # means records were lost and the metrics would rank a partial
            # pull - a lost wrong-case replay scores 100% over the remaining
            # cases. Same contract as the hot path's record-count check.
            return {
                "statuses": statuses,
                "join_stats": join_stats,
                "timings": timings,
                "reload": "restart",
                "incomplete": True,
                "incomplete_reason": (
                    f"replay records for the restart run: {len(joined)} of "
                    f"{len(cases)} cases joined (join_stats={json.dumps(join_stats)})"
                ),
            }
        rows, bank_stats = apply_bank(rows, bank)
        classified = attribute(
            [CaseInput.from_dict(r, i) for i, r in enumerate(rows)],
            oracle_policy,
            dataset_hash="",  # stamped by the caller via the spec
        )
        summary = summarize(classified)
        return {
            "rows": classified,
            "summary": summary,
            "statuses": statuses,
            "join_stats": join_stats,
            "bank_stats": bank_stats,
            "timings": timings,
            "reload": "restart",
        }
    finally:
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _management_headers(env: RunnerEnv) -> dict[str, str]:
    """Management API headers; bearer auth reuses the --api-key-env contract."""
    headers = {"Accept": "application/json"}
    if env.api_key_env and os.environ.get(env.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[env.api_key_env]}"
    return headers


_HTTP_RESTART_REQUIRED = 409  # the management API answers 409 RESTART_REQUIRED


def _get_config_etag(management_url: str, env: RunnerEnv, timeout: float) -> str:
    """Read GET /api/v1/config and return its ETag (H).

    The management API requires optimistic concurrency on every config
    mutation: a PUT without `If-Match` is answered 428 PRECONDITION_REQUIRED,
    and a stale ETag is answered 412/409 CONFIG_CHANGED. The ETag is the
    quoted sha256 of the persisted source config document.
    """
    request = urllib.request.Request(
        management_url.rstrip("/") + "/api/v1/config",
        headers=_management_headers(env),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            etag = (response.headers.get("ETag") or "").strip()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunnerError(
            f"cannot GET config from management API at {management_url}: {exc}"
        ) from exc
    if not etag:
        raise RunnerError("management API GET /api/v1/config returned no ETag")
    return etag


def _put_router_config(
    management_url: str, env: RunnerEnv, yaml_text: str, timeout: float
) -> tuple[int, dict]:
    """PUT /api/v1/config with replace semantics (H).

    Proves the read with the current document's ETag (If-Match; see
    _get_config_etag). Returns (http_status, response_json). Connection
    failures raise RunnerError; non-2xx replies are returned for the caller
    to classify (409 RESTART_REQUIRED -> HotReloadRejectedError).
    """
    etag = _get_config_etag(management_url, env, timeout)
    body = json.dumps({"yaml": yaml_text}).encode("utf-8")
    request = urllib.request.Request(
        management_url.rstrip("/") + "/api/v1/config",
        data=body,
        headers={
            **_management_headers(env),
            "Content-Type": "application/json",
            "If-Match": etag,
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
            return int(response.status), payload if isinstance(payload, dict) else {}
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            payload = {}
        return int(exc.code), payload if isinstance(payload, dict) else {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunnerError(
            f"cannot PUT config to management API at {management_url}: {exc}"
        ) from exc


def _wait_config_activation(
    management_url: str, env: RunnerEnv, timeout_s: float
) -> None:
    """Poll GET /api/v1/config/hash until activation_status is "active" (H).

    The PUT response usually reports "active" already (the server waits up to
    20s for the runtime publication), so this only spins for the pending/202
    case. "failed" raises HotReloadRejectedError; a lapsed deadline raises
    RunnerTimeoutError. No fixed sleep is used as a completion signal: this polls
    the router's own publication status.
    """
    if timeout_s <= 0:
        raise RunnerTimeoutError(
            "trial exceeded the budget waiting for config activation"
        )
    deadline = time.monotonic() + timeout_s
    url = management_url.rstrip("/") + "/api/v1/config/hash"
    while True:
        request = urllib.request.Request(url, headers=_management_headers(env))
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            status = (
                payload.get("activation_status") if isinstance(payload, dict) else None
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ):
            status = None  # transient; keep polling until the deadline
        if status == "active":
            return
        if status == "failed":
            raise HotReloadRejectedError(
                "config activation failed (management API reports activation_status=failed)"
            )
        if time.monotonic() >= deadline:
            raise RunnerTimeoutError(
                f"config activation not active within {timeout_s:.0f}s"
            )
        time.sleep(0.25)


def candidate_tag(yaml_text: str) -> str:
    """H: short content hash naming one candidate within the replay store."""
    return hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()[:8]


def request_id_for_case(tag: str, case_id: str) -> str:
    """H: the x-request-id sent for one case of one candidate."""
    return f"{REQUEST_ID_PREFIX}{tag}-{case_id}"


class HotReloadSession:
    """One router + backend serving many candidate evaluations (H).

    The router boots once from a scratch copy of `boot_source` (the baseline
    config, or the first candidate when no baseline is given) and every
    further candidate is swapped in through `PUT /api/v1/config`. A rejected
    PUT (409) or a failed activation raises HotReloadRejectedError;
    `evaluate_with_fallback` then runs that candidate through the restart
    path and leaves the session stopped, so the next candidate re-boots hot.
    """

    def __init__(
        self,
        *,
        env: RunnerEnv,
        boot_source: str | Path,
        router_model: str,
        family: str,
        request_timeout: float = 180.0,
    ):
        if env.reload_mode != "hot":
            raise RunnerError("HotReloadSession requires RunnerEnv.reload_mode='hot'")
        self.env = env
        self.boot_source = Path(boot_source)
        self.router_model = router_model
        self.family = family
        self.request_timeout = request_timeout
        self.boot_config = env.work_dir / "hot-router-config.yaml"
        self.status_file = env.work_dir / "router-runtime.json"
        self._active_text: str | None = None
        self._procs: list[subprocess.Popen] = []
        # per-boot ports from _patch_ports: one gateway/management pair serves
        # every candidate; candidates are re-patched onto them before a PUT.
        self._router_port: int | None = None
        self._management_port: int | None = None
        self._listener_ports: list[int] | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return bool(self._procs)

    def ensure_started(self) -> None:
        """Boot the backend + router from a per-boot port-patched copy; no-op when up."""
        if self.running:
            return
        env = self.env
        env.work_dir.mkdir(parents=True, exist_ok=True)
        self.boot_config, router_port, management_port = _patch_ports(
            self.boot_source, env.work_dir
        )
        boot_text = self.boot_config.read_text(encoding="utf-8")
        boot_yaml = yaml.safe_load(boot_text)
        self._router_port = router_port
        self._management_port = management_port
        self._listener_ports = [
            int(listener["port"]) for listener in (boot_yaml.get("listeners") or [])
        ]
        self.status_file.unlink(missing_ok=True)

        mock = subprocess.Popen(
            [
                env.python_bin,
                env.mock_backend_script,
                "--port",
                str(env.backend_port),
                *env.mock_extra_args,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._procs.append(mock)
        time.sleep(1.0)
        router_env = dict(os.environ)
        if env.hf_home:
            router_env["HF_HOME"] = env.hf_home
        router_env.update(env.env_extra)
        router = subprocess.Popen(
            [
                env.router_bin,
                "-config",
                str(self.boot_config),
                "-gateway",
                "standalone",
                "-api-port",
                str(management_port),
            ],
            # The handle is owned by the spawned router process.
            stdout=open(  # noqa: SIM115
                env.work_dir / "router.log", "w", encoding="utf-8"
            ),
            stderr=subprocess.STDOUT,
            env=router_env,
            start_new_session=True,
        )
        self._procs.append(router)
        try:
            _wait_model_runtime_ready(
                f"http://127.0.0.1:{management_port}",
                env.ready_timeout_s,
                router,
                status_file=self.status_file,
            )
        except RunnerError:
            self.stop()
            raise
        self._active_text = boot_text

    def stop(self) -> None:
        """Kill the process groups (the spawned model runtime dies with them)."""
        for proc in self._procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        self._procs = []
        self._active_text = None

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        *,
        candidate_config: str | Path,
        cases: list[dict[str, Any]],
        oracle_policy: OraclePolicy,
        bank: dict[tuple[str, str], Any],
        trial_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate one candidate on the live session: PUT -> wait active ->
        send tagged requests -> pull this candidate's records by request id.

        Returns the same shape as `evaluate_candidate` plus `timings` and
        `reload: "hot"`; an incomplete record pull (fewer or duplicate rows
        than cases) sets `incomplete` + `incomplete_reason` instead of
        failing, so the caller can record status=incomplete.
        """
        if not self.running:
            raise RunnerError(
                "hot reload session is not running; call ensure_started() first"
            )
        env = self.env
        management_url = f"http://127.0.0.1:{self._management_port}"
        router_url = f"http://127.0.0.1:{self._router_port}"
        deadline = (
            time.monotonic() + trial_timeout_s if trial_timeout_s is not None else None
        )

        def remaining_s() -> float | None:
            return None if deadline is None else deadline - time.monotonic()

        def check_deadline(stage: str) -> None:
            left = remaining_s()
            if left is not None and left <= 0:
                raise RunnerTimeoutError(
                    f"trial exceeded {trial_timeout_s:.0f}s budget before {stage}"
                )

        yaml_text_raw = Path(candidate_config).read_text(encoding="utf-8")
        tag = candidate_tag(yaml_text_raw)
        yaml_text = _apply_session_ports(yaml_text_raw, self._listener_ports)
        timings = {"switch_s": 0.0, "request_s": 0.0, "fetch_s": 0.0}

        check_deadline("config switch")
        if yaml_text == self._active_text:
            pass  # the boot config (baseline or first candidate): already serving
        else:
            switch_started = time.monotonic()
            status, payload = _put_router_config(
                management_url, env, yaml_text, timeout=30.0
            )
            if status == _HTTP_RESTART_REQUIRED:
                detail = (
                    payload.get("error", {}).get("message")
                    if isinstance(payload.get("error"), dict)
                    else payload.get("message")
                )
                raise HotReloadRejectedError(
                    f"hot reload rejected with HTTP 409: {detail or 'incompatible change'}"
                )
            if status not in (200, 201, 202):
                detail = payload.get("message") or payload.get("error")
                raise RunnerError(f"config PUT failed with HTTP {status}: {detail}")
            activation = str(payload.get("activation_status") or "").lower()
            if activation == "failed":
                raise HotReloadRejectedError(
                    "config activation failed (PUT reported activation_status=failed)"
                )
            # "active" confirms instantly in _wait_config_activation; "" (a
            # legacy server without a runtime registry) and "pending" poll the
            # hash endpoint until the publication reports active.
            left = remaining_s()
            if left is not None and left <= 0:
                raise RunnerTimeoutError(
                    f"trial exceeded {trial_timeout_s:.0f}s budget before config activation"
                )
            _wait_config_activation(
                management_url, env, left if left is not None else env.ready_timeout_s
            )
            self._active_text = yaml_text
            timings["switch_s"] = time.monotonic() - switch_started

        cases_by_request_id: dict[str, CaseJoin] = {}
        statuses: dict[str, int] = {}
        request_started = time.monotonic()
        for case in cases:
            check_deadline(f"case {case['case_id']}")
            rid = request_id_for_case(tag, str(case["case_id"]))
            cases_by_request_id[rid] = CaseJoin(
                case_id=str(case["case_id"]),
                prompt=str(case["prompt"]),
                truth=str(case["truth"]),
            )
            statuses[str(case["case_id"])] = _send_prompt(
                router_url,
                self.router_model,
                str(case["prompt"]),
                self.request_timeout,
                request_id=rid,
            )
        timings["request_s"] = time.monotonic() - request_started

        check_deadline("replay fetch")
        fetch_started = time.monotonic()
        records = fetch_replays(management_url, api_key_env=env.api_key_env)
        timings["fetch_s"] = time.monotonic() - fetch_started
        rows, join_stats = records_to_cases_by_request_id(
            records, cases_by_request_id, family=self.family
        )
        result: dict[str, Any] = {
            "statuses": statuses,
            "join_stats": join_stats,
            "timings": timings,
            "reload": "hot",
            "candidate_tag": tag,
        }
        if len(rows) != len(cases):
            # H: the shared store must hold exactly one record per tagged
            # request; anything else means records were lost or duplicated.
            result["incomplete"] = True
            result["incomplete_reason"] = (
                f"replay records for candidate {tag}: {len(rows)} rows for {len(cases)} cases "
                f"(join_stats={json.dumps(join_stats)})"
            )
            return result
        banked, bank_stats = apply_bank(rows, bank)
        classified = attribute(
            [CaseInput.from_dict(r, i) for i, r in enumerate(banked)],
            oracle_policy,
            dataset_hash="",  # stamped by the caller via the spec
        )
        result["rows"] = classified
        result["summary"] = summarize(classified)
        result["bank_stats"] = bank_stats
        return result

    def evaluate_with_fallback(
        self,
        *,
        candidate_config: str | Path,
        cases: list[dict[str, Any]],
        oracle_policy: OraclePolicy,
        bank: dict[tuple[str, str], Any],
        trial_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Hot evaluate; on a rejected PUT, run the restart path once.

        The session is stopped before the fallback (the restart path needs
        the ports) and left stopped afterwards, so the next candidate
        re-boots hot from the boot source. The returned result carries
        `reload: "restart-fallback"` and `fallback_reason` for the ledger.
        """
        try:
            return self.evaluate(
                candidate_config=candidate_config,
                cases=cases,
                oracle_policy=oracle_policy,
                bank=bank,
                trial_timeout_s=trial_timeout_s,
            )
        except HotReloadRejectedError as exc:
            self.stop()
            result = evaluate_candidate(
                candidate_config=candidate_config,
                cases=cases,
                oracle_policy=oracle_policy,
                bank=bank,
                env=self.env,
                router_model=self.router_model,
                family=self.family,
                request_timeout=self.request_timeout,
                trial_timeout_s=trial_timeout_s,
            )
            result["reload"] = "restart-fallback"
            result["fallback_reason"] = str(exc)
            return result


def attribution_delta(
    baseline_summary: dict, candidate_summary: dict
) -> dict[str, int]:
    """Fixed-error counts of a candidate against the baseline summary."""

    def count(summary: dict, key: str) -> int:
        return int(summary.get("class_counts", {}).get(key, 0))

    e1_b, e1_c = count(baseline_summary, "E1_signal"), count(
        candidate_summary, "E1_signal"
    )
    e2_b, e2_c = count(baseline_summary, "E2_policy"), count(
        candidate_summary, "E2_policy"
    )
    mm_b, mm_c = count(baseline_summary, "decision_mismatch"), count(
        candidate_summary, "decision_mismatch"
    )
    return {
        "e1_fixed": max(0, e1_b - e1_c),
        "e2_fixed": max(0, e2_b - e2_c),
        "decision_mismatch_fixed": max(0, mm_b - mm_c),
        "e3_remaining": count(candidate_summary, "E3_capability"),
    }


# A1: where each contracted replay metric lives in the run summary. The
# metric name is the ledger key, the summary key is the runner's own field.
REPLAY_SUMMARY_KEYS = {"routed_accuracy": "accuracy"}

# E2: ledger key of the replay cost proxy. It is a secondary metric (recorded
# alongside objective.replay_metric whenever cost_model is declared), never a
# ranking metric: the replay tier ranks on the contracted replay_metric only.
COST_PROXY_METRIC = "routed_cost_index"


def routed_cost_index(
    rows: list[dict], cost_model: dict[str, float]
) -> tuple[float | None, list[str]]:
    """E2: mean relative cost of the routed models over answered samples.

    Only answered rows count (`answer_ok` is not None; unanswered /
    no_bank_entry rows carry no routing outcome and are excluded). A routed
    model missing from cost_model makes the index incomputable - unknown
    cost is never counted as zero (the sr-bench rule) - and the missing
    model names are returned so the caller can record the reason.
    Returns (index, unknown_models); index is None when nothing usable
    remains (no answered rows or any unknown model).
    """
    known: list[float] = []
    unknown: set[str] = set()
    for row in rows:
        if row.get("answer_ok") is None:
            continue
        model = row.get("routed")
        cost = cost_model.get(model)
        if cost is None:
            unknown.add(str(model))
        else:
            known.append(float(cost))
    if unknown or not known:
        return None, sorted(unknown)
    return round(sum(known) / len(known), 6), []


def replay_constraint_evidence(
    constraints,
    candidate_index: float | None,
    baseline_index: float | None,
) -> tuple[list[dict], bool]:
    """E2: evaluate the declared constraints against the cost proxy.

    Every declared constraint is proxied: the replay tier has exactly one
    economic measurement (routed_cost_index), so the declared live metric
    name is preserved for provenance and `proxy` records the substitution.
    `bound: "baseline"` resolves against the baseline arm's proxy value
    (None when no baseline config was evaluated -> not_evaluated, never a
    guess). Returns (evidence list, constraint_warning) where the warning
    is True when any constraint is violated under the proxy.
    """
    evidence: list[dict] = []
    warning = False
    for constraint in constraints:
        resolved = resolve_constraint(
            {
                "name": constraint.name,
                "metric": constraint.metric,
                "op": constraint.op,
                "bound": constraint.bound,
                "tolerance_pct": constraint.tolerance_pct,
                # replay proxy substitution: both arms measured on the proxy
                "baseline": baseline_index if constraint.bound == "baseline" else None,
                "candidate": candidate_index,
                "proxy": COST_PROXY_METRIC,
            }
        )
        if resolved["status"] == "violated":
            resolved["reason"] = (
                f"constraint {constraint.name} violated (replay cost proxy)"
            )
            warning = True
        evidence.append(resolved)
    return evidence, warning


def replay_metric_value(summary: dict, replay_metric: str) -> float | None:
    """A contracted replay metric (objective.replay_metric) from a run summary."""
    summary_key = REPLAY_SUMMARY_KEYS.get(replay_metric)
    if summary_key is None:
        raise ContractError(
            f"unknown replay metric {replay_metric!r}; known: {sorted(REPLAY_SUMMARY_KEYS)}"
        )
    value = summary.get(summary_key)
    return None if value is None else round(float(value), 6)


def candidate_accuracy(summary: dict, metric_key: str) -> float | None:
    """The objective metric for a candidate: routed-outcome accuracy (live tier)."""
    accuracy = summary.get("accuracy")
    return None if accuracy is None else round(float(accuracy), 6)
