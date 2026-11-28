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
- Every replay-tier request carries `rl-<request_tag>-<case_id>`. Fresh
  private memory uses the candidate's short content hash. Shared stores and
  repeat evaluations in one hot session add a unique trial suffix so that
  earlier records cannot satisfy a later request.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from .adapters.router_replay import (
    CaseJoin,
    fetch_replays,
    records_to_cases,
    records_to_cases_by_request_id,
    records_to_rows,
)
from .agentic import SwitchEvidence, agentic_gate_results
from .attribution import CaseInput, attribute, summarize
from .bank import apply_bank
from .contract import ContractError, OraclePolicy
from .loop_evidence import HTTP_OK
from .session_attribution import analyze_sessions, ordered_sessions
from .session_cases import TOOL_LOOP_PHASE
from .verdict import resolve_constraint

# HTTP response classes a replay row may carry: 2xx served, 4xx rejected.
_HTTP_SUCCESS_CEILING = 300
_HTTP_CLIENT_ERROR = 400


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


def _wait_gateway_ready(router_url: str, timeout_s: float) -> None:
    """Wait for the router's gateway listener to accept connections.

    The startup-status probe only covers the model runtime; a mom-v1 config
    initializes many signals and decisions before the gateway binds, so a
    caller that starts posting completions right after the runtime reports
    ready can hit ECONNREFUSED even though the router process is alive.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                router_url + "/v1/models", timeout=5.0
            ) as response:
                if response.status == HTTP_OK:
                    return
        except urllib.error.HTTPError:
            return  # the gateway is bound; a 4xx still proves it
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise RunnerError(f"gateway listener not ready within {timeout_s}s")


def _send_prompt(
    router_url: str,
    model: str,
    prompt: str,
    timeout: float,
    request_id: str | None = None,
    http_errors: dict[str, dict] | None = None,
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
        if http_errors is not None and request_id is not None:
            # Only retain the protocol error fields, never the request body.
            try:
                payload = json.loads(exc.read(8192))
            except (ValueError, UnicodeError):
                payload = {}
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                http_errors[request_id] = {
                    key: value[:512]
                    for key in ("code", "type", "message")
                    if isinstance(value := error.get(key), str)
                }
            elif isinstance(error, str):
                http_errors[request_id] = {"message": error[:512]}
            exc.close()
        return int(exc.code)


def _stop_procs(procs: list[subprocess.Popen]) -> None:
    """Terminate a booted stack: SIGTERM the process groups, then SIGKILL
    whatever survived (the spawned model runtime dies with its group)."""
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


def probe_records(
    env: RunnerEnv,
    config: str | Path,
    cases: list[dict[str, Any]],
    *,
    router_model: str = "vllm-sr/auto",
    request_timeout: float = 180.0,
    trial_timeout_s: float | None = None,
    request_id_prefix: str | None = None,
    http_errors: dict[str, dict] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float]]:
    """The one probe primitive of the replay tier: boot the stack, send every
    case once, pull the replay records, stop the stack.

    Shared by evaluate_candidate (grading join) and the cold-start label
    observation (observe_labels): readiness wait, per-case deadline checks,
    the timing keys and the cleanup behave identically for both. When
    `request_id_prefix` is given, every request carries
    `x-request-id: <prefix><case_id>` (kept verbatim by the record), so the
    caller can join records to cases by request id.

    trial_timeout_s (A5, budget.max_minutes_per_trial * 60) bounds the whole
    trial: readiness wait + request loop + replay fetch. Exceeding it raises
    RunnerTimeoutError; processes are always cleaned up in `finally`.
    Returns (records, statuses, timings).
    """
    env.work_dir.mkdir(parents=True, exist_ok=True)
    patched_config, router_port, management_port = _patch_ports(
        Path(config), env.work_dir
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
        # The gateway listener binds only after the router finishes wiring
        # its services, which on slow stacks trails the model-runtime
        # readiness probe by minutes; posting the first completion before
        # the bind fails with ECONNREFUSED. Wait like the hot-reload
        # session does (same budget the readiness wait used).
        _wait_gateway_ready(router_url, ready_budget)
        timings["switch_s"] = time.monotonic() - switch_started

        statuses: dict[str, int] = {}
        request_started = time.monotonic()
        for case in cases:
            check_deadline(f"case {case['case_id']}")
            request_id = (
                None
                if request_id_prefix is None
                else f"{request_id_prefix}{case['case_id']}"
            )
            statuses[case["case_id"]] = _send_prompt(
                router_url,
                router_model,
                case["prompt"],
                request_timeout,
                request_id=request_id,
                http_errors=http_errors,
            )
        timings["request_s"] = time.monotonic() - request_started

        check_deadline("replay fetch")
        fetch_started = time.monotonic()
        records = fetch_replays(management_url, api_key_env=env.api_key_env)
        timings["fetch_s"] = time.monotonic() - fetch_started
        return records, statuses, timings
    finally:
        _stop_procs(procs)


def _replay_failures(
    rows: list[dict],
    cases_by_request_id: dict[str, CaseJoin],
    statuses: dict[str, int],
    http_errors: dict[str, dict],
) -> list[dict]:
    """Check completeness by case before grading, including explicit gateway rejections."""
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    failures = []
    for rid, case in cases_by_request_id.items():
        matches = by_case[case.case_id]
        status = statuses.get(case.case_id)
        if (
            len(matches) == 1
            and isinstance(matches[0].get("routed"), str)
            and matches[0]["routed"].strip()
            and status is not None
            and HTTP_OK <= status < _HTTP_SUCCESS_CEILING
            and matches[0].get("lifecycle_state") != "failed"
            and not (
                isinstance(matches[0].get("response_status"), int)
                and matches[0]["response_status"] >= _HTTP_CLIENT_ERROR
            )
        ):
            continue
        evidence = {
            "case_id": case.case_id,
            "request_id": rid,
            "http_status": status,
            "replay_count": len(matches),
            "replays": [
                {
                    k: row[k]
                    for k in (
                        "request_id",
                        "matched_decision",
                        "routed",
                        "response_status",
                        "terminal_reason",
                        "lifecycle_state",
                    )
                    if k in row
                }
                for row in matches
            ],
        }
        if rid in http_errors:
            evidence["http_error"] = http_errors[rid]
        failures.append(evidence)
    return failures


def _replay_capture(
    rows: list[dict],
    cases_by_request_id: dict[str, CaseJoin],
    statuses: dict[str, int],
    http_errors: dict[str, dict],
    router_log: Path,
) -> dict:
    failures = _replay_failures(rows, cases_by_request_id, statuses, http_errors)
    if not failures:
        return {}
    # Some protocol adapters return only a generic 503 body. The router's
    # structured rejection log carries the policy code and cause. Never
    # attribute a log entry to a case by its ordering or timestamp.
    by_request = {
        rid: failure
        for failure in failures
        for rid in [
            failure["request_id"],
            *(r.get("request_id") for r in failure["replays"]),
        ]
        if rid
    }
    try:
        with router_log.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if "selection_rejected" not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (
                    not isinstance(event, dict)
                    or event.get("event", event.get("msg")) != "selection_rejected"
                ):
                    continue
                failure = by_request.get(event.get("request_id"))
                if failure is not None:
                    failure.setdefault("router_rejections", []).append(
                        {
                            k: event[k][:512]
                            for k in ("request_id", "code", "error")
                            if isinstance(event.get(k), str)
                        }
                    )
    except OSError:
        pass  # HTTP and replay evidence remains available without a router log.
    return {
        "incomplete": True,
        "incomplete_reason": (
            f"{len(rows)} replay rows for {len(cases_by_request_id)} cases; "
            f"{len(failures)} case(s) lack exactly one successful routed request"
        ),
        "replay_failures": failures,
    }


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

    Delegates the stack lifecycle to probe_records and adds the grading join:
    records -> cases (prompt join) -> answer bank -> attribution.
    """
    yaml_text = Path(candidate_config).read_text(encoding="utf-8")
    tag = candidate_tag(yaml_text)
    memory_store = _in_memory_replay(yaml_text)
    request_tag = tag if memory_store else _trial_request_tag(tag)
    http_errors = {}
    records, statuses, timings = probe_records(
        env,
        candidate_config,
        cases,
        router_model=router_model,
        request_timeout=request_timeout,
        trial_timeout_s=trial_timeout_s,
        request_id_prefix=request_id_for_case(request_tag, ""),
        http_errors=http_errors,
    )
    cases_by_request_id = {
        request_id_for_case(request_tag, str(c["case_id"])): CaseJoin(
            case_id=str(c["case_id"]),
            prompt=c["prompt"],
            truth=c["truth"],
            category=c.get("category") or "",
        )
        for c in cases
    }
    rows, join_stats = records_to_cases_by_request_id(
        records, cases_by_request_id, family=family
    )
    # Only an isolated, explicitly in-memory restart store whose records
    # have no caller IDs (empty / router-generated UUIDs) can fall back to
    # the legacy prompt join. A different caller's tag is never evidence.
    if not rows and records and memory_store:
        ids = [r.get("request_id") or r.get("RequestID") for r in records]
        if all(
            not rid
            or (
                isinstance(rid, str)
                and re.fullmatch(
                    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", rid
                )
            )
            for rid in ids
        ):
            rows, join_stats = records_to_cases(
                records, list(cases_by_request_id.values()), family=family
            )
    capture = _replay_capture(
        rows, cases_by_request_id, statuses, http_errors, env.work_dir / "router.log"
    )
    if capture:
        return dict(
            capture,
            candidate_tag=tag,
            request_tag=request_tag,
            statuses=statuses,
            timings=timings,
            join_stats=join_stats,
            reload="restart",
        )
    rows, bank_stats = apply_bank(rows, bank)
    classified = attribute(
        [CaseInput.from_dict(r, i) for i, r in enumerate(rows)],
        oracle_policy,
        dataset_hash="",  # stamped by the caller via the spec
    )
    summary = summarize(classified)
    return {
        "candidate_tag": tag,
        "request_tag": request_tag,
        "rows": classified,
        "summary": summary,
        "statuses": statuses,
        "join_stats": join_stats,
        "bank_stats": bank_stats,
        "timings": timings,
        "reload": "restart",
    }


def _post_messages(
    router_url: str,
    model: str,
    messages: list[Any],
    timeout: float,
    request_id: str | None = None,
    session_identity: str | None = None,
    conversation_identity: bool = True,
) -> int:
    """Send one recorded request through the router, identity headers on.

    The route-replay tier replays a conversation in order, so every request
    carries the conversation's `x-session-id` and `x-conversation-id`
    (conversation-scoped protection binds to both, see AG1/W1) plus the
    candidate-scoped `x-request-id` the replay store keeps verbatim. A case
    with `session_only_identity` replays a client that sends only the
    session header (Z7: the missing-conversation-header scenario - the
    protection's conversation identity resolves missing and the protection
    skips the request entirely).
    """
    body = json.dumps({"model": model, "max_tokens": 64, "messages": messages}).encode(
        "utf-8"
    )
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["x-request-id"] = request_id
    if session_identity:
        headers["x-session-id"] = session_identity
        if conversation_identity:
            headers["x-conversation-id"] = session_identity
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


def probe_route_replay(
    env: RunnerEnv,
    config: str | Path,
    cases: list[dict[str, Any]],
    *,
    router_model: str = "vllm-sr/auto",
    request_timeout: float = 180.0,
    trial_timeout_s: float | None = None,
    request_id_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float]]:
    """The route-replay probe primitive (AG3): boot the stack once, replay
    every session case's request sequence in order with the original
    identity headers, pull this candidate's replay records, stop the stack.

    Session cases carry `requests` (recorded message lists, replay order);
    single-request cases carry `prompt`. The recorded history is fixed - a
    changed candidate cannot alter later requests - so the pulled records
    measure routing only, never quality. Records are filtered to this
    probe's `x-request-id` prefix when one is given (the replay store has
    no delete endpoint). Returns (records, statuses, timings).
    """
    env.work_dir.mkdir(parents=True, exist_ok=True)
    patched_config, router_port, management_port = _patch_ports(
        Path(config), env.work_dir
    )
    management_url = f"http://127.0.0.1:{management_port}"
    router_url = f"http://127.0.0.1:{router_port}"
    deadline = (
        time.monotonic() + trial_timeout_s if trial_timeout_s is not None else None
    )

    def check_deadline(stage: str) -> None:
        if deadline is not None and time.monotonic() > deadline:
            raise RunnerTimeoutError(
                f"trial exceeded {trial_timeout_s:.0f}s budget before {stage}"
            )

    procs: list[subprocess.Popen] = []
    timings = {"switch_s": 0.0, "request_s": 0.0, "fetch_s": 0.0}
    try:
        status_file = patched_config.parent / "router-runtime.json"
        if status_file.exists():
            status_file.unlink(missing_ok=True)
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
            stdout=open(  # noqa: SIM115
                Path(env.work_dir) / "router.log", "w", encoding="utf-8"
            ),
            stderr=subprocess.STDOUT,
            env=router_env,
            start_new_session=True,
        )
        procs.append(router)
        _wait_model_runtime_ready(
            management_url, env.ready_timeout_s, router, status_file=status_file
        )
        _wait_gateway_ready(router_url, env.ready_timeout_s)
        timings["switch_s"] = time.monotonic() - switch_started

        statuses: dict[str, int] = {}
        request_started = time.monotonic()
        for case in cases:
            identity = str(case.get("session_id") or case["case_id"])
            if isinstance(case.get("requests"), list):
                requests = case["requests"]
            else:
                requests = [
                    {
                        "messages": [
                            {"role": "user", "content": case["prompt"]},
                        ]
                    }
                ]
            for index, request in enumerate(requests):
                check_deadline(f"case {case['case_id']} request {index}")
                request_id = (
                    None
                    if request_id_prefix is None
                    else f"{request_id_prefix}{case['case_id']}-{index}"
                )
                statuses[f"{case['case_id']}-{index}"] = _post_messages(
                    router_url,
                    router_model,
                    request["messages"],
                    request_timeout,
                    request_id=request_id,
                    session_identity=identity,
                    conversation_identity=not case.get("session_only_identity"),
                )
        timings["request_s"] = time.monotonic() - request_started

        check_deadline("replay fetch")
        fetch_started = time.monotonic()
        records = fetch_replays(management_url, api_key_env=env.api_key_env)
        if request_id_prefix is not None:
            records = [
                record
                for record in records
                if str(record.get("request_id") or "").startswith(request_id_prefix)
            ]
        timings["fetch_s"] = time.monotonic() - fetch_started
        return records, statuses, timings
    finally:
        _stop_procs(procs)


def route_replay_phases(
    cases: list[dict[str, Any]], request_id_prefix: str
) -> dict[str, str]:
    """Map each replayed request id to its phase (the arriving request's
    position in its conversation: user turn vs tool loop), computed from the
    cases' own recorded messages (the replay record carries no phase)."""
    phases: dict[str, str] = {}
    for case in cases:
        if isinstance(case.get("requests"), list):
            for index, request in enumerate(case["requests"]):
                phases[f"{request_id_prefix}{case['case_id']}-{index}"] = str(
                    request.get("phase") or ""
                )
        else:
            phases[f"{request_id_prefix}{case['case_id']}-0"] = ""
    return phases


def route_replay_evidence(
    records: list[dict[str, Any]],
    *,
    phases_by_request_id: dict[str, str],
    gates: dict[str, Any] | None = None,
    baseline_events: dict[str, int] | None = None,
    cost_model: dict[str, float] | None = None,
    strong_model: str | None = None,
) -> dict[str, Any]:
    """AG3: routing evidence from one candidate's replay records.

    The recorded history is fixed, so this is routing evidence only - no
    quality claim is derivable. Carries the session attribution (lane
    flaps, rejected selections, interrupted tool loops), the tool-loop
    switch count (adjacent routed-model changes whose arriving request sat
    in the tool-loop phase), the per-session switch average, the model
    share distribution, the first-turn decision distribution, the optional
    strong-model share and the cost proxy over the routed models.

    `gates` (an `agentic_gates` mapping) resolves through AG7's
    `agentic_gate_results` against this evidence; the baseline arm's event
    counts anchor a `baseline` interruption bound.
    """
    rows = records_to_rows(records)
    analysis = analyze_sessions(rows)
    tool_loop_switches = 0
    drift_releases = 0
    for requests in ordered_sessions(rows).values():
        routed = [row for row in requests if str(row.get("routed") or "")]
        for previous, current in pairwise(routed):
            phase = phases_by_request_id.get(str(current.get("request_id") or ""))
            if phase == TOOL_LOOP_PHASE and previous.get("routed") != current.get(
                "routed"
            ):
                tool_loop_switches += 1
                # Z5: the arriving record's protection trace marks the
                # in-design drift releases (#5567) the zero ceiling must
                # not judge; untraced switches stay on the unprotected side
                policy = current.get("session_policy")
                if isinstance(policy, dict) and policy.get("decision_drift") is True:
                    drift_releases += 1
    multi_request = [
        session for session in analysis["sessions"] if session["requests_recorded"] > 1
    ]
    mean_switches = (
        analysis["event_counts"]["lane_flap"] / len(multi_request)
        if multi_request
        else None
    )
    model_counts: dict[str, int] = {}
    for row in rows:
        model = str(row.get("routed") or "")
        if model:
            model_counts[model] = model_counts.get(model, 0) + 1
    routed_total = sum(model_counts.values())
    model_shares = (
        {model: count / routed_total for model, count in sorted(model_counts.items())}
        if routed_total
        else {}
    )
    first_turn_decisions: dict[str, int] = {}
    for requests in ordered_sessions(rows).values():
        decision = str(requests[0].get("matched_decision") or "") if requests else ""
        if decision:
            first_turn_decisions[decision] = first_turn_decisions.get(decision, 0) + 1
    strong_share: float | None = None
    if strong_model and routed_total:
        strong_share = model_counts.get(strong_model, 0) / routed_total
    cost_index: float | None = None
    cost_unknown: list[str] = []
    if cost_model:
        costs: list[float] = []
        unknown: set[str] = set()
        for row in rows:
            model = str(row.get("routed") or "")
            if not model:
                continue
            entry = cost_model.get(model)
            if entry is None:
                unknown.add(model)
            else:
                costs.append(float(entry))
        if unknown or not costs:
            cost_unknown = sorted(unknown)
        else:
            cost_index = round(sum(costs) / len(costs), 6)
    summary = {
        "records": len(records),
        "sessions": len(analysis["sessions"]),
        "model_switches": analysis["event_counts"]["lane_flap"],
        "tool_loop_switches": tool_loop_switches,
        "decision_drift_switches": drift_releases,
        "selection_rejected": analysis["event_counts"]["selection_failed"],
        "tool_loop_interrupted": analysis["event_counts"]["tool_loop_interrupted"],
        "mean_switches_per_session": mean_switches,
        "model_shares": model_shares,
        "first_turn_decisions": dict(sorted(first_turn_decisions.items())),
        "strong_model_share": strong_share,
        "cost_index": cost_index,
    }
    reasons: list[str] = []
    if cost_unknown:
        reasons.append(
            "cost proxy incomplete: routed model(s) not in cost_model: "
            + ", ".join(cost_unknown)
        )
    gate_results = agentic_gate_results(
        gates or {},
        candidate_continuity={
            "model_switches_by_phase": {TOOL_LOOP_PHASE: tool_loop_switches},
            "mean_switches_per_task": mean_switches,
        },
        candidate_events=analysis["event_counts"],
        baseline_events=baseline_events,
        # Z5: the switch evidence this tier derives itself - the drift
        # split from the records' protection traces, one session = one task
        candidate_switches=SwitchEvidence(
            tool_loop_switches=float(tool_loop_switches),
            decision_drift=float(drift_releases),
            sessions=len(analysis["sessions"]),
        ),
    )
    return {
        "summary": summary,
        "gate_results": gate_results,
        "reasons": reasons,
        "analysis": analysis,
    }


def evaluate_route_replay(
    *,
    candidate_config: str | Path,
    cases: list[dict[str, Any]],
    env: RunnerEnv,
    router_model: str = "vllm-sr/auto",
    request_timeout: float = 180.0,
    trial_timeout_s: float | None = None,
    request_id_prefix: str | None = None,
    gates: dict[str, Any] | None = None,
    baseline_events: dict[str, int] | None = None,
    cost_model: dict[str, float] | None = None,
    strong_model: str | None = None,
) -> dict[str, Any]:
    """AG3: one candidate on the route-replay tier - boot the stack, replay
    the recorded sessions in order, and turn the pulled records into routing
    evidence and stability-gate results. No quality metric is produced:
    the recorded history cannot change with the candidate."""
    records, statuses, timings = probe_route_replay(
        env,
        candidate_config,
        cases,
        router_model=router_model,
        request_timeout=request_timeout,
        trial_timeout_s=trial_timeout_s,
        request_id_prefix=request_id_prefix,
    )
    evidence = route_replay_evidence(
        records,
        phases_by_request_id=route_replay_phases(cases, request_id_prefix or ""),
        gates=gates,
        baseline_events=baseline_events,
        cost_model=cost_model,
        strong_model=strong_model,
    )
    return {
        "summary": evidence["summary"],
        "gate_results": evidence["gate_results"],
        "reasons": evidence["reasons"],
        "statuses": statuses,
        "timings": timings,
        "records_count": len(records),
        "reload": "restart",
    }


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


def _in_memory_replay(yaml_text: str) -> bool:
    doc = yaml.safe_load(yaml_text) or {}
    services = (doc.get("global") or {}).get("services") or {}
    store = services.get("router_replay") or {}
    return isinstance(store, dict) and store.get("store_backend") == "memory"


def _trial_request_tag(tag: str) -> str:
    return f"{tag}.{uuid.uuid4().hex}"


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
        self._seen_tags: set[str] = set()
        self._boot_memory_store = False
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
            # Same slow-stack guard as probe_records: the gateway binds
            # after the services finish wiring, behind the readiness probe.
            _wait_gateway_ready(f"http://127.0.0.1:{router_port}", env.ready_timeout_s)
        except RunnerError:
            self.stop()
            raise
        self._active_text = boot_text
        self._boot_memory_store = _in_memory_replay(boot_text)
        self._seen_tags.clear()

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
        http_errors = {}
        # Reserve the identity before the first send. A timeout can leave
        # partial rows in the store even when the trial never returns.
        deterministic = (
            self._boot_memory_store
            and _in_memory_replay(yaml_text_raw)
            and tag not in self._seen_tags
        )
        request_tag = tag if deterministic else _trial_request_tag(tag)
        self._seen_tags.add(tag)
        request_started = time.monotonic()
        for case in cases:
            check_deadline(f"case {case['case_id']}")
            rid = request_id_for_case(request_tag, str(case["case_id"]))
            cases_by_request_id[rid] = CaseJoin(
                case_id=str(case["case_id"]),
                prompt=str(case["prompt"]),
                truth=str(case["truth"]),
                category=str(case.get("category") or ""),
            )
            statuses[str(case["case_id"])] = _send_prompt(
                router_url,
                self.router_model,
                str(case["prompt"]),
                self.request_timeout,
                request_id=rid,
                http_errors=http_errors,
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
            "request_tag": request_tag,
        }
        capture = _replay_capture(
            rows,
            cases_by_request_id,
            statuses,
            http_errors,
            env.work_dir / "router.log",
        )
        if capture:
            result.update(capture)
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

    e1_b, e1_c = (
        count(baseline_summary, "E1_signal"),
        count(candidate_summary, "E1_signal"),
    )
    e2_b, e2_c = (
        count(baseline_summary, "E2_policy"),
        count(candidate_summary, "E2_policy"),
    )
    mm_b, mm_c = (
        count(baseline_summary, "decision_mismatch"),
        count(candidate_summary, "decision_mismatch"),
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
