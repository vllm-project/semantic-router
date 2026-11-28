"""Agentic task driver (AG2, Z14).

Productizes the AG0/W2 pilot (``swe_driver.py`` + ``ag0_header_server.py``):
drive a real agent client (headless Claude Code) over a task file - SWE-bench
rows - through a routing target (the Router, or a single-model endpoint), one
independent working directory and session per task.

The driver never talks to the client's requests itself; a local header
forwarding proxy sits between the client and the target:

- it forwards every request verbatim to the target and relays the response;
- it records, per request, the *native* identity headers Claude Code sends
  (W2: ``x-claude-code-session-id`` always, ``x-claude-code-agent-id`` on
  delegated sub-agent requests only; the neutral headers are never sent by
  claude-code, so the driver injects nothing);
- it injects a deterministic per-request ``x-request-id`` when the client
  sent none - the client-side join key Z9 needs, because the Router does not
  log agent headers into its replay records. The recorded manifest is
  consumed by ``recipe-lab diagnose --agent-requests`` unchanged.

Judging (patch correctness through the SWE-bench official harness) lives in
``swebench_judge.py`` and runs as a separate batch step (``judge-agentic``);
a run directory produced here is its input.

Endpoints, paths, the claude binary and every environment knob are
parameters - the pilot's hard-coded values are only defaults.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from typing import Any

from .contract import ContractError

# AG0 pilot guidance, verbatim: dependencies are not installed in the base
# checkout, so the agent is told to fix by editing and verify by reading.
GUIDANCE = (
    "You are working in a checkout of {repo} at a historical base commit. "
    "Dependencies and test environments are NOT installed, so do not attempt to run "
    "the full test suite; if you run a quick targeted check and it fails on missing "
    "imports, rely on code reading instead. Fix the issue described below by editing "
    "the source. Work through it with the usual tools (read, edit, grep, git diff). "
    "When done, summarize the change in one short paragraph.\n\n"
    "=== ISSUE ===\n{problem}\n"
)

DEFAULT_CLONE_URL_TEMPLATE = "https://github.com/{repo}.git"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

SESSION_HEADER = "x-claude-code-session-id"  # fallback: the neutral header
AGENT_HEADER = "x-claude-code-agent-id"
REQUEST_ID_HEADER = "x-request-id"


class HeaderForwardProxy:
    """Local forwarding proxy between the agent client and the target.

    One instance serves the whole run; ``current_task`` names the task the
    driver is driving (sequential), so every forwarded request is journalled
    under it. Port 0 binds an ephemeral port; ``url()`` is what the client's
    ``ANTHROPIC_BASE_URL`` points at.
    """

    def __init__(
        self, target: str, journal_dir: Path, *, request_id_prefix: str = ""
    ) -> None:
        self.request_id_prefix = request_id_prefix
        self.target = target.rstrip("/")
        self.journal_dir = Path(journal_dir)
        self.current_task: str | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _proxy(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                _task, request_id, record = outer._journal(
                    task=outer.current_task,
                    path=self.path,
                    headers=dict(self.headers.items()),
                    has_body=bool(body),
                )
                headers = {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower() not in HOP_BY_HOP
                }
                if not any(name.lower() == REQUEST_ID_HEADER for name in headers):
                    headers[REQUEST_ID_HEADER] = request_id
                request = urllib.request.Request(
                    outer.target + self.path,
                    data=body if body else None,
                    headers=headers,
                    method=self.command,
                )
                try:
                    with urllib.request.urlopen(request, timeout=None) as response:
                        payload = response.read()
                        status = response.status
                        response_headers = list(response.getheaders())
                except urllib.error.HTTPError as exc:
                    payload = exc.read()
                    status = exc.code
                    response_headers = list(exc.headers.items() or [])
                except OSError as exc:
                    outer._record_outcome(
                        record, status=None, error=f"target unreachable: {exc}"
                    )
                    payload = json.dumps(
                        {"error": {"type": "proxy_error", "message": str(exc)}}
                    ).encode()
                    status = 502
                    response_headers = [("Content-Type", "application/json")]
                outer._record_outcome(
                    record,
                    status=status,
                    model_usage=self._pick(response_headers, "x-vsr-model-usage"),
                )
                self.send_response(status)
                for name, value in response_headers:
                    if str(name).lower() in HOP_BY_HOP:
                        continue
                    self.send_header(str(name), str(value))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            @staticmethod
            def _pick(headers: list[Any], name: str) -> str | None:
                for key, value in headers:
                    if str(key).lower() == name:
                        return str(value)
                return None

            def do_GET(self) -> None:  # health checks reach the target as-is
                self._proxy()

            def do_POST(self) -> None:
                self._proxy()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def begin_task(self, instance_id: str) -> None:
        with self._lock:
            self.current_task = instance_id
            self._seq = 0
            self._records = []
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            (self.journal_dir / f"{instance_id}.jsonl").write_text("", encoding="utf-8")

    def end_task(self) -> list[dict[str, Any]]:
        with self._lock:
            self.current_task = None
            return list(self._records)

    def _journal(
        self,
        *,
        task: str | None,
        path: str,
        headers: dict[str, str],
        has_body: bool,
    ) -> tuple[str | None, str, dict[str, Any]]:
        lowered = {str(k).lower(): str(v) for k, v in headers.items()}
        session = lowered.get(SESSION_HEADER) or lowered.get("x-session-id")
        agent = lowered.get(AGENT_HEADER)
        client_request_id = lowered.get(REQUEST_ID_HEADER)
        with self._lock:
            self._seq += 1
            if client_request_id:
                request_id = client_request_id
                injected = False
            else:
                request_id = (
                    f"{self.request_id_prefix}{task or 'untagged'}-{self._seq:04d}"
                )
                injected = True
            record = {
                "task": task,
                "request_id": request_id,
                "request_id_injected": injected,
                "agent": "sub" if agent else "main",
                "agent_header": agent,
                "session_header": session,
                "path": path,
                "has_body": has_body,
                "ts": time.time(),
            }
            self._records.append(record)
            self._append_journal(record)
            return task, request_id, record

    def _record_outcome(
        self,
        record: dict[str, Any],
        *,
        status: int | None,
        error: str | None = None,
        model_usage: str | None = None,
    ) -> None:
        with self._lock:
            record["response_status"] = status
            record["response_error"] = error
            record["model_usage"] = model_usage
            self._append_journal(record)

    def _append_journal(self, record: dict[str, Any]) -> None:
        if self.current_task is None:
            return
        path = self.journal_dir / f"{self.current_task}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sh(cmd: list[str], cwd: str | None, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout + "\n" + proc.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except FileNotFoundError as exc:
        return 127, str(exc)


def prepare_repo(
    task: dict[str, Any], work: Path, clone_url_template: str
) -> tuple[str, str]:
    """One independent checkout per task, at the task's base commit."""
    inst = str(task["instance_id"])
    repo_dir = work / inst
    if not (repo_dir / ".git").exists():
        code, log = _sh(
            [
                "git",
                "clone",
                "--filter=blob:none",
                clone_url_template.format(repo=task["repo"]),
                str(repo_dir),
            ],
            cwd=None,
            timeout=1800,
        )
        if code != 0:
            return "clone_failed", log
    code, log = _sh(
        ["git", "checkout", "--force", str(task["base_commit"])],
        cwd=str(repo_dir),
        timeout=300,
    )
    if code != 0:
        return "checkout_failed", log
    return "ready", ""


def build_client_env(
    *,
    proxy_url: str,
    api_key: str,
    model: str | None,
    small_fast_model: str | None,
    max_output_tokens: int,
    extra_env: list[str],
) -> dict[str, str]:
    """The client's environment: the proxy is the base URL; identity headers
    are the client's own (W2) - the driver injects none."""
    env = {
        "ANTHROPIC_BASE_URL": proxy_url,
        "ANTHROPIC_API_KEY": api_key,
        "API_TIMEOUT_MS": "900000",
        # vLLM rejects (context + output) > max-model-len with a 400 instead
        # of clamping like the Anthropic API (AG0 F0c): cap the output budget.
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
    }
    if model:
        env["ANTHROPIC_MODEL"] = model
    if small_fast_model:
        env["ANTHROPIC_SMALL_FAST_MODEL"] = small_fast_model
    for pair in extra_env:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"--env expects 'KEY=VALUE', got {pair!r}")
        env[name.strip()] = value
    return env


def git_patch(repo_dir: Path, base_commit: str) -> str:
    """Capture the final worktree, including new files and agent commits."""

    def git(*args: str, env: dict[str, str] | None = None) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            timeout=120,
            check=True,
        ).stdout

    # Seed the private index with what the agent tracked, including staged
    # files that match an ignore rule. Then snapshot the actual worktree.
    with tempfile.TemporaryDirectory(prefix="recipe-lab-index-") as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        try:
            staged = git("write-tree").decode("ascii").strip()
            git("read-tree", staged, env=env)
            git("add", "--all", "--", ".", env=env)
            return git(
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                base_commit,
                "--",
                env=env,
            ).decode("utf-8")
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise ContractError(
                f"could not capture patch for {repo_dir}: {detail}"
            ) from exc
        except (subprocess.TimeoutExpired, UnicodeDecodeError) as exc:
            raise ContractError(
                f"could not capture patch for {repo_dir}: {exc}"
            ) from exc


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as scratch:
            scratch.write(data)
            temp = Path(scratch.name)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def completed_checkpoint(path: Path, instance_id: str) -> dict[str, Any] | None:
    """Only reuse a complete client result. Retry interrupted/preparation failures."""
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None
    if (
        isinstance(result, dict)
        and result.get("instance_id") == instance_id
        and result.get("status") == "done"
        and type(result.get("exit_code")) is int
        and isinstance(result.get("model_patch"), str)
        and isinstance(result.get("claude"), dict)
    ):
        return result
    return None


def run_task(
    task: dict[str, Any],
    *,
    args: Any,
    proxy: HeaderForwardProxy,
) -> dict[str, Any]:
    """Drive one task end to end; resumable (an existing result is kept)."""
    inst = str(task["instance_id"])
    out = Path(args.out)
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{inst}.json"
    if hasattr(args, "client_base_env"):
        cached = completed_checkpoint(result_path, inst)
        if cached is not None:
            return cached
    elif result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    status, prep_log = prepare_repo(task, out / "tasks", args.clone_url_template)
    if status != "ready":
        result = {
            "instance_id": inst,
            "status": status,
            "prep_log": prep_log,
            "model_patch": "",
        }
        atomic_write(
            result_path,
            json.dumps(result, ensure_ascii=False, indent=1).encode("utf-8"),
        )
        return result

    if hasattr(args, "client_base_env") and (
        result_path.exists() or (out / "requests" / f"{inst}.jsonl").exists()
    ):
        # A prior interrupted leg may have left untracked edits. Only clean
        # the immediate, managed task checkout; single-arm inputs stay untouched.
        repo = (out / "tasks" / inst).resolve()
        if repo.parent != (out.resolve() / "tasks") or not (repo / ".git").exists():
            raise ContractError("invalid retry checkout")
        code, detail = _sh(["git", "clean", "-fdx", "--", "."], str(repo), 120)
        if code:
            raise ContractError(f"could not reset interrupted checkout: {detail}")
    proxy.begin_task(inst)
    prompt = GUIDANCE.format(repo=task["repo"], problem=task["problem_statement"])
    env = dict(getattr(args, "client_base_env", os.environ))
    env.update(
        build_client_env(
            proxy_url=proxy.url(),
            api_key=args.api_key,
            model=args.model,
            small_fast_model=args.small_fast_model,
            max_output_tokens=args.max_output_tokens,
            extra_env=list(args.env or []),
        )
    )
    cmd = [
        args.claude_bin,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        str(args.max_turns),
        "--dangerously-skip-permissions",
    ]
    repo_dir = out / "tasks" / inst
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
            env=env,
            check=False,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        raw = exc.stdout or b""
        # TimeoutExpired carries captured output as bytes even with text=True.
        stdout = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        stderr = "timeout"
    elapsed = round(time.time() - started, 1)
    journal = proxy.end_task()

    try:
        claude = json.loads(stdout)
        if not isinstance(claude, dict):
            claude = {"raw": str(claude)[-4000:]}
    except json.JSONDecodeError:
        claude = {"raw": stdout[-4000:]}
    sessions = sorted({r["session_header"] for r in journal if r.get("session_header")})
    result = {
        "instance_id": inst,
        "status": "done",
        "exit_code": exit_code,
        "seconds": elapsed,
        "target": args.target,
        "claude": claude,
        "model_patch": git_patch(repo_dir, str(task["base_commit"])),
        "session_ids": sessions,
        "stderr_tail": stderr,
    }
    atomic_write(
        result_path, json.dumps(result, ensure_ascii=False, indent=1).encode("utf-8")
    )
    return result


def load_tasks(path: str | Path, only: str | None) -> list[dict[str, Any]]:
    tasks = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = set()
    for task in tasks:
        inst = task.get("instance_id") if isinstance(task, dict) else None
        if (
            not isinstance(inst, str)
            or not inst.strip()
            or inst in {".", ".."}
            or Path(inst).name != inst
            or PureWindowsPath(inst).name != inst
            or ":" in inst
        ):
            raise ContractError(
                "each task needs a single, non-empty instance_id path component"
            )
        key = os.path.normcase(inst)
        if key in seen:
            raise ContractError(f"duplicate task instance_id: {inst}")
        seen.add(key)
    if only:
        tasks = [t for t in tasks if t.get("instance_id") == only]
    return tasks


def agent_request_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Z9: the run's sub-agent request manifest, probe-compatible.

    Every journalled request the client sent with the agent identity header
    (W2: sub-agent requests only) is named by its injected/fwd request id,
    the key ``recipe-lab diagnose --agent-requests`` joins replay rows by.
    """
    entries: list[dict[str, Any]] = []
    journal_dir = Path(run_dir) / "requests"
    for journal in sorted(journal_dir.glob("*.jsonl")):
        seen: set[str] = set()
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row.get("request_id") or "")
            if row.get("agent") == "sub" and request_id and request_id not in seen:
                seen.add(request_id)
                entries.append(
                    {
                        "request_id": request_id,
                        "agent": "sub",
                        "instance_id": row.get("task"),
                    }
                )
    return {"agent_requests": entries}


def task_result_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """sr-bench-shaped per-task rows for one run directory.

    Same fields the sr-bench engine writes per completed case (engine.py's
    ``_completed_case``): ``case_id`` / ``status`` / binary ``correct`` /
    ``score`` / ``call_count`` / ``latency_s`` / ``cost_usd``. Until the
    judge runs, ``correct``/``score`` are None - never guessed. Cost comes
    verbatim from the client's own result JSON when it reports one.
    """
    rows: list[dict[str, Any]] = []
    journal_dir = Path(run_dir) / "requests"
    for result_path in sorted((Path(run_dir) / "results").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        inst = str(result.get("instance_id") or result_path.stem)
        journal = journal_dir / f"{inst}.jsonl"
        call_count = 0
        if journal.exists():
            call_count = sum(
                1
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("response_status") is not None
            )
        claude = result.get("claude")
        claude = claude if isinstance(claude, dict) else {}
        cost = claude.get("total_cost_usd")
        usage = claude.get("usage")
        completed = result.get("status") == "done" and result.get("exit_code") == 0
        rows.append(
            {
                "case_id": inst,
                "benchmark": "swe-bench",
                "status": "completed" if completed else "failed",
                "correct": None,
                "score": None,
                "cost_usd": cost if isinstance(cost, (int, float)) else None,
                "usage": usage if isinstance(usage, dict) else None,
                "call_count": call_count,
                "latency_s": result.get("seconds"),
                "details": {
                    "instance_id": inst,
                    "session_ids": result.get("session_ids"),
                },
            }
        )
    return rows


def run_driver(args: Any) -> int:
    """The ``run-agentic`` command body."""
    tasks = load_tasks(args.tasks, args.only)
    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ExitStack() as stack:
        if getattr(args, "retry_policy", None):
            # PLC0415: the optional policy module loads only when the run
            # declares a retry policy.
            from .agentic_retry import retry_executor  # noqa: PLC0415

            execute = stack.enter_context(retry_executor(args))
        else:
            if (out / "retry-manifest.json").exists():
                raise ContractError("this run requires its original --retry-policy")
            proxy = HeaderForwardProxy(args.target, out / "requests")
            proxy.start()
            stack.callback(proxy.stop)

            def execute(task):
                return run_task(task, args=args, proxy=proxy)

        for index, task in enumerate(tasks, 1):
            inst = str(task["instance_id"])
            print(f"[{index}/{len(tasks)}] {inst} start", flush=True)
            result = execute(task)
            print(
                f"[{index}/{len(tasks)}] {inst} -> {result.get('status')} "
                f"exit={result.get('exit_code')} {result.get('seconds')}s",
                flush=True,
            )
            if result.get("status") == "done" and result.get("exit_code") == 0:
                ok += 1
    manifest_path = out / "agent-requests.json"
    manifest_path.write_text(
        json.dumps(agent_request_manifest(out), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    rows_path = out / "task-results.json"
    rows_path.write_text(
        json.dumps(task_result_rows(out), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"DRIVER-DONE ok={ok}/{len(tasks)} results={out}")
    return 0
