"""Z15 / V6: the agentic mock end-to-end acceptance - the four V6 scenarios
on the fully mocked stack (real router, real decision model, AG8 scripted
agent, mock OpenAI backend).

§7 V6 runs four scenarios and locks them as integration tests:

1. cold start: ``profile-agentic`` + ``bootstrap-agentic`` generate the
   candidate -> ``run --tier route-replay`` ranks it -> a live four-arm
   comparison (candidate C, all-weak baseline B, single-model references
   S/W and the random mix R) through the real ``verify_live`` machinery ->
   PASS. The candidate routes complex tasks to the strong model and simple
   tasks to the cheap one, and its task success rate is strictly above R.
2. tuning: the protection-disabled baseline (the doc's 'deliberately
   disable protection') is the baseline; ``diagnose`` and
   ``attribute-session`` read its real replay records (honestly empty of
   session rows - a disabled protection resolves no identity);
   ``propose-session-policy`` hands the unprotected switches back as the
   W1-class cause (re-enabling enforcement is not a knob); the remedy's
   route-replay shows the unprotected tool-loop residual at zero and it
   wins the live stability gate.
3. negative: the enforcement-opt-out candidate (the drifting decision on
   ``adaptations.mode: observe``) is eliminated on the route-replay tier
   under the negative contract's strict drift ceiling (0.1 - its single
   release re-baselines the lock, landing the per-task rate exactly on the
   default 0.2) and FAILs the live verdict through the agentic gates fed
   from its own arm's Z4-fetched replay rows.
4. sub agent: the generic-client config (no agent identity header) is
   diagnosed "sub-agents cannot keep a separate identity" from the AG8
   probe's client-side request manifest, and ``propose-session-policy``
   answers with the agent-header candidate.

The sr-bench harness layer is a scripted stand-in (``LiveArmCli``): the
protocol shape mirrors SrbenchCli, but every arm run is REAL traffic - the
MoM arms boot their own router processes and the AG8 scripted agent drives
the fixture sessions through them; the reference arms drive the same agent
straight into the mock backend. Per-task success comes from the scripted
completions (the cheap model abandons the complex tasks, the strong one
never does), so the metric, the paired statistics, the continuity sections
and the random mix R all derive from actual routing behavior. The verdict,
the agentic gates (Z4 pulls each arm's replay records from its own router
management API) and the R-arm statistics are the production machinery,
exercised for real.

Two documented divergences from §6 are load-bearing:

- the agentic profile measures models, not lanes, so bootstrap-agentic's
  shared unmeasured-lane selection (Z10) gives every ranked lane the same
  model. The V6 template keeps ``simple_general_local`` outside the lane
  set (bootstrap preserves non-lane decisions verbatim), so the cold-start
  config splits cheap/simple from strong/complex structurally.
- AG6's proposer fixes broken identity by ADDING the missing header; it
  has no "enable protection" candidate (Z6 hands enforcement back as a
  W1/W2-class cause, never a knob). The tuning baseline therefore
  disables the protection outright (the doc's letter) and the scenario
  asserts the W1 handoff plus the remedy's zero unprotected residual.
  Two real-stack findings shaped the design: an unmapped identity header
  silently falls back to the default name the client sends (so the
  protection engages anyway), an empty name is rejected by the loader,
  and an observe-mode enforcement opt-out keeps its releases TRACED as
  decision drift - the unprotected residual only exists when the
  protection never runs.

Determinism (criterion 5): the candidate arm's fixture set runs twice and
the AG8 trajectory comparison must come back empty; ``diagnose`` is run
twice over the same records and its diagnosis.json must be byte-identical.

Skips unless ``RECIPE_LAB_ROUTER_BIN`` points at a router built from the
stack top; the classifier artifact (vllm-sr/Vela-2.0-0.8B, the AG8
real-stack decision model) resolves through ``RECIPE_LAB_HF_HOME``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from recipe_lab.agentic import session_random_mix_stats
from recipe_lab.adapters.router_replay import fetch_replays, records_to_rows
from recipe_lab.contract import ExperimentSpec
from recipe_lab.live import verify_live

pytestmark = [pytest.mark.integration]

RECIPE_LAB_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = RECIPE_LAB_ROOT / "examples" / "smoke"
FIXTURES_DIR = RECIPE_LAB_ROOT / "tests" / "fixtures"
V6_FIXTURES = FIXTURES_DIR / "v6_agentic_fixtures.json"
V6_TEMPLATE = FIXTURES_DIR / "agent_recipe_v6.yaml"

MODEL_A = "google/gemini-2.5-flash-lite"  # cheap; abandons the complex tasks
MODEL_B = "google/gemini-3.1-pro"  # expensive; completes everything
METRIC_KEY = "capability.score"
ROUTER_MODEL = "vllm-sr/auto"

# The mid-loop drift note makes one session release the tool lock as a
# traced decision drift (in-design per #5567); with five live conversations
# and up to two releases in the drift session the per-task rate tops the
# 0.2 default, so the workload declares the smoke's own ceiling (documented
# in the spec, never silently widened).
DRIFT_CEILING = 0.5

COMPLEX_TASKS = ("long-loop", "sub-agent", "correction", "drift-loop")

PRICING = {
    MODEL_A: {"prompt_per_1m": 1.0, "completion_per_1m": 1.0},
    MODEL_B: {"prompt_per_1m": 1.0, "completion_per_1m": 4.0},
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fixtures() -> dict:
    return json.loads(V6_FIXTURES.read_text(encoding="utf-8"))


def _normalized_model(raw: str) -> str:
    """The model behind a transcript entry's model field (the probe prefixes
    header-sourced models with ``hdr:`` and body-sourced ones with ``body:``)."""
    value = str(raw or "")
    for prefix in ("hdr:", "body:"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _task_success(transcript: list[dict], fixtures: dict) -> dict[str, bool]:
    """Per-fixture task success from the probe transcript: the main agent's
    final content must be the scripted completion, not the abandon text."""
    tasks = fixtures["tasks"]
    final_by_fixture: dict[str, str] = {}
    for entry in transcript:
        if entry.get("agent") != "main":
            continue
        if entry.get("status") != 200:
            final_by_fixture[entry["fixture"]] = ""
            continue
        final_by_fixture[entry["fixture"]] = str(entry.get("content") or "")
    success: dict[str, bool] = {}
    for session in fixtures["sessions"]:
        name = session["fixture"]
        expected = str(tasks[session["task"]]["final_content"])
        success[name] = final_by_fixture.get(name) == expected
    return success


def _spend_usd(
    transcript: list[dict], pricing: dict[str, dict], dominant_model: str
) -> float:
    """Assumed-priced spend over the transcript's usage. The router relays
    the backend's model in the response body, so the transcript names the
    routed model; anything unrecognized prices at the arm's dominant model
    (the spend is descriptive evidence, never a gate)."""
    total = 0.0
    for entry in transcript:
        usage = entry.get("usage") or {}
        model = _normalized_model(entry.get("model") or "") or dominant_model
        rates = pricing.get(model) or pricing.get(dominant_model)
        if rates is None or not usage:
            continue
        total += (
            float(usage.get("prompt_tokens") or 0) * rates["prompt_per_1m"]
            + float(usage.get("completion_tokens") or 0) * rates["completion_per_1m"]
        ) / 1e6
    return total


# ---- recorded route cases (the offline mirror of the probe's client loop) ----


def _phase_of(task: dict, step: int) -> str:
    """The client-side phase of one task turn (mirrors the router's rule:
    a request whose history carries tool turns arrives in the tool loop)."""
    if step == 0:
        return "user_turn"
    return "tool_loop"


def _session_requests(
    fixtures: dict, session: dict, *, sub: bool = False
) -> list[dict]:
    """The recorded request sequence of one fixture session - the same
    growing-history shape the live probe sends, so route-replay replays the
    exact recorded conversations."""
    tasks = fixtures["tasks"]
    task = tasks[session["subtask_task"] if sub else session["task"]]
    messages: list[dict] = [{"role": "user", "content": task["task_prompt"]}]
    steps = [int(s) for s in task.get("tool_call_steps") or []]
    names = list(task.get("tool_names") or [])
    tool_results = fixtures.get("tool_results") or {}
    sub_tool = session.get("sub_agent_tool")
    correction_after = session.get("user_correction_after_step")
    requests: list[dict] = []
    for step in range(12):
        requests.append(
            {
                "phase": _phase_of(task, step),
                "messages": copy.deepcopy(messages),
            }
        )
        if step in steps:
            name = names[steps.index(step) % len(names)] if names else "read_file"
            args = (task.get("tool_args") or {}).get(name, {})
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{step}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
            if name == sub_tool and not sub:
                requests.extend(_session_requests(fixtures, session, sub=True))
            result = (
                "(replaced by the sub-agent's final answer)"
                if name == sub_tool
                else str(tool_results.get(name, "ok"))
            )
            messages.append(
                {"role": "tool", "tool_call_id": f"call-{step}", "content": result}
            )
            if not sub and correction_after == step:
                messages.append(
                    {"role": "user", "content": str(session.get("correction_text", ""))}
                )
            continue
        break
    return requests


# ---- one arm router ----


class RouterArm:
    """One MoM arm router kept alive across runs (the arm IS a real router).

    Replay records accumulate on the arm for its whole lifetime; every
    consumer filters them by the run's session-id prefix (Z4) or this
    module's explicit request-id prefixes.
    """

    def __init__(
        self,
        name: str,
        config_path: Path,
        router_bin: str,
        hf_home: str | None,
        work_dir: Path,
    ) -> None:
        from recipe_lab.runner import _patch_ports

        self.name = name
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        patched, router_port, management_port = _patch_ports(
            Path(config_path), self.work_dir
        )
        self.router_url = f"http://127.0.0.1:{router_port}"
        self.management_url = f"http://127.0.0.1:{management_port}"
        env = dict(os.environ)
        if hf_home:
            env["HF_HOME"] = hf_home
        log_path = self.work_dir / "router.log"
        self.proc = subprocess.Popen(
            [
                router_bin,
                "-config",
                str(patched),
                "-gateway",
                "standalone",
                "-api-port",
                str(management_port),
            ],
            stdout=log_path.open("ab"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            self._wait_ready()
            self.config_hash = self._fetch_config_hash()
        except BaseException:
            self.stop()
            raise

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 600.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} router exited early (see {self.work_dir / 'router.log'})"
                )
            try:
                with urllib.request.urlopen(
                    self.management_url + "/api/v1/inventory/model-runtime",
                    timeout=2.0,
                ) as response:
                    # match the raw text (the conftest's probe): re-serializing
                    # would inject separators the router never emits
                    text = response.read().decode("utf-8", "replace")
                if '"ready":true' in text.replace(" ", ""):
                    return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1.0)
        raise RuntimeError(f"{self.name} router not ready within 600s")

    def _fetch_config_hash(self) -> str:
        payload = _http_json(self.management_url + "/api/v1/config/hash", 10.0)
        digest = str(payload.get("active_runtime_hash", ""))
        if len(digest) != 64:
            raise RuntimeError(
                f"{self.name} router config hash is not 64 hex: {digest!r}"
            )
        return digest

    def records(self) -> list[dict]:
        return fetch_replays(self.management_url)

    def stop(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---- the scripted sr-bench stand-in: real arms, real scoring ----


class LiveArmCli:
    """Scripted stand-in for SrbenchCli whose runs are real arms.

    The protocol surface mirrors exactly what ``verify_live`` consumes
    (setup / targets / show / run_manifest / candidate_plan / compare /
    report); the execution underneath is real: MoM targets are live
    ``RouterArm`` routers and the AG8 scripted agent drives the fixture
    sessions through them, single targets drive the same agent straight
    into the mock backend. Per-task scores, spends and continuity sections
    come from that traffic; the verdict and gates above them are the
    production machinery.
    """

    def __init__(
        self,
        *,
        registry: dict[str, dict],
        mom_targets: dict[str, RouterArm],
        mock_endpoint: str,
        fixture_file: Path,
        pricing: dict[str, dict],
    ) -> None:
        self.registry = registry
        self.mom_targets = mom_targets
        self.mock_endpoint = mock_endpoint
        self.fixture_file = Path(fixture_file)
        self.pricing = pricing
        self.runs: dict[str, dict] = {}
        self._seq = 0
        self._fixtures_cache: dict | None = None
        if str(SMOKE_DIR) not in sys.path:
            sys.path.insert(0, str(SMOKE_DIR))

    # -- SrbenchCli protocol --

    def run_command(self, args: list[str]) -> dict:
        if args[:2] != ["setup", "--benchmark"]:
            raise AssertionError(f"unexpected run_command {args!r}")
        return {
            "version": "sr-bench-1.0",
            "benchmarks": [
                {"benchmark": args[args.index("--benchmark") + 1], "installed": True}
            ],
        }

    def targets(self) -> list[dict]:
        return [dict(entry, id=target) for target, entry in self.registry.items()]

    def show(self, run_id: str) -> dict:
        return self.runs[run_id]["doc"]

    def run_manifest(self, manifest_path: str | Path) -> dict:
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        targets = raw.get("targets") or []
        mom = [t for t in targets if isinstance(t, dict) and t.get("kind") == "mom"]
        if mom:
            return self._run_mom(mom[0]["id"], copy.deepcopy(raw))
        single = [
            t for t in targets if isinstance(t, dict) and t.get("kind") == "single"
        ]
        return self._run_reference(single[0]["id"], copy.deepcopy(raw))

    def candidate_plan(
        self,
        baseline_run_id: str,
        *,
        targets,
        mode: str = "live",
        name=None,
        hypothesis="",
    ) -> dict:
        if mode == "preview":
            # the preview is diagnostic-only; the arms carry all the evidence
            return {"id": self._next_id("preview"), "status": "skipped"}
        target = targets[0] if isinstance(targets, list) and targets else targets
        return self._run_mom(str(target), {"targets": [{"id": str(target)}]})

    def compare(self, base_id: str, cand_id: str) -> dict:
        base, cand = self.runs[base_id], self.runs[cand_id]
        names = sorted(base["task_scores"])

        def section(run: dict) -> dict:
            return {
                "id": run["target_id"],
                "kind": "mom",
                "accuracy": run["accuracy"],
                METRIC_KEY: run["accuracy"],
                "total_spend_usd": run["spend"],
                "output_diagnostics": {
                    "planned_cases": len(names),
                    "result_cases": len(names),
                    "output_limit_cases": 0,
                    "no_answer_cases": 0,
                },
                "continuity": run["continuity"],
            }

        return {
            "baseline": {METRIC_KEY: base["accuracy"], "total_spend_usd": base["spend"]},
            "candidate": {
                METRIC_KEY: cand["accuracy"],
                "total_spend_usd": cand["spend"],
            },
            "baseline_targets": [section(base)],
            "candidate_targets": [section(cand)],
            "baseline_selected_target_id": base["target_id"],
            # no comparisons[]: five paired cases are too few for the paired
            # CI decision to clear any bar, so the verdict stays on the
            # point-estimate path (the R arm and the gates carry the rest)
        }

    def report(self, run_id: str, output_path: str | Path) -> dict:
        run = self.runs[run_id]
        section = {
            "id": run["target_id"],
            "accuracy": run["accuracy"],
            METRIC_KEY: run["accuracy"],
            "total_spend_usd": run["spend"],
            "latency_p50_s": 0.5,
            "output_diagnostics": {
                "planned_cases": len(run["task_scores"]),
                "result_cases": len(run["task_scores"]),
                "output_limit_cases": 0,
                "no_answer_cases": 0,
                "subject_calls": {"total": run["subject_calls"]},
            },
        }
        if run.get("continuity"):
            section["continuity"] = run["continuity"]
        return {"summary": {"targets": [section]}}

    # -- execution --

    def _fixtures(self) -> dict:
        if self._fixtures_cache is None:
            self._fixtures_cache = json.loads(
                self.fixture_file.read_text(encoding="utf-8")
            )
        return self._fixtures_cache

    def _next_id(self, kind: str) -> str:
        self._seq += 1
        return f"v6-{kind}-{self._seq}"

    def _drive(self, router_url: str, run_id: str, model: str) -> list[dict]:
        """The AG8 scripted agent client, in process, through one endpoint."""
        from agentic_probe import AgentRunner

        fixtures = self._fixtures()
        runner = AgentRunner(
            router=router_url,
            # no management API on a direct drive; the replay dump degrades
            # to an empty list on its own
            mgmt="http://127.0.0.1:1",
            fixtures=fixtures,
            run_tag="run1",
            session_prefix=run_id,
            sub_agent_header="x-claude-code-agent-id",
            extra_headers={},
            max_tokens=64,
            max_steps=12,
            model=model,
        )
        for session in fixtures["sessions"]:
            runner.run_session(session)
        return runner.transcript

    def _run_mom(self, target_id: str, manifest: dict) -> dict:
        arm = self.mom_targets[target_id]
        run_id = self._next_id(target_id.replace("/", "-"))
        transcript = self._drive(arm.router_url, run_id, ROUTER_MODEL)
        rows = self._arm_rows(arm, run_id)
        continuity = self._continuity(rows, run_id)
        task_scores = {
            name: 1.0 if ok else 0.0
            for name, ok in _task_success(transcript, self._fixtures()).items()
        }
        accuracy = sum(task_scores.values()) / len(task_scores)
        routed_counts: dict[str, int] = {}
        for row in rows:
            model = str(row.get("routed") or "")
            if model:
                routed_counts[model] = routed_counts.get(model, 0) + 1
        dominant = (
            max(routed_counts, key=routed_counts.get) if routed_counts else target_id
        )
        self.runs[run_id] = {
            "target_id": target_id,
            "doc": {"id": run_id, "status": "completed", "manifest": manifest},
            "task_scores": task_scores,
            "accuracy": accuracy,
            "spend": _spend_usd(transcript, self.pricing, dominant),
            "continuity": continuity,
            "subject_calls": sum(
                1 for entry in transcript if entry.get("status") == 200
            ),
            "rows": rows,
        }
        return {"id": run_id, "status": "completed"}

    def _run_reference(self, alias: str, manifest: dict) -> dict:
        run_id = self._next_id(alias.replace("/", "-"))
        transcript = self._drive(self.mock_endpoint, run_id, alias)
        task_scores = {
            name: 1.0 if ok else 0.0
            for name, ok in _task_success(transcript, self._fixtures()).items()
        }
        accuracy = sum(task_scores.values()) / len(task_scores)
        self.runs[run_id] = {
            "target_id": alias,
            "doc": {"id": run_id, "status": "completed", "manifest": manifest},
            "task_scores": task_scores,
            "accuracy": accuracy,
            "spend": _spend_usd(transcript, self.pricing, alias),
            "continuity": None,
            "subject_calls": sum(
                1 for entry in transcript if entry.get("status") == 200
            ),
            "rows": [],
        }
        return {"id": run_id, "status": "completed"}

    def _arm_rows(self, arm: RouterArm, run_id: str) -> list[dict]:
        """Z4's own filter: the run's rows, from the arm's replay store."""
        from recipe_lab.adapters.router_replay import session_rows_for_run

        return session_rows_for_run(arm.records(), run_id)

    def _continuity(self, rows: list[dict], run_id: str) -> dict:
        """AG7's continuity section from one arm run's replay rows.

        Sessions are the run's conversation identities; a session's phase
        comes from its fixture layout (turn 0 is the user turn, every later
        turn of a tool task arrives in the tool loop).
        """
        fixtures = self._fixtures()
        tool_tasks = {
            session["fixture"]
            for session in fixtures["sessions"]
            if fixtures["tasks"][session["task"]].get("tool_call_steps")
        }
        by_session: dict[str, list[dict]] = {}
        prefix = f"{run_id}-"
        for row in rows:
            session = row.get("session") or {}
            conversation = str(session.get("conversation_id") or "")
            if not conversation.startswith(prefix):
                continue
            fixture = conversation[len(prefix) :].rsplit("-run1", 1)[0]
            by_session.setdefault(fixture, []).append(row)
        switches_by_phase: dict[str, int] = {}
        per_task: list[int] = []
        multi = 0
        for fixture in sorted(by_session):
            requests = sorted(
                by_session[fixture],
                key=lambda row: int(row.get("session", {}).get("turn_index") or 0),
            )
            routed = [str(row.get("routed") or "") for row in requests if row.get("routed")]
            if len(requests) > 1:
                multi += 1
            task_switches = 0
            for index in range(1, len(routed)):
                if routed[index] == routed[index - 1]:
                    continue
                task_switches += 1
                phase = "user_turn" if index == 0 else "tool_loop"
                if index > 0 and fixture not in tool_tasks:
                    phase = "user_turn"
                switches_by_phase[phase] = switches_by_phase.get(phase, 0) + 1
            per_task.append(task_switches)
        switched = sum(1 for count in per_task if count > 0)
        mean = sum(per_task) / len(per_task) if per_task else 0.0
        return {
            "session_mode": "session_aware",
            "multi_request_tasks": multi,
            "model_switches": sum(per_task),
            "switched_tasks": switched,
            "unswitched_tasks": len(per_task) - switched,
            "mean_switches_per_task": mean,
            "model_switches_by_phase": switches_by_phase,
        }


# ---- module fixtures ----


@pytest.fixture(scope="module")
def router_bin() -> str:
    value = os.environ.get("RECIPE_LAB_ROUTER_BIN", "")
    if not value:
        pytest.skip("RECIPE_LAB_ROUTER_BIN is not set; real router required")
    return value


@pytest.fixture(scope="module")
def recipe_lab_bin(tmp_path_factory) -> Path:
    bin_dir = tmp_path_factory.mktemp("recipe-lab-bin")
    wrapper = bin_dir / "recipe-lab"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec env PYTHONPATH="{RECIPE_LAB_ROOT}" "{sys.executable}" '
        '-m recipe_lab.cli "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    return bin_dir


@pytest.fixture(scope="module")
def stack(tmp_path_factory, router_bin):
    """The shared V6 stack: the mock backend (V6 fixtures), the recorded
    route cases, the contract spec, the per-model run reports and the arm
    registry (lazy-booted routers, stopped at teardown)."""
    fixtures = _fixtures()
    stack: dict = {"arms": {}}
    mock_port = _free_port()
    mock_proc = subprocess.Popen(
        [
            sys.executable,
            str(SMOKE_DIR / "mock_backend.py"),
            "--port",
            str(mock_port),
            "--agent-fixtures",
            str(V6_FIXTURES),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _wait_mock() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                _http_json(f"http://127.0.0.1:{mock_port}/v1/models", 2.0)
                return
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                time.sleep(0.2)
        raise RuntimeError("mock backend not ready within 30s")

    try:
        _wait_mock()
        if str(SMOKE_DIR) not in sys.path:
            sys.path.insert(0, str(SMOKE_DIR))  # the AG8 probe import
        work = tmp_path_factory.mktemp("v6-work")

        cases_path = work / "cases.jsonl"
        rows = []
        for session in fixtures["sessions"]:
            rows.append(
                {
                    "case_id": session["fixture"],
                    "session_id": f"v6-{session['fixture']}",
                    "requests": _session_requests(fixtures, session),
                }
            )
        cases_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        spec: dict = {
            "api_version": "recipe-lab/v1alpha1",
            "objective": {
                "metric_key": METRIC_KEY,
                "direction": "maximize",
                "min_improvement_pct": 10.0,
                "min_cases": 2,
                "replay_metric": "routed_accuracy",
            },
            "workloads": [
                {
                    "name": "agentic-smoke",
                    "kind": "agentic",
                    "benchmark": "terminal-bench-2.1",
                    "profile": "quick",
                    "repetitions": 1,
                    "split_rule": "scripted AG8 sessions, one task each",
                    "agentic_gates": {
                        "max_decision_drift_switches_per_task": DRIFT_CEILING,
                        # numeric ceiling: the tuning scenario's
                        # protection-disabled baseline resolves no session
                        # identity, so a "baseline" interruption pin could
                        # never evaluate
                        "max_tool_loop_interrupted": 0,
                    },
                    "splits": {
                        "frozen_test": {
                            "manifest": "cases.jsonl",
                            "content_hash": _sha256_file(cases_path),
                        }
                    },
                }
            ],
            "attribution": {"oracle_policy": {"domain_code": MODEL_B}},
            "references": {"strong": MODEL_B, "weak": MODEL_A},
            "cost_basis": {
                "source": "provider_pricing",
                "assumed_pricing": {
                    MODEL_A: {"prompt_per_1m": 1.0, "completion_per_1m": 1.0},
                    MODEL_B: {"prompt_per_1m": 1.0, "completion_per_1m": 4.0},
                },
            },
            "budget": {"rounds": 1, "max_minutes_per_trial": 15},
            "gates": [],
        }
        spec_path = work / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

        # the negative scenario's contract: the AG7 default gates with one
        # declared override - a strict drift ceiling (0.1) the switching
        # candidate is judged against (its single release re-baselines the
        # lock, so the per-task rate lands exactly on the default 0.2 and
        # only a stricter ceiling separates it from the accepted config)
        negative_spec = copy.deepcopy(spec)
        negative_spec["workloads"][0]["agentic_gates"] = {
            "max_decision_drift_switches_per_task": 0.1,
        }
        spec_default_path = work / "spec-negative.yaml"
        spec_default_path.write_text(
            yaml.safe_dump(negative_spec, sort_keys=False), encoding="utf-8"
        )

        reports = {}
        for name, accuracy, spend, calls in (
            (MODEL_A, 0.4, 4.0, 8),
            (MODEL_B, 1.0, 16.0, 24),
        ):
            path = work / f"report-{name.replace('/', '-')}.json"
            path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "targets": [
                                {
                                    "id": name,
                                    "kind": "single",
                                    "accuracy": accuracy,
                                    "total": len(fixtures["sessions"]),
                                    "total_spend_usd": spend,
                                    "cache_neutral_cost_usd": spend * 1.2,
                                    "latency_p50_s": 1.0,
                                    "output_diagnostics": {
                                        "planned_cases": len(fixtures["sessions"]),
                                        "output_limit_cases": 0,
                                        "subject_calls": {"total": calls},
                                    },
                                }
                            ]
                        }
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            reports[name] = path

        yield {
            "router_bin": router_bin,
            "mock_endpoint": f"http://127.0.0.1:{mock_port}",
            "work": work,
            "cases_path": cases_path,
            "spec_path": spec_path,
            "spec_default_path": spec_default_path,
            "fixtures": fixtures,
            "reports": reports,
            "arms": {},
        }
    finally:
        for arm in stack["arms"].values():
            arm.stop()
        mock_proc.terminate()
        try:
            mock_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock_proc.kill()


# ---- config builders and command helpers ----


def _endpoint_hostport(mock_endpoint: str) -> str:
    return mock_endpoint.split("//", 1)[1]


def _write_config(stack: dict, config: dict, name: str) -> Path:
    path = stack["work"] / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def _load(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _bind_mock_and_return(stack: dict, config: dict) -> dict:
    """Bind every backend ref to the shared mock and return the config."""
    endpoint = _endpoint_hostport(stack["mock_endpoint"])
    for model in config.get("providers", {}).get("models") or []:
        for ref in model.get("backend_refs") or []:
            ref["endpoint"] = endpoint
    return config


def _protection(config: dict) -> dict:
    return config["global"]["router"]["learning"]["protection"]


def _disabled_protection_config(stack: dict) -> Path:
    """The tuning baseline: the cold-start config with protection disabled
    (§7 V6 scenario 2's 'deliberately disable protection', the doc's
    letter). Every in-loop switch lands in the unprotected residual; the
    trade-off is that a disabled protection resolves no session identity,
    so the arm's replay records carry no session fields."""
    disabled = copy.deepcopy(_load(stack["work"] / "cold-start.yaml"))
    _protection(disabled)["enabled"] = False
    return _write_config(stack, disabled, "disabled-protection")


def _observe_opt_out_config(stack: dict) -> Path:
    """The negative candidate: the cold-start config with the drifting
    decision opted out of learning enforcement (``adaptations.mode:
    observe`` on the decision the mid-loop drift arrives at). The identity
    still resolves - so the Z4 rows keep their session evidence - and the
    real stack proved the releases stay traced as decision drift, so the
    candidate is eliminated through the DEFAULT drift ceiling (0.2), which
    its 2-releases-per-5-sessions rate exceeds."""
    broken = copy.deepcopy(_load(stack["work"] / "cold-start.yaml"))
    for decision in broken["routing"]["decisions"]:
        if decision["name"] == "simple_general_local":
            decision["adaptations"] = {"mode": "observe"}
    return _write_config(stack, broken, "negative-observe")


def _decision_models(config: dict) -> dict[str, str]:
    return {
        d["name"]: d["modelRefs"][0]["model"] for d in config["routing"]["decisions"]
    }


def _arm(stack: dict, name: str, config_path: Path) -> RouterArm:
    """One shared arm router per name, booted lazily and kept alive."""
    arm = stack["arms"].get(name)
    if arm is None:
        arm = RouterArm(
            name,
            config_path,
            stack["router_bin"],
            os.environ.get("RECIPE_LAB_HF_HOME") or None,
            stack["work"] / f"arm-{name}",
        )
        stack["arms"][name] = arm
    return arm


def _mom_entry(stack: dict, target_id: str, config_path: Path) -> dict:
    """A registry entry whose config hash is the ARM ROUTER's own runtime
    hash - the same binding sr-bench freezes at target registration."""
    arm = _arm(stack, target_id, config_path)
    return {
        "kind": "mom",
        "mode": "live",
        "config_hash": arm.config_hash,
        "session_mode": "session_aware",
        "arm": arm,
    }


def _run_recipe_lab(recipe_lab_bin: Path, argv: list[str], timeout: int = 1800) -> str:
    proc = subprocess.run(
        [str(recipe_lab_bin / "recipe-lab"), *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return proc.stdout


def _run_route_replay(
    stack: dict,
    recipe_lab_bin: Path,
    *,
    candidates: Path,
    ledger: Path,
    out: Path,
    spec_path: Path | None = None,
) -> list[dict]:
    argv = [
        "run",
        "--tier",
        "route-replay",
        "--spec",
        str(spec_path or stack["spec_path"]),
        "--cases",
        str(stack["cases_path"]),
        "--candidates",
        str(candidates),
        "--proposer",
        "bootstrap",
        "--router-bin",
        stack["router_bin"],
        "--mock-backend",
        str(SMOKE_DIR / "mock_backend.py"),
        "--python-bin",
        sys.executable,
        "--backend-port",
        str(_free_port()),
        "--ready-timeout",
        "600",
        "--output-dir",
        str(out),
        "--ledger",
        str(ledger),
        "--round",
        "1",
        "--router-model",
        ROUTER_MODEL,
    ]
    if os.environ.get("RECIPE_LAB_HF_HOME"):
        argv += ["--hf-home", os.environ["RECIPE_LAB_HF_HOME"]]
    _run_recipe_lab(recipe_lab_bin, argv)
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _route_evidence(records: list[dict]) -> dict:
    route_records = [r for r in records if r["tier"] == "route_replay"]
    assert route_records, "route-replay appended no ledger record"
    return route_records[0]


def _verify(
    stack: dict,
    *,
    registry: dict[str, dict],
    candidate_target: str,
    baseline_target: str,
    work_dir: Path,
    spec_path: Path | None = None,
) -> tuple[dict, LiveArmCli]:
    """The live four-arm comparison through the real verify_live.

    The baseline manifest's MoM target is the baseline arm; the candidate
    target rides ``targets``; the single-model references come from the
    registry. Both arms' replay records are fetched by verify_live itself
    (Z4) from the arms' management endpoints.
    """
    mom_targets = {
        target: entry["arm"]
        for target, entry in registry.items()
        if entry.get("kind") == "mom"
    }
    cli = LiveArmCli(
        registry=registry,
        mom_targets=mom_targets,
        mock_endpoint=stack["mock_endpoint"],
        fixture_file=V6_FIXTURES,
        pricing=PRICING,
    )
    spec = ExperimentSpec.from_dict(_load(spec_path or stack["spec_path"]))
    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest = work_dir / "baseline-manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "manifest_id": "v6-baseline",
                "dataset": {
                    "path": "cases.jsonl",
                    "sha256": _sha256_file(stack["cases_path"]),
                },
                "targets": [
                    {"id": baseline_target, "kind": "mom"},
                    {"id": MODEL_B, "kind": "single", "mode": "live"},
                ],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    candidate_arm = registry[candidate_target]["arm"]
    baseline_arm = registry[baseline_target]["arm"]
    result = verify_live(
        spec=spec,
        cli=cli,
        baseline_run_id=None,
        baseline_manifest=str(baseline_manifest),
        targets=[candidate_target],
        repetitions=1,
        measurements=[],
        work_dir=work_dir,
        workload_split="frozen_test",
        workload=spec.select_workload(None),
        candidate_management_endpoint=candidate_arm.management_url,
        baseline_management_endpoint=baseline_arm.management_url,
    )
    return result, cli


def _task_first_models(rows: list[dict], run_id: str) -> dict[str, str]:
    """Per-task (fixture) first-turn routed model from one run's rows."""
    by_fixture: dict[str, list[tuple[int, str]]] = {}
    prefix = f"{run_id}-"
    for row in rows:
        session = row.get("session") or {}
        conversation = str(session.get("conversation_id") or "")
        if not conversation.startswith(prefix):
            continue
        fixture = conversation[len(prefix) :].rsplit("-run1", 1)[0]
        model = str(row.get("routed") or "")
        turn = int(session.get("turn_index") or 0)
        by_fixture.setdefault(fixture, []).append((turn, model))
    first: dict[str, str] = {}
    for fixture, turns in by_fixture.items():
        turns.sort()
        if turns and turns[0][1]:
            first[fixture] = turns[0][1]
    return first


# ---- scenario 1: cold start ----


def test_v6_scenario_1_cold_start(stack, recipe_lab_bin, tmp_path):
    """bootstrap-agentic -> route-replay -> live four-arm -> PASS, with the
    candidate routing complex tasks to MODEL_B, simple ones to MODEL_A, and
    a task success rate strictly above the random mix R."""
    profile_path = stack["work"] / "agentic-profile.json"
    _run_recipe_lab(
        recipe_lab_bin,
        [
            "profile-agentic",
            "--reports",
            str(stack["reports"][MODEL_A]),
            "--reports",
            str(stack["reports"][MODEL_B]),
            "--output",
            str(profile_path),
        ],
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    by_model = {entry["model"]: entry for entry in profile["models"]}
    assert by_model[MODEL_A]["success_rate"] == pytest.approx(0.4)
    assert by_model[MODEL_B]["success_rate"] == pytest.approx(1.0)

    cands = stack["work"] / "cands"
    _run_recipe_lab(
        recipe_lab_bin,
        [
            "bootstrap-agentic",
            "--spec",
            str(stack["spec_path"]),
            "--profile",
            str(profile_path),
            "--template",
            str(V6_TEMPLATE),
            "--client",
            "generic",
            "--lambda",
            "0",
            "--out",
            str(cands),
        ],
    )
    sidecar = json.loads(
        (cands / "agentic-config.bootstrap.json").read_text(encoding="utf-8")
    )
    lanes = {lane["decision"]: lane["model"] for lane in sidecar["lanes"]}
    # the ranked code lanes take the accuracy winner (lambda 0); the
    # preserved non-lane decision keeps the cheap model (Z10's uniform
    # unmeasured-lane selection + the template's structural split)
    assert lanes["domain_code_complex"] == MODEL_B
    assert lanes["domain_code"] == MODEL_B
    assert sidecar["client"] == "generic"
    assert sidecar["scope"] == "conversation"
    assert sidecar["identity_headers"] == {
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    }
    assert sidecar["adaptation_enabled"] is False
    assert sidecar["progress_gate"] == {"mode": "observe"}

    cold_start = _write_config(
        stack,
        _bind_mock_and_return(stack, _load(cands / "agentic-config.yaml")),
        "cold-start",
    )
    config = _load(cold_start)
    protection = _protection(config)
    assert protection["enabled"] is True
    assert protection["scope"] == "conversation"
    assert protection["identity"]["headers"] == {
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    }
    decisions = _decision_models(config)
    assert decisions["domain_code_complex"] == MODEL_B
    assert decisions["domain_code"] == MODEL_B
    assert decisions["simple_general_local"] == MODEL_A

    # route-replay tier: the recorded sessions rank; the drift release stays
    # traced (in-design), so the unprotected residual is zero and no gate
    # is violated
    ledger = stack["work"] / "ledger-cold-start.jsonl"
    records = _run_route_replay(
        stack,
        recipe_lab_bin,
        candidates=cold_start.parent / cold_start.name,
        ledger=ledger,
        out=stack["work"] / "route-replay-cold-start",
    )
    record = _route_evidence(records)
    assert record.get("status") is None, (
        f"reasons={record['reasons']} gates={record['gates']} "
        f"notes={record['notes']}"
    )
    evidence = json.loads(record["notes"].split("routing-evidence=", 1)[1])
    assert evidence["tool_loop_switches"] == evidence["decision_drift_switches"]
    assert evidence["first_turn_decisions"].get("simple_general_local", 0) >= 1
    assert (
        evidence["first_turn_decisions"].get("domain_code_complex", 0)
        + evidence["first_turn_decisions"].get("domain_code", 0)
        >= 1
    )

    # determinism (criterion 5): two runs of the same fixture set through
    # the candidate arm produce identical trajectories
    arm = _arm(stack, "cold-start", cold_start)
    from agentic_probe import AgentRunner, compare_transcripts

    def _drive(tag: str) -> list[dict]:
        runner = AgentRunner(
            router=arm.router_url,
            mgmt=arm.management_url,
            fixtures=stack["fixtures"],
            run_tag=tag,
            session_prefix=f"v6-det-{tag}",
            sub_agent_header="x-claude-code-agent-id",
            extra_headers={},
            max_tokens=64,
            max_steps=12,
        )
        for session in stack["fixtures"]["sessions"]:
            runner.run_session(session)
        return runner.transcript

    assert compare_transcripts(_drive("det1"), _drive("det2")) == []

    # the live four-arm comparison: C vs the all-weak baseline, S/W
    # references and R computed from the observed shares. The candidate arm
    # is the SAME router the determinism drives used (one arm per config)
    all_weak = _load(cold_start)
    for decision in all_weak["routing"]["decisions"]:
        decision["modelRefs"][0]["model"] = MODEL_A
    all_weak_path = _write_config(stack, all_weak, "all-weak")
    registry = {
        "mom-baseline": _mom_entry(stack, "mom-baseline", all_weak_path),
        "mom-cold-start": {
            "kind": "mom",
            "mode": "live",
            "config_hash": arm.config_hash,
            "session_mode": "session_aware",
            "arm": arm,
        },
        MODEL_A: {"kind": "single", "mode": "live"},
        MODEL_B: {"kind": "single", "mode": "live"},
    }
    result, cli = _verify(
        stack,
        registry=registry,
        candidate_target="mom-cold-start",
        baseline_target="mom-baseline",
        work_dir=tmp_path / "verify-cold-start",
    )
    assert result["verdict"] == "PASS", result["reasons"]
    gates = result["agentic"]["gates"]
    assert gates["max_unprotected_tool_loop_switches"] == "pass"
    assert gates["max_selection_rejected"] == "pass"
    # Y4: the verdict names the config the candidate arm actually served
    assert result["config_sha256"] == "sha256:" + arm.config_hash

    # the four arms' observed behavior, from the stub's recorded runs
    cand_runs = [
        run
        for run in cli.runs.values()
        if run["target_id"] == "mom-cold-start" and run["rows"]
    ]
    assert cand_runs, "the candidate arm produced no replay rows"
    cand_run = cand_runs[-1]
    first_models = _task_first_models(cand_run["rows"], _run_id_of(cand_run))
    for name in COMPLEX_TASKS:
        assert first_models.get(name) == MODEL_B, (name, first_models)
    assert first_models.get("simple") == MODEL_A

    reference_runs = {
        run["target_id"]: run for run in cli.runs.values() if run["target_id"] in PRICING
    }
    assert reference_runs[MODEL_A]["accuracy"] == pytest.approx(0.4)
    assert reference_runs[MODEL_B]["accuracy"] == pytest.approx(1.0)

    # the R arm: share-weighted reference expectation over the candidate's
    # per-task first-turn model shares - strictly below the candidate
    shares: dict[str, float] = {}
    for model in first_models.values():
        shares[model] = shares.get(model, 0) + 1
    total = sum(shares.values())
    shares = {model: count / total for model, count in shares.items()}
    r_stats = session_random_mix_stats(
        cand_run["task_scores"],
        {
            MODEL_A: dict(reference_runs[MODEL_A]["task_scores"]),
            MODEL_B: dict(reference_runs[MODEL_B]["task_scores"]),
        },
        shares,
    )
    assert r_stats is not None
    assert r_stats["candidate_accuracy"] == pytest.approx(1.0)
    assert r_stats["candidate_accuracy"] > r_stats["expected_r_accuracy"]


def _run_id_of(run: dict) -> str:
    return run["doc"]["id"]


# ---- scenario 2: tuning ----


def test_v6_scenario_2_tuning(stack, recipe_lab_bin, tmp_path):
    """The protection-disabled baseline (the doc's 'deliberately disable
    protection') -> diagnose + attribute-session -> propose-session-policy
    (the W1 handoff) -> the remedy's route-replay (unprotected residual 0)
    -> the live stability gate."""
    cold_start = stack["work"] / "cold-start.yaml"
    disabled_path = _disabled_protection_config(stack)

    # route-replay tier: every in-loop switch lands in the unprotected
    # residual - a disabled protection traces nothing
    ledger = stack["work"] / "ledger-disabled.jsonl"
    records = _run_route_replay(
        stack,
        recipe_lab_bin,
        candidates=disabled_path,
        ledger=ledger,
        out=stack["work"] / "route-replay-disabled",
    )
    record = _route_evidence(records)
    evidence = json.loads(record["notes"].split("routing-evidence=", 1)[1])
    assert evidence["tool_loop_switches"] >= 1
    assert evidence["decision_drift_switches"] == 0
    summary_path = stack["work"] / "route-replay-disabled-summary.json"
    summary_path.write_text(json.dumps(evidence, indent=1), encoding="utf-8")

    # diagnose over the disabled arm's real records (AG4 phase split), twice,
    # byte-identical (criterion 5). The arm is driven live first: the
    # route-replay tier uses its own ephemeral routers, so the arm's replay
    # store only carries what the probe sent it.
    disabled_arm = _arm(stack, "disabled", disabled_path)
    from agentic_probe import AgentRunner

    tune_runner = AgentRunner(
        router=disabled_arm.router_url,
        mgmt=disabled_arm.management_url,
        fixtures=stack["fixtures"],
        run_tag="tune1",
        session_prefix="v6-tune",
        sub_agent_header="x-claude-code-agent-id",
        extra_headers={},
        max_tokens=64,
        max_steps=12,
    )
    for session in stack["fixtures"]["sessions"]:
        tune_runner.run_session(session)
    rows_path = stack["work"] / "diag-rows.jsonl"
    rows = records_to_rows(disabled_arm.records())
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    diag_a = stack["work"] / "diag-a"
    diag_b = stack["work"] / "diag-b"
    for out in (diag_a, diag_b):
        _run_recipe_lab(
            recipe_lab_bin,
            [
                "diagnose",
                "--records",
                str(rows_path),
                "--config",
                str(disabled_path),
                "--output-dir",
                str(out),
                "--source",
                "V6 tuning baseline (protection-disabled arm)",
            ],
        )
    assert (diag_a / "diagnosis.json").read_bytes() == (
        diag_b / "diagnosis.json"
    ).read_bytes()
    diagnosis = json.loads((diag_a / "diagnosis.json").read_text(encoding="utf-8"))
    # a protection-disabled arm resolves no session identity, so the AG4
    # agentic section (phase distributions, separation, failures) is
    # honestly absent - the W1 shape's own evidence
    assert "agentic" not in diagnosis
    assert diagnosis["decision_coverage"]["total_decisions"] == 3

    # attribute-session over the same records: a protection-disabled config
    # resolves no session identity, so the session attribution honestly
    # reports zero session rows - the W1 shape's own evidence
    sess_dir = stack["work"] / "session-report"
    _run_recipe_lab(
        recipe_lab_bin,
        [
            "attribute-session",
            "--records",
            str(rows_path),
            "--output-dir",
            str(sess_dir),
        ],
    )
    session_events = json.loads(
        (sess_dir / "session-events.json").read_text(encoding="utf-8")
    )
    # the session attribution still reads the wire identity: the disabled
    # arm's records carry the conversation fields, and the drift-loop's
    # unprotected switch shows up as the lane flap event
    assert any(event["class"] == "lane_flap" for event in session_events)

    # propose-session-policy over the disabled config: Z6's cause model
    # hands the unprotected switches back as the W1-class cause (re-enabling
    # enforcement is not a protection knob), so no candidate file is
    # written and the remedy is the handoff's own prescription - the
    # enforced cold-start config
    spp_out = stack["work"] / "spp"
    spp_stdout = _run_recipe_lab(
        recipe_lab_bin,
        [
            "propose-session-policy",
            "--config",
            str(disabled_path),
            "--replay-summary",
            str(summary_path),
            "--output-dir",
            str(spp_out),
        ],
    )
    spp = json.loads(spp_stdout)
    handoff_reasons = [h["reason"] for h in spp["handoffs"]]
    assert "unprotected_tool_loop_switches" in handoff_reasons, spp["handoffs"]
    assert not spp["candidates"], "re-enabling enforcement became a knob"

    # the remedy's route-replay: the protection engages, the in-loop
    # switches are all traced drift releases, the unprotected residual is
    # zero (criterion 2)
    tuned_ledger = stack["work"] / "ledger-tuned.jsonl"
    tuned_records = _run_route_replay(
        stack,
        recipe_lab_bin,
        candidates=cold_start,
        ledger=tuned_ledger,
        out=stack["work"] / "route-replay-tuned",
    )
    tuned_record = _route_evidence(tuned_records)
    assert tuned_record.get("status") is None, tuned_record["reasons"]
    tuned_evidence = json.loads(
        tuned_record["notes"].split("routing-evidence=", 1)[1]
    )
    assert tuned_evidence["tool_loop_switches"] == tuned_evidence["decision_drift_switches"]

    # live comparison: baseline (protection disabled) vs the remedy. The two
    # arms complete the same tasks (protection changes stability, not
    # accuracy), so the verdict need not clear the accuracy bar - criterion
    # 2 is the stability gate: the tuned arm's unprotected residual is zero.
    registry = {
        "mom-disabled": _mom_entry(stack, "mom-disabled", disabled_path),
        "mom-tuned": {
            "kind": "mom",
            "mode": "live",
            "config_hash": stack["arms"]["cold-start"].config_hash,
            "session_mode": "session_aware",
            "arm": stack["arms"]["cold-start"],
        },
        MODEL_A: {"kind": "single", "mode": "live"},
        MODEL_B: {"kind": "single", "mode": "live"},
    }
    result, cli = _verify(
        stack,
        registry=registry,
        candidate_target="mom-tuned",
        baseline_target="mom-disabled",
        work_dir=tmp_path / "verify-tuning",
    )
    gates = result["agentic"]["gates"]
    assert gates["max_unprotected_tool_loop_switches"] == "pass", result["reasons"]
    # the gates judged the arms' own rows: the disabled baseline's rows
    # carry its switches (the wire identity rides the records even with the
    # protection off - only the traces are gone)
    baseline_rows = result["agentic"]["replay_evidence"]["baseline"]
    assert baseline_rows["rows"] and baseline_rows["rows"] >= 1
    tuned_runs = [
        run
        for run in cli.runs.values()
        if run["target_id"] == "mom-tuned" and run["rows"]
    ]
    assert tuned_runs, "the tuned arm produced no replay rows"


# ---- scenario 3: negative ----


def test_v6_scenario_3_negative(stack, recipe_lab_bin, tmp_path):
    """The enforcement-opt-out candidate is eliminated on the route-replay
    tier (the DEFAULT drift ceiling) and FAILs the live verdict through the
    agentic gates fed from its own arm's Z4-fetched replay rows."""
    cold_start = stack["work"] / "cold-start.yaml"
    cold_arm = _arm(stack, "cold-start", cold_start)
    negative_path = _observe_opt_out_config(stack)

    # eliminated: under the negative contract's strict drift ceiling (0.1)
    # the candidate's traced drift release over five sessions exceeds the
    # per-task rate, and the route-replay tier records the violation
    ledger = stack["work"] / "ledger-negative.jsonl"
    records = _run_route_replay(
        stack,
        recipe_lab_bin,
        candidates=negative_path,
        ledger=ledger,
        out=stack["work"] / "route-replay-negative",
        spec_path=stack["spec_default_path"],
    )
    record = _route_evidence(records)
    assert record.get("status") == "gate-violated", (
        f"reasons={record['reasons']} gates={record['gates']} "
        f"notes={record['notes']}"
    )
    assert record["gates"]["max_decision_drift_switches_per_task"] == "fail"

    # live: the same candidate FAILs against the protected config - the
    # candidate arm's own replay rows carry the drift releases, and the
    # default contract's stability gate vetoes the verdict
    registry = {
        "mom-cold-start": {
            "kind": "mom",
            "mode": "live",
            "config_hash": cold_arm.config_hash,
            "session_mode": "session_aware",
            "arm": cold_arm,
        },
        "mom-negative": _mom_entry(stack, "mom-negative", negative_path),
        MODEL_A: {"kind": "single", "mode": "live"},
        MODEL_B: {"kind": "single", "mode": "live"},
    }
    result, cli = _verify(
        stack,
        registry=registry,
        candidate_target="mom-negative",
        baseline_target="mom-cold-start",
        work_dir=tmp_path / "verify-negative",
        spec_path=stack["spec_default_path"],
    )
    assert result["verdict"] == "FAIL", result["reasons"]
    gates = result["agentic"]["gates"]
    assert gates["max_decision_drift_switches_per_task"] == "fail"
    assert "agentic stability gate exceeded" in " ".join(result["reasons"])
    # the verdict judged the candidate arm's own rows
    negative_runs = [
        run
        for run in cli.runs.values()
        if run["target_id"] == "mom-negative" and run["rows"]
    ]
    assert negative_runs, "the negative arm produced no replay rows"


# ---- scenario 4: sub agent ----


def test_v6_scenario_4_sub_agent(stack, recipe_lab_bin, tmp_path):
    """No agent identity header configured: the diagnosis reports the
    sub-agents' identity as not separate, and the proposer answers with the
    agent-header candidate."""
    cold_start = stack["work"] / "cold-start.yaml"
    # a DEDICATED arm: the Z9 join compares each sub-agent request's
    # identity hash against the MOST COMMON identity among the other rows,
    # so the store must carry only this run's traffic
    arm = _arm(stack, "sub4", cold_start)

    # one dedicated probe run of the sub-agent fixture session only - the
    # manifest names this run's sub-agent requests, and the main turns of
    # the same conversation provide the identity the sub turns must share
    from agentic_probe import AgentRunner, agent_request_manifest

    runner = AgentRunner(
        router=arm.router_url,
        mgmt=arm.management_url,
        fixtures=stack["fixtures"],
        run_tag="sub4",
        session_prefix="v6-sub4",
        sub_agent_header="x-claude-code-agent-id",
        extra_headers={},
        max_tokens=64,
        max_steps=12,
    )
    for session in stack["fixtures"]["sessions"]:
        if session.get("sub_agent_tool"):
            runner.run_session(session)
    manifest_path = stack["work"] / "agent-requests-sub4.json"
    manifest_path.write_text(
        json.dumps(agent_request_manifest(runner.transcript), indent=1),
        encoding="utf-8",
    )
    assert manifest_path.read_text(encoding="utf-8").count("request_id") >= 1

    rows = records_to_rows(arm.records())
    rows_path = stack["work"] / "diag-rows-sub4.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    diag = stack["work"] / "diag-sub4"
    _run_recipe_lab(
        recipe_lab_bin,
        [
            "diagnose",
            "--records",
            str(rows_path),
            "--config",
            str(cold_start),
            "--output-dir",
            str(diag),
            "--agent-requests",
            str(manifest_path),
            "--source",
            "V6 sub-agent scenario (generic client, agent header unmapped)",
        ],
    )
    diagnosis = json.loads((diag / "diagnosis.json").read_text(encoding="utf-8"))
    separation = ((diagnosis.get("agentic") or {}).get("agent_separation")) or {}
    assert separation.get("status") == "not_separate", separation

    # the proposer answers with the agent-header candidate
    spp_out = stack["work"] / "spp-sub4"
    _run_recipe_lab(
        recipe_lab_bin,
        [
            "propose-session-policy",
            "--config",
            str(cold_start),
            "--diagnosis",
            str(diag / "diagnosis.json"),
            "--output-dir",
            str(spp_out),
        ],
    )
    candidates = sorted(spp_out.glob("candidate-*.yaml"))
    assert candidates, "no agent-header candidate proposed"
    agent_candidate = _load(candidates[0])
    headers = _protection(agent_candidate)["identity"]["headers"]
    assert headers.get("agent") == "x-claude-code-agent-id"
