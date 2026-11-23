"""Z7: the route-replay tier against the REAL router (AG3 + AG6验收).

The fake-router tests (test_route_replay.py) validate the evidence
plumbing; this one validates the chain end to end on a router built from
the stack top: recorded sessions replayed through `run --tier route-replay`,
the agent recipe's decisions classifying through the real decision model,
session protection holding the tool loop, and the Z5 cause split computing
from the replay records' own protection traces.

Three candidates replay the same recorded session (one code task whose
tool loop is joined mid-way by two deliberately general requests - the
drift prompts are unambiguous so they classify general):

- protected: the agent recipe with conversation-scoped protection on -
  the in-loop requests hard-lock onto the baseline model, the general
  requests release the lock as decision drift (traced decision_drift), so
  the unprotected residual stays at 0 (gate passes) while the per-task
  drift rate exceeds its ceiling (gate fails) - the two gates judge the
  same switches by different causes.
- protection disabled: no traces ride the records, so every in-loop
  switch lands in the unprotected residual (gate fails) and the drift
  gate sees zero (passes) - the "decision_drift counted separately"
  proof is this contrast.
- missing conversation header: the protected config replayed with
  `session_only_identity` cases (only x-session-id on the wire) - the
  conversation identity resolves missing, the protection skips the
  request entirely, and the gate fails exactly like the disabled one.

Skips unless RECIPE_LAB_ROUTER_BIN points at a router built from the
stack top; the classifier artifact (vllm-sr/Vela-2.0-0.8B, the AG8
real-stack decision model) downloads through RECIPE_LAB_HF_HOME. The
general-in-loop classification is the one model-behavior dependency: the
prompts are deliberately unambiguous, and the H20 run verifies it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

RECIPE_LAB_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def router_bin() -> str:
    value = os.environ.get("RECIPE_LAB_ROUTER_BIN", "")
    if not value:
        pytest.skip("RECIPE_LAB_ROUTER_BIN is not set; real router required")
    return value


@pytest.fixture(scope="module")
def recipe_lab_bin(tmp_path_factory) -> Path:
    """A dir with the CLI wrapper the run command is invoked through."""
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


def _run_recipe_lab(recipe_lab_bin: Path, argv: list[str]) -> None:
    proc = subprocess.run(
        [str(recipe_lab_bin / "recipe-lab"), *argv],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _candidate_config(
    tmp_path: Path, backend_port: int, name: str, *, protection: bool
) -> Path:
    """The agent recipe's decisions on the AG8 real-stack infrastructure."""
    fixture = yaml.safe_load(
        (RECIPE_LAB_ROOT / "tests" / "fixtures" / "agent_recipe_min.yaml").read_text(
            encoding="utf-8"
        )
    )
    replay_plugin = {
        "type": "router_replay",
        "configuration": {
            "capture_request_body": True,
            "capture_response_body": True,
            "enabled": True,
            "max_body_bytes": 8192,
            "max_records": 1000,
            "max_tool_trace_steps": 100,
        },
    }
    decisions = []
    for decision in fixture["routing"]["decisions"]:
        entry = dict(decision)
        entry["plugins"] = [dict(replay_plugin)]
        decisions.append(entry)
    backend = f"127.0.0.1:{backend_port}"

    def backend_ref(model: str) -> dict:
        return {
            "name": model,
            "provider_model_id": model,
            "api_format": "openai",
            "backend_refs": [
                {
                    "name": "mock-backend",
                    "endpoint": backend,
                    "protocol": "http",
                    "provider": "vllm",
                    "weight": 1,
                }
            ],
        }

    backend_refs = [
        backend_ref("qwen/qwen3.6-rocm"),
        backend_ref("google/gemini-2.5-flash-lite"),
        backend_ref("google/gemini-3.1-pro"),
    ]
    # the recipe's modelCards name more models than this candidate provides;
    # the router rejects a modelCard no provider model backs, so keep only
    # the cards the mock backend actually serves
    provided = {model["name"] for model in backend_refs}
    model_cards = [
        card
        for card in fixture["routing"]["modelCards"]
        if card.get("name") in provided
    ]
    config = {
        "version": "v0.3",
        "listeners": [
            {"name": "http", "address": "0.0.0.0", "port": 8899, "timeout": "1200s"}
        ],
        "global": {
            "router": {
                "learning": {
                    "enabled": True,
                    "adaptation": {"enabled": False},
                    "protection": {
                        "enabled": protection,
                        "scope": "conversation",
                        "identity": {
                            "headers": {
                                "session": "x-session-id",
                                "conversation": "x-conversation-id",
                            }
                        },
                        "tuning": {"idle_timeout_seconds": 300},
                    },
                }
            },
            "model_catalog": {
                "deployments": {
                    "primary": {
                        "provider": "model_runtime",
                        # the AG8 real-stack decision model; the d3 family's
                        # CPU golden check is not ready to gate readiness here
                        "artifact": "vllm-sr/Vela-2.0-0.8B",
                    }
                },
                "system": {"decision_model": {"deployment": "primary"}},
            },
        },
        "providers": {"models": backend_refs},
        "routing": {
            "modelCards": model_cards,
            "signals": fixture["routing"]["signals"],
            "decisions": decisions,
        },
        "entrypoints": [{"model_names": ["vllm-sr/auto"], "recipe": "default"}],
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


BASE_TASK = (
    "Debug a segfault in our JSON parser when a required field is missing; "
    "find the root cause."
)
DRIFT_TASK = "What is the capital of France?"


def _session_requests() -> list[dict]:
    """One recorded session: a code task whose loop is joined by two
    deliberately general requests (the drift pair)."""

    def loop_turn(call_id: str) -> dict:
        # OpenAI chat schema: type and arguments are required on tool calls
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }

    user = {"role": "user", "content": BASE_TASK}
    loop_a = [
        user,
        loop_turn("t1"),
        {"role": "tool", "tool_call_id": "t1", "content": "parser.py:41"},
    ]
    loop_b = [
        *loop_a,
        loop_turn("t2"),
        {"role": "tool", "tool_call_id": "t2", "content": "line 44"},
        {"role": "user", "content": DRIFT_TASK},
        loop_turn("t3"),
        {"role": "tool", "tool_call_id": "t3", "content": "noted"},
    ]
    loop_c = [
        *loop_b,
        loop_turn("t4"),
        {"role": "tool", "tool_call_id": "t4", "content": "still noted"},
        {"role": "user", "content": DRIFT_TASK},
        loop_turn("t5"),
        {"role": "tool", "tool_call_id": "t5", "content": "done"},
    ]
    return [
        {"phase": "user_turn", "messages": [user]},
        {"phase": "tool_loop", "messages": loop_a},
        {"phase": "tool_loop", "messages": loop_b},
        {"phase": "tool_loop", "messages": loop_c},
    ]


def _write_cases(path: Path, *, session_only: bool) -> None:
    case: dict = {
        "case_id": "code-then-drift",
        "session_id": "conv-route-real-1",
        "requests": _session_requests(),
    }
    if session_only:
        case["session_only_identity"] = True
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_spec(path: Path) -> None:
    path.write_text(
        "api_version: recipe-lab/v1alpha1\n"
        "objective:\n"
        "  metric_key: capability.score\n"
        "  direction: maximize\n"
        "  min_improvement_pct: 1.0\n"
        "  replay_metric: routed_accuracy\n"
        "workload:\n"
        "  manifest: cases.jsonl\n"
        "  content_hash: sha256:" + "0" * 64 + "\n"
        "  kind: agentic\n"
        "  benchmark: terminal-bench-2.1\n"
        "  profile: quick\n"
        "  repetitions: 1\n"
        "attribution:\n"
        "  oracle_policy:\n"
        "    domain_code: local/model-a\n"
        "budget:\n"
        "  rounds: 1\n"
        "  max_minutes_per_trial: 15\n",
        encoding="utf-8",
    )


def _run_tier(
    recipe_lab_bin: Path,
    router_bin: str,
    work: Path,
    *,
    candidates_dir: Path,
    cases_path: Path,
    ledger_path: Path,
    backend_port: int,
    hf_home: str | None,
) -> None:
    argv = [
        "run",
        "--tier",
        "route-replay",
        "--spec",
        str(work / "spec.yaml"),
        "--cases",
        str(cases_path),
        "--candidates",
        str(candidates_dir),
        "--proposer",
        "bootstrap",
        "--router-bin",
        router_bin,
        "--mock-backend",
        str(RECIPE_LAB_ROOT / "examples" / "smoke" / "mock_backend.py"),
        "--python-bin",
        sys.executable,
        # the mock keeps an explicit port; every candidate's backend_refs
        # must reference this same one or the router dials a dead endpoint
        "--backend-port",
        str(backend_port),
        "--ready-timeout",
        "600",
        "--output-dir",
        str(work / "out"),
        "--ledger",
        str(ledger_path),
        "--round",
        "1",
        "--router-model",
        "vllm-sr/auto",
    ]
    if hf_home:
        argv += ["--hf-home", hf_home]
    _run_recipe_lab(recipe_lab_bin, argv)


def _gates_by_candidate(ledger_path: Path) -> dict[str, dict[str, str]]:
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        Path(record["candidate"]["ref"]).stem: record["gates"] for record in records
    }


def test_route_replay_real_judges_the_causes(
    router_bin: str,
    recipe_lab_bin: Path,
    hf_home: str | None,
    tmp_path: Path,
) -> None:
    backend_port = _free_port()
    protected = _candidate_config(tmp_path, backend_port, "protected", protection=True)
    unprotected = _candidate_config(
        tmp_path, backend_port, "unprotected", protection=False
    )
    candidates = tmp_path / "cands"
    candidates.mkdir()
    protected.rename(candidates / "a-protected.yaml")
    unprotected.rename(candidates / "b-unprotected.yaml")
    _write_spec(tmp_path / "spec.yaml")
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, session_only=False)
    ledger = tmp_path / "ledger.jsonl"
    _run_tier(
        recipe_lab_bin,
        router_bin,
        tmp_path,
        candidates_dir=candidates,
        cases_path=cases,
        ledger_path=ledger,
        backend_port=backend_port,
        hf_home=hf_home,
    )
    gates = _gates_by_candidate(ledger)
    protected_gates = gates["a-protected"]
    unprotected_gates = gates["b-unprotected"]
    # protected: the loop hard-locked and the drift releases carry their own
    # cause - the unprotected residual stays clean while the per-task drift
    # ceiling trips on the same switches
    assert protected_gates["max_unprotected_tool_loop_switches"] == "pass"
    assert protected_gates["max_decision_drift_switches_per_task"] == "fail"
    # unprotected: no traces ride the records, so the same in-loop switches
    # are all unprotected and the drift gate sees zero - the two gates
    # judging the same switches by different causes is the Z5 split's proof
    assert unprotected_gates["max_unprotected_tool_loop_switches"] == "fail"
    assert unprotected_gates["max_decision_drift_switches_per_task"] == "pass"


def test_route_replay_real_missing_conversation_header_fails(
    router_bin: str,
    recipe_lab_bin: Path,
    hf_home: str | None,
    tmp_path: Path,
) -> None:
    backend_port = _free_port()
    protected = _candidate_config(tmp_path, backend_port, "protected", protection=True)
    candidates = tmp_path / "cands"
    candidates.mkdir()
    protected.rename(candidates / "a-protected.yaml")
    _write_spec(tmp_path / "spec.yaml")
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, session_only=True)
    ledger = tmp_path / "ledger.jsonl"
    _run_tier(
        recipe_lab_bin,
        router_bin,
        tmp_path,
        candidates_dir=candidates,
        cases_path=cases,
        ledger_path=ledger,
        backend_port=backend_port,
        hf_home=hf_home,
    )
    gates = _gates_by_candidate(ledger)["a-protected"]
    # the conversation identity resolves missing, so the protection never
    # engages and every in-loop switch is unprotected (the W1 failure shape)
    assert gates["max_unprotected_tool_loop_switches"] == "fail"
    assert gates["max_decision_drift_switches_per_task"] == "pass"
