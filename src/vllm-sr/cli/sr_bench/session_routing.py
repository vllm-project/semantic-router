from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cli.routing_preview import case_request_fields

from .adapters import AdapterContext, BenchmarkAdapter
from .canonical import canonical
from .contracts import (
    DEFAULT_LIMITS,
    MAX_CONCURRENCY,
    SESSION_AWARE,
    validate_request_params,
)
from .dataset_io import MAX_ROWS
from .grading import SESSION_OBSERVATION_RESULT
from .sources import _write_dataset
from .transport import CallFailure

BENCHMARK_ID: str = "live-session-routing-v1"
OBSERVATION_VERSION: str = "native-live-session-routing-v1"
SELECTION_VERSION: str = "native-session-routing-scenarios-v1"
SOURCE_REVISION: str = "a6d23eaff862a9d24dc10ccb778ad0618a839ef7"
SOURCE_URL: str = (
    "https://github.com/vllm-project/semantic-router/blob/"
    + SOURCE_REVISION
    + "/bench/agentic_routing_live_benchmark.py"
)
SOURCE_PATH: Path = Path(__file__).resolve().parents[4] / "bench"
SCENARIOS: tuple[str, ...] = (
    "balanced",
    "tool-heavy",
    "frontier-heavy",
    "idle-heavy",
    "stateful-heavy",
    "drift-heavy",
)
MAX_SESSIONS: int = MAX_CONCURRENCY
MAX_TURNS: int = DEFAULT_LIMITS["max_calls_per_case"]
MAX_PAUSE_SECONDS: float = float(DEFAULT_LIMITS["max_run_seconds"])
DEFAULT_THRESHOLDS: dict[str, int | float] = {
    "min_success_rate": 0.0,
    "max_p95_latency_ms": 0.0,
    "max_tool_loop_violations": -1,
}


def phase_for_turn(*, scenario: str, turn: int) -> str:
    if scenario == "tool-heavy":
        return "tool_loop" if turn % 5 in (2, 3, 4) else "user_turn"
    if scenario == "frontier-heavy":
        return "frontier_turn" if turn % 4 in (1, 2) else "user_turn"
    if scenario == "idle-heavy":
        return "idle_boundary" if turn > 0 and turn % 4 == 0 else "user_turn"
    if scenario == "stateful-heavy":
        return "provider_state" if turn > 0 and turn % 5 in (1, 2) else "user_turn"
    if scenario == "drift-heavy":
        return "topic_drift" if turn > 0 and turn % 6 in (1, 2) else "user_turn"
    return "tool_loop" if turn % 6 in (2, 3) else "user_turn"


def prompt_for_phase(
    *, scenario: str, session_index: int, turn: int, phase: str
) -> str:
    base = f"Session {session_index}, turn {turn}. "
    if phase == "tool_loop":
        return (
            base
            + "Use the provided tool result and continue the investigation in one concise sentence."
        )
    if phase == "provider_state":
        return (
            base
            + "Continue from the previous response state and preserve the same analysis thread."
        )
    if phase == "topic_drift":
        return (
            base
            + "The task direction changed: now compare the routing policy against a simpler baseline."
        )
    if phase == "frontier_turn":
        return (
            base
            + "Solve a harder planning step with careful reasoning but keep the answer short."
        )
    if phase == "idle_boundary":
        return (
            base
            + "This follows an idle pause. Re-evaluate the next best model for a fresh subtask."
        )
    return base + f"Summarize the next action for the {scenario} agent workload."


def build_messages(
    *, session_key: str, turn: int, phase: str, prompt: str
) -> list[dict[str, Any]]:
    system = {
        "role": "system",
        "content": "You are a concise benchmark assistant. Answer in one short sentence.",
    }
    if phase != "tool_loop":
        return [system, {"role": "user", "content": prompt}]

    tool_call_id = f"call_{session_key}_{turn}"
    return [
        system,
        {"role": "user", "content": "Look up the current routing task status."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "lookup_task_status",
                        "arguments": json.dumps({"task": session_key}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"status": "tool result ready", "turn": turn}),
        },
        {"role": "user", "content": prompt},
    ]


def _finite_nonnegative(value: object, *, label: str, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{label} must be finite and between zero and {maximum}")
    return float(value)


def _workload_spec(
    *,
    scenario: str,
    session_index: int,
    turns: int,
    turn_delay_seconds: float,
    idle_pause_seconds: float,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    session_key = f"live-router-{scenario}-{session_index:04d}"
    saved_turns: list[dict[str, Any]] = []
    for turn in range(turns):
        phase = phase_for_turn(scenario=scenario, turn=turn)
        prompt = prompt_for_phase(
            scenario=scenario, session_index=session_index, turn=turn, phase=phase
        )
        pause = (
            idle_pause_seconds
            if phase == "idle_boundary" and idle_pause_seconds > 0
            else turn_delay_seconds
        )
        saved_turns.append(
            {
                "turn": turn,
                "phase": phase,
                "prompt": prompt,
                "messages": build_messages(
                    session_key=session_key, turn=turn, phase=phase, prompt=prompt
                ),
                "pause_seconds": pause if turn > 0 else 0.0,
            }
        )
    return {
        "version": OBSERVATION_VERSION,
        "scenario": scenario,
        "session_index": session_index,
        "session_key": session_key,
        "turns": saved_turns,
        "request_settings": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        },
        "timing": {
            "turn_delay_seconds": turn_delay_seconds,
            "idle_pause_seconds": idle_pause_seconds,
        },
    }


def _case_spec(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata")
    spec = metadata.get("session_routing") if isinstance(metadata, dict) else None
    if not isinstance(spec, dict) or not isinstance(spec.get("request_settings"), dict):
        raise ValueError("Session routing requires its frozen native workload")
    settings = spec["request_settings"]
    timing = spec.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Session routing requires frozen timing settings")
    expected = _workload_spec(
        scenario=spec.get("scenario"),
        session_index=spec.get("session_index"),
        turns=len(spec.get("turns", [])),
        turn_delay_seconds=timing.get("turn_delay_seconds"),
        idle_pause_seconds=timing.get("idle_pause_seconds"),
        max_tokens=settings.get("max_tokens"),
        temperature=settings.get("temperature"),
        timeout=settings.get("timeout"),
    )
    if canonical(spec) != canonical(expected):
        raise ValueError("Session-routing case differs from its frozen native workload")
    if (
        case["id"]
        != f"{BENCHMARK_ID}/{spec['scenario']}/session-{spec['session_index']:04d}"
        or case["messages"] != spec["turns"][0]["messages"]
    ):
        raise ValueError("Session-routing case identity or initial messages changed")
    return spec


def preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    spec = _case_spec(case)
    if manifest["profile"] == "standard":
        raise ValueError(
            "Session-routing observations do not establish a Standard holdout"
        )
    if case_request_fields(case) or "answer" in case:
        raise ValueError(
            "Session-routing observations do not accept request overrides or answers"
        )
    turns = spec["turns"]
    settings = spec["request_settings"]
    limits = manifest["limits"]
    if (
        not turns
        or len(turns) > MAX_TURNS
        or limits["max_calls_per_case"] < len(turns)
        or limits["total_timeout_s"] != settings["timeout"]
    ):
        raise ValueError(
            "Session-routing schedule exceeds its frozen call or timeout limits"
        )
    if manifest["output_policy"] != "bounded":
        raise ValueError("Session-routing observations require bounded output")
    selected = {
        cell["target_id"]
        for cell in manifest.get("execution_cells", [])
        if cell["case_id"] == case["id"]
    }
    for target in manifest["targets"]:
        if "execution_cells" in manifest and target["id"] not in selected:
            continue
        if target.get("session_mode") != SESSION_AWARE:
            raise ValueError(
                "Session-routing observations require session_aware targets"
            )
        effective = {**manifest["sampling"], **target.get("request_params", {})}
        if effective != {key: settings[key] for key in ("max_tokens", "temperature")}:
            raise ValueError(
                "Session-routing generation differs from its frozen native settings"
            )
    options = manifest.setdefault("benchmark_options", {})
    if not isinstance(options, dict):
        raise ValueError("benchmark_options must be an object")
    config = options.setdefault(BENCHMARK_ID, {})
    if not isinstance(config, dict) or set(config) - {"thresholds"}:
        raise ValueError(
            "Session-routing options do not support Responses state, retry, baseline, or recovery controls"
        )
    thresholds = config.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - DEFAULT_THRESHOLDS.keys():
        raise ValueError("Session-routing thresholds contain an unsupported field")
    thresholds = {**DEFAULT_THRESHOLDS, **thresholds}
    for name, value in thresholds.items():
        if name == "max_tool_loop_violations":
            if type(value) is not int or value < -1:
                raise ValueError(
                    "max_tool_loop_violations must be -1 or a nonnegative integer"
                )
        else:
            maximum = 1.0 if name == "min_success_rate" else float("inf")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} must be a finite nonnegative threshold")
    config["thresholds"] = thresholds


def _pause(context: AdapterContext, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        if context.cancelled():
            raise CallFailure("Run cancelled or wall-time budget exhausted")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    turns = _case_spec(case)["turns"]
    for turn in turns:
        _pause(context, turn["pause_seconds"])
        response = context.call(messages=turn["messages"])
        if not response.get("output_complete", True):
            raise CallFailure(
                "Session-routing output limit reached; observations are incomplete"
            )
    return {
        "answer": None,
        "correct": None,
        "score": None,
        "details": {
            "session_observation": {
                "version": SESSION_OBSERVATION_RESULT,
                "planned_turns": len(turns),
                "completed_turns": len(turns),
            }
        },
    }


def adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        id=BENCHMARK_ID,
        title="Live session-routing observation",
        kind="session-observation",
        source_url=SOURCE_URL,
        version=OBSERVATION_VERSION,
        execute=execute,
        preflight=preflight,
        requires_answer=False,
        result_kind=SESSION_OBSERVATION_RESULT,
    )


def prepare(
    *,
    store: Path,
    scenarios: tuple[str, ...] = SCENARIOS,
    sessions: int = 8,
    turns: int = 12,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: float = 60.0,
    turn_delay_seconds: float = 0.0,
    idle_pause_seconds: float = 0.0,
    profile: str = "quick",
) -> dict[str, Any]:
    if profile not in {"smoke", "quick"}:
        raise ValueError(
            "Session-routing preparation supports smoke/quick observations only"
        )
    if (
        not isinstance(scenarios, tuple)
        or not scenarios
        or len(set(scenarios)) != len(scenarios)
        or set(scenarios) - set(SCENARIOS)
    ):
        raise ValueError("Choose one or more unique native session-routing scenarios")
    if type(sessions) is not int or not 1 <= sessions <= MAX_SESSIONS:
        raise ValueError(f"sessions must be an integer from 1 to {MAX_SESSIONS}")
    if type(turns) is not int or not 1 <= turns <= MAX_TURNS:
        raise ValueError(f"turns must be an integer from 1 to {MAX_TURNS}")
    if len(scenarios) * sessions > MAX_ROWS:
        raise ValueError("Session-routing cases exceed the dataset size limit")
    if (
        type(max_tokens) is not int
        or not 1 <= max_tokens <= DEFAULT_LIMITS["max_output_tokens"]
    ):
        raise ValueError("max_tokens exceeds the supported output cap")
    timeout_value = _finite_nonnegative(
        timeout, label="timeout", maximum=MAX_PAUSE_SECONDS
    )
    if timeout_value == 0:
        raise ValueError("timeout must be positive")
    turn_delay = _finite_nonnegative(
        turn_delay_seconds, label="turn_delay_seconds", maximum=MAX_PAUSE_SECONDS
    )
    idle_pause = _finite_nonnegative(
        idle_pause_seconds, label="idle_pause_seconds", maximum=MAX_PAUSE_SECONDS
    )
    settings = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    validate_request_params(
        params=settings,
        limits={"max_output_tokens": DEFAULT_LIMITS["max_output_tokens"]},
    )
    settings["temperature"] = float(temperature)
    source = {
        "url": SOURCE_URL,
        "revision": SOURCE_REVISION,
        "partition": "agentic-routing-scenarios",
        "files": [
            {
                "name": "agentic_routing_live_benchmark.py",
                "sha256": hashlib.sha256(
                    (SOURCE_PATH / "agentic_routing_live_benchmark.py").read_bytes()
                ).hexdigest(),
            },
            {
                "name": "session_routing.py",
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        ],
        "normalizer": OBSERVATION_VERSION,
        "revision_verification": "declared-native-revision-with-bundled-workload-digests",
        "selection": {
            "scenarios": list(scenarios),
            "sessions": sessions,
            "turns": turns,
            "request_settings": {**settings, "timeout": timeout_value},
            "timing": {
                "turn_delay_seconds": turn_delay,
                "idle_pause_seconds": idle_pause,
            },
        },
    }

    def cases() -> Iterator[dict[str, Any]]:
        for scenario in scenarios:
            for session_index in range(sessions):
                spec = _workload_spec(
                    scenario=scenario,
                    session_index=session_index,
                    turns=turns,
                    turn_delay_seconds=turn_delay,
                    idle_pause_seconds=idle_pause,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    timeout=timeout_value,
                )
                yield {
                    "id": (f"{BENCHMARK_ID}/{scenario}/session-{session_index:04d}"),
                    "benchmark": BENCHMARK_ID,
                    "messages": spec["turns"][0]["messages"],
                    "metadata": {
                        "split": "dev",
                        "stratum": scenario,
                        "session_routing": spec,
                    },
                }

    return _write_dataset(
        root=store,
        cases=cases(),
        profile=profile,
        seed=0,
        sources={BENCHMARK_ID: source},
        custom=True,
        selection=SELECTION_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare frozen live session-routing observation cases"
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--turn-delay-seconds", type=float, default=0.0)
    parser.add_argument("--idle-pause-seconds", type=float, default=0.0)
    parser.add_argument("--profile", choices=("smoke", "quick"), default="quick")
    args = parser.parse_args()
    try:
        prepared = prepare(
            store=args.store,
            scenarios=tuple(args.scenario or SCENARIOS),
            sessions=args.sessions,
            turns=args.turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            turn_delay_seconds=args.turn_delay_seconds,
            idle_pause_seconds=args.idle_pause_seconds,
            profile=args.profile,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(prepared, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
