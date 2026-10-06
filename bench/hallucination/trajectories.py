import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "agent_trajectory.v1"
DEFAULT_FIXTURE = Path(__file__).parent / "testdata" / "agent_trajectories.json"

# Must match defaultHallucinationTokenLabels in
# src/semantic-router/pkg/classification/hallucination_detector_endpoint.go.
SPAN_LABELS = frozenset(
    {
        "HALLUCINATED",
        "unsupported",
        "contradicted",
        "unverifiable",
        "contradiction",
        "fabricated_reference",
        "unsupported_addition",
    }
)
OUTSIDE_LABELS = frozenset({"SUPPORTED", "O"})

_DOCUMENT_KEYS = frozenset({"format", "trajectories"})
_TRAJECTORY_KEYS = frozenset({"id", "request", "steps"})
_TOOL_KEYS = frozenset({"role", "name", "arguments", "output"})
_ASSISTANT_KEYS = frozenset({"role", "text", "unsupported_spans"})
_SPAN_KEYS = frozenset({"start", "end", "text", "label", "score"})


class TrajectoryFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ToolStep:
    name: str
    output: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantStep:
    text: str
    unsupported_spans: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Trajectory:
    id: str
    request: str
    steps: tuple[ToolStep | AssistantStep, ...]


@dataclass(frozen=True)
class StepCheck:
    trajectory_id: str
    step: int
    question: str
    context: str
    answer: str
    gold_spans: tuple[dict[str, Any], ...]


def load_trajectories(path: Path | str = DEFAULT_FIXTURE) -> list[Trajectory]:
    where = str(path)
    document = _object(
        json.loads(Path(path).read_text(encoding="utf-8")),
        allowed=_DOCUMENT_KEYS,
        required=_DOCUMENT_KEYS,
        where=where,
    )
    if document["format"] != FORMAT:
        raise _error(where, f"format must be {FORMAT!r}, got {document['format']!r}")
    raw_trajectories = document["trajectories"]
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise _error(where, "trajectories must be a non-empty list")

    trajectories = [
        _trajectory(raw, where=f"trajectories[{index}]")
        for index, raw in enumerate(raw_trajectories)
    ]
    ids = [trajectory.id for trajectory in trajectories]
    duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if duplicates:
        raise _error(where, f"duplicate trajectory ids {duplicates}")
    return trajectories


def validate_spans(spans: object, text: str, where: str) -> tuple[dict[str, Any], ...]:
    """Check gold spans against the token_spans.v1 rules for one message."""
    if not isinstance(spans, list):
        raise _error(where, "unsupported_spans must be a list")
    seen: set[tuple[str, int, int]] = set()
    for index, raw in enumerate(spans):
        at = f"{where}.unsupported_spans[{index}]"
        span = _object(
            raw, allowed=_SPAN_KEYS, required=_SPAN_KEYS - {"score"}, where=at
        )
        start, end, label = span["start"], span["end"], span["label"]
        if not (_is_int(start) and _is_int(end) and 0 <= start < end <= len(text)):
            raise _error(
                at,
                f"offsets [{start}, {end}) must satisfy "
                f"0 <= start < end <= {len(text)} code points",
            )
        if span["text"] != text[start:end]:
            raise _error(
                at,
                f"text {span['text']!r} != message[{start}:{end}] {text[start:end]!r}",
            )
        if isinstance(label, str) and label in OUTSIDE_LABELS:
            raise _error(at, f"{label!r} marks supported text and cannot be a span")
        if not isinstance(label, str) or label not in SPAN_LABELS:
            raise _error(at, f"unknown label {label!r}")
        score = span.get("score")
        if score is not None and not (_is_number(score) and 0 <= score <= 1):
            raise _error(at, f"score {score!r} must be a number in [0, 1]")
        key = (label, start, end)
        if key in seen:
            raise _error(at, f"duplicate span {key}")
        seen.add(key)
    return tuple(spans)


def step_checks(trajectory: Trajectory) -> list[StepCheck]:
    """One detector input per assistant step, grounded in the tool output before it."""
    outputs: list[str] = []
    checks: list[StepCheck] = []
    for index, step in enumerate(trajectory.steps):
        if isinstance(step, ToolStep):
            outputs.append(step.output)
            continue
        checks.append(
            StepCheck(
                trajectory_id=trajectory.id,
                step=index,
                question=trajectory.request,
                # Same separator the response stage uses to join tool results
                # (req_filter_fact_check.go), so steps see what the router sees.
                context="\n\n".join(outputs),
                answer=step.text,
                gold_spans=step.unsupported_spans,
            )
        )
    return checks


def _trajectory(raw: object, where: str) -> Trajectory:
    trajectory = _object(
        raw, allowed=_TRAJECTORY_KEYS, required=_TRAJECTORY_KEYS, where=where
    )
    raw_steps = trajectory["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise _error(where, "steps must be a non-empty list")

    steps: list[ToolStep | AssistantStep] = [
        _step(raw_step, where=f"{where}.steps[{index}]")
        for index, raw_step in enumerate(raw_steps)
    ]
    if not any(isinstance(step, AssistantStep) for step in steps):
        raise _error(where, "has no assistant step to check")
    return Trajectory(
        id=_text(trajectory, key="id", where=where),
        request=_text(trajectory, key="request", where=where),
        steps=tuple(steps),
    )


def _step(raw: object, where: str) -> ToolStep | AssistantStep:
    role = raw.get("role") if isinstance(raw, dict) else None
    if role == "tool":
        step = _object(
            raw, allowed=_TOOL_KEYS, required=_TOOL_KEYS - {"arguments"}, where=where
        )
        arguments = step.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _error(where, "arguments must be an object")
        return ToolStep(
            name=_text(step, key="name", where=where),
            output=_text(step, key="output", where=where),
            arguments=arguments,
        )
    if role == "assistant":
        step = _object(
            raw, allowed=_ASSISTANT_KEYS, required=_ASSISTANT_KEYS, where=where
        )
        text = _text(step, key="text", where=where)
        return AssistantStep(
            text=text,
            unsupported_spans=validate_spans(
                step["unsupported_spans"], text=text, where=where
            ),
        )
    raise _error(where, f"role must be 'tool' or 'assistant', got {role!r}")


def _object(
    raw: object, allowed: frozenset[str], required: frozenset[str], where: str
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _error(where, "must be an object")
    # A misspelled key would otherwise drop its value, and a step whose spans
    # were dropped reads as supported.
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _error(where, f"unknown keys {unknown}")
    missing = sorted(required - set(raw))
    if missing:
        raise _error(where, f"missing keys {missing}")
    return raw


def _text(raw: dict[str, Any], key: str, where: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise _error(where, f"{key} must be a non-empty string")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _error(where: str, message: str) -> TrajectoryFormatError:
    return TrajectoryFormatError(f"{where}: {message}")
