"""Deterministic synthetic workloads for single-state and batch Decision APIs."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from pydantic import ValidationError

from .cases import MODELS
from decision_runtime.contracts import (
    MAX_BATCH_DECISIONS,
    MAX_BATCH_QUESTIONS,
    MAX_BATCH_STATES,
    SystemOneBatchRequest,
    SystemOneRequest,
)


def wire_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class RequestSpec:
    body: bytes
    request: SystemOneRequest | SystemOneBatchRequest
    state_id: str | None
    decisions: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


@dataclass(frozen=True)
class WorkloadCase:
    id: str
    question_count: int
    state_count: int
    old_singles: tuple[RequestSpec, ...]
    new_request: RequestSpec

    @property
    def decisions(self) -> int:
        return self.question_count * self.state_count

    @property
    def same_wire_bytes(self) -> bool:
        return (
            self.state_count == 1 and self.old_singles[0].body == self.new_request.body
        )


_TOPICS = ("billing", "shipping", "account", "service", "returns", "access")
_CHANNELS = ("chat", "email", "phone")


def _question(index: int, variant: int) -> dict[str, object]:
    kind = (index + variant) % 3
    if kind == 0:
        return {
            "type": "noul",
            "instructions": f"Does the message require an action for check {index}?",
            "criteria": {"true": "An action is needed", "false": "No action is needed"},
        }
    if kind == 1:
        return {
            "type": "choice",
            "instructions": f"Choose the handling queue for check {index}.",
            "criteria": {
                "billing": "Payment and refund requests",
                "service": "Product and account requests",
                "delivery": "Shipment requests",
            },
        }
    return {
        "type": "score",
        "instructions": f"Rate the urgency for check {index}.",
        "criteria": ["Routine", "Same day", "Immediate"],
    }


def generate_cases(
    model: str,
    old_model_id: str,
    *,
    question_count: int,
    state_count: int,
    variants: int,
    seed: int,
) -> tuple[WorkloadCase, ...]:
    """Generate strict requests before traffic; no benchmark data is sampled live."""

    if model not in MODELS:
        raise ValueError("unsupported Decision model")
    if not old_model_id.strip():
        raise ValueError("old model ID must not be blank")
    if not 1 <= question_count <= MAX_BATCH_QUESTIONS:
        raise ValueError("question count exceeds the batch contract")
    if not 1 <= state_count <= MAX_BATCH_STATES:
        raise ValueError("state count exceeds the batch contract")
    if question_count * state_count > MAX_BATCH_DECISIONS:
        raise ValueError("questions times states exceeds the batch decision limit")
    if variants < 1:
        raise ValueError("variants must be positive")

    rng = random.Random(seed + question_count * 1_000_003 + state_count * 10_007)
    cases = []
    for variant in range(variants):
        case_id = f"q{question_count:04d}_s{state_count:04d}_v{variant:03d}"
        questions = {
            f"q{index:04d}": _question(index, variant)
            for index in range(question_count)
        }
        states = [
            {
                "id": f"s{index:04d}",
                "state": {
                    "message": (
                        f"Synthetic {_TOPICS[rng.randrange(len(_TOPICS))]} request "
                        f"{variant}-{index}; please review the next action."
                    ),
                    "channel": _CHANNELS[rng.randrange(len(_CHANNELS))],
                    "priority_hint": index % 3,
                },
            }
            for index in range(state_count)
        ]
        old_singles = []
        try:
            for state in states:
                payload = {
                    "model": old_model_id,
                    "state": state["state"],
                    "questions": questions,
                }
                old_singles.append(
                    RequestSpec(
                        body=wire_bytes(payload),
                        request=SystemOneRequest.model_validate(payload),
                        state_id=str(state["id"]),
                        decisions=question_count,
                    )
                )
            if state_count == 1:
                new_payload = {
                    "model": model,
                    "state": states[0]["state"],
                    "questions": questions,
                }
                new_request = SystemOneRequest.model_validate(new_payload)
                new_spec = RequestSpec(
                    body=wire_bytes(new_payload),
                    request=new_request,
                    state_id=str(states[0]["id"]),
                    decisions=question_count,
                )
            else:
                new_payload = {"model": model, "states": states, "questions": questions}
                new_request = SystemOneBatchRequest.model_validate(new_payload)
                new_spec = RequestSpec(
                    body=wire_bytes(new_payload),
                    request=new_request,
                    state_id=None,
                    decisions=question_count * state_count,
                )
        except ValidationError as error:
            raise ValueError(
                "generated case violates the Decision request contract"
            ) from error
        cases.append(
            WorkloadCase(
                id=case_id,
                question_count=question_count,
                state_count=state_count,
                old_singles=tuple(old_singles),
                new_request=new_spec,
            )
        )
    return tuple(cases)


def cohort_sha256(cases: tuple[WorkloadCase, ...]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.id.encode("ascii"))
        digest.update(b"\x00")
        for spec in (*case.old_singles, case.new_request):
            digest.update(spec.body)
            digest.update(b"\x00")
    return digest.hexdigest()
