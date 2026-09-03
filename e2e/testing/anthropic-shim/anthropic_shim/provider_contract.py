"""Closed Messages provider-body contract for the Anthropic E2E simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContractViolationError(Exception):
    message: str
    field: str | None = None

    def __str__(self) -> str:
        return self.message


def _load_contract() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "schema_contract.json"
    with path.open(encoding="utf-8") as contract_file:
        document = json.load(contract_file)
    return document["protocols"]["anthropic_messages"]


_CONTRACT = _load_contract()


def validate_provider_request(body: Any) -> dict[str, Any]:
    """Validate the Messages envelope while preserving nested official unions."""

    if not isinstance(body, dict):
        raise ContractViolationError("request body must be a JSON object")

    allowed = set(_CONTRACT["official_request_fields"])
    allowed.update(_CONTRACT.get("extension_request_fields", []))
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ContractViolationError(
            f"unknown request field: {unknown[0]}",
            field=unknown[0],
        )

    missing = [
        field for field in _CONTRACT["required_request_fields"] if field not in body
    ]
    if missing:
        raise ContractViolationError(
            f"missing required request field: {missing[0]}",
            field=missing[0],
        )

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ContractViolationError("model must be a non-empty string", field="model")
    max_tokens = body.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 0
    ):
        raise ContractViolationError(
            "max_tokens must be a non-negative integer", field="max_tokens"
        )
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractViolationError(
            "messages must be a non-empty array", field="messages"
        )
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ContractViolationError("stream must be a boolean", field="stream")
    return body


def request_field_inventory() -> set[str]:
    return set(_CONTRACT["official_request_fields"]) | set(
        _CONTRACT.get("extension_request_fields", [])
    )
