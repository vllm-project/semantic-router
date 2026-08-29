"""Closed provider-body contracts for the OpenAI-compatible E2E simulator."""

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


def _load_contracts() -> dict[str, dict[str, Any]]:
    path = Path(__file__).with_name("schema_contract.json")
    with path.open(encoding="utf-8") as contract_file:
        document = json.load(contract_file)
    return document["protocols"]


_CONTRACTS = _load_contracts()


def _matches_json_type(value: Any, kind: str) -> bool:
    if kind == "array":
        return isinstance(value, list)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "string":
        return isinstance(value, str)
    raise ValueError(f"unknown JSON type in provider contract: {kind}")


def validate_provider_request(protocol: str, body: Any) -> dict[str, Any]:
    """Validate the provider envelope without rewriting nested wire objects."""

    if not isinstance(body, dict):
        raise ContractViolationError("request body must be a JSON object")

    contract = _CONTRACTS[protocol]
    allowed = set(contract["official_request_fields"])
    allowed.update(contract.get("extension_request_fields", []))
    provider_fields = contract.get("provider_request_fields", {})
    allowed.update(provider_fields)
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ContractViolationError(
            f"unknown request field: {unknown[0]}",
            field=unknown[0],
        )

    missing = [
        field for field in contract["required_request_fields"] if field not in body
    ]
    if missing:
        raise ContractViolationError(
            f"missing required request field: {missing[0]}",
            field=missing[0],
        )

    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ContractViolationError("model must be a non-empty string", field="model")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ContractViolationError("stream must be a boolean", field="stream")

    for field, accepted_types in provider_fields.items():
        if field not in body:
            continue
        if not any(_matches_json_type(body[field], kind) for kind in accepted_types):
            expected = " or ".join(accepted_types)
            raise ContractViolationError(
                f"{field} must be {expected}",
                field=field,
            )

    if protocol == "openai_chat_completions":
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ContractViolationError(
                "messages must be a non-empty array", field="messages"
            )
    return body


def request_field_inventory(protocol: str) -> set[str]:
    contract = _CONTRACTS[protocol]
    return (
        set(contract["official_request_fields"])
        | set(contract.get("extension_request_fields", []))
        | set(contract.get("provider_request_fields", {}))
    )


def protocol_request_field_inventory(protocol: str) -> set[str]:
    """Return neutral-protocol fields, excluding provider-only extensions."""

    contract = _CONTRACTS[protocol]
    return set(contract["official_request_fields"]) | set(
        contract.get("extension_request_fields", [])
    )


def provider_request_field_inventory(protocol: str) -> set[str]:
    return set(_CONTRACTS[protocol].get("provider_request_fields", {}))
