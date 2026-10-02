"""The supported Router IntentRequest envelope, shared by CLI and sr-bench."""

import copy
import json

PROMPT_FIELDS = {
    "tools",
    "functions",
    "tool_choice",
    "function_call",
    "response_format",
}
REQUEST_FIELDS = PROMPT_FIELDS | {
    "text",
    "messages",
    "model",
    "metadata",
    "options",
    "preview_context",
    "max_tokens",
    "max_completion_tokens",
}
MESSAGE_FIELDS = {
    "role",
    "name",
    "content",
    "tool_calls",
    "tool_call_id",
    "function_call",
    "refusal",
    "reasoning_content",
}
CONTEXT_FIELDS = {"session_id", "conversation_id", "sampling_seed"}
OPTION_FIELDS = {
    "return_probabilities",
    "confidence_threshold",
    "include_explanation",
    "trace",
}
MAX_METADATA_ENTRIES = 32
MAX_METADATA_KEY_BYTES = 128
MAX_METADATA_VALUE_BYTES = 1024
MAX_IDENTITY_BYTES = 1024


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"{label} contains unsupported fields or is not an object")


def _metadata(value):
    if (
        not isinstance(value, dict)
        or len(value) > MAX_METADATA_ENTRIES
        or any(
            not isinstance(key, str)
            or not key
            or len(key.encode()) > MAX_METADATA_KEY_BYTES
            or not isinstance(item, str)
            or len(item.encode()) > MAX_METADATA_VALUE_BYTES
            for key, item in value.items()
        )
    ):
        raise ValueError("Request metadata must be a bounded map of strings")


def build_preview_request(request, *, model=None, preview_context=None, trace=None):
    """Validate the supported subset; never silently discard completion fields."""
    _object(request, REQUEST_FIELDS, "Preview request")
    payload = copy.deepcopy(request)
    if "preview_context" in payload:
        _object(payload["preview_context"], CONTEXT_FIELDS, "Preview context")
    if "options" in payload:
        _object(payload["options"], OPTION_FIELDS, "Preview options")
    if model is not None:
        payload["model"] = model
    if preview_context is not None:
        payload["preview_context"] = {
            **payload.get("preview_context", {}),
            **preview_context,
        }
    if trace is not None:
        payload["options"] = {**payload.get("options", {}), "trace": trace}
    for name in ("text", "model"):
        if name in payload and not isinstance(payload[name], str):
            raise ValueError(f"Preview {name} must be a string")
    if "messages" in payload:
        if not isinstance(payload["messages"], list):
            raise ValueError("Preview messages must be an array")
        for message in payload["messages"]:
            _object(message, MESSAGE_FIELDS, "Preview message")
            if not isinstance(message.get("role"), str) or not message["role"].strip():
                raise ValueError("Each Preview message requires a role")
            for name in ("name", "tool_call_id"):
                if name in message and not isinstance(message[name], str):
                    raise ValueError(f"Message {name} must be a string")
            if "tool_calls" in message and not isinstance(message["tool_calls"], list):
                raise ValueError("Message tool_calls must be an array")
    for name in ("tools", "functions"):
        if name in payload and not isinstance(payload[name], list):
            raise ValueError(f"Preview {name} must be an array")
    if "metadata" in payload:
        _metadata(payload["metadata"])
    if "options" in payload:
        _object(payload["options"], OPTION_FIELDS, "Preview options")
    if "preview_context" in payload:
        context = payload["preview_context"]
        _object(context, CONTEXT_FIELDS, "Preview context")
        for name in ("session_id", "conversation_id"):
            if name in context and (
                not isinstance(context[name], str)
                or len(context[name].encode()) > MAX_IDENTITY_BYTES
                or any(c in context[name] for c in ("\r", "\n", "\x00"))
            ):
                raise ValueError(
                    "Preview identities must be bounded strings without control separators"
                )
        if "sampling_seed" in context and (
            isinstance(context["sampling_seed"], bool)
            or not isinstance(context["sampling_seed"], int)
            or not -(2**63) <= context["sampling_seed"] < 2**63
        ):
            raise ValueError("Preview sampling_seed must be a signed 64-bit integer")
    json.dumps(payload, allow_nan=False)
    return payload


def case_request_fields(case):
    """Only explicit request inputs; benchmark metadata may contain answer keys."""
    fields = {name: case[name] for name in PROMPT_FIELDS if name in case}
    if "request_metadata" in case:
        fields["metadata"] = case["request_metadata"]
    return build_preview_request(fields)
