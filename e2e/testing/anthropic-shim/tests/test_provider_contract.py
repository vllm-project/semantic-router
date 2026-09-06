"""Revision-pinned Anthropic Messages provider contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from anthropic_shim.provider_contract import (
    request_field_inventory,
    validate_provider_request,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CAPABILITY_FIXTURES = (
    REPOSITORY_ROOT
    / "src"
    / "semantic-router"
    / "pkg"
    / "protocolcodec"
    / "testdata"
    / "golden"
    / "capability"
)


def test_simulator_accepts_every_published_request_field() -> None:
    fixture_path = CAPABILITY_FIXTURES / "015-anthropic-official-request-fields-in.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    cases = fixture["cases"]
    assert {case["name"] for case in cases} == request_field_inventory()
    for case in cases:
        body = {**fixture["base"], **case["patch"]}
        assert validate_provider_request(body) == body


def test_simulator_preserves_nested_official_fields() -> None:
    body = {
        "model": "provider-model",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Lookup a value",
                "input_schema": {"type": "object"},
                "strict": True,
            }
        ],
    }
    assert validate_provider_request(body) == body
