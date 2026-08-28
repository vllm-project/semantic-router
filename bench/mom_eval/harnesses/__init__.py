"""Lightweight harness stubs for core-suite layers."""

from __future__ import annotations

ROBUSTNESS_PROMPTS = [
    {"id": "en_direct", "language": "en", "tools": False},
    {"id": "es_direct", "language": "es", "tools": False},
    {"id": "en_tools", "language": "en", "tools": True},
    {"id": "long_context", "language": "en", "tools": False, "long_context": True},
]

SAFETY_PROMPTS = [
    {"id": "jailbreak_ignore", "category": "jailbreak"},
    {"id": "harmful_request", "category": "harmful"},
    {"id": "policy_bypass", "category": "jailbreak"},
]

PII_PROMPTS = [
    {"id": "email_leak", "category": "pii"},
    {"id": "ssn_leak", "category": "pii"},
]

LATENCY_PROMPTS = [
    {"id": "short_qa", "tokens": 32},
    {"id": "medium_qa", "tokens": 256},
    {"id": "tool_call", "tokens": 128, "tools": True},
]
