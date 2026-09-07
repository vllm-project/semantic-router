"""Stable identifiers shared by evaluation contracts and adapters."""

from __future__ import annotations

SCHEMA_VERSION = "evaluation.v1"

TRACK_IDS = (
    "routing",
    "model_pool",
    "joint",
    "agentic",
    "multimodal",
    "preference",
    "safety",
    "capacity",
)

BUILTIN_SUITE_IDS = (
    "evaluation-smoke",
    "live-mom-core",
    "live-agent-tasks",
    "live-fault-recovery",
    "live-multimodal",
    "live-hard-policy",
    "live-production-experiment",
    "live-capacity",
)
