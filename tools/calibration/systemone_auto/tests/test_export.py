"""Deployment renaming preserves the measured model pool and fitted heads."""

from __future__ import annotations

import copy

import pytest
from systemone_auto.export import bind_policy
from systemone_auto.metrics import FEATURE_NAMES


def policy():
    return {
        "schema_version": "systemone-policy/v1",
        "feature_names": FEATURE_NAMES,
        "heads": {
            "kai": {"eos": {"weights": [0.1] * 11}},
            "eos": {"kai": {"weights": [0.2] * 11}},
        },
        "actions": {
            name: {"model": name, "identity": {"model_id": name, "revision": "a" * 40}}
            for name in ("kai", "eos")
        },
        "training": {},
    }


def test_binding_renames_all_references_without_repointing_or_mutating_input():
    value = policy()
    original = copy.deepcopy(value)
    result = bind_policy(
        value,
        {
            "kai": {"stage": "fast", "model": "native-small"},
            "eos": {"stage": "upgrade", "model": "native-other"},
        },
    )
    assert value == original
    assert result["heads"]["fast"]["upgrade"] == original["heads"]["kai"]["eos"]
    assert (
        result["actions"]["fast"]["identity"] == original["actions"]["kai"]["identity"]
    )
    assert result["actions"]["fast"]["model"] == "native-small"


def test_binding_cannot_silently_remove_a_candidate_or_merge_stages():
    with pytest.raises(ValueError, match="every measured action"):
        bind_policy(policy(), {"kai": {"stage": "fast", "model": "small"}})
    with pytest.raises(ValueError, match="unique"):
        bind_policy(
            policy(),
            {name: {"stage": "fast", "model": name} for name in ("kai", "eos")},
        )
