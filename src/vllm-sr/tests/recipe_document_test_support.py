"""A configuration document around a test's signals and decisions."""

from __future__ import annotations

import copy
from typing import Any

TEST_MODEL_PROVIDERS = {
    "defaults": {"model": "test_model"},
    "models": [
        {
            "name": "test_model",
            "backend_refs": [
                {"name": "ep1", "weight": 1, "endpoint": "localhost:8000"}
            ],
        }
    ],
}


def default_recipe_document(parts: dict[str, Any]) -> dict[str, Any]:
    """A v0.5 document whose default recipe holds the signals and decisions in
    parts, served by test_model; the other keys of parts stay top-level."""

    document = dict(parts)
    recipe = {"name": "default"}
    for key in ("signals", "decisions"):
        if key in document:
            recipe[key] = document.pop(key)
    return {
        "version": "v0.5",
        **document,
        "providers": copy.deepcopy(TEST_MODEL_PROVIDERS),
        "recipes": [recipe],
    }
