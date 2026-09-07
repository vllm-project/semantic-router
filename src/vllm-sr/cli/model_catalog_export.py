"""Emit the packaged virtual-model catalog for internal consumers."""

from __future__ import annotations

import json
from typing import Any

from cli.model_catalog import (
    DEFAULT_CHANNEL,
    _load_catalog_document,
)


def packaged_model_catalog_document() -> dict[str, Any]:
    """Return every catalog channel without consulting a working-tree config."""

    _, latest = _load_catalog_document(DEFAULT_CHANNEL)
    defaults = latest.get("defaults") or {}
    document = {
        "schema_version": latest["schema_version"],
        "catalogs": [
            {
                "catalog_version": latest["catalog_version"],
                "channel": latest["channel"],
                "default_model": defaults.get("model", ""),
                "enabled_models": list(defaults.get("enabled") or []),
                "default_intelligence_index": defaults.get("intelligence_index", ""),
            }
        ],
    }
    for resource in (
        "protocols",
        "providers",
        "reasoning_families",
        "models",
        "benchmarks",
        "evaluations",
        "evaluation_coverage",
        "indices",
        "index_results",
    ):
        document[resource] = latest.get(resource, [])
    return document


def main() -> None:
    print(json.dumps(packaged_model_catalog_document(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
