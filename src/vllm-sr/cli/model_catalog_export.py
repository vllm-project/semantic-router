"""Emit the packaged virtual-model catalog for internal consumers."""

from __future__ import annotations

import json
from typing import Any

from cli.model_catalog import catalog_model_to_dict, load_all_model_catalogs


def packaged_model_catalog_document() -> dict[str, Any]:
    """Return every catalog channel without consulting a working-tree config."""

    catalogs = load_all_model_catalogs()
    return {
        "catalogs": [
            {
                "catalog_version": catalog.version,
                "channel": catalog.channel,
                "default_model": catalog.default_model,
                "enabled_models": list(catalog.enabled_models),
            }
            for catalog in catalogs
        ],
        "models": [
            catalog_model_to_dict(model)
            for catalog in catalogs
            for model in catalog.models
        ],
    }


def main() -> None:
    print(json.dumps(packaged_model_catalog_document(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
