#!/usr/bin/env python3
"""Assert that Helm preserves canonical deployment budgets and recipe bindings."""

import sys
from pathlib import Path

import yaml


def check(rendered_path: str) -> None:
    fixture = Path(__file__).parent / "testdata/model-runtime-values.yaml"
    expected = yaml.safe_load(fixture.read_text())["configOverride"]
    documents = yaml.safe_load_all(Path(rendered_path).read_text())
    configs = [
        yaml.safe_load(document["data"]["config.yaml"])
        for document in documents
        if document
        and document.get("kind") == "ConfigMap"
        and "config.yaml" in document.get("data", {})
    ]
    assert len(configs) == 1, "expected one canonical Router ConfigMap"
    actual = configs[0]
    assert (
        actual["global"]["model_catalog"] == expected["global"]["model_catalog"]
    ), "model catalog changed"
    assert actual["routing"] == expected["routing"], "default bindings changed"
    assert actual["recipes"] == expected["recipes"], "recipe bindings changed"


if __name__ == "__main__":
    check(sys.argv[1])
