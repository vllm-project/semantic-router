#!/usr/bin/env python3
"""Assert that Helm appends extraContainers to the Router pod unchanged."""

import sys
from pathlib import Path

import yaml


def check(rendered_path: str) -> None:
    fixture = Path(__file__).parent / "testdata/extra-containers-values.yaml"
    values = yaml.safe_load(fixture.read_text())
    deployments = [
        document
        for document in yaml.safe_load_all(Path(rendered_path).read_text())
        if document and document.get("kind") == "Deployment"
    ]
    assert len(deployments) == 1, "expected one Router Deployment"
    pod = deployments[0]["spec"]["template"]["spec"]
    containers = pod["containers"]
    assert containers[0]["name"] == "semantic-router", "router must stay first"
    assert containers[1:] == values["extraContainers"], "extraContainers changed"
    for volume in values["extraVolumes"]:
        assert volume in pod["volumes"], f"extra volume missing: {volume['name']}"


if __name__ == "__main__":
    check(sys.argv[1])
