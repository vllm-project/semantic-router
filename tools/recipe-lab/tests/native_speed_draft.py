"""Explicit native check: set RECIPE_LAB_V05_CLI and RECIPE_LAB_V05_ROUTER."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
import yaml
from test_speed_draft import FAST, LAB, PUBLIC, STRONG, providers


@pytest.mark.parametrize(
    (
        "strong_has_tools",
        "fast_has_tools",
        "tool_model",
        "reasoning_order",
        "fast_order",
        "agent_streaming",
    ),
    [
        (True, False, STRONG, [STRONG], [FAST, STRONG], False),
        (False, True, FAST, [STRONG, FAST], [FAST], False),
        (True, True, STRONG, [STRONG], [FAST], False),
        (True, True, STRONG, [STRONG], [FAST], True),
    ],
)
def test_compile_two_backends_with_real_offline_cli_and_router(
    tmp_path,
    strong_has_tools,
    fast_has_tools,
    tool_model,
    reasoning_order,
    fast_order,
    agent_streaming,
):
    """Opt in with the same installed CLI and native Router as the target deployment."""
    cli = os.environ.get("RECIPE_LAB_V05_CLI")
    router = os.environ.get("RECIPE_LAB_V05_ROUTER")
    assert cli and router, "both real binaries are required; do not substitute a stub"
    doc = providers()
    strong, fast = doc["providers"]["models"]
    if not strong_has_tools:
        strong["capabilities"].remove("tools")
    if fast_has_tools:
        fast["capabilities"].append("tools")
    doc["recipes"] = [{"name": "default"}]
    provider_input = tmp_path / "providers.yaml"
    provider_input.write_text(yaml.safe_dump(doc, sort_keys=False), "utf-8")
    out = tmp_path / "draft"
    cp = subprocess.run(
        [
            sys.executable,
            "-B",
            str(LAB / "recipe_lab/speed_draft.py"),
            "--providers",
            str(provider_input),
            "--strong",
            STRONG,
            "--fast",
            FAST,
            "--tool-model",
            tool_model,
            "--public-model",
            PUBLIC,
            "--vllm-sr",
            cli,
            "--router",
            router,
            "--out",
            str(out),
            *(["--agent-streaming"] if agent_streaming else []),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert yaml.safe_load(provider_input.read_text("utf-8")) == doc
    assert (out / "bindings.yaml").is_file() and (out / "manifest.json").is_file()
    output_bytes = (out / "config.yaml").read_bytes()
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["sha256"]["config.yaml"] == hashlib.sha256(output_bytes).hexdigest()
    assert manifest["gateway_started"] is False
    assert manifest["business_quality_measured"] is False
    cfg = yaml.safe_load(output_bytes)
    assert cfg["version"] == "v0.5"
    entry = next(e for e in cfg["entrypoints"] if PUBLIC in e["model_names"])
    recipe = next(r for r in cfg["recipes"] if r["name"] == entry["recipe"])
    bindings = {d["name"]: d["modelRefs"] for d in recipe["decisions"]}
    if agent_streaming:
        assert all(
            d["reliability"]["total_timeout"] == "0s" for d in recipe["decisions"]
        )
        assert any("total_timeout" in change for change in manifest["changes"])
    assert (
        yaml.safe_load((out / "bindings.yaml").read_text("utf-8"))
        == manifest["bindings"]
        == bindings
    )
    assert {lane: [r["model"] for r in refs] for lane, refs in bindings.items()} == {
        "tools": [tool_model],
        "reasoning": reasoning_order,
        "fast": fast_order,
    }
