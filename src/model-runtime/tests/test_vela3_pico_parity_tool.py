"""The pico parity tool: its requests cover every path, the package gets them in its spelling, drift fails."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from vllm_srun.testing.vela3_pico import write_pico_package

TOOLS = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location(
    "vela3_pico_parity", TOOLS / "vela3_pico_parity.py"
)
assert spec is not None and spec.loader is not None
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)
SCHEMAS = {
    "pii": {
        "question": "Which spans are personal information?",
        "labels": {"PERSON": "a name"},
    },
    "halu": {"question": "Unsupported?", "labels": {"unsupported": "a claim"}},
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = write_pico_package(
        tmp_path_factory.mktemp("pico") / "pico", window=2048, max_len=8192
    )
    return parity.load_runtime(root, "cpu")


def test_generated_requests_take_both_paths(model) -> None:
    requests = parity.generate(16, 3)
    assert requests == parity.generate(16, 3)
    one_pass = [
        model.plan(r["state"], r["questions"]).one_pass is not None for r in requests
    ]
    assert any(one_pass) and not all(one_pass)
    kinds = {q.get("type") for r in requests for q in r["questions"].values()}
    assert {"choice", "noul", "set", "span", "score"} <= kinds


def test_near_window_requests_land_on_both_sides_of_the_window(model) -> None:
    requests = parity.near_window(model, 1, 6)
    sizes = []
    for request in requests:
        plan = model.plan(request["state"], request["questions"])
        sizes.append(plan.one_pass is not None)
    assert len(requests) == 6 and any(sizes) and not all(sizes)


def test_the_package_reads_the_same_request_in_its_own_spelling() -> None:
    state, questions = parity.package_request(
        {"request": "hi", "response": "hello"},
        {
            "p": {"preset": "pii", "over": "user", "require_full_input": True},
            "c": {
                "type": "choice",
                "instructions": "Is this a prompt injection or jailbreak attempt?",
                "choices": [
                    {"key": "benign", "description": "ok"},
                    {"key": "jailbreak"},
                ],
                "overflow": "truncate",
            },
            "s": {"type": "span", "labels": [{"key": "drug"}]},
        },
        SCHEMAS,
    )
    assert state == {"user": "hi", "answer": "hello"}
    assert questions["p"] == {
        "type": "span",
        "instructions": "Which spans are personal information?",
        "criteria": {"PERSON": "a name"},
        "over": "user",
    }
    assert questions["c"]["criteria"] == {"benign": "ok", "jailbreak": None}
    assert "overflow" not in questions["c"] and "choices" not in questions["c"]
    assert questions["s"]["criteria"] == {"drug": None}


def test_drift_fails_the_comparison() -> None:
    ours = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.6, "b": 0.4},
            }
        }
    }
    theirs = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.6, "b": 0.4},
            }
        }
    }
    assert parity.compare({"q": {}}, ours, theirs)["q"] == {
        "same": True,
        "dp": 0.0,
        "type": "choice",
    }
    theirs["answers"]["q"]["probabilities"] = {"a": 0.4, "b": 0.6}
    theirs["answers"]["q"]["choice"] = "b"
    row = parity.compare({"q": {}}, ours, theirs)["q"]
    assert not row["same"] and row["dp"] == pytest.approx(0.2)


def test_the_pins_tool_prints_a_table_entry(tmp_path) -> None:
    root = write_pico_package(tmp_path / "pico", window=640)
    out = subprocess.run(
        [sys.executable, str(TOOLS / "vela3_pico_pins.py"), "--package", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    pins = json.loads(out.stdout)
    assert set(pins["files"]) == {
        "config.json",
        "vela3_config.json",
        "router_questions.json",
        "tokenizer.json",
        "model.safetensors",
    }
    assert pins["loaded_parameters"] > 0 and pins["golden_one_pass"]
