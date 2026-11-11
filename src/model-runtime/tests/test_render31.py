"""The render31 port against the Vela 3.1 package's own ``render31.py``, on its recorded outputs."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

import pytest
from vllm_srun.families import render31

CASES = json.loads(
    files("vllm_srun.testing").joinpath("data/render31_cases.json").read_text("utf-8")
)


def _port(function: str, arguments: list[Any]) -> Any:
    if function == "to_row":
        state, question = arguments
        text, options = render31.decision(question)
        return {"part": render31.text(state), "text": text, "options": options}
    return getattr(render31, function)(*arguments)


def test_the_cases_come_from_the_released_renderer() -> None:
    assert CASES["package"]["sha256"]["render31.py"] == (
        "d0c8ebc0896c7d3b942f5eba98cdd247ef01e08d329237015bef4fc55009c8ad"
    )
    assert render31.VERSION == "render31/1"
    assert len(CASES["render31"]) == 39


@pytest.mark.parametrize(
    "case",
    CASES["render31"],
    ids=[f"{i}-{c['function']}" for i, c in enumerate(CASES["render31"])],
)
def test_every_function_renders_as_the_package_does(case: dict[str, Any]) -> None:
    if "error" in case:
        kind, _, message = case["error"].partition(": ")
        assert kind == "ValueError"
        with pytest.raises(ValueError) as raised:
            _port(case["function"], case["arguments"])
        assert str(raised.value) == message
        return
    got = _port(case["function"], case["arguments"])
    # Byte for byte: the same strings, and the same option order.
    assert json.dumps(got, ensure_ascii=False) == json.dumps(
        case["result"], ensure_ascii=False
    )


def test_names_only_stay_names_only_and_nothing_is_filled_in() -> None:
    builtin = {"options": {"health": "medicine", "math": "mathematics"}}
    asked = {"type": "choice", "criteria": {"math": None, "law": "rules"}}
    assert render31.options(asked, builtin) == {"math": "", "law": "rules"}
    assert render31.options({"type": "choice"}, builtin) == {
        "health": "medicine",
        "math": "mathematics",
    }
    labels = {"criteria": {"unsupported": None}}
    assert render31.span_labels(labels, {"labels": {"unsupported": "a claim"}}) == {
        "unsupported": ""
    }
