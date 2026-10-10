"""The reference examples checker (``tools/reference_examples.py``)."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reference_examples.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("reference_examples", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_shows_every_question_type(tool):
    pairs = tool.examples(tool.REFERENCE.read_text(encoding="utf-8"))
    kinds = {
        question["type"]
        for request, _response in pairs
        for question in request["questions"].values()
    }
    assert kinds == {"choice", "noul", "score", "set", "span"}
    for request, response in pairs:
        for question_id, question in request["questions"].items():
            if question["type"] == "set":
                assert question_id in response["sets"]
            else:
                assert question_id in response["answers"]


def test_numbers_match_within_the_tolerance_and_extra_fields_are_ignored(tool):
    documented = {"answers": {"q": {"type": "noul", "noul": 0.97}}}
    actual = {"answers": {"q": {"type": "noul", "noul": 0.9712}}, "usage": {}}
    assert tool.differences(documented, actual, 0.02) == []
    drift = tool.differences(documented, {"answers": {"q": {"type": "noul"}}}, 0.02)
    assert drift == ["$.answers.q.noul: missing"]
    assert tool.differences(
        documented, {"answers": {"q": {"type": "noul", "noul": 0.9}}}, 0.02
    ) == ["$.answers.q.noul: 0.9 is not within 0.02 of 0.97"]
    assert tool.differences([1, 2], [1], 0.02) == ["$: 1 items, the page shows 2"]
    assert tool.differences({"x": True}, {"x": 1}, 0.02) == ["$.x: 1 != True"]
