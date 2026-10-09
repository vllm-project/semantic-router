"""Strict System One questions either cover the supplied state or fail explicitly.

The fixtures use short context limits to exercise the same schema overhead and
window boundaries as the published packages. Token usage alone is not evidence
that all input was read: span coverage is checked against the planned words.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from starlette.testclient import TestClient
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.families.vela2.family import Vela2Family
from vllm_srun.plugins.base import EngineOptions, PackageRef, SurfacePlan
from vllm_srun.runtime import Runtime
from vllm_srun.testing import decision1
from vllm_srun.testing.vela2 import write_decoder_package, write_encoder_package

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

QUESTION_TYPES = {
    "choice": {
        "type": "choice",
        "instructions": "What is the task?",
        "criteria": {"code": "Programming", "other": "Other work"},
    },
    "score": {
        "type": "score",
        "instructions": "How difficult is the task?",
        "criteria": ["Easy", "Hard"],
    },
    "noul": {
        "type": "noul",
        "instructions": "Does the input contain personal information?",
    },
    "set": {
        "type": "set",
        "instructions": "Which kinds of personal information occur?",
        "criteria": {"person": "A person's name", "email": "An email address"},
    },
    "span": {
        "type": "span",
        "instructions": "Locate personal information.",
        "criteria": {"email": "An email address"},
    },
}


@pytest.fixture(scope="module")
def strict_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("full-input")
    packages = {
        "encoder": write_encoder_package(root / "encoder", max_length=1024),
        "decoder": write_decoder_package(root / "decoder", max_length=1024),
    }
    models = {}
    for name, path in packages.items():
        family = Vela2Family()
        verified = family.verify(PackageRef(path))
        spec = family.describe(verified)
        accelerator = CPUAccelerator()
        engine = NativeEngine().load(
            spec, accelerator, accelerator.devices()[0], EngineOptions()
        )
        models[name] = family.load(verified, spec, engine)
    return models


def question(kind, **overrides):
    return {**deepcopy(QUESTION_TYPES[kind]), **overrides}


def finish(model, plan):
    results = model.run(plan.items) if plan.items else []
    return model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), results
    )


def long_text(model, minimum):
    text = "The report contains ordinary service details. "
    while len(model.tokens.ids(text)) < minimum:
        text += "The report contains ordinary service details. "
    return text + " Contact alice@example.com."


@pytest.fixture(scope="module")
def decision_clients(tmp_path_factory, qwen3_runtime):
    root = tmp_path_factory.mktemp("full-input-decision1")
    package = decision1.write_fixture(root / "decision1", "vela-encoder", 0)
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device="cpu"),))
    )
    runtime.start(background=False)
    try:
        yield {
            "decision1": TestClient(create_app(runtime)),
            "decision2": TestClient(create_app(qwen3_runtime)),
        }
    finally:
        runtime.stop()


@pytest.mark.parametrize("member", ["decision1", "decision2"])
def test_strict_decision_api_preserves_answers_and_emits_proof(
    decision_clients, member
):
    client = decision_clients[member]
    body = {
        "state": "Email alice@example.com.",
        "questions": {kind: question(kind) for kind in ("choice", "score", "noul")},
    }
    original = client.post("/v1/decisions", json=body)
    assert original.status_code == 200
    for item in body["questions"].values():
        item["require_full_input"] = True
    strict = client.post("/v1/decisions", json=body)
    assert strict.status_code == 200
    response = strict.json()
    for answer in response["answers"].values():
        assert "error" not in answer
        assert answer.pop("input_coverage") == "complete"
    assert response == original.json()


@pytest.mark.parametrize("member", ["encoder", "decoder"])
@pytest.mark.parametrize("kind", QUESTION_TYPES)
def test_full_input_is_an_explicit_boolean_for_every_question(
    strict_models, member, kind
):
    model = strict_models[member]
    for value in (False, True):
        plan = model.reader.read(
            "Email alice@example.com.",
            {"task": question(kind, require_full_input=value)},
        )
        assert plan.errors == {}
        assert len(plan.questions) == 1


@pytest.mark.parametrize("kind", QUESTION_TYPES)
@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_non_boolean_full_input_is_not_silently_coerced(strict_models, kind, value):
    plan = strict_models["decoder"].reader.read(
        "Email alice@example.com.",
        {"task": question(kind, require_full_input=value)},
    )
    assert plan.errors["task"]["error"] == "invalid_question"
    assert plan.questions == []


@pytest.mark.parametrize("kind", QUESTION_TYPES)
def test_full_input_cannot_be_combined_with_truncation(strict_models, kind):
    plan = strict_models["decoder"].reader.read(
        "Email alice@example.com.",
        {"task": question(kind, require_full_input=True, overflow="truncate")},
    )
    assert plan.errors["task"]["error"] == "invalid_question"


@pytest.mark.parametrize("member", ["encoder", "decoder"])
def test_strict_short_answers_preserve_the_published_layout(strict_models, member):
    model = strict_models[member]
    state = "Email alice@example.com."
    # The schema and all supplied text fit, so admission must not change the
    # numerical path, question order, sidecars, thresholds or input accounting.
    questions = {"pii": question("span"), "present": question("noul")}
    original = finish(model, model.plan(state, questions))
    for value in (False, True):
        flagged = {
            name: {**item, "require_full_input": value}
            for name, item in questions.items()
        }
        response = finish(model, model.plan(state, flagged))
        for answer in response["answers"].values():
            if value:
                assert answer.pop("input_coverage") == "complete"
            else:
                assert "input_coverage" not in answer
        assert response == original


@pytest.mark.parametrize("member", ["encoder", "decoder"])
def test_strict_set_proof_covers_the_set_and_every_label(strict_models, member):
    model = strict_models[member]
    result = finish(
        model,
        model.plan(
            "Email alice@example.com.",
            {"pii_categories": question("set", require_full_input=True)},
        ),
    )
    assert result["sets"]["pii_categories"]["input_coverage"] == "complete"
    for name in ("person", "email"):
        answer = result["answers"][f"pii_categories.{name}"]
        assert "error" not in answer
        assert answer["input_coverage"] == "complete"


@pytest.mark.parametrize("kind", ["choice", "score", "noul", "set"])
def test_decoder_rejects_state_that_fits_without_its_question_schema(
    strict_models, kind
):
    model = strict_models["decoder"]
    state = long_text(model, 950)
    assert len(model.tokens.ids(state)) < model.package.max_input_tokens
    permissive = model.plan(state, {"task": question(kind)})
    assert permissive.items and not permissive.errors
    strict = model.plan(state, {"task": question(kind, require_full_input=True)})
    result = finish(model, strict)
    assert result["answers"]["task"]["error"] == "max_length_exceeded"
    assert "input_coverage" not in result["answers"]["task"]
    assert strict.items == []


@pytest.mark.parametrize("member", ["encoder", "decoder"])
@pytest.mark.parametrize("long_role", ["context", "request", "answer"])
def test_grounding_covers_every_supplied_part_not_only_the_target(
    strict_models, member, long_role
):
    model = strict_models[member]
    state = {
        "context": "The meeting is on Friday. " * 12,
        "request": "When is the meeting? " * 12,
        "answer": "The meeting is on Friday. " * 12,
    }
    state[long_role] = long_text(model, 950)
    # Even when over names only the answer, its truth depends on the context
    # and request. Covering one part cannot establish the full-input promise.
    plan = model.plan(
        state,
        {
            "grounded": {
                "type": "noul",
                "instructions": "Is the answer entirely supported by the context?",
                "over": "answer",
                "require_full_input": True,
            }
        },
    )
    answer = finish(model, plan)["answers"]["grounded"]
    assert answer["error"] == "max_length_exceeded"
    assert "input_coverage" not in answer
    assert plan.items == []


@pytest.mark.parametrize("member", ["encoder", "decoder"])
def test_span_windows_prove_coverage_including_the_last_word(strict_models, member):
    model = strict_models[member]
    state = long_text(model, 1500)
    plan = model.plan(
        state,
        {"pii": question("span", require_full_input=True)},
        scan=4096,
    )
    assert not plan.errors and plan.items
    (row,) = plan.rows
    words = row.parts[0].words
    assert words is not None
    if member == "encoder":
        covered = np.concatenate([item.word_index for item in plan.items])
    else:
        windows = [tree for tree in plan.items if tree.window]
        assert windows
        covered = np.concatenate(
            [block.word_index for tree in windows for block in tree.blocks]
        )
    assert set(covered.tolist()) == set(range(len(words)))
    assert int(words.offsets[-1][1]) == len(state)
    assert any(state[start:end] == "alice@example.com" for start, end in words.offsets)


@pytest.mark.parametrize("member", ["encoder", "decoder"])
@pytest.mark.parametrize("long_role", ["context", "request"])
def test_span_windows_cannot_hide_clipped_grounding_context(
    strict_models, member, long_role
):
    model = strict_models[member]
    state = {
        "context": "The meeting is on Friday.",
        "request": "When is the meeting?",
        "answer": long_text(model, 1500),
    }
    state[long_role] = long_text(model, 950)
    plan = model.plan(
        state,
        {
            "unsupported": {
                "type": "span",
                "instructions": "Locate claims unsupported by the supplied context.",
                "criteria": {"unsupported": "An unsupported claim"},
                "over": "answer",
                "require_full_input": True,
            }
        },
        scan=4096,
    )
    answer = finish(model, plan)["answers"]["unsupported"]
    assert answer["error"] == "max_length_exceeded"
    assert "input_coverage" not in answer
    assert plan.items == []


@pytest.mark.parametrize("member", ["encoder", "decoder"])
@pytest.mark.parametrize("kind", ["noul", "span"])
def test_scan_limit_covers_parts_outside_over(strict_models, member, kind):
    model = strict_models[member]
    state = {"request": "Email alice@example.com.", "context": long_text(model, 1500)}
    plan = model.plan(
        state,
        {"pii": question(kind, over="request", require_full_input=True)},
        scan=1024,
    )
    answer = finish(model, plan)["answers"]["pii"]
    assert answer["error"] == "scan_budget_exceeded"
    assert "input_coverage" not in answer
    assert plan.items == []


@pytest.mark.parametrize("member", ["encoder", "decoder"])
def test_a_word_start_does_not_prove_coverage_of_an_oversized_word(
    strict_models, member
):
    model = strict_models[member]
    state = "Identifier" * 1024
    count = len(model.tokens.ids(state))
    assert count > model.package.max_input_tokens
    plan = model.plan(
        state,
        {"entities": question("span", require_full_input=True)},
        scan=count + 1,
    )
    # A Span head scores the first subword. Its presence in one window does
    # not show that the rest of this single long word was visible there.
    answer = finish(model, plan)["answers"]["entities"]
    assert answer["error"] == "max_length_exceeded"
    assert "input_coverage" not in answer
    assert plan.items == []


def test_decoder_word_coverage_cannot_hide_unread_leading_tokens(strict_models):
    model = strict_models["decoder"]
    state = " " * 100 + long_text(model, 1500)
    permissive = model.plan(state, {"entities": question("span")}, scan=4096)
    assert permissive.items and not permissive.errors
    part = permissive.rows[0].parts[0]
    assert part.words is not None and part.words.first[0] > 0
    strict = model.plan(
        state,
        {"entities": question("span", require_full_input=True)},
        scan=4096,
    )
    answer = finish(model, strict)["answers"]["entities"]
    assert answer["error"] == "max_length_exceeded"
    assert "input_coverage" not in answer
    assert strict.items == []


def test_decoder_rejection_is_per_question_and_span_coverage_is_not_transferable(
    strict_models,
):
    model = strict_models["decoder"]
    state = long_text(model, 1500)
    plan = model.plan(
        state,
        {
            "complete_presence": question("noul", require_full_input=True),
            "best_effort_presence": question("noul"),
            "locations": question("span", require_full_input=True),
        },
        scan=4096,
    )
    result = finish(model, plan)
    assert result["answers"]["complete_presence"]["error"] == "max_length_exceeded"
    assert "input_coverage" not in result["answers"]["complete_presence"]
    assert "error" not in result["answers"]["best_effort_presence"]
    assert "input_coverage" not in result["answers"]["best_effort_presence"]
    assert "error" not in result["answers"]["locations"]
    assert result["answers"]["locations"]["input_coverage"] == "complete"
