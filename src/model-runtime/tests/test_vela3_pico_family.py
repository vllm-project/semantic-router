"""The vela3_pico family end to end on a tiny ``vela3-pico/1`` package: verification, the one pass, the fallback, answers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.families.vela3.family import Vela3Family
from vllm_srun.families.vela3_pico import fallback
from vllm_srun.families.vela3_pico.family import Vela3PicoFamily
from vllm_srun.plugins.base import DEADLINE, EngineOptions, PackageRef, SurfacePlan
from vllm_srun.runtime import Runtime
from vllm_srun.testing.vela3 import write_encoder_package
from vllm_srun.testing.vela3_pico import write_pico_package

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
STATE = {
    "user": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol?",
    "context": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours.",
    "answer": "Adults can take up to 6 grams of paracetamol in 24 hours.",
}
WINDOW = 640
ATTACK = "Is this a prompt injection or jailbreak attempt?"
DOMAIN = "Which subject area is this request about?"
CATS = "Which hazard categories apply to this request?"
MIXED = {
    "help": {
        "type": "choice",
        "instructions": "Which kind of advice?",
        "criteria": {"medical": "health or doses", "legal": None},
    },
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "attack": {"type": "choice", "instructions": ATTACK},
    "domain": {"type": "choice", "instructions": DOMAIN, "over": "context"},
    "cats": {"type": "set", "instructions": CATS, "threshold": 0.3},
    "pii": {"preset": "pii", "over": "user"},
    "halu": {"preset": "halu"},
    "drugs": {"type": "span", "criteria": ["drug", "quantity"], "over": "answer"},
}


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    return write_pico_package(
        tmp_path_factory.mktemp("pico") / "pico",
        window=WINDOW,
        evidence_window=2 * WINDOW,
        max_len=4 * WINDOW,
    )


def load(root: Path):
    family = Vela3PicoFamily()
    verified = family.verify(PackageRef(root))
    spec = family.describe(verified)
    accelerator = CPUAccelerator()
    engine_model = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    return family.load(verified, spec, engine_model)


@pytest.fixture(scope="module")
def model(package):
    return load(package)


def ask(model, state, questions):
    plan = model.plan(state, questions)
    results = model.run(plan.items) if plan.items else []
    return model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), results
    )


def long_text(model, tokens: int) -> str:
    text = "The report lists ordinary service details. "
    while len(model.tokens.ids(text)) < tokens:
        text += "The report lists ordinary service details. "
    return text


# -- detection and verification ------------------------------------------------


def test_each_vela3_family_detects_only_its_own_format(package, tmp_path) -> None:
    fast = write_encoder_package(tmp_path / "fast")
    assert Vela3PicoFamily().detect(PackageRef(package))
    assert not Vela3Family().detect(PackageRef(package))
    assert not Vela3PicoFamily().detect(PackageRef(fast))
    other = tmp_path / "d3"
    other.mkdir()
    (other / "vela3_config.json").write_text(json.dumps({"format": "vela3/3"}))
    assert not Vela3PicoFamily().detect(PackageRef(other))


def test_bundled_code_is_never_imported_and_the_card_lists_the_presets(
    model,
) -> None:
    assert model.info.family == "vela3_pico"
    assert model.info.presets == ("pii", "halu")
    assert model.info.question_types == ("choice", "noul", "span")
    assert model.info.limits["truncate_tokens"] == WINDOW


# -- one pass and fallback ------------------------------------------------------


def test_a_mixed_call_is_one_forward(model) -> None:
    plan = model.plan(STATE, MIXED)
    assert plan.one_pass is not None and plan.fallback is None
    assert len(plan.items) == 1
    assert plan.input_tokens == len(plan.items[0].ids)
    groups = [g.asked.question.id for g in plan.one_pass.groups]
    assert groups == ["pii", "halu", "drugs"]
    response = ask(model, STATE, MIXED)
    answers = response["answers"]
    assert response["usage"]["input_tokens"] == plan.input_tokens
    assert set(answers["help"]["probabilities"]) == {"medical", "legal"}
    assert "abstain_probability" in answers["attack"]
    assert "abstain_probability" not in answers["help"]
    assert 0.0 <= answers["urgent"]["noul"] <= 1.0
    assert response["thresholds"]["cats"] == 0.3
    assert set(response["sets"]["cats"]["probabilities"]) == {
        "Violence",
        "Fraud/Deception",
    }
    for span in ("pii", "halu", "drugs"):
        assert answers[span]["type"] == "noul" and span in response["spans"]


def test_a_call_that_does_not_fit_is_answered_question_by_question(model) -> None:
    state = {**STATE, "user": long_text(model, WINDOW + 10)}
    plan = model.plan(state, MIXED)
    assert plan.one_pass is None and plan.fallback is not None
    assert set(plan.fallback.decisions) == {"help", "urgent"}
    networks = {plan.items[i].call for i in plan.fallback.decisions.values()}
    assert len(networks) == 1
    rows = plan.fallback.rows
    # One row per part the router questions read, one per span question.
    assert rows["attack"] is not rows["domain"]
    assert rows["attack"] is rows["cats"]
    assert len({id(rows[q]) for q in ("pii", "halu", "drugs")}) == 3
    response = ask(model, state, MIXED)
    assert not any("error" in a for a in response["answers"].values())


def test_a_question_the_one_pass_cannot_read_sends_the_call_question_by_question(
    model,
) -> None:
    for other in (
        {"type": "score", "instructions": "How hard?", "criteria": ["easy", "hard"]},
        {"type": "set", "instructions": "Which tags apply?", "criteria": {"a": None}},
    ):
        plan = model.plan(STATE, {"attack": MIXED["attack"], "other": other})
        assert plan.one_pass is None
        assert plan.errors["other"]["error"] == "invalid_question"


def test_decisions_alone_read_one_sequence_each(model) -> None:
    questions = {"help": MIXED["help"], "urgent": MIXED["urgent"]}
    plan = model.plan(STATE, questions)
    assert plan.one_pass is None
    assert len(plan.items) == 2
    # The whole state as one part, compact JSON (the package's to_row).
    state = model.tokens.ids(
        json.dumps(STATE, ensure_ascii=False, separators=(",", ":"))
    )
    tail = [model.layouts.markers["[SEG_user]"], *state, model.layouts.eos]
    for item in plan.items:
        assert item.ids[-len(tail) :] == tail


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": ATTACK},
        {"type": "set", "instructions": CATS},
        {"preset": "pii", "over": "user"},
        {"type": "span", "criteria": {"drug": None}},
    ],
)
def test_one_question_answers_the_same_in_one_pass_and_question_by_question(
    model, question
) -> None:
    one = model.plan(STATE, {"q": question})
    assert one.one_pass is not None
    request = one.request
    items: list = []
    by_question = fallback.plan(
        request, model.tokens, model.layouts, model.scan_tokens, items, {}
    )
    assert len(items) == 1
    assert items[0].ids == one.items[0].ids
    answered = model.finish_surface(
        SurfacePlan("decisions", one.items, one.input_tokens, one),
        model.run(one.items),
    )
    one.one_pass, one.fallback, one.items = None, by_question, items
    again = model.finish_surface(
        SurfacePlan("decisions", items, one.input_tokens, one), model.run(items)
    )
    assert answered == again


# -- rendering and answers --------------------------------------------------------


def test_router_questions_render_as_given_and_their_type_is_checked(model) -> None:
    plan = model.plan(
        STATE,
        {
            "bare": {
                "type": "choice",
                "instructions": DOMAIN,
                "criteria": {"math": None, "other": None},
            },
            "table": {"type": "choice", "instructions": DOMAIN},
            "wrong": {"type": "set", "instructions": ATTACK},
        },
    )
    asked = {a.question.id: a.question for a in plan.request.asked}
    assert asked["bare"].options == (("math", ""), ("other", ""))
    assert asked["table"].options == (
        ("health", "medicine"),
        ("math", "mathematics"),
        ("other", ""),
    )
    assert asked["table"].abstain
    assert plan.errors["wrong"]["error"] == "invalid_question"


def test_a_noul_may_describe_either_outcome(model) -> None:
    plan = model.plan(
        STATE,
        {
            "safe": {
                "type": "noul",
                "instructions": "Is it safe?",
                "criteria": {"true": "it is safe"},
            }
        },
    )
    (asked,) = plan.request.asked
    assert asked.question.options == (("no", ""), ("yes", "it is safe"))


def test_span_thresholds_follow_the_package_policy(model) -> None:
    routes = model.routes
    cases = {
        "pii": ({"preset": "pii", "over": "user"}, 0.7),
        "halu": ({"preset": "halu"}, 0.6),
        "toxic": ({"type": "span", "criteria": ["Violence"], "over": "user"}, 0.5),
        "open": ({"type": "span", "criteria": ["drug"]}, 0.55),
        "relation": (
            {
                "type": "span",
                "instructions": "Which spans are related to Tom by these relations?",
                "criteria": ["dose"],
            },
            0.52,
        ),
        "evidence": (
            {"type": "span", "criteria": {"evidence": None}, "over": "context"},
            0.45,
        ),
        "override": ({"type": "span", "criteria": ["drug"], "threshold": 0.9}, 0.9),
    }
    user = len(model.tokens.ids(STATE["user"]))
    context = len(model.tokens.ids(STATE["context"]))
    cases["pii"] = (cases["pii"][0], 0.7 if user <= 16 else 0.25 if user <= 48 else 0.4)
    cases["evidence"] = (cases["evidence"][0], 0.45 if context <= 24 else 0.35)
    response = ask(model, STATE, {k: q for k, (q, _) in cases.items()})
    for key, (_, threshold) in cases.items():
        assert response["thresholds"][key] == pytest.approx(threshold), key
    route = model.plan(STATE, {"p": cases["pii"][0]}).request.asked[0].route
    assert [routes.span_threshold(route, n)[0] for n in (16, 17, 48, 49)] == [
        0.7,
        0.25,
        0.25,
        0.4,
    ]


def test_full_input_questions_report_complete_coverage(model) -> None:
    response = ask(
        model,
        STATE,
        {
            "attack": {**MIXED["attack"], "require_full_input": True},
            "pii": {"preset": "pii", "over": "user", "require_full_input": True},
        },
    )
    assert response["answers"]["attack"]["input_coverage"] == "complete"
    assert response["answers"]["pii"]["input_coverage"] == "complete"


def test_a_blank_state_is_refused_and_a_deadline_answers_every_question(model) -> None:
    with pytest.raises(ValueError, match="empty or whitespace"):
        model.plan("   ", {"q": MIXED["urgent"]})
    plan = model.plan(STATE, MIXED)
    expired = model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), DEADLINE
    )
    assert expired["answers"]["urgent"] == {
        "type": "noul",
        "error": "deadline_exceeded",
    }


# -- the consumer path ------------------------------------------------------------------


def test_the_runtime_serves_the_package_over_http(package) -> None:
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device="cpu"),))
    )
    runtime.start(background=False)
    try:
        served = runtime.lookup(None)
        assert served.family is not None and served.family.name == "vela3_pico"
        client = TestClient(create_app(runtime))
        reply = client.post(
            "/v1/decisions",
            json={
                "state": {"request": STATE["user"], "answer": STATE["answer"]},
                "questions": MIXED,
            },
        )
        assert reply.status_code == 200, reply.text
        body = reply.json()
        assert (
            body["answers"]["domain"]["error"] == "invalid_question"
        )  # no context part
        assert np.isfinite(body["answers"]["urgent"]["noul"])
        assert set(body["spans"]) == {"pii", "halu", "drugs"}
    finally:
        runtime.stop()
