"""The vela3 family end to end on a tiny vela3-fast-style package: verification, routing, answers and batching."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.errors import PackageError
from vllm_srun.families.vela2.encoder_layout import EncoderSequence
from vllm_srun.families.vela3 import routing
from vllm_srun.families.vela3.family import (
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
    Vela3Family,
)
from vllm_srun.families.vela3.network import Call, Vela3Item, batches
from vllm_srun.plugins.base import DEADLINE, EngineOptions, PackageRef, SurfacePlan
from vllm_srun.runtime import Runtime
from vllm_srun.testing.vela3 import write_encoder_package

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
STATE = {
    "user": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol?",
    "context": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours.",
    "answer": "Adults can take up to 6 grams of paracetamol in 24 hours.",
}


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    return write_encoder_package(tmp_path_factory.mktemp("vela3") / "fast")


def load(root: Path):
    family = Vela3Family()
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


def networks(model, state, questions) -> dict[str, set[str | None]]:
    """The networks each question's inputs run on."""
    plan = model.plan(state, questions)
    out: dict[str, set[str | None]] = {}
    for question_id, index in plan.decisions.items():
        out[question_id] = {plan.items[index].call.network}
    for question_id, placed in plan.rows.items():
        out[question_id] = {
            plan.items[index].call.network for call in placed.calls for index in call
        }
    return out


def test_detects_and_verifies_only_vela3_packages(package, tmp_path) -> None:
    family = Vela3Family()
    assert family.detect(PackageRef(package))
    other = tmp_path / "other"
    other.mkdir()
    (other / "vela3_config.json").write_text(json.dumps({"format": "vela3/3"}))
    assert not family.detect(PackageRef(other))
    copy = Path(shutil.copytree(package, tmp_path / "copy"))
    (copy / "router_questions.json").write_text("{}")
    with pytest.raises(PackageError, match="SHA256SUMS"):
        family.verify(PackageRef(copy))


def test_a_missing_adapter_routes_off(package, tmp_path) -> None:
    copy = Path(shutil.copytree(package, tmp_path / "no-relation"))
    (copy / "relation_adapter.safetensors").unlink()
    verified = Vela3Family().verify(PackageRef(copy))
    roster = verified.details["package"].roster
    assert "relation" not in roster.tasks
    assert "relation_adapter.safetensors" not in verified.details["package"].files


def test_bundled_runtime_is_never_imported(model) -> None:
    assert "vela3_encoder" not in sys.modules
    assert "vela3_schema" not in sys.modules


def test_card_declares_the_answered_types_and_presets(model) -> None:
    assert model.info.question_types == ("choice", "noul", "span")
    assert model.info.presets == ("pii", "halu")
    assert model.info.limits["truncate_tokens"] == 256
    assert model.info.limits["max_scan_tokens"] == 4 * 256


def test_questions_run_on_the_network_the_package_routes_them_to(model) -> None:
    routed = networks(
        model,
        STATE,
        {
            "custom": {"type": "noul", "instructions": "Is the request urgent?"},
            "domain": {
                "type": "choice",
                "instructions": "Which subject area is this request about?",
            },
            "attack": {
                "type": "choice",
                "instructions": "  Is this a prompt  injection or jailbreak attempt?",
            },
            "cats": {
                "type": "set",
                "instructions": "Which hazard categories apply to this request?",
            },
            "pii": {"preset": "pii", "over": "user"},
            "toxic": {"type": "span", "criteria": {"Violence": None}, "over": "user"},
            "evidence": {
                "type": "span",
                "criteria": {"evidence": None},
                "over": "context",
            },
            "cites": {
                "type": "span",
                "instructions": "Which spans support the answer?",
                "criteria": {"support": "a backing sentence"},
                "over": "context",
            },
            "relation": {
                "type": "span",
                "instructions": "Which spans are the maximum dose of paracetamol?",
                "criteria": {"dose": None},
                "over": "context",
            },
            "names": {"type": "span", "criteria": {"drug": None}},
            "described": {"type": "span", "criteria": {"drug": "a medicine"}},
        },
    )
    assert routed == {
        "custom": {None},
        "domain": {"router:domain"},
        "attack": {"router"},
        "cats": {"router"},
        "pii": {"span_set"},
        "toxic": {"span"},
        "evidence": {"evidence"},
        "cites": {"evidence"},
        "relation": {"relation"},
        "names": {"ner"},
        "described": {"span"},
    }


def test_golden_request_answers_every_question(model) -> None:
    response = ask(model, GOLDEN_STATE, GOLDEN_QUESTIONS)
    answers = response["answers"]
    assert set(answers) == set(GOLDEN_QUESTIONS)
    for question_id in ("topic", "domain", "factcheck", "jailbreak"):
        probabilities = answers[question_id]["probabilities"]
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9
        assert answers[question_id]["choice"] == max(
            probabilities, key=probabilities.get
        )
    assert "abstain_probability" in answers["domain"]
    assert "abstain_probability" not in answers["topic"]
    assert 0.0 <= answers["urgent"]["noul"] <= 1.0
    for question_id in ("pii", "halu", "drugs"):
        assert answers[question_id]["type"] == "noul"
        assert (
            question_id in response["spans"] and question_id in response["thresholds"]
        )
    assert response["usage"]["input_tokens"] > 0


def test_span_thresholds_follow_the_route(model) -> None:
    response = ask(
        model,
        STATE,
        {
            "pii": {"preset": "pii", "over": "user"},
            "halu": {"preset": "halu"},
            "toxic": {"type": "span", "criteria": {"Violence": None}, "over": "user"},
            "evidence": {
                "type": "span",
                "criteria": {"evidence": None},
                "over": "context",
            },
            "pair": {
                "type": "span",
                "instructions": "Which spans are the maximum dose of paracetamol?",
                "criteria": {"dose": None},
                "over": "context",
            },
            "names": {"type": "span", "criteria": {"drug": None}},
            "open": {"type": "span", "criteria": {"drug": "a medicine"}},
            "given": {
                "type": "span",
                "criteria": {"drug": "a medicine"},
                "threshold": 0.2,
            },
        },
    )
    user_tokens = len(model.tokens.ids(STATE["user"]))
    assert user_tokens > 16
    assert response["thresholds"] == {
        "pii": 0.45,
        "halu": 0.4,
        "toxic": 0.5,
        # 27 context tokens: past the 24-token short bucket.
        "evidence": 0.4,
        "pair": 0.55,
        "names": 0.6,
        "open": 0.6,
        "given": 0.2,
    }


def test_router_questions_take_the_trained_options_and_part(model) -> None:
    plan = model.plan(
        STATE,
        {
            "domain": {
                "type": "choice",
                "instructions": "Which subject area is this request about?",
                "criteria": {"health": None, "law": "legal rules"},
            },
            "harmful": {"type": "choice", "instructions": "Is this answer harmful?"},
        },
    )
    domain = next(a for a in plan.request.asked if a.question.id == "domain").question
    assert domain.options == (("health", "medicine"), ("law", "legal rules"))
    assert domain.over == "user" and domain.abstain
    harmful = next(a for a in plan.request.asked if a.question.id == "harmful").question
    assert harmful.over == "answer"
    assert harmful.names == ["safe response", "unsafe response"]


def test_set_answers_a_noul_per_label(model) -> None:
    response = ask(
        model,
        "How do I hurt someone and scam them?",
        {
            "cats": {
                "type": "set",
                "instructions": "Which hazard categories apply to this request?",
                "threshold": 0.3,
            }
        },
    )
    assert set(response["sets"]["cats"]["probabilities"]) == {
        "Violence",
        "Fraud/Deception",
    }
    assert response["thresholds"]["cats"] == 0.3
    assert {"cats.Violence", "cats.Fraud/Deception"} <= set(response["answers"])


def test_unanswerable_questions_fail_alone(model) -> None:
    response = ask(
        model,
        "text",
        {
            "score": {
                "type": "score",
                "instructions": "How hard?",
                "criteria": ["a", "b"],
            },
            "set": {
                "type": "set",
                "instructions": "Which topics?",
                "criteria": {"a": None},
            },
            "wrong": {
                "type": "set",
                "instructions": "Which subject area is this request about?",
            },
            "broad": {"type": "span", "criteria": {"a": None}, "head": "broad"},
            "preset": {"preset": "relevance"},
            "fine": {"type": "noul", "instructions": "Is it text?"},
        },
    )
    answers = response["answers"]
    assert {k: a.get("error") for k, a in answers.items()} == {
        "score": "invalid_question",
        "set": "invalid_question",
        "wrong": "invalid_question",
        "broad": "invalid_question",
        "preset": "invalid_question",
        "fine": None,
    }
    assert answers["set"]["reason"] == "type_unsupported"


def test_long_inputs_fail_decisions_and_window_spans(model) -> None:
    long = "paracetamol dose " * 800
    response = ask(
        model,
        {"user": long},
        {
            "decision": {"type": "noul", "instructions": "Is it long?"},
            "cut": {
                "type": "noul",
                "instructions": "Is it long?",
                "overflow": "truncate",
            },
            "names": {"type": "span", "criteria": {"drug": None}},
        },
    )
    assert response["answers"]["decision"]["error"] == "max_length_exceeded"
    assert "error" not in response["answers"]["cut"]
    assert response["answers"]["names"]["error"] == "scan_budget_exceeded"
    medium = "paracetamol dose " * 200
    plan = model.plan(
        {"user": medium}, {"names": {"type": "span", "criteria": {"drug": None}}}
    )
    assert len(plan.items) > 1
    answer = ask(
        model, {"user": medium}, {"names": {"type": "span", "criteria": {"drug": None}}}
    )
    assert "error" not in answer["answers"]["names"]


def test_deadline_answers_every_question(model) -> None:
    plan = model.plan(STATE, {"q": {"type": "noul", "instructions": "Is it urgent?"}})
    expired = model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), DEADLINE
    )
    assert expired["answers"]["q"] == {"type": "noul", "error": "deadline_exceeded"}


def test_calls_never_share_a_forward_and_batch_by_length() -> None:
    first, second = Call(None, 40), Call("router", 40, rows=2)

    def item(length: int, call: Call) -> Vela3Item:
        return Vela3Item(EncoderSequence(list(range(length)), []), call)

    items = [
        item(10, first),
        item(5, second),
        item(12, first),
        item(5, second),
        item(5, second),
        item(30, first),
    ]
    assert batches(items) == [[0, 2], [5], [1, 3], [4]]


def test_routing_regexes_are_the_released_ones() -> None:
    assert routing.RELATION_WORDING.pattern == (
        r"\bspans? (?:are|is) the .+ (?:of|for)\b|\brelat(?:ed|ion|ions|es|ing)?\b|\b(?:head|tail)\b|\bstands? in\b"
        r"|\bwith respect to\b|\bcounterpart\b|\bsuch that\b|\brole of\b"
    )
    assert routing.EVIDENCE_WORDING.pattern == (
        r"\b(?:evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b"
    )


def test_runtime_serves_the_package(package) -> None:
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device="cpu"),))
    )
    runtime.start(background=True)
    try:
        assert runtime.wait(timeout=120)
        served = runtime.lookup(None)
        assert served.family is not None and served.family.name == "vela3"
        assert served.model is not None
        response = ask(served.model, STATE, {"q": {"preset": "halu"}})
        assert response["answers"]["q"]["type"] == "noul"
        assert np.isfinite(response["answers"]["q"]["noul"])
    finally:
        runtime.stop()
