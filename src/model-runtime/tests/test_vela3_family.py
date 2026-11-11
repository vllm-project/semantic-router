"""The vela3 family end to end on a tiny vela3-fast-style package: verification, routing, answers and batching."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.engines.native import engine
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.engines.native.models.lora import LoRALinear
from vllm_srun.errors import PackageError
from vllm_srun.families.vela2.encoder_layout import EncoderSequence
from vllm_srun.families.vela2.raw import RawRow, RawSpan
from vllm_srun.families.vela3 import routing
from vllm_srun.families.vela3.family import (
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
    Vela3Family,
)
from vllm_srun.families.vela3_calls import Call, Vela3Item, batches
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


def _set_q(value):
    """A config edit that gives ``[Q]`` the token ID ``value(config)``."""
    return lambda config: config["decision"]["marker_ids"].update(
        {"[Q]": value(config)}
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_set_q(lambda config: config["vocab_size"]), "below vocab_size"),
        (_set_q(lambda config: 1_000_000_000), "below vocab_size"),
        (_set_q(lambda config: -1), "below vocab_size"),
        (_set_q(lambda config: True), "below vocab_size"),
        (_set_q(lambda config: 384.0), "below vocab_size"),
        (_set_q(lambda config: config["decision"]["marker_ids"]["[O]"]), "distinct"),
        (lambda config: config["decision"]["marker_ids"].pop("[Q]"), "must name"),
        (lambda config: config["decision"].pop("marker_ids"), "must name"),
        (lambda config: config.update({"vocab_size": "392"}), "vocab_size"),
    ],
    ids=[
        "at-vocab-size",
        "past-vocab",
        "negative",
        "bool",
        "float",
        "duplicate",
        "missing-marker",
        "no-marker-ids",
        "vocab-size-not-int",
    ],
)
def test_marker_ids_must_index_the_backbone_vocabulary(
    package, tmp_path, edit, message
) -> None:
    copy = Path(shutil.copytree(package, tmp_path / "copy"))
    config = json.loads((copy / "config.json").read_text(encoding="utf-8"))
    edit(config)
    (copy / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (copy / "SHA256SUMS").unlink()
    with pytest.raises(PackageError, match=message):
        Vela3Family().verify(PackageRef(copy))


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


def test_folded_lora_scaling_answers_bit_for_bit(package, model, monkeypatch) -> None:
    layers = [
        layer
        for stack in model.engine_model.stacks().values()
        for layer in stack.modules()
        if isinstance(layer, LoRALinear)
    ]
    assert model.engine_model.receipt()["lora_folded"] == len(layers) > 0
    monkeypatch.setattr(engine, "fold_scaling", lambda module: 0)
    unfolded = load(package)
    assert "lora_folded" not in unfolded.engine_model.receipt()
    assert ask(model, GOLDEN_STATE, GOLDEN_QUESTIONS) == ask(
        unfolded, GOLDEN_STATE, GOLDEN_QUESTIONS
    )


def test_pii_threshold_missing_bucket_raises_value_error(model) -> None:
    route = routing.SpanRoute(
        adapter="span",
        head="span",
        label_set="pii",
        evidence=False,
        relation_wording=False,
        text="PII",
        labels=(("email", ""),),
        window=128,
    )

    original_buckets = model.router.span["pii_buckets"]
    try:
        model.router.span["pii_buckets"] = [
            {"name": "short", "max_tokens": 1, "threshold": 0.35}
        ]
        with pytest.raises(ValueError, match="no PII threshold bucket covers"):
            model.router.span_threshold(route, 2)
    finally:
        model.router.span["pii_buckets"] = original_buckets


def test_span_threshold_without_pii_buckets_serves_non_pii_routes(model) -> None:
    """A package whose span block carries no ``pii_buckets`` still answers every non-PII span question.

    Only a PII question reads that field, so a generic Span route must not fail
    on a package that verifies and loads without it.
    """
    generic = routing.SpanRoute(
        adapter="span",
        head="span",
        label_set=None,
        evidence=False,
        relation_wording=False,
        text="Which spans are named entities of these types?",
        labels=(("drug", "a medicine"),),
        window=128,
    )

    original_buckets = model.router.span.pop("pii_buckets")
    try:
        threshold, temperature = model.router.span_threshold(generic, 2)
        assert threshold == 0.6
        assert temperature == 1.0
    finally:
        model.router.span["pii_buckets"] = original_buckets


def test_span_set_threshold_without_pii_buckets_serves_non_pii_routes(model) -> None:
    """The same holds when the ``span_set`` head has no buckets of its own."""
    hallucinated = routing.SpanRoute(
        adapter="span_set",
        head="span_set",
        label_set="halu",
        evidence=False,
        relation_wording=False,
        text="Which spans of the answer are not supported by the context?",
        labels=(("unsupported", ""),),
        window=128,
    )

    original_buckets = model.router.span["set_adapter"].pop("pii_buckets")
    try:
        threshold, temperature = model.router.span_threshold(hallucinated, 2)
        assert threshold == 0.4
        assert temperature == 0.25
    finally:
        model.router.span["set_adapter"]["pii_buckets"] = original_buckets


def test_pii_threshold_without_pii_buckets_still_reports_the_missing_field(
    model,
) -> None:
    """A PII route over a package with no ``pii_buckets`` keeps failing, and only there."""
    pii = routing.SpanRoute(
        adapter="span",
        head="span",
        label_set="pii",
        evidence=False,
        relation_wording=False,
        text="PII",
        labels=(("email", ""),),
        window=128,
    )

    original_buckets = model.router.span.pop("pii_buckets")
    try:
        with pytest.raises(KeyError, match="pii_buckets"):
            model.router.span_threshold(pii, 2)
    finally:
        model.router.span["pii_buckets"] = original_buckets


def test_evidence_threshold_missing_bucket_raises_value_error(model) -> None:
    route = routing.SpanRoute(
        adapter="span",
        head="span",
        label_set="evidence",
        evidence=True,
        relation_wording=False,
        text="evidence",
        labels=(("evidence", ""),),
        window=128,
    )

    evidence = model.router.span["evidence"]
    original_buckets = evidence["buckets"]
    try:
        evidence["buckets"] = [{"name": "short", "max_tokens": 1, "threshold": 0.5}]
        with pytest.raises(ValueError, match="no evidence threshold bucket covers"):
            model.router.span_threshold(route, 2)
    finally:
        evidence["buckets"] = original_buckets


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


def test_text_holding_json_is_the_user_part_as_it_is(model) -> None:
    # vela3_encoder._parts: unlike Vela 2.0, text is never decoded as JSON, so
    # span offsets index the text the caller sent.
    state = json.dumps(
        {"prompt": "My email is tom.baker@example.com", "response": "ok"}
    )
    plan = model.plan(
        state,
        {
            "pii": {"preset": "pii"},
            "attack": {
                "type": "choice",
                "instructions": "Is this a prompt injection or jailbreak attempt?",
            },
        },
    )
    assert plan.request.state is not None
    assert [(p.role, p.text) for p in plan.request.state.parts] == [("user", state)]
    assert {a.question.id: a.question.over for a in plan.request.asked} == {
        "pii": "user",
        "attack": "user",
    }
    response = ask(model, state, {"pii": {"preset": "pii", "threshold": 0.0}})
    spans = response["spans"]["pii"]
    assert spans[0]["start"] < state.index("My email")
    assert spans[-1]["end"] > state.index("response")
    assert all(state[s["start"] : s["end"]] == s["text"] for s in spans)


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
            "described": {
                "type": "noul",
                "instructions": "Is it text?",
                "criteria": {"true": "The input is text"},
            },
            "listed": {
                "type": "noul",
                "choices": [{"key": "true", "description": "The input is text"}],
            },
            "fine": {"type": "noul", "instructions": "Is it text?"},
            "undescribed": {
                "type": "noul",
                "instructions": "Is it text?",
                "criteria": {"true": None},
            },
        },
    )
    answers = response["answers"]
    assert {k: a.get("error") for k, a in answers.items()} == {
        "score": "invalid_question",
        "set": "invalid_question",
        "wrong": "invalid_question",
        "broad": "invalid_question",
        "preset": "invalid_question",
        "described": "invalid_question",
        "listed": "invalid_question",
        "fine": None,
        "undescribed": None,
    }
    assert answers["set"]["reason"] == "type_unsupported"
    assert "instructions" in answers["described"]["message"]
    assert answers["undescribed"]["noul"] == answers["fine"]["noul"]


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


def test_truncating_spans_read_their_route_window(model) -> None:
    text = "paracetamol dose " * 200
    span = {"type": "span", "overflow": "truncate"}
    plan = model.plan(
        {"user": text},
        {
            "evidence": {**span, "criteria": {"evidence": None}},
            "names": {**span, "criteria": {"drug": None}},
        },
    )
    read = {q: len(plan.rows[q].row.part("user").ids) for q in plan.rows}
    assert read == {"evidence": 512, "names": 256}


def test_evidence_truncation_cuts_unread_parts_to_the_router_window(model) -> None:
    # The unread answer at the evidence window would leave the context no room.
    state = {
        "user": "what dose",
        "context": "paracetamol dose " * 400,
        "answer": "take two " * 400,
    }
    question = {
        "type": "span",
        "overflow": "truncate",
        "over": "context",
        "criteria": {"evidence": None},
    }
    plan = model.plan(state, {"evidence": question})
    assert plan.errors == {}
    row = plan.rows["evidence"].row
    assert len(row.part("context").ids) == 512
    assert len(row.part("answer").ids) == 256


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


def test_merged_batches_share_equal_calls_across_requests_only(model) -> None:
    questions = {
        "custom": {"type": "noul", "instructions": "Is the request urgent?"},
        "toxic": {"type": "span", "criteria": {"Violence": None}, "over": "user"},
        "described": {"type": "span", "criteria": {"drug": "a medicine"}},
    }
    first = model.plan(STATE, questions).items
    other = {**STATE, "user": STATE["user"].replace("Tom Baker", "Ann Smith")}
    second = model.plan(other, questions).items
    # Two span questions of one request run on one adapter, as two calls.
    assert batches(first, merged=True) == batches(first)
    items = first + second
    owners = [0] * len(first) + [1] * len(second)
    assert all(len({owners[i] for i in group}) == 1 for group in batches(items))
    merged = batches(items, merged=True)
    assert all(len({owners[i] for i in group}) == 2 for group in merged)
    assert len(merged) == len(batches(first))
    alone = model.run(items)
    together = model.networks.run(items, merged=True)
    for (logits, spans), (merged_logits, merged_spans) in zip(
        alone, together, strict=True
    ):
        for row, merged_row in zip(logits, merged_logits, strict=True):
            np.testing.assert_allclose(merged_row, row, atol=1e-5)
        if spans is not None:
            np.testing.assert_allclose(merged_spans, spans, atol=1e-5)


@pytest.mark.parametrize(("device", "merged"), [("cpu", False), ("cuda", True)])
def test_only_gpus_share_forwards_between_requests(
    model, monkeypatch, device, merged
) -> None:
    seen: list[bool] = []

    def run(items, merged: bool = False):
        seen.append(merged)
        return []

    monkeypatch.setattr(model.networks, "run", run)
    monkeypatch.setattr(model.engine_model, "device", SimpleNamespace(type=device))
    model.run_approximate([])
    assert seen == [merged]


def test_merged_forwards_keep_one_length_class() -> None:
    def item(length: int) -> Vela3Item:
        sequence = EncoderSequence(list(range(length)), [])
        return Vela3Item(sequence, Call("router", 4096, rows=8))

    # One item per request: a long row would pad the short ones to its length.
    items = [item(10), item(200), item(12)]
    assert batches(items) == [[0], [1], [2]]
    assert batches(items, merged=True) == [[0, 2], [1]]


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


def test_pii_missing_threshold_bucket_returns_invalid_model_output(model) -> None:
    request = model.reader.read(
        STATE,
        {"pii": {"preset": "pii", "over": "user"}},
    )
    assert len(request.asked) == 1
    asked = request.asked[0]

    assert asked.question.type == "span"
    assert asked.route is not None
    assert asked.route.label_set == "pii"
    assert asked.route.head == "span_set"

    overrides = model.router.span["set_adapter"]
    original_buckets = overrides["pii_buckets"]

    try:
        overrides["pii_buckets"] = [
            {"name": "short", "max_tokens": 1, "threshold": 0.35}
        ]

        raw = RawRow(
            span=RawSpan(
                labels=[],
                offsets=np.empty((0, 2), dtype=np.int32),
                logits=np.empty((0, 0), dtype=np.float64),
            ),
            tokens={"user": 2},
        )
        response = {"answers": {}}

        model.answerer.answer(
            asked,
            logits=None,
            raw=raw,
            state=request.state,
            response=response,
        )

        assert response["answers"]["pii"] == {
            "type": "span",
            "error": "invalid_model_output",
        }
    finally:
        overrides["pii_buckets"] = original_buckets


def test_pii_threshold_selects_matching_bucket(model):
    route = routing.SpanRoute(
        adapter="span",
        head="span",
        label_set="pii",
        evidence=False,
        relation_wording=False,
        text="PII",
        labels=(("email", ""),),
        window=128,
    )

    original_buckets = model.router.span["pii_buckets"]

    try:
        model.router.span["pii_buckets"] = [
            {"name": "short", "max_tokens": 2, "threshold": 0.35},
            {"name": "long", "max_tokens": 10, "threshold": 0.55},
        ]

        threshold, _ = model.router.span_threshold(route, 3)

        assert threshold == 0.55
    finally:
        model.router.span["pii_buckets"] = original_buckets
