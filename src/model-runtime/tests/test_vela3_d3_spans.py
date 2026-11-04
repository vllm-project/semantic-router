"""Span questions of the Vela 3.0 d3 sizes: validation, routing, thresholds, rendering, windows and answers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from vllm_srun.errors import QuestionError
from vllm_srun.families.decision3.prompt import render
from vllm_srun.families.vela3_d3 import span_render, spans
from vllm_srun.families.vela3_d3.package import SpanConfig
from vllm_srun.families.vela3_d3.span_policy import Policy
from vllm_srun.families.vela3_d3.span_read import SpanReader, combine
from vllm_srun.families.vela3_d3.span_render import Renderer, Window
from vllm_srun.plugins.base import AdapterGate
from vllm_srun.testing import vela3_d3 as fixture

from .conftest import start_runtime

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

STATE = {
    "user": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose?",
    "answer": "Adults can take up to 6 grams of paracetamol in 24 hours.",
}
SPAN_CONFIG = fixture.vela3_config({})["span"]


def no_limit(length):
    return None


def no_definitions(names, only):
    return {}


def policy(without=()) -> Policy:
    return Policy(
        SpanConfig.parse({k: v for k, v in SPAN_CONFIG.items() if k not in without}),
        no_definitions,
    )


# ------------------------------------------------------------------ routing and thresholds


@pytest.mark.parametrize(
    ("names", "instructions", "route"),
    [
        (["EMAIL", "PERSON"], None, "span"),  # a router label set
        (["evidence"], None, "span"),
        (["answer"], "Which spans are named entities?", "span"),
        (["drug"], None, "ner"),
        (["drug"], "Which spans are the dose of the drug?", "relation"),
        (["drug"], "Which spans are related to the dose?", "relation"),
        (
            ["drug"],
            "Which spans relate to the dose?",
            "span",
        ),  # "relate" is not in the released pattern
        (["drug"], "Which spans support the claim?", "span"),
        (["drug"], "Which named entities appear?", "ner"),
        (["drug"], "Mark the drugs.", "span"),
    ],
)
def test_routes_follow_the_released_rules(names, instructions, route) -> None:
    p = policy()
    assert p.route(p.label_set(names), names, instructions) == route


def test_without_a_relation_or_ner_head_questions_take_the_span_head() -> None:
    assert (
        policy(without=("relation",)).route(None, ["x"], "Which spans relate to y?")
        == "span"
    )
    assert policy(without=("ner", "relation")).route(None, ["x"], None) == "span"


def test_thresholds_follow_the_label_set_and_the_route() -> None:
    p = policy()
    named = [("drug", "")]
    described = [("drug", "a medicine")]
    assert p.threshold("span", "pii", named, 10) == 0.7
    assert p.threshold("span", "pii", named, 17) == 0.6
    assert p.threshold("span", "evidence", named, 10) == 0.5
    assert p.threshold("span", "halu", named, 10) == 0.45
    assert p.threshold("span", None, described, 10) == 0.55
    assert p.threshold("span", None, named, 10) == 0.5
    assert p.threshold("ner", None, described, 10) == 0.6
    assert p.threshold("ner", None, named, 10) == 0.65
    assert p.threshold("relation", None, named, 10) == 0.4
    assert p.threshold("relation", None, named * 2, 10) == 0.45


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({"type": "span", "criteria": ["a"], "preset": "pii"}, "takes only"),
        ({"type": "span", "criteria": ["a"], "scope": "x"}, "do not take"),
        ({"type": "span", "criteria": ["a"], "head": "broad"}, "broad"),
        ({"type": "span", "criteria": ["a"], "labels": [{"key": "b"}]}, "not both"),
        ({"type": "span", "criteria": ["a"], "threshold": 1.5}, "threshold"),
        ({"type": "span", "criteria": ["a"], "threshold": True}, "threshold"),
        ({"type": "span", "criteria": ["a"], "over": "context"}, "over"),
        ({"type": "span", "criteria": ["a", "a"]}, "distinct"),
        ({"type": "span", "criteria": [f"l{i}" for i in range(256)]}, "1 to 255"),
        ({"type": "span"}, "criteria"),
        ({"type": "span", "criteria": ["a"], "instructions": 3}, "string"),
        ({"type": "span", "criteria": ["a"], "require_full_input": "yes"}, "boolean"),
    ],
)
def test_malformed_span_questions_are_refused(question, message) -> None:
    planner = spans.Spans(policy(), Renderer(None, []), SpanReader(None, {}))
    with pytest.raises(QuestionError, match=message):
        planner.plan("q", question, STATE, no_limit)


def test_a_span_question_needs_text_parts() -> None:
    with pytest.raises(QuestionError, match="text state"):
        spans.parts_of({"user": 3})
    assert spans.parts_of("text") == [("user", "text")]


# ------------------------------------------------------------------ rendering and windows


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Path:
    return fixture.write_fixture(
        tmp_path_factory.mktemp("spans") / "vela3-spans", "text", 13
    )


@pytest.fixture(scope="module")
def runtime(written):
    runtime = start_runtime(
        written, base_path=str(written.parent / f"{written.name}-base")
    )
    yield runtime
    runtime.stop()


@pytest.fixture(scope="module")
def model(runtime):
    return runtime.lookup(None).model


def test_a_window_is_the_d3_prompt_with_word_and_option_positions(model) -> None:
    renderer = model.spans.renderer
    parts = list(STATE.items())
    labels = [("dose", "an amount of a drug"), ("drug", "")]
    window = renderer.window(
        parts, "answer", "Which spans?", labels, STATE["answer"], 0
    )
    state = f"user:\n{STATE['user']}\n\nanswer:\n{STATE['answer']}"
    user = (
        f"State:\n{state}\n\nQuestion:\nWhich spans?\n\nOptions:\n"
        f"{model.codes[0]}: dose: an amount of a drug\n{model.codes[1]}: drug\n\n"
        "Reply with only the code of the best option."
    )
    encoding = model.tokenizer.encode(render(user), add_special_tokens=False)
    assert window.ids == tuple(encoding.ids)
    rendered = render(user)
    text = STATE["answer"]
    words = [text[a:b] for a, b in window.word_offsets]
    assert words[:4] == ["Adults", "can", "take", "up"]
    for (a, _b), token in zip(window.word_offsets, window.word_tokens, strict=True):
        start, end = encoding.offsets[token]
        assert rendered[start:end].strip() and text[a] in rendered[start:end]
    first = rendered[encoding.offsets[window.option_ranges[0][0]][0] :]
    assert first.lstrip().startswith(f"{model.codes[0]}:")


def test_long_targets_read_in_word_aligned_windows(model) -> None:
    renderer = model.spans.renderer
    text = " ".join(f"w{i}" for i in range(3000))
    windows, count = renderer.windows(
        [("user", text)], "user", "Which spans?", [("x", "")]
    )
    assert count > span_render.MAX_TARGET and len(windows) > 1
    covered = {offsets for window in windows for offsets in window.word_offsets}
    assert min(covered)[0] == 0 and max(covered)[1] == len(text)
    assert all(len(window.word_offsets) for window in windows)


def test_windows_start_a_stride_apart_on_word_starts() -> None:
    assert span_render.windows(np.arange(5000), 5000) == [
        (0, 1799),
        (1536, 3335),
        (3072, 4871),
        (4608, 5000),
    ]
    assert span_render.windows(np.arange(100), 100) == [(0, 100)]


def test_label_sets_take_their_trained_question_and_descriptions() -> None:
    schema = SPAN_CONFIG["schemas"]["pii"]
    routed = policy().resolve([("EMAIL", ""), ("PERSON", "")], None)
    assert (routed.label_set, routed.question) == ("pii", schema["question"])
    assert routed.labels == [
        ("EMAIL", schema["labels"]["EMAIL"]),
        ("PERSON", schema["labels"]["PERSON"]),
    ]
    open_labels = policy().resolve([("thing", "")], None)
    assert open_labels.question == SPAN_CONFIG["default_question"]


def test_open_labels_without_descriptions_leave_the_ner_head_unless_it_takes_names() -> (
    None
):
    config = {**SPAN_CONFIG, "ner": {**SPAN_CONFIG["ner"], "names_only_route": "span"}}
    strict = Policy(SpanConfig.parse(config), no_definitions)
    drug = [("drug", "")]
    assert strict.resolve(drug, None).route == "span"
    assert policy().resolve(drug, None).route == "ner"
    defined = Policy(
        SpanConfig.parse(config), lambda names, only: {"drug": "a medicine"}
    ).resolve(drug, None)
    assert (defined.route, defined.labels) == ("ner", [("drug", "a medicine")])


def test_window_scores_average_per_word() -> None:
    a = Window((), (0, 1), ((0, 1), (2, 3)), ())
    b = Window((), (0, 1), ((2, 3), (4, 5)), ())
    offsets, values = combine(
        [a, b], [np.array([[1.0], [2.0]]), np.array([[4.0], [6.0]])], 1
    )
    assert offsets.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert values[:, 0].tolist() == [1.0, 3.0, 6.0]


# ------------------------------------------------------------------ answers


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


def test_span_answers_follow_the_contract(runtime) -> None:
    question = {"type": "span", "criteria": {"dose": "an amount"}}
    status, body = call(
        runtime,
        {
            "state": STATE,
            "questions": {
                "all": {**question, "threshold": 0.0},
                "none": {**question, "threshold": 1.0},
            },
        },
    )
    assert status == 200
    assert body["thresholds"] == {"all": 0.0, "none": 1.0}
    assert body["spans"]["none"] == []
    (whole,) = body["spans"]["all"]
    text = STATE["answer"]
    assert (whole["start"], whole["end"], whole["text"]) == (0, len(text), text)
    assert 0 < whole["probability"] < 1
    assert body["answers"]["all"]["type"] == "noul"
    assert body["answers"]["all"]["noul"] == body["answers"]["none"]["noul"] > 0


def test_a_span_question_that_requires_its_full_input_reads_it_completely(
    runtime,
) -> None:
    question = {"type": "span", "criteria": {"dose": "an amount"}}
    status, body = call(
        runtime,
        {
            "state": STATE,
            "questions": {
                "full": {**question, "require_full_input": True},
                "plain": question,
            },
        },
    )
    assert status == 200
    assert body["answers"]["full"]["input_coverage"] == "complete"
    assert "input_coverage" not in body["answers"]["plain"]
    assert body["spans"]["full"] == body["spans"]["plain"]


def test_each_route_runs_its_adapter_under_the_noncausal_mask(
    model, monkeypatch
) -> None:
    seen: list[tuple[AdapterGate | None, bool]] = []
    forward = model.engine_model.forward

    def spy(batch):
        seen.append((batch.adapter, batch.noncausal))
        return forward(batch)

    monkeypatch.setattr(model.engine_model, "forward", spy)
    questions = {
        "pii": {"type": "span", "criteria": ["EMAIL", "PERSON"], "over": "user"},
        "ner": {"type": "span", "criteria": {"drug": "a medicine"}},
        "rel": {
            "type": "span",
            "criteria": ["dose"],
            "instructions": "Which spans are the dose of the drug?",
        },
    }
    plan = model.plan(STATE, questions)
    assert [item.route for item in plan.items] == ["span", "ner", "relation"]
    model.run(list(plan.items))
    assert [(gate.name, flag) for gate, flag in seen] == [
        ("span", True),
        ("ner", True),
        ("relation", True),
    ]


def test_the_head_reads_word_first_tokens_and_option_line_means(model) -> None:
    import torch
    from vllm_srun.plugins.base import ForwardBatch

    question = {"type": "span", "criteria": {"dose": "an amount", "drug": None}}
    (item,) = model.plan(STATE, {"s": question}).items
    (window,) = item.windows
    length = len(window.ids)
    output = model.engine_model.forward(
        ForwardBatch(
            input_ids=torch.tensor([window.ids]),
            attention_mask=torch.ones((1, length), dtype=torch.long),
            gather=torch.arange(length)[None],
            query=torch.tensor([length - 1]),
            lengths=[length],
            adapter=AdapterGate("ner" if item.route == "ner" else "span"),
            noncausal=True,
        )
    )
    hidden = output.gathered[0]
    words = hidden[list(window.word_tokens)]
    labels = torch.stack([hidden[a : b + 1].mean(0) for a, b in window.option_ranges])
    head = model.spans.reader.heads[item.route]
    with torch.inference_mode():
        want = head(words, labels).float().cpu().numpy()
    offsets, got = model.spans.reader.logits(item)
    assert offsets.tolist() == [list(o) for o in window.word_offsets]
    assert np.array_equal(got, want.astype(np.float64))


def test_span_questions_add_no_input_tokens(model) -> None:
    decisions = {"x": {"type": "noul", "instructions": "Is this hard?"}}
    span = {"s": {"type": "span", "criteria": ["thing"]}}
    alone = model.plan("Merge two lists.", decisions)
    assert (
        model.plan("Merge two lists.", {**decisions, **span}).input_tokens
        == alone.input_tokens
    )
    assert model.plan("Merge two lists.", span).input_tokens == 0


def test_decisions_beside_span_questions_answer_as_before(runtime) -> None:
    decisions = {
        "attack": {"type": "noul", "instructions": "Is this an attack?"},
        "x": {"type": "noul", "instructions": "Is this hard?"},
    }
    _, alone = call(runtime, {"state": "Merge two lists.", "questions": decisions})
    span = {"type": "span", "criteria": ["thing"]}
    _, mixed = call(
        runtime, {"state": "Merge two lists.", "questions": {**decisions, "s": span}}
    )
    assert {k: mixed["answers"][k] for k in decisions} == alone["answers"]
    assert list(mixed["answers"]) == ["attack", "x", "s"]
    assert mixed["usage"]["input_tokens"] == alone["usage"]["input_tokens"]


def test_presets_ask_the_trained_label_sets(runtime, model) -> None:
    schemas = SPAN_CONFIG["schemas"]
    questions = {
        "pii": {"preset": "pii", "over": "user", "require_full_input": True},
        "halu": {"preset": "halu", "type": "span", "threshold": 0.3},
    }
    status, body = call(runtime, {"state": STATE, "questions": questions})
    assert status == 200
    assert set(body["spans"]) == {"pii", "halu"}
    assert body["thresholds"]["halu"] == 0.3
    assert body["answers"]["pii"]["input_coverage"] == "complete"
    plan = model.plan(STATE, questions)
    by_id = {item.question_id: item for item in plan.items}
    assert by_id["pii"].names == tuple(schemas["pii"]["labels"])
    assert by_id["pii"].label_set == "pii"
    assert by_id["halu"].names == tuple(schemas["halu"]["labels"])
    assert runtime.lookup(None).card([])["presets"] == ["pii", "halu"]


@pytest.mark.parametrize(
    ("question", "reason", "message"),
    [
        ({"preset": "toxicity"}, "preset_unknown", "preset must be one of"),
        ({"preset": "pii", "criteria": ["a"]}, "question_malformed", "takes only"),
        ({"preset": "pii", "type": "choice"}, "question_malformed", "span question"),
        (
            {"type": "span", "criteria": ["a"], "overflow": "truncate"},
            "question_malformed",
            "window",
        ),
    ],
)
def test_malformed_presets_and_overflows_fail_alone(
    runtime, question, reason, message
) -> None:
    other = {"type": "noul", "instructions": "Is this hard?"}
    status, body = call(
        runtime, {"state": STATE, "questions": {"q": question, "x": other}}
    )
    assert status == 200
    answer = body["answers"]["q"]
    assert (answer["error"], answer["reason"]) == ("invalid_question", reason)
    assert message in answer["message"]


def test_span_questions_read_their_target_in_windows(runtime) -> None:
    question = {"type": "span", "criteria": ["dose"], "overflow": "window"}
    status, body = call(runtime, {"state": STATE, "questions": {"q": question}})
    assert status == 200 and "q" in body["spans"]


def test_span_windows_go_through_the_base_length_check(runtime) -> None:
    # Only the target part is windowed: the other parts go whole into every window, so a long
    # user part with a short target overflows the base's limit (4,096 tokens on the fixture).
    long_user = " ".join(["alpha"] * 1000)
    state = {"user": long_user, "answer": "Adults can take up to 4 grams."}
    questions = {
        "over_answer": {"type": "span", "criteria": ["dose"], "over": "answer"},
        "over_user": {"type": "span", "criteria": ["dose"], "over": "user"},
        "decision": {"type": "noul", "instructions": "Is this hard?"},
    }
    status, body = call(runtime, {"state": state, "questions": questions})
    assert status == 200
    refused = body["answers"]["over_answer"]
    assert (refused["error"], refused["reason"]) == (
        "max_length_exceeded",
        "question_tokens",
    )
    assert "over_answer" not in body.get("spans", {})
    assert body["answers"]["decision"]["error"] == "max_length_exceeded"
    assert "error" not in body["answers"]["over_user"]
    assert "over_user" in body["spans"]


@pytest.mark.parametrize("question_cap", [None, 262_144])
def test_a_span_target_over_the_span_budget_is_refused_before_it_runs(
    runtime, model, monkeypatch, question_cap
) -> None:
    # vela3-lite, -mini and -flash's bases name no validated length: their question cap is the
    # backbone's 262,144 positions, and the span budget must not follow it.
    if question_cap is not None:
        monkeypatch.setattr(model, "question_cap", lambda: (question_cap, False))
    forwards: list[object] = []
    forward = model.engine_model.forward
    monkeypatch.setattr(
        model.engine_model,
        "forward",
        lambda batch: forwards.append(batch) or forward(batch),
    )
    state = {"user": "What is the dose?", "answer": " ".join(["alpha"] * 6000)}
    questions = {"long": {"type": "span", "criteria": ["dose"], "over": "answer"}}
    status, body = call(runtime, {"state": state, "questions": questions})
    assert status == 200
    refused = body["answers"]["long"]
    assert (refused["error"], refused["reason"]) == (
        "scan_budget_exceeded",
        "scan_budget",
    )
    assert forwards == []


def test_a_request_of_span_questions_only_is_answered(runtime) -> None:
    status, body = call(
        runtime,
        {
            "state": "Call Tom at 555-0100.",
            "questions": {"s": {"type": "span", "criteria": ["PERSON"]}},
        },
    )
    assert status == 200
    assert set(body["spans"]) == {"s"}


def test_invalid_span_questions_fail_alone(runtime) -> None:
    status, body = call(
        runtime,
        {
            "state": "text",
            "questions": {
                "bad": {"type": "span", "criteria": ["a"], "head": "broad"},
                "ok": {"type": "span", "criteria": ["a"]},
            },
        },
    )
    assert status == 200
    assert body["answers"]["bad"]["error"] == "invalid_question"
    assert "bad" not in body.get("spans", {})
    assert "ok" in body["spans"]
