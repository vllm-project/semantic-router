"""The pico reader and one-pass assembler against the package's own ``render_call``, ``one_pass`` and ``encode``.

The cases were recorded from the released package by ``tools/pico_render_cases.py`` on its own
``router_questions.json`` and ``vela3_config.json``, with the byte tokenizer; every token ID, index and
rendered string must be the package's, byte for byte.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

import numpy as np
import pytest
from vllm_srun.families.vela2.encoder_layout import batch_indices
from vllm_srun.families.vela2.layout import Tokens, read_parts
from vllm_srun.families.vela2.request import Plan
from vllm_srun.families.vela3_pico import fallback, sequence
from vllm_srun.families.vela3_pico.request import DECISION, ROUTER, SPAN, Reader
from vllm_srun.families.vela3_pico.routing import Routes
from vllm_srun.families.vela3_shared import layout
from vllm_srun.testing.vela3_pico import BYTE_MARKERS, BYTE_SPECIAL, byte_tokenizer

CASES = json.loads(
    files("vllm_srun.testing").joinpath("data/render31_cases.json").read_text("utf-8")
)
TABLES = CASES["tables"]


def _tokens() -> Tokens:
    tokenizer = byte_tokenizer()
    return Tokens(
        encode=lambda value: tokenizer.encode(value, add_special_tokens=False),
        ids=lambda value: (
            tokenizer.encode(value, add_special_tokens=False).ids if value else []
        ),
        tokenizer=tokenizer,
    )


TOKENS = _tokens()
READER = Reader(
    Routes(
        {"span": TABLES["span"], "router": TABLES["router"]},
        TABLES["router_questions"],
    )
)


def api(questions: dict[str, Any]) -> dict[str, Any]:
    """The request as System One spells it: options or labels named in a list become null descriptions."""
    out = {}
    for qid, asked in questions.items():
        question = asked
        criteria = question.get("criteria")
        if question.get("type") in ("choice", "set") and isinstance(criteria, list):
            question = {
                **question,
                "criteria": dict.fromkeys(criteria),
            }
        labels = question.get("labels")
        if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
            question = {
                **{k: v for k, v in question.items() if k != "labels"},
                "criteria": dict.fromkeys(labels),
            }
        out[qid] = question
    return out


def ids_of(case: dict[str, Any]) -> list[int]:
    return list(range(len(case)))


@pytest.mark.parametrize("case", CASES["render_call"], ids=ids_of(CASES["render_call"]))
def test_questions_render_as_the_package_renders_a_call(case: dict[str, Any]) -> None:
    request = READER.read(case["state"], api(case["questions"]))
    result = case.get("result")
    assert request.one_pass == (result is not None)
    if result is None:
        # A Score question, or a Set question that is not a router question:
        # the package answers question by question, where they are invalid.
        assert set(request.errors) == {"level"} or set(request.errors) == {"tags"}
        return
    assert not request.errors
    parts, rows, meta = result
    assert request.state is not None
    assert [[p.role, p.text] for p in request.state.parts] == parts
    roles = [p.role for p in request.state.parts]
    by_id = {a.question.id: a for a in request.asked}
    for row in rows:
        asked = by_id[row["id"]]
        question = asked.question
        assert question.text == row["text"]
        if row["type"] == "span":
            assert asked.path == SPAN
            assert list(question.options) == list(row["labels"].items())
            assert question.over == row["over"]
            assert asked.route is not None and asked.route.label_set == meta[row["id"]]
            continue
        assert [tuple(x) for x in row["options"].items()] == list(question.options)
        assert question.abstain == row["abstain"]
        if asked.path == DECISION:
            assert row["type"] == "choice"
            assert row["over"] == (roles[0] if len(roles) == 1 else roles)
        else:
            assert asked.path == ROUTER and question.type == row["type"]
            assert question.over == row["over"]


def build(case: dict[str, Any]) -> sequence.OnePass | None:
    request = READER.read(case["state"], api(case["questions"]))
    assert request.state is not None
    parts = read_parts(
        Plan(request.state, request.question_ids, [a.question for a in request.asked]),
        TOKENS,
    )
    pass_layout = sequence.Layout(
        dict(BYTE_MARKERS),
        BYTE_SPECIAL["bos_token_id"],
        BYTE_SPECIAL["eos_token_id"],
        case["max_len"],
    )
    return sequence.build(request.asked, parts, TOKENS, pass_layout)


@pytest.mark.parametrize("case", CASES["one_pass"], ids=ids_of(CASES["one_pass"]))
def test_one_pass_is_the_package_sequence_byte_for_byte(case: dict[str, Any]) -> None:
    one = build(case)
    recorded = case["result"]
    if recorded is None:
        assert one is None
        return
    assert one is not None
    seq = one.sequence
    assert seq.ids == recorded["ids"]
    assert len(seq.ids) == recorded["input_tokens"]
    indices = batch_indices([seq])
    for name in ("q_index", "opt_index", "unit_index", "ent_index"):
        assert indices[name].tolist() == recorded[name], name
    assert [
        (q.question.id, q.question.names, len(q.options)) for q in seq.questions
    ] == [(q["qid"], q["names"], q["logits"]) for q in recorded["questions"]]
    assert [g.asked.question.id for g in one.groups] == list(recorded["spans"])
    for group in one.groups:
        span = recorded["spans"][group.asked.question.id]
        assert group.asked.question.names == span["names"]
        assert np.asarray(group.offsets).tolist() == span["offsets"]
        assert group.words[1] - group.words[0] == span["units"]
        assert group.labels[1] - group.labels[0] == span["labels"]
        assert group.target_tokens == span["target_tokens"]


def test_the_recorded_calls_cover_the_fit_boundary() -> None:
    budgets = [(c["max_len"], c["result"] is None) for c in CASES["one_pass"]]
    assert sum(missing for _, missing in budgets) == 7
    assert sum(not missing for _, missing in budgets) == 14


@pytest.mark.parametrize("case", CASES["decisions"], ids=ids_of(CASES["decisions"]))
def test_decisions_are_the_package_decision_inputs(case: dict[str, Any]) -> None:
    config = {"decision": {"marker_ids": BYTE_MARKERS}, **BYTE_SPECIAL}
    layouts = fallback.Layouts(
        markers=dict(BYTE_MARKERS),
        bos=BYTE_SPECIAL["bos_token_id"],
        eos=BYTE_SPECIAL["eos_token_id"],
        max_len=case["max_len"],
        window=layout(config, 8192, 512),
        evidence=layout(config, 32768, 512),
    )
    request = READER.read(case["state"], api({"q": case["question"]}))
    assert not request.errors and request.asked[0].path == DECISION
    state = TOKENS.ids(
        case["state"]
        if isinstance(case["state"], str)
        else json.dumps(case["state"], ensure_ascii=False, separators=(",", ":"))
    )
    got = fallback._decision(request.asked[0], state, TOKENS, layouts)
    if "error" in case:
        assert case["error"].startswith("Unsupported")
        assert got is None
        return
    assert got is not None
    recorded = case["result"]
    assert got.ids == recorded["ids"]
    (placed,) = got.questions
    assert placed.query == recorded["q_pos"]
    assert placed.options == recorded["opt_pos"]
    assert list(placed.pool) == recorded["pool"]
    assert placed.question.names == recorded["names"]
