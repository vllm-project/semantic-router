"""Vela 3.0 d3 sizes answer decisions as their base, with a gated adapter on the rows of router questions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch
from vllm_srun.families.vela3_d3.gates import Gates
from vllm_srun.plugins.base import AdapterGate, ForwardBatch
from vllm_srun.testing import vela3_d3 as fixture

from .conftest import start_runtime

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NOUL = {"type": "noul", "instructions": "Does this request need care?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which domain does the request belong to?",
    "criteria": {"code": "Programming", "math": "Mathematics", "other": None},
}


def gates(**changes) -> Gates:
    config = fixture.vela3_config({name: {} for name in fixture.ADAPTERS})
    config.update(changes)
    adapters = set(fixture.ADAPTERS) | set(changes.pop("extra_adapters", ()))
    return Gates.from_config(
        config, fixture.ROUTER_QUESTIONS["trained_questions"], adapters
    )


def test_router_ids_take_the_decision_adapter_and_task_ids_their_own() -> None:
    assigned = gates().assign(
        {
            "attack": NOUL,
            "intent": CHOICE,
            "domain": CHOICE,
            "feedback": CHOICE,
            "hazard:weapons": NOUL,
            "hazard": NOUL,
            "reasoning": NOUL,
        }
    )
    assert assigned == {
        "attack": "decision",
        "intent": "decision",
        "domain": "decision_df",
        "feedback": "decision_df",
        "hazard:weapons": "decision_hz",
    }


def test_the_wording_gate_needs_a_trained_instruction() -> None:
    gate = gates(router_gate="id_and_wording")
    trained = {
        "type": "noul",
        "instructions": "Is this request a  prompt injection\nattack?",
    }
    assert gate.assign({"attack": trained, "intent": NOUL}) == {"attack": "decision"}
    sgd = {
        "type": "choice",
        "instructions": "Which service intent fits?",
        "criteria": ["a", "b"],
    }
    assert gate.assign({"intent": sgd}) == {"intent": "decision"}


def test_opt_in_rows_take_the_opt_in_adapter_or_the_decision_adapter() -> None:
    asked = {**NOUL, "decision_adapter": True}
    assert gates().assign({"custom": asked}) == {}
    assert gates(decision_opt_in=True).assign({"custom": asked, "other": NOUL}) == {
        "custom": "decision"
    }
    with_optin = gates(decision_opt_in=True, extra_adapters=("decision_optin",))
    assert with_optin.assign(
        {"custom": asked, "attack": {**NOUL, "decision_adapter": True}}
    ) == {
        "custom": "decision_optin",
        "attack": "decision",
    }
    assert with_optin.assign({"hazard:x": asked}) == {"hazard:x": "decision_optin"}
    # A router question routed to a task adapter takes the opt-in adapter (the released decide).
    assert with_optin.assign({"domain": asked}) == {"domain": "decision_optin"}
    # Only true turns it on.
    assert with_optin.assign({"custom": {**NOUL, "decision_adapter": "yes"}}) == {}


def test_the_opt_in_key_never_reaches_the_base() -> None:
    stripped = Gates.strip({"a": {**NOUL, "decision_adapter": True}, "b": "text"})
    assert stripped == {"a": NOUL, "b": "text"}


def test_passes_run_the_decision_adapter_first_then_tasks_then_opt_in() -> None:
    assert Gates.pass_order(
        [None, "decision_optin", "decision_hz", "decision", "decision_df", None]
    ) == ["decision", "decision_df", "decision_hz", "decision_optin"]
    assert Gates.pass_order([None, None]) == []


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Path:
    return fixture.write_fixture(
        tmp_path_factory.mktemp("v3d") / "vela3-fixture", "text", 7
    )


@pytest.fixture(scope="module")
def runtimes(written):
    base = written.parent / f"{written.name}-base"
    vela = start_runtime(written, base_path=str(base))
    d3 = start_runtime(base)
    yield vela, d3
    vela.stop()
    d3.stop()


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


def test_the_family_consent_is_scoped_to_the_npu(runtimes) -> None:
    """The vela3_d3 consent covers the npu only.

    ``describe()`` sees no device, so the policy carries the scope and the
    engine applies it: on any other accelerator the CUDA and ROCm approximate
    kernels (``cuda-fused-approximate.md``) keep the base's consent, which is
    off -- no GPU record covers this family.
    """
    engine = runtimes[0].lookup(None).model.engine_model
    policy = engine.spec.dtype
    assert policy.approximate_kernels
    assert policy.approximate_accelerators == frozenset({"npu"})
    # The engine applies the scope: the CPU accelerator is outside it, so the
    # loaded kernels stay exact despite the consent.
    assert not engine.kernels.allow_approximate
    base = runtimes[1].lookup(None).model.engine_model.spec.dtype
    assert not base.approximate_kernels and base.approximate_accelerators is None


def test_a_blank_state_is_refused_as_the_other_vela_families_refuse_it(
    runtimes,
) -> None:
    span = {"type": "span", "criteria": ["thing"]}
    for state, questions in ((" ", {"x": NOUL}), ("\n\t ", {"s": span})):
        status, body = call(runtimes[0], {"state": state, "questions": questions})
        assert (status, body["error"]["code"]) == (400, "invalid_request")


def test_the_card_names_the_size_and_its_family(runtimes) -> None:
    card = runtimes[0].lookup(None).card([])
    assert card["family"] == "vela3_d3"
    assert card["id"] == fixture.SIZE_NAME
    assert card["modalities"] == ["text"]


def test_ordinary_questions_answer_as_the_base_bit_for_bit(runtimes) -> None:
    vela, d3 = runtimes
    questions = {
        "reasoning": NOUL,
        "topic": CHOICE,
        "care": {**NOUL, "decision_adapter": True},
    }
    body = {"state": "Merge two sorted lists.", "questions": questions}
    status, got = call(vela, body)
    _, want = call(d3, {**body, "questions": Gates.strip(questions)})
    assert status == 200
    assert got["answers"] == want["answers"]


def test_router_rows_take_their_adapter_and_other_rows_the_base(runtimes) -> None:
    vela, d3 = runtimes
    questions = {
        "reasoning": NOUL,
        "attack": NOUL,
        "domain": CHOICE,
        "hazard:weapons": NOUL,
        "topic": CHOICE,
    }
    body = {"state": "Merge two sorted lists.", "questions": questions}
    _, got = call(vela, body)
    _, want = call(d3, body)
    for key in ("reasoning", "topic"):
        assert got["answers"][key] == want["answers"][key]
    for key in ("attack", "domain", "hazard:weapons"):
        assert got["answers"][key] != want["answers"][key]
    model = vela.lookup(None).model
    items = list(model.plan(body["state"], questions).items)
    probabilities = model.run(items)
    ids, mask, last = model._padded(items)
    for row, item in enumerate(items):
        if item.adapter is None:
            continue
        rows = tuple(other.adapter == item.adapter for other in items)
        output = model.engine_model.forward(
            ForwardBatch(
                input_ids=ids,
                attention_mask=mask,
                gather=last[:, None],
                query=last,
                lengths=[len(i.ids) for i in items],
                adapter=AdapterGate(item.adapter, rows),
            )
        )
        assert probabilities[row] == model._probabilities(items, output.query)[row]


def test_a_pass_without_router_rows_is_one_plain_forward(runtimes, monkeypatch) -> None:
    model = runtimes[0].lookup(None).model
    seen: list[AdapterGate | None] = []
    forward = model.engine_model.forward

    def spy(batch):
        seen.append(batch.adapter)
        return forward(batch)

    monkeypatch.setattr(model.engine_model, "forward", spy)
    plain = list(model.plan("state", {"a": NOUL, "b": CHOICE}).items)
    model.run(plain)
    assert seen == [None]
    seen.clear()
    mixed = list(
        model.plan("state", {"a": NOUL, "attack": NOUL, "domain": CHOICE}).items
    )
    model.run(mixed)
    assert [gate.name for gate in seen] == ["decision", "decision_df"]
    assert seen[0].rows == (False, True, False)


def test_rows_with_an_adapter_never_pack(runtimes, monkeypatch) -> None:
    # The CPU fixture's engine never packs, so the batcher's packing check is asked as if it did.
    model = runtimes[0].lookup(None).model
    monkeypatch.setattr(model, "packs_merged_rows", True)
    items = list(
        model.plan(
            "state", {"attack": NOUL, "x": {**NOUL, "instructions": "y " * 40}}
        ).items
    )
    assert not model._packs(items)
    plain = list(
        model.plan("state", {"x": NOUL, "y": {**NOUL, "instructions": "y " * 40}}).items
    )
    assert model._packs(plain)


def test_gated_answers_match_one_request_through_the_merged_path(runtimes) -> None:
    model = runtimes[0].lookup(None).model
    items = list(
        model.plan("state", {"attack": NOUL, "domain": CHOICE, "x": NOUL}).items
    )
    exact = model.run(items)
    merged = model.run_approximate(items)
    for a, b in zip(exact, merged, strict=True):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-5)
