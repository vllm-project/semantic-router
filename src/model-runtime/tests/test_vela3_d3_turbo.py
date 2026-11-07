"""vela3-turbo: d3-turbo's state-once answers, then the router rows again with their adapter on (noncausal prompts)."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
import torch
from vllm_srun.config import EngineArgs
from vllm_srun.errors import PackageError
from vllm_srun.families.decision3.family import BATCH_SIZE
from vllm_srun.families.vela3_d3.family import Vela3D3Family
from vllm_srun.families.vela3_d3.gates import Gates
from vllm_srun.families.vela3_d3.model import adapter_of
from vllm_srun.plugins.base import (
    AdapterGate,
    ForwardBatch,
    Job,
    PackageRef,
    RegistryOptions,
)
from vllm_srun.scheduler.batching import Batcher, SharedContext
from vllm_srun.testing import vela3_d3 as fixture

from .conftest import start_runtime

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NOUL = {"type": "noul", "instructions": "Does this request need care?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which domain does the request belong to?",
    "criteria": {"code": "Programming", "math": "Mathematics", "other": None},
}
STATE = "Merge two sorted lists."


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Path:
    return fixture.write_fixture(
        tmp_path_factory.mktemp("turbo") / "vela3-turbo", "turbo", 9
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


def test_ordinary_questions_answer_as_d3_turbo_bit_for_bit(runtimes) -> None:
    vela, d3 = runtimes
    body = {"state": STATE, "questions": {"reasoning": NOUL, "topic": CHOICE}}
    _, got = call(vela, body)
    _, want = call(d3, body)
    assert got["answers"] == want["answers"]
    assert got["usage"] == want["usage"]


def replay_reference(model, items, name):
    ids, mask, last = model._padded(items)
    output = model.engine_model.forward(
        ForwardBatch(
            input_ids=ids,
            attention_mask=mask,
            gather=last[:, None],
            query=last,
            lengths=[len(item.ids) for item in items],
            adapter=AdapterGate(name),
            noncausal=True,
        )
    )
    return model._probabilities(items, output.query)


def test_router_rows_are_answered_again_on_noncausal_prompts(runtimes) -> None:
    vela, d3 = runtimes
    questions = {"reasoning": NOUL, "attack": NOUL, "domain": CHOICE, "intent": CHOICE}
    body = {"state": STATE, "questions": questions}
    _, got = call(vela, body)
    _, want = call(d3, body)
    assert got["answers"]["reasoning"] == want["answers"]["reasoning"]
    model = vela.lookup(None).model
    plan = model.plan(STATE, questions)
    replays = [item for item in plan.items if adapter_of(item) is not None]
    assert [(item.question_id, item.adapter) for item in replays] == [
        ("attack", "decision"),
        ("domain", "decision_df"),
        ("intent", "decision"),
    ]
    assert all(item.state is None for item in replays)
    base = {item.question_id: item for item in plan.items if adapter_of(item) is None}
    assert all(item.ids == base[item.question_id].ids for item in replays)
    values = {}
    for name in ("decision", "decision_df"):
        rows = [item for item in replays if item.adapter == name]
        for item, value in zip(rows, replay_reference(model, rows, name), strict=True):
            values[item.question_id] = value
    for key in ("attack", "intent", "domain"):
        item = next(item for item in replays if item.question_id == key)
        assert got["answers"][key] == model.answer(item, values[key])
        assert got["answers"][key] != want["answers"][key]
    extra = sum(len(item.ids) for item in replays)
    assert got["usage"]["input_tokens"] == want["usage"]["input_tokens"] + extra


def test_replays_run_after_the_state_once_pass_in_chunks_of_the_batch_size(
    runtimes,
) -> None:
    model = runtimes[0].lookup(None).model
    questions = {"plain": NOUL} | {f"hazard:{i}": NOUL for i in range(BATCH_SIZE + 2)}
    items = list(model.plan(STATE, questions).items)
    batches = model.exact_batches(items)
    assert batches[0] == list(range(BATCH_SIZE + 3))
    assert batches[1:] == [
        list(range(BATCH_SIZE + 3, 2 * BATCH_SIZE + 3)),
        list(range(2 * BATCH_SIZE + 3, 2 * BATCH_SIZE + 5)),
    ]
    merged = model.run_approximate(items)
    exact = [v for batch in batches for v in model.run([items[i] for i in batch])]
    order = [i for batch in batches for i in batch]
    for index, value in zip(order, exact, strict=True):
        assert torch.allclose(
            torch.tensor(merged[index]), torch.tensor(value), atol=1e-5
        )


def test_span_questions_and_replays_leave_the_state_once_pass_whole(runtimes) -> None:
    vela, _ = runtimes
    model = vela.lookup(None).model
    span = {"type": "span", "criteria": ["thing"]}
    questions = {"reasoning": NOUL, "topic": CHOICE, "attack": NOUL, "s": span}
    items = list(model.plan(STATE, questions).items)
    state = next(item.state for item in items if item.state is not None)
    assert model.shared_context(items, None) == len(state.ids)
    own = [item for item in items if item.state is None]
    assert own and model.shared_context(own, None) == 0
    batcher = Batcher(EngineArgs(), SharedContext())
    batcher.bind(model)
    [batch] = batcher.plan([Job(items=items, deadline=None, enqueued=0.0)], None)
    assert batch.shared_prefix == len(state.ids)
    _, together = call(vela, {"state": STATE, "questions": questions})
    rest = {key: question for key, question in questions.items() if key != "s"}
    _, alone = call(vela, {"state": STATE, "questions": rest})
    assert {key: together["answers"][key] for key in rest} == alone["answers"]
    assert "s" in together["spans"]


def test_images_box_and_track_requests_take_the_base_alone(runtimes, png_url) -> None:
    vela, d3 = runtimes
    body = {
        "state": STATE,
        "questions": {"attack": NOUL, "x": NOUL},
        "images": [png_url(64, 64)],
    }
    _, got = call(vela, body)
    _, want = call(d3, body)
    assert got["answers"] == want["answers"]
    model = vela.lookup(None).model
    box = {"type": "box", "instructions": "Where is the cat?"}
    plan = model.plan(STATE, {"attack": NOUL, "where": box})
    assert all(adapter_of(item) is None for item in plan.items)


def test_a_noncausal_forward_reads_the_whole_prompt_on_a_state_once_backbone(
    runtimes,
) -> None:
    from vllm_srun.engines.native.models.qwen3_5 import noncausal_masks

    engine_model = runtimes[0].lookup(None).model.engine_model
    ids = torch.tensor([[5, 9, 12, 7, 3, 4], [0, 0, 8, 11, 6, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    last = torch.tensor([5, 5])

    def forward(noncausal: bool) -> torch.Tensor:
        return engine_model.forward(
            ForwardBatch(
                input_ids=ids,
                attention_mask=mask,
                gather=last[:, None],
                query=last,
                lengths=[6, 4],
                noncausal=noncausal,
            )
        ).query

    full, linear = noncausal_masks(mask, True)
    with torch.inference_mode():
        want = engine_model.backbone(ids, mask, masks={"full": full, "linear": linear})
    assert torch.equal(forward(True), want[:, -1])
    assert not torch.equal(forward(False), want[:, -1])


def test_span_questions_run_beside_turbo_decisions_but_refuse_images(
    runtimes, png_url
) -> None:
    vela, d3 = runtimes
    span = {"type": "span", "criteria": ["thing"]}
    body = {"state": STATE, "questions": {"x": NOUL, "s": span}}
    _, got = call(vela, body)
    _, want = call(d3, {"state": STATE, "questions": {"x": NOUL}})
    assert got["answers"]["x"] == want["answers"]["x"]
    assert "s" in got["spans"]
    _, imaged = call(vela, {**body, "images": [png_url(64, 64)]})
    assert imaged["answers"]["s"]["error"] == "invalid_question"
    assert "x" in imaged["answers"] and "error" not in imaged["answers"]["x"]


def test_a_blank_state_is_refused_as_the_other_vela_families_refuse_it(
    runtimes,
) -> None:
    span = {"type": "span", "criteria": ["thing"]}
    for state, questions in ((" ", {"x": NOUL}), ("\n\t ", {"s": span})):
        status, body = call(runtimes[0], {"state": state, "questions": questions})
        assert (status, body["error"]["code"]) == (400, "invalid_request")


BOX = {"type": "box", "instructions": "Where is the cat?"}


def test_box_requests_plan_spans_and_strip_the_opt_in_key(runtimes) -> None:
    # A Box question sends the request through plan_resolved: the base alone for the decisions
    # (adapters off, the opt-in key stripped), span questions as on any other request.
    vela, d3 = runtimes
    span = {"type": "span", "criteria": ["thing"]}
    care = {**NOUL, "decision_adapter": True}
    questions = {"attack": NOUL, "care": care, "where": BOX, "s": span}
    status, got = call(vela, {"state": STATE, "questions": questions})
    assert status == 200
    base_questions = Gates.strip({"attack": NOUL, "care": care, "where": BOX})
    _, want = call(d3, {"state": STATE, "questions": base_questions})
    for key in ("attack", "care", "where"):
        assert got["answers"][key] == want["answers"][key]
    assert "error" not in got["answers"]["care"]
    assert "s" in got["spans"] and "error" not in got["answers"]["s"]
    assert list(got["answers"]) == list(questions)


def test_a_blank_state_is_refused_on_box_requests_too(runtimes) -> None:
    status, body = call(
        runtimes[0], {"state": "  ", "questions": {"x": NOUL, "where": BOX}}
    )
    assert (status, body["error"]["code"]) == (400, "invalid_request")


def test_the_opt_in_key_is_stripped_for_the_base(runtimes) -> None:
    vela, d3 = runtimes
    questions = {"care": {**NOUL, "decision_adapter": True}}
    _, got = call(vela, {"state": STATE, "questions": questions})
    _, want = call(d3, {"state": STATE, "questions": Gates.strip(questions)})
    assert got["answers"] == want["answers"]


def test_only_the_d3_sizes_take_the_opt_in_key(runtimes) -> None:
    vela, d3 = runtimes
    questions = {"care": {**NOUL, "decision_adapter": True}}
    _, got = call(vela, {"state": STATE, "questions": questions})
    status, refused = call(d3, {"state": STATE, "questions": questions})
    assert "error" not in got["answers"]["care"]
    assert status == 400
    assert "do not take ['decision_adapter']" in refused["error"]["message"]


def test_a_state_once_base_needs_noncausal_replays(written, tmp_path) -> None:
    root = tmp_path / "vela3-turbo"
    shutil.copytree(written, root)
    path = root / "vela3_config.json"
    config = json.loads(path.read_text())
    config["decisions"] = {**(config.get("decisions") or {}), "attention": "causal"}
    path.write_text(json.dumps(config))
    fixture.write_sums(root)
    base = written.parent / f"{written.name}-base"
    family = Vela3D3Family(RegistryOptions(base_path=base))
    with pytest.raises(PackageError, match=r"decisions\.attention 'noncausal'"):
        family.verify(PackageRef(root))
