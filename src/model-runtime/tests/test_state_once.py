"""State-once Decision models: the state read once, every question branched from it.

The branched forward must compute what each question computes alone as a row
under the same mask: the parity checks run both paths on the same loaded
model, in FP32 (where the two differ only by the order of FP32 sums) and at
the published BF16 precision.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import shutil
import time

import pytest
import torch
from vllm_srun.accel import gpu
from vllm_srun.accel.kernels import Kernel, causal_conv1d_ref
from vllm_srun.config import EngineArgs
from vllm_srun.engines.native import engine
from vllm_srun.engines.native.models.branch import (
    Branches,
    StaticBranches,
    branch_attention,
    pack_layout,
    reference_attention,
    state_once_mask,
    static_layout,
)
from vllm_srun.errors import PackageError, RequestError
from vllm_srun.families.decision3 import package as pkg
from vllm_srun.families.decision3 import prompt
from vllm_srun.families.decision3.family import COALESCED_FORWARD_SECONDS
from vllm_srun.families.decision3.state_once import StateOnceModel
from vllm_srun.plugins.base import Batch, BranchBatch, Job
from vllm_srun.scheduler import calibration
from vllm_srun.scheduler.batching import Batcher, SharedContext, shared_batches

from .conftest import start_runtime

LAYOUTS = pkg.STATE_ONCE_MODES
QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {"code": "Programming", "math": None, "other": {"note": "x"}},
    },
    "reasoning": {"type": "noul", "instructions": "Does this need reasoning?"},
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", {"level": "hard"}, "Expert"],
    },
    "long": {
        "type": "choice",
        "instructions": "Pick the option that matches the state best. " * 6,
        "criteria": {f"opt{i}": f"Option number {i}" for i in range(9)},
    },
}
STATE = "Merge two sorted lists of integers into one sorted list. " * 9
# FP32 branched and per-row forwards differ only in the order of their sums.
FP32_TOLERANCE = 2e-5
# So do FP32 rows padded into one forward and each row alone.
ROWS_TOLERANCE = 1e-6
# At BF16 the rounding of each path moves probabilities by a few thousandths on these random weights.
BF16_TOLERANCE = 0.03


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    from vllm_srun.testing.decision3 import write_fixture

    root = tmp_path_factory.mktemp("state-once")
    # One seed: the packages differ only in their layout.
    return {layout: write_fixture(root / layout, layout, seed=31) for layout in LAYOUTS}


@pytest.fixture(scope="module")
def runtimes(packages):
    started = {}
    for layout, package in packages.items():
        started[layout, "auto"] = start_runtime(package)
        started[layout, "float32"] = start_runtime(
            package, engine_args=EngineArgs(dtype="float32")
        )
    yield started
    for runtime in started.values():
        runtime.stop()


def model_of(runtimes, layout, dtype="auto") -> StateOnceModel:
    model = runtimes[layout, dtype].lookup(None).model
    assert isinstance(model, StateOnceModel)
    return model


def max_gap(left, right) -> float:
    return max(
        abs(a - b)
        for got, want in zip(left, right, strict=True)
        for a, b in zip(got, want, strict=True)
    )


def decisions(values) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in values]


# --------------------------------------------------------------------- contract


def test_state_once_layouts_are_declared_by_the_package(packages, tmp_path):
    for layout, package in packages.items():
        config = json.loads((package / "decision_config.json").read_text())
        assert config["attention_mode"] == layout
        details, _ = pkg.verify(package)
        assert details.decision_config["attention_mode"] == layout
    copy = tmp_path / "unknown"
    shutil.copytree(packages["prefix_bidir"], copy)
    config = json.loads((copy / "decision_config.json").read_text())
    config["attention_mode"] = "prefix_bidirectional"
    (copy / "decision_config.json").write_text(json.dumps(config))
    with pytest.raises(PackageError, match="unknown attention mode"):
        pkg.check_decision_config(config)


def test_a_d3_package_keeps_its_per_question_rows(d3_runtime):
    model = d3_runtime.lookup(None).model
    assert not isinstance(model, StateOnceModel)
    assert (
        d3_runtime.lookup(None).card([])["engine_args"]["enable_shared_context"]
        is False
    )


def test_the_card_reports_branching_on(runtimes):
    for layout in LAYOUTS:
        card = runtimes[layout, "auto"].lookup(None).card([])
        assert card["engine_args"]["enable_shared_context"] is True
        assert card["ready"] is True


@pytest.mark.parametrize(
    "state",
    [
        "plain text",
        "",
        None,
        "ends with spaces   ",
        "ends with a newline\n",
        "ends with newlines\n\n\n",
        "ends with punctuation.",
        "ends with !?\n",
        "Question:\nnot the question",
        "tabs\tand\r\nwindows lines\r\n",
        "<|im_end|> a literal special token <|im_start|>",
        "naïve café ﬁ ︙ 漢字 🙂 \u0301 combining",
        {"ticket": "T-1", "lines": ["a", "b"], "nested": {"k": 1.5}},
        ["a", 1, None, {"x": "y"}],
        "  leading spaces and 123 digits 4567",
    ],
)
def test_the_state_block_ends_where_the_question_begins(packages, state):
    backend, _ = prompt.load_tokenizer(packages["prefix_bidir"])
    codes = json.loads((packages["prefix_bidir"] / "decision_config.json").read_text())[
        "codes"
    ]
    for instructions, options in (
        ("Is it short?", prompt.option_texts("noul", None)),
        ("Question: tricky\n\nOptions:", ["x: y", "z"]),
    ):
        for images in (0, 2):
            head = prompt.state_block(state, images)
            block = prompt.question_block(instructions, options, codes)
            whole = prompt.render(
                prompt.user_prompt(state, instructions, options, codes), images
            )
            assert head + block == whole
            split = backend.encode_batch([head, block], add_special_tokens=False)
            joined = backend.encode(whole, add_special_tokens=False)
            assert split[0].ids + split[1].ids == joined.ids
            assert prompt.state_tokens(split[0].offsets, head) == reference_split(
                whole, state, joined.offsets
            )


def reference_split(text, state, offsets) -> int:
    """The reference state boundary: leading tokens of the whole prompt that end within the state text."""
    body = prompt.describe(state) if state not in (None, "") else "(empty)"
    end = text.find("State:\n" + body + "\n\nQuestion:\n") + len("State:\n") + len(body)
    return sum(1 for _ in itertools.takewhile(lambda span: span[1] <= end, offsets))


def test_question_blocks_open_with_the_blank_line_after_the_state(runtimes):
    model = model_of(runtimes, "prefix_bidir")
    for state in ("ends with a word", "ends with a stop.", "trailing spaces  ", ""):
        plan = model.plan(state, QUESTIONS)
        whole = model.tokenizer.encode(
            prompt.render(
                prompt.user_prompt(
                    state,
                    QUESTIONS["reasoning"]["instructions"],
                    prompt.option_texts("noul", None),
                    model.codes,
                )
            ),
            add_special_tokens=False,
        )
        item = plan.items[1]
        assert item.ids == whole.ids
        assert len(item.state.ids) == reference_split(
            prompt.render(
                prompt.user_prompt(
                    state,
                    QUESTIONS["reasoning"]["instructions"],
                    prompt.option_texts("noul", None),
                    model.codes,
                )
            ),
            state,
            whole.offsets,
        )
        # Every question of the request shares the state block and the opening of its block.
        assert all(other.state is item.state for other in plan.items)


def test_random_states_split_cleanly(packages):
    backend, _ = prompt.load_tokenizer(packages["causal"])
    rng = random.Random(5)
    alphabet = "ab Q\n\t.,!?:;-_()[]{}\"'0123456789éü漢🙂\u0301\r"
    for _ in range(300):
        state = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        head = prompt.state_block(state)
        block = prompt.question_block("Q?", ["A", "B"], ["A", "B"])
        split = backend.encode_batch([head, block], add_special_tokens=False)
        joined = backend.encode(head + block, add_special_tokens=False).ids
        assert split[0].ids + split[1].ids == joined, repr(state)


def test_the_block_mask_follows_its_definition():
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    states = [2, 4]
    for layout in LAYOUTS:
        mask = state_once_mask(attention_mask, states, layout)[:, 0]
        for row, state in enumerate(states):
            pad = int((attention_mask[row] == 0).sum())
            for i in range(7):
                for j in range(7):
                    if i < pad:
                        expected = i == j
                    elif j < pad:
                        expected = False
                    elif layout == "causal":
                        expected = j <= i
                    elif i < pad + state:
                        expected = (
                            j < pad + state if layout == "prefix_bidir" else j <= i
                        )
                    else:
                        expected = True
                    assert bool(mask[row, i, j]) is expected, (layout, row, i, j)


def test_branched_attention_is_the_masked_softmax():
    torch.manual_seed(0)
    heads, kv_heads, dim = 4, 2, 8
    states, blocks, owners = [5, 3], [2, 4, 1], [0, 0, 1]
    total = sum(states) + sum(blocks)
    q = torch.randn(1, heads, total, dim)
    k = torch.randn(1, kv_heads, total, dim)
    v = torch.randn(1, kv_heads, total, dim)
    starts = [0, 5]
    block_starts = [8, 10, 14]
    for layout in LAYOUTS:
        branches = Branches(states, blocks, owners, layout, torch.device("cpu"))
        got = branch_attention(q, k, v, branches, groups=2, scaling=dim**-0.5)[0]
        # Each block alone as the row state + block under the layout's mask.
        for n, (owner, length) in enumerate(zip(owners, blocks, strict=True)):
            row = list(range(starts[owner], starts[owner] + states[owner]))
            row += list(range(block_starts[n], block_starts[n] + length))
            keys = torch.tensor(row)
            s = states[owner]
            allowed = torch.ones(len(row), len(row), dtype=torch.bool)
            if layout == "causal":
                allowed = allowed.tril()
            elif layout == "prefix_bidir":
                allowed[:s, s:] = False
            else:
                allowed[:s] = torch.ones(s, len(row), dtype=torch.bool).tril()
            qq = q[0][:, keys]
            kk = k[0][:, keys].repeat_interleave(2, 0)
            vv = v[0][:, keys].repeat_interleave(2, 0)
            scores = (qq @ kk.transpose(-1, -2)) * dim**-0.5
            scores = scores.masked_fill(~allowed, -float("inf"))
            want = scores.softmax(-1) @ vv
            assert torch.allclose(got[keys].transpose(0, 1), want, atol=1e-5)


def test_a_causal_segment_is_aligned_at_the_end_of_its_keys():
    torch.manual_seed(1)
    q = torch.randn(4, 2, 8)
    k, v = torch.randn(10, 1, 8), torch.randn(10, 1, 8)
    out = reference_attention(
        q, k, v, [0, 4], [0, 10], causal=True, scale=0.5, groups=2
    )
    for row in range(4):
        # Query ``row`` of the last 4 keys' queries sees the first 6 + row + 1 keys.
        seen = 6 + row + 1
        scores = (q[row] @ k[:seen, 0].T) * 0.5
        want = scores.softmax(-1) @ v[:seen, 0]
        assert torch.allclose(out[row], want, atol=1e-6)


# ----------------------------------------------------------------------- parity


@pytest.mark.parametrize("layout", LAYOUTS)
def test_branches_equal_rows_in_fp32(runtimes, layout, png_url):
    model = model_of(runtimes, layout, "float32")
    for plan in (
        model.plan(STATE, QUESTIONS),
        model.plan_images(STATE, QUESTIONS, [png_url(96, 64, 3), png_url(64, 64, 4)]),
    ):
        items = list(plan.items)
        branched, rows = model.run(items), model.run_rows(items)
        assert max_gap(branched, rows) < FP32_TOLERANCE


def wrapping_conv(index_range: int):
    """causal-conv1d's convolution with a row's offset (``batch_id * x_batch_stride``)
    taken in ``index_range`` elements, as its 32-bit offsets are: a row past the range
    reads and writes other rows' memory and its own output stays as allocated (NaN)."""

    def conv(hidden_states, weight, bias=None, activation=None):
        batch, channels, length = hidden_states.shape
        size = channels * length
        # Channel-last memory, as the gated-delta layers pass their projections.
        memory = hidden_states.transpose(1, 2).flatten()
        out = torch.full_like(memory, float("nan"))
        for row in range(batch):
            start = row * size % index_range
            segment = memory[start : start + size]
            window = segment.view(1, length, channels).transpose(1, 2)
            convolved = causal_conv1d_ref(window, weight, bias, activation)
            out[start : start + size] = convolved.transpose(1, 2).flatten()
        return out.view(batch, length, channels).transpose(1, 2)

    return conv


def flat(values) -> torch.Tensor:
    return torch.tensor([p for row in values for p in row], dtype=torch.float64)


def test_padded_rows_answer_as_each_row_alone_in_fp32(runtimes, monkeypatch):
    """``run_rows`` left-pads every question of a state to its longest in one forward: in
    FP32 each row answers as it does alone, also where the forward holds more elements
    than the convolution kernel's offsets reach."""
    model = model_of(runtimes, "prefix_bidir", "float32")
    questions = {
        "short": {"type": "noul", "instructions": "Is it code?"},
        "long": {"type": "noul", "instructions": "Read the request carefully. " * 60},
        "domain": QUESTIONS["domain"],
        "last": {"type": "noul", "instructions": "Math?"},
    }
    items = list(model.plan(STATE, questions).items)
    assert len(items) == len(questions)
    alone = flat(model.run_rows([item])[0] for item in items)
    rows = flat(model.run_rows(items))
    torch.testing.assert_close(rows, alone, rtol=0, atol=ROWS_TOLERANCE)
    backbone = model.engine_model.backbone
    gated = next(layer for layer in backbone.layers if layer.kind == "linear_attention")
    channels = gated.linear_attn.conv_dim
    width = max(len(item.ids) for item in items)
    # The kernel's index range scaled to these rows: only the last row's offset passes
    # it, landing a fifth of the way into the first row and a third of the way into a
    # token's channels, so what it writes there is not the first row's own values.
    landing = width // 5 * channels + channels // 3
    index_range = (len(items) - 1) * width * channels - landing
    conv = gpu._wrap_conv(wrapping_conv(index_range))
    monkeypatch.setitem(
        model.engine_model.kernels.available,
        "causal_conv1d",
        [Kernel("causal_conv1d", conv, "causal-conv1d", exact=True)],
    )
    monkeypatch.setattr(gpu, "CONV_ELEMENTS", index_range // 2)
    each = flat(model.run_rows([item])[0] for item in items)
    torch.testing.assert_close(each, alone, rtol=0, atol=0)
    rows = flat(model.run_rows(items))
    torch.testing.assert_close(rows, alone, rtol=0, atol=ROWS_TOLERANCE)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_branches_equal_rows_with_videos(runtimes, layout, png_url, mp4_url):
    pytest.importorskip("cv2")
    model = model_of(runtimes, layout, "float32")
    if not model.video_inputs:
        pytest.skip("video decoding is unavailable")
    plan = model.plan_images(
        "What happens?", QUESTIONS, [png_url(64, 48, 5)], [mp4_url(96, 64, frames=8)]
    )
    items = list(plan.items)
    assert max_gap(model.run(items), model.run_rows(items)) < FP32_TOLERANCE


@pytest.mark.parametrize("layout", LAYOUTS)
def test_branches_match_rows_at_the_published_precision(runtimes, layout, png_url):
    model = model_of(runtimes, layout)
    for plan in (
        model.plan(STATE, QUESTIONS),
        model.plan_images(STATE, QUESTIONS, [png_url(96, 64, 3)]),
    ):
        items = list(plan.items)
        branched, rows = model.run(items), model.run_rows(items)
        assert max_gap(branched, rows) < BF16_TOLERANCE


def test_the_layout_decides_what_attends_to_what(runtimes):
    answers = {
        layout: model_of(runtimes, layout, "float32").run(
            list(model_of(runtimes, layout, "float32").plan(STATE, QUESTIONS).items)
        )
        for layout in LAYOUTS
    }
    for left, right in (
        ("prefix_bidir", "causal"),
        ("prefix_bidir", "prefix_causal"),
        ("prefix_causal", "causal"),
    ):
        assert max_gap(answers[left], answers[right]) > 10 * FP32_TOLERANCE


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_question_answers_the_same_whatever_else_is_asked(runtimes, layout):
    model = model_of(runtimes, layout, "float32")
    alone = model.run(
        list(model.plan(STATE, {"reasoning": QUESTIONS["reasoning"]}).items)
    )
    together = model.run(list(model.plan(STATE, QUESTIONS).items))
    assert max_gap(alone, [together[1]]) < FP32_TOLERANCE


@pytest.mark.parametrize("layout", LAYOUTS)
def test_requests_share_a_branched_forward(runtimes, layout, png_url, monkeypatch):
    model = model_of(runtimes, layout, "float32")
    plans = [
        model.plan(STATE, QUESTIONS),
        model.plan_images("What is shown?", QUESTIONS, [png_url(80, 60, 6)]),
        model.plan("Short.", {"reasoning": QUESTIONS["reasoning"]}),
        model.plan_images(
            "Two images.", QUESTIONS, [png_url(64, 64, 7), png_url(48, 80, 8)]
        ),
    ]
    alone = {}
    for plan in plans:
        for item, value in zip(plan.items, model.run(list(plan.items)), strict=True):
            alone[id(item)] = value
    # Questions of different requests interleaved: each still branches from its own state.
    items = [
        item
        for group in zip(*(plan.items for plan in plans[:2]), strict=True)
        for item in group
    ]
    items += [item for plan in plans[2:] for item in plan.items]
    calls = []
    branch = model.engine_model.branch

    def counted(batch):
        calls.append((len(batch.states), len(batch.blocks)))
        return branch(batch)

    monkeypatch.setattr(model.engine_model, "branch", counted)
    together = model.run_shared(items, 1)
    assert calls == [(4, len(items))]
    assert max_gap([alone[id(item)] for item in items], together) < FP32_TOLERANCE


def test_a_lone_question_runs_as_its_row(runtimes, png_url, monkeypatch):
    model = model_of(runtimes, "prefix_bidir", "float32")
    calls = []
    branch = model.engine_model.branch

    def counted(batch):
        calls.append(len(batch.blocks))
        return branch(batch)

    monkeypatch.setattr(model.engine_model, "branch", counted)
    single = [
        item
        for state in ("Short.", STATE, "Another state about a refund.")
        for item in model.plan(state, {"reasoning": QUESTIONS["reasoning"]}).items
    ]
    # A lone question runs as its row, with no branching.
    alone = model.run(single[:1])
    assert calls == []
    assert max_gap(alone, model.run_rows(single[:1])) < FP32_TOLERANCE
    # Several questions branch, whether of one state or of several; so does an image in the state.
    assert max_gap(model.run(single), model.run_rows(single)) < FP32_TOLERANCE
    many = list(model.plan(STATE, QUESTIONS).items)
    model.run(many)
    image = list(
        model.plan_images(
            "Shown?", {"r": QUESTIONS["reasoning"]}, [png_url(64, 64, 2)]
        ).items
    )
    model.run(image)
    assert calls == [len(single), len(many), 1]


def test_bf16_decisions_agree_on_many_requests(runtimes):
    model = model_of(runtimes, "prefix_bidir")
    rng = random.Random(9)
    words = ["sorted", "list", "merge", "python", "error", "refund", "ticket", "fast"]
    branched, rows = [], []
    for _ in range(12):
        state = " ".join(rng.choice(words) for _ in range(rng.randint(5, 120)))
        items = list(model.plan(state, QUESTIONS).items)
        branched += model.run(items)
        rows += model.run_rows(items)
    gaps = [
        max(abs(a - b) for a, b in zip(left, right, strict=True))
        for left, right in zip(branched, rows, strict=True)
    ]
    assert max(gaps) < BF16_TOLERANCE
    changed = [
        index
        for index, (left, right) in enumerate(zip(branched, rows, strict=True))
        if decisions([left]) != decisions([right])
    ]
    # A changed decision is a near-tie of the per-row answer.
    for index in changed:
        top = sorted(rows[index], reverse=True)
        assert top[0] - top[1] < 2 * BF16_TOLERANCE


# ------------------------------------------------------------------- requests


def test_the_state_counts_once_in_the_input_tokens(runtimes):
    runtime = runtimes["prefix_bidir", "auto"]
    model = model_of(runtimes, "prefix_bidir")
    status, body = call(runtime, {"state": STATE, "questions": QUESTIONS})
    assert status == 200
    plan = model.plan(STATE, QUESTIONS)
    rows = [len(item.ids) for item in plan.items]
    state = len(plan.items[0].state.ids)
    assert body["usage"]["input_tokens"] == state + sum(row - state for row in rows)
    assert set(body["answers"]) == set(QUESTIONS)


def usage_of(model, state, questions) -> int:
    """The state block once plus each accepted question's block: what a state-once request counts."""
    plan = model.plan(state, questions)
    if not plan.items:
        return 0
    shared = len(plan.items[0].state.ids)
    return shared + sum(len(item.ids) - shared for item in plan.items)


def test_a_state_once_request_counts_its_state_once_and_d3_counts_every_row(
    runtimes, d3_runtime
):
    model = model_of(runtimes, "prefix_bidir")
    plan = model.plan(STATE, QUESTIONS)
    state = len(plan.items[0].state.ids)
    # Each question's cost is its block and a share of the state: the shares add up to the state once.
    assert sum(item.cost for item in plan.items) == usage_of(model, STATE, QUESTIONS)
    shares = [item.cost - (len(item.ids) - state) for item in plan.items]
    assert sum(shares) == state and max(shares) - min(shares) <= 1
    status, once = call(
        runtimes["prefix_bidir", "auto"], {"state": STATE, "questions": QUESTIONS}
    )
    assert status == 200
    assert once["usage"]["input_tokens"] == usage_of(model, STATE, QUESTIONS)
    rows = d3_runtime.lookup(None).model.plan(STATE, QUESTIONS)
    assert [list(item.ids) for item in rows.items] == [
        list(item.ids) for item in plan.items
    ]
    status, every = call(d3_runtime, {"state": STATE, "questions": QUESTIONS})
    assert status == 200
    assert every["usage"]["input_tokens"] == sum(len(item.ids) for item in rows.items)
    assert (
        every["usage"]["input_tokens"] - once["usage"]["input_tokens"]
        == (len(QUESTIONS) - 1) * state
    )


def test_refused_questions_add_no_tokens(runtimes, monkeypatch):
    runtime = runtimes["prefix_bidir", "auto"]
    model = model_of(runtimes, "prefix_bidir")
    short = {"type": "noul", "instructions": "Is it short?"}
    long = {"type": "noul", "instructions": "word " * 400}
    alone = model.plan("state", {"a": short})
    monkeypatch.setattr(model, "limit", len(alone.items[0].ids) + 50)
    status, body = call(
        runtime, {"state": "state", "questions": {"l": long, "a": short}}
    )
    assert status == 200
    assert body["answers"]["l"]["error"] == "max_length_exceeded"
    # The answered question carries the whole state.
    assert body["usage"]["input_tokens"] == len(alone.items[0].ids)
    status, body = call(
        runtime, {"state": "state", "questions": {"l": long, "m": long}}
    )
    assert status == 200
    assert {answer["error"] for answer in body["answers"].values()} == {
        "max_length_exceeded"
    }
    assert body["usage"]["input_tokens"] == 0
    literal = {"type": "noul", "instructions": None}
    status, body = call(runtime, {"state": "state", "questions": {"x": literal}})
    assert status == 400
    assert "usage" not in body


def test_requests_sharing_forwards_each_count_their_own_state(runtimes):
    runtime = runtimes["prefix_bidir", "auto"]
    model = model_of(runtimes, "prefix_bidir")
    names = list(QUESTIONS)
    bodies = [
        {
            "state": f"{STATE} {index}" * (1 + index % 3),
            "questions": {name: QUESTIONS[name] for name in names[: 1 + index % 4]},
        }
        for index in range(8)
    ]
    before = runtime.metrics.input_tokens._value.get()

    async def burst():
        return await asyncio.gather(*(runtime.call("decisions", b) for b in bodies))

    counted = 0
    for body, (status, answer) in zip(bodies, asyncio.run(burst()), strict=True):
        assert status == 200
        assert answer["usage"]["input_tokens"] == usage_of(
            model, body["state"], body["questions"]
        )
        counted += answer["usage"]["input_tokens"]
    # The input-token metric counts what the responses report.
    assert runtime.metrics.input_tokens._value.get() - before == counted


def test_each_named_state_counts_once(runtimes):
    runtime = runtimes["prefix_bidir", "auto"]
    model = model_of(runtimes, "prefix_bidir")
    other = "A different state about refunds. " * 5
    status, body = call(
        runtime,
        {
            "state": STATE,
            "questions": QUESTIONS,
            "states": {"other": {"state": other, "questions": QUESTIONS}},
        },
    )
    assert status == 200
    assert body["usage"]["input_tokens"] == usage_of(model, STATE, QUESTIONS)
    assert body["states"]["other"]["usage"]["input_tokens"] == usage_of(
        model, other, QUESTIONS
    )


def test_an_image_is_part_of_the_state(runtimes, png_url, monkeypatch):
    runtime = runtimes["prefix_bidir", "auto"]
    model = model_of(runtimes, "prefix_bidir")
    image = png_url(120, 90, 2)
    text = call(runtime, {"state": "What is shown?", "questions": QUESTIONS})[1]
    one = call(
        runtime, {"state": "What is shown?", "questions": QUESTIONS, "images": [image]}
    )[1]
    from vllm_srun.families.decision3 import images as img

    tokens = img.input_tokens(model.settings, 120, 90)
    # Each image adds its envelope (vision start, end) and its tokens, once per request.
    assert one["usage"]["input_tokens"] - text["usage"]["input_tokens"] == tokens + 2
    assert one["answers"] != text["answers"]
    calls = []
    tower = model.engine_model.tower_features

    def counted(name, pixel_values, grids):
        calls.append(len(grids))
        return tower(name, pixel_values, grids)

    model.feature_cache = type(model.feature_cache)(model.feature_cache.max_bytes)
    monkeypatch.setattr(model.engine_model, "tower_features", counted)
    questions = {f"q{i}": QUESTIONS["reasoning"] for i in range(8)}
    status, _ = call(
        runtime, {"state": "Shown?", "questions": questions, "images": [image]}
    )
    assert status == 200
    assert calls == [1]


def test_question_errors_stay_per_question(runtimes, png_url, monkeypatch):
    runtime = runtimes["causal", "auto"]
    model = model_of(runtimes, "causal")
    body = {
        "state": "x",
        "questions": {
            "ok": QUESTIONS["reasoning"],
            "literal": {"type": "noul", "instructions": "Has <|image_pad|>?"},
        },
        "images": [png_url(64, 64, 1)],
    }
    status, answer = call(runtime, body)
    assert status == 200
    assert answer["answers"]["literal"]["error"] == "invalid_question"
    assert "error" not in answer["answers"]["ok"]
    short = {"type": "noul", "instructions": "Is it short?"}
    long = {"type": "noul", "instructions": "word " * 400}
    alone = model.plan("state", {"a": short})
    monkeypatch.setattr(model, "limit", len(alone.items[0].ids) + 50)
    plan = model.plan("state", {"l": long, "a": short})
    assert [item.question_id for item in plan.items] == ["a"]
    assert plan.errors["l"]["error"] == "max_length_exceeded"


def test_concurrent_requests_all_answer(runtimes, png_url):
    runtime = runtimes["prefix_bidir", "auto"]

    async def burst():
        bodies = [
            {"state": f"{STATE} {index}", "questions": QUESTIONS} for index in range(6)
        ] + [
            {"state": "Shown?", "questions": QUESTIONS, "images": [png_url(64, 48, 10)]}
        ]
        return await asyncio.gather(*(runtime.call("decisions", b) for b in bodies))

    for status, body in asyncio.run(burst()):
        assert status == 200
        assert all("error" not in answer for answer in body["answers"].values())


# -------------------------------------------------------------------- batching


class _Items:
    def __init__(self, costs):
        self.items = [type("I", (), {"ids": [0] * c, "cost": c})() for c in costs]


def test_shared_batches_pack_whole_requests():
    jobs = [
        Job(items=_Items(costs).items, deadline=None, enqueued=0.0)
        for costs in (
            [300, 20, 20],
            [500],
            [100, 100],
            [900, 900],
        )
    ]
    trees = [
        Batch(parts=[(job, list(range(len(job.items))))], shared_prefix=7)
        for job in jobs
    ]
    batches = shared_batches(trees, budget=1000, max_rows=4)
    assert [[id(job) for job, _ in batch.parts] for batch in batches] == [
        [id(jobs[3])],
        [id(jobs[1]), id(jobs[0])],
        [id(jobs[2])],
    ]
    assert all(batch.shared_prefix == 7 for batch in batches)
    every = sorted(
        (id(job), tuple(indices)) for batch in batches for job, indices in batch.parts
    )
    assert every == sorted((id(job), tuple(range(len(job.items)))) for job in jobs)


def test_the_batcher_merges_requests_of_a_state_once_model(runtimes, monkeypatch):
    model = model_of(runtimes, "prefix_bidir")
    # Pin the cost per token that sizes merged forwards, so the host's speed cannot split the plan.
    monkeypatch.setitem(
        model._token_seconds, "text", COALESCED_FORWARD_SECONDS / 10_000
    )
    batcher = Batcher(EngineArgs(), SharedContext())
    batcher.bind(model)
    plans = [model.plan(f"state {index}", QUESTIONS) for index in range(3)]
    jobs = [Job(items=list(plan.items), deadline=None, enqueued=0.0) for plan in plans]
    batches = batcher.plan(jobs, None)
    assert len(batches) == 1 and batches[0].shared_prefix > 0
    assert sorted(id(job) for job, _ in batches[0].parts) == sorted(map(id, jobs))


def test_calibration_rows_branch_from_states_of_their_own(runtimes, monkeypatch):
    model = model_of(runtimes, "prefix_bidir")
    items = model.calibration_items(4, 256)
    assert len({id(item.state) for item in items}) == 4
    for item in items:
        assert len(item.ids) == item.cost == 256
        assert tuple(item.ids[: len(item.state.ids)]) == item.state.ids
        assert 0 < len(item.state.ids) < 256
    branched: list[BranchBatch] = []
    branch = model.engine_model.branch

    def recorded(batch):
        branched.append(batch)
        return branch(batch)

    monkeypatch.setattr(model.engine_model, "branch", recorded)
    values = model.run_approximate(items)
    assert [len(value) for value in values] == [2] * 4
    assert len(branched) == 1 and len(branched[0].states) == 4
    assert sum(map(len, (*branched[0].states, *branched[0].blocks))) == 4 * 256


@pytest.fixture(scope="module")
def validated_runtime(tmp_path_factory):
    from vllm_srun.testing.decision3 import VALIDATED_LENGTH, write_fixture

    package = write_fixture(
        tmp_path_factory.mktemp("state-once-validated") / "prefix_bidir",
        "prefix_bidir",
        seed=31,
        validated=True,
    )
    runtime = start_runtime(package)
    yield runtime, VALIDATED_LENGTH
    runtime.stop()


def test_questions_past_the_validated_length_need_long_context(validated_runtime):
    runtime, validated = validated_runtime
    card = runtime.lookup(None).card([])
    assert (
        card["limits"]["validated_input_tokens"]
        == validated
        < card["limits"]["max_input_tokens"]
    )
    first = next(iter(QUESTIONS))
    body = {"state": "word " * validated, "questions": QUESTIONS}
    status, answer = call(runtime, body)
    assert status == 200
    for key in QUESTIONS:
        assert answer["answers"][key]["error"] == "max_length_exceeded"
        assert answer["answers"][key]["reason"] == "question_tokens"
        assert "long context" in answer["answers"][key]["message"]
    status, answer = call(runtime, {**body, "options": {"long_context": True}})
    assert status == 200
    assert all("error" not in answer["answers"][key] for key in QUESTIONS)
    short = {"state": "Merge two sorted lists.", "questions": {first: QUESTIONS[first]}}
    status, answer = call(runtime, short)
    assert status == 200 and "error" not in answer["answers"][first]


def test_composed_questions_take_the_validated_length_too(validated_runtime, png_url):
    from vllm_srun.plugins.decisions import LONG_CONTEXT

    runtime, validated = validated_runtime
    model = runtime.lookup(None).model
    questions = {"box": {"type": "box", "instructions": "the red square"}}
    image = [png_url(64, 48, 71)]
    planned = model.plan_composed("word " * validated, questions, image, None, {})
    assert planned.errors["box"]["reason"] == "question_tokens" and not planned.items
    token = LONG_CONTEXT.set(True)
    try:
        planned = model.plan_composed("word " * validated, questions, image, None, {})
    finally:
        LONG_CONTEXT.reset(token)
    assert not planned.errors and len(planned.items) == 1


def test_a_state_once_card_caps_the_request_at_the_context_length(runtimes, d3_runtime):
    limits = runtimes["prefix_bidir", "auto"].lookup(None).card([])["limits"]
    assert limits["max_request_tokens"] == limits["max_input_tokens"]
    assert "max_request_tokens" not in d3_runtime.lookup(None).card([])["limits"]


def test_a_request_over_its_token_cap_is_refused_whole(runtimes, png_url):
    from vllm_srun.plugins.decisions import LONG_CONTEXT

    runtime = runtimes["prefix_bidir", "auto"]
    model = model_of(runtimes, "prefix_bidir")
    cap = model.info.limits["max_request_tokens"]
    one = model.plan(STATE, {"q0": QUESTIONS["long"]}).input_tokens
    block = (
        model.plan(
            STATE, {"q0": QUESTIONS["long"], "q1": QUESTIONS["long"]}
        ).input_tokens
        - one
    )
    # The state once plus count blocks is just over the cap; one block fewer is not.
    count = (cap - (one - block)) // block + 1
    many = {f"q{index}": QUESTIONS["long"] for index in range(count)}
    fits = dict(itertools.islice(many.items(), count - 1))
    assert model.plan(STATE, fits).input_tokens <= cap
    status, answer = call(runtime, {"state": STATE, "questions": fits})
    assert status == 200 and all("error" not in a for a in answer["answers"].values())
    for options in ({}, {"options": {"long_context": True}}):
        status, answer = call(runtime, {"state": STATE, "questions": many, **options})
        assert status == 400, answer
        assert answer["error"]["code"] == "invalid_request"
        assert answer["error"]["reason"] == "request_tokens"
        assert answer["error"]["limit"] == cap
    status, answer = call(runtime, {"state": STATE, "questions": QUESTIONS})
    assert status == 200 and all("error" not in a for a in answer["answers"].values())
    box = {"box": {"type": "box", "instructions": "the red square"}}
    for planned in (
        lambda: model.plan_composed(STATE, many, [], None, {}),
        lambda: model.plan_composed(
            STATE, {**many, **box}, [png_url(64, 48, 71)], None, {}
        ),
    ):
        token = LONG_CONTEXT.set(True)
        try:
            with pytest.raises(RequestError) as refused:
                planned()
        finally:
            LONG_CONTEXT.reset(token)
        assert refused.value.reason == "request_tokens" and refused.value.limit == cap


@pytest.mark.parametrize("layout", LAYOUTS)
def test_default_engine_arguments_calibrate_the_token_budget(
    packages, layout, monkeypatch
):
    """A GPU that reports its memory calibrates the token budget a state-once package's engine arguments leave
    unset, from branched forwards; the CPU never calibrates, so it poses here as a small device class.
    """
    small = calibration.DeviceClass("small", 0, 2048, 16)
    monkeypatch.setattr(calibration, "device_class", lambda device: small)
    runtime = start_runtime(packages[layout])
    try:
        served = runtime.lookup(None)
        assert served.limits_source == "the small device class, calibrated"
        assert served.args.max_num_seqs == 16
        assert (
            calibration.MIN_CALIBRATED_TOKENS // 4
            <= served.args.max_num_batched_tokens
            <= 2048
        )
        assert (
            served.scheduler.batcher.max_num_batched_tokens
            == served.args.max_num_batched_tokens
        )
        status, body = call(runtime, {"state": STATE, "questions": QUESTIONS})
        assert status == 200
        assert all("error" not in body["answers"][key] for key in QUESTIONS)
    finally:
        runtime.stop()


def _branch_inputs(model, questions):
    """Three requests' states and question blocks, as the engine receives a merged batch."""
    states, blocks, owners, items = [], [], [], []
    for owner in range(3):
        plan = model.plan(f"{STATE} {owner}", questions)
        state = list(plan.items[0].state.ids)
        states.append(state)
        for entry in plan.items:
            blocks.append(list(entry.ids[len(state) :]))
            owners.append(owner)
            items.append(entry)
    return states, blocks, owners, items


@pytest.mark.parametrize(
    ("joined", "exact", "merged"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (False, True, True),
    ],
    ids=[
        "branched",
        "branched-exact",
        "joined",
        "joined-exact",
        "merged",
        "merged-exact",
    ],
)
def test_a_static_layout_answers_as_the_packed_branches_do(
    runtimes, joined, exact, merged
):
    """FP32 on the CPU through the per-segment path: bucket padding, trash and filler segments, joined rows,
    merged attention and the last layer's readout rows leave every block's answer as ``Branches`` gives it.
    """
    model = model_of(runtimes, "prefix_bidir", "float32")
    first = next(iter(QUESTIONS))
    questions = {first: QUESTIONS[first]} if joined else QUESTIONS
    states, blocks, owners, items = _branch_inputs(model, questions)
    backbone = model.engine_model._state_once()
    window = max(
        layer.linear_attn.conv_kernel_size
        for layer in backbone.layers
        if layer.kind == "linear_attention"
    )
    shape, values = static_layout(
        states, blocks, owners, window, joined=joined, exact=exact, merged=merged
    )
    assert shape.joined == joined and shape.merged == merged
    assert shape.rows >= sum(map(len, (*states, *blocks)))
    view = StaticBranches(shape, backbone.layout, *pack_layout(shape, values), values)
    with torch.inference_mode():
        embeds = backbone.embed(view.ids[None])
        hidden = backbone.forward_branches(
            embeds, view.positions.expand(3, -1), view, view.last
        )[0, : len(blocks)]
    packed = model.engine_model.branch(BranchBatch(states, blocks, owners, None))
    assert (
        max_gap(
            model._probabilities(items, hidden), model._probabilities(items, packed)
        )
        < FP32_TOLERANCE
    )


def test_batches_of_one_bucket_share_a_static_shape():
    states = [[5] * 300, [6] * 290]
    blocks = [[7] * 40, [8] * 41, [9] * 30]
    shape, values = static_layout(states, blocks, [0, 0, 1], 4)
    other, _ = static_layout(
        [[1] * 299, [2] * 291], [[3] * 39, [4] * 42, [5] * 30], [0, 0, 1], 4
    )
    assert other == shape
    for name, size in {**shape.int32_sizes(), **shape.int64_sizes()}.items():
        assert len(values[name]) == size, name
    # Every chunk of every segment once, in order; the other entries name zero-length fillers.
    chunks = list(
        zip(values["chunks_all"][::2], values["chunks_all"][1::2], strict=True)
    )
    lengths = [b - a for a, b in itertools.pairwise(values["rule_all"])]
    real = [
        (segment, chunk)
        for segment, length in enumerate(lengths)
        for chunk in range(-(-length // 64))
    ]
    assert [entry for entry in chunks if lengths[entry[0]]] == real
    assert all(chunk == 0 for segment, chunk in chunks if not lengths[segment])
    exact, exact_values = static_layout(states, blocks, [0, 0, 1], 4, exact=True)
    assert (
        exact.rows == 300 + 290 + 40 + 41 + 30
        and exact_values["ids"][:300].tolist() == [5] * 300
    )
    merged, merged_values = static_layout(states, blocks, [0, 0, 1], 4, merged=True)
    assert merged.queries == merged.state_segments + merged.block_segments
    # The states' segments (trash and fillers included) open the call, each over its own rows.
    heads = merged.state_segments + 1
    assert merged_values["cu_blocks"][:heads] == values["cu_states"]
    assert merged_values["cu_keys"][:heads] == values["cu_states"]
    assert merged_values["block_keys"][: shape.state_rows].tolist() == list(
        range(shape.state_rows)
    )
    ints32, ints64 = pack_layout(shape, values)
    assert ints32.dtype == torch.int32 and ints64.dtype == torch.long
    assert ints32.tolist() == [
        int(v) for name in shape.int32_sizes() for v in values[name]
    ]
    assert ints64.tolist() == [
        int(v) for name in shape.int64_sizes() for v in values[name]
    ]


def test_a_forward_that_captured_a_graph_leaves_the_cost_estimate(
    runtimes, monkeypatch
):
    model = model_of(runtimes, "prefix_bidir")
    items = list(model.plan(f"{STATE} estimate", QUESTIONS).items)
    captures = itertools.count()
    monkeypatch.setattr(
        model.engine_model, "graph_captures", lambda: next(captures), raising=False
    )
    before = dict(model._token_seconds)
    model.run(items)
    assert model._token_seconds == before
    monkeypatch.setattr(model.engine_model, "graph_captures", lambda: 3, raising=False)
    model.run(items)
    assert model._token_seconds != before


def test_narrow_backbones_replay_branched_forwards_to_longer_rows():
    class Backbone:
        def __init__(self, hidden: int):
            self.config = {"hidden_size": hidden}

    assert (
        engine.branch_graph_tokens(Backbone(1024)) == engine.NARROW_BRANCH_GRAPH_TOKENS
    )
    assert (
        engine.branch_graph_tokens(Backbone(2048)) == engine.NARROW_BRANCH_GRAPH_TOKENS
    )
    assert engine.branch_graph_tokens(Backbone(2560)) == engine.BRANCH_GRAPH_TOKENS
    assert engine.branch_graph_tokens(object()) == engine.BRANCH_GRAPH_TOKENS


def test_a_request_served_alone_does_not_hold_the_batching_window(
    runtimes, monkeypatch
):
    runtime = runtimes["prefix_bidir", "auto"]
    scheduler = runtime.lookup(None).scheduler
    alone: list[bool] = []
    submit = scheduler.submit_group

    def recorded(*args, **kwargs):
        alone.append(kwargs.get("alone", False))
        return submit(*args, **kwargs)

    monkeypatch.setattr(scheduler, "submit_group", recorded)
    first = next(iter(QUESTIONS))
    body = {"state": f"{STATE} alone", "questions": {first: QUESTIONS[first]}}
    assert call(runtime, body)[0] == 200
    assert alone == [True]

    async def burst():
        return await asyncio.gather(
            *(
                runtime.call("decisions", {**body, "state": f"{STATE} burst {index}"})
                for index in range(3)
            )
        )

    assert all(status == 200 for status, _ in asyncio.run(burst()))
    # Concurrent requests may join one another's batch: they hold the window as before.
    assert len(alone) > 1 and not any(alone[1:])


def test_a_burst_planned_over_more_than_the_window_shares_one_forward(
    runtimes, monkeypatch
):
    runtime = runtimes["prefix_bidir", "auto"]
    served = runtime.lookup(None)
    monkeypatch.setattr(served.scheduler.limits, "batch_window_ms", 200.0)
    # Pin the cost per token that sizes merged forwards, so the host's speed cannot split the burst.
    monkeypatch.setitem(
        served.model._token_seconds, "text", COALESCED_FORWARD_SECONDS / 10_000
    )
    plan = served.model.plan_surface

    def slow(surface, request):
        # Request i is planned i * 80 ms late: the last past the 200 ms window, well within three of it.
        time.sleep(0.08 * int(request.body["state"].rsplit(" ", 1)[1]))
        return plan(surface, request)

    monkeypatch.setattr(served.model, "plan_surface", slow)
    forwards: list[int] = []
    branched = served.model._branched

    def counted(items):
        forwards.append(len(items))
        return branched(items)

    monkeypatch.setattr(served.model, "_branched", counted)
    first = next(iter(QUESTIONS))

    async def burst():
        return await asyncio.gather(
            *(
                runtime.call(
                    "decisions",
                    {
                        "state": f"{STATE} slow {index}",
                        "questions": {first: QUESTIONS[first]},
                    },
                )
                for index in range(4)
            )
        )

    assert all(status == 200 for status, _ in asyncio.run(burst()))
    assert forwards == [4]
