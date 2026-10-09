"""Both Decision 1.0 runtimes end to end on tiny packages, and bit for bit against the bundled numerics on CPU.

The references rebuild the packages' bundled runtime from Transformers and
PyTorch modules (``Qwen3_5TextModel`` in FP32; one ``ModernBertModel`` per
question type and ``nn.TransformerEncoderLayer`` heads with the fused fast
path off), with the released physical batches.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
import torch
from torch import nn
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.families.decision1 import package as pkg
from vllm_srun.families.decision1 import qwen, vela
from vllm_srun.families.decision1.questions import KINDS
from vllm_srun.heads.candidate import load_head, logits
from vllm_srun.plugins.base import Job
from vllm_srun.plugins.decisions import RenderedItem
from vllm_srun.profiles.batching import BatchingProfile
from vllm_srun.profiles.exact import ExactProfile
from vllm_srun.runtime import Runtime
from vllm_srun.scheduler.planner import padded
from vllm_srun.testing.decision1 import write_package
from vllm_srun.text import segments

STATE = (
    "Write a Python function that merges two sorted lists and explain its running time."
)
QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {"code": "Programming", "math": "Mathematics", "other": None},
    },
    "reasoning": {
        "type": "noul",
        "instructions": "Does this need multi-step reasoning?",
    },
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", "Moderate", {"level": "Hard"}],
    },
}
# Eleven questions: two physical batches, every type, mixed lengths.
MANY = {
    **{
        f"c{i}": {
            "type": "choice",
            "instructions": "Which area? " * (i + 1),
            "criteria": {"code": "Programming", "math": None, "x": "y" * (i + 1)},
        }
        for i in range(4)
    },
    **{
        f"n{i}": {"type": "noul", "instructions": "Is it hard? " * (i + 1)}
        for i in range(4)
    },
    **{
        f"s{i}": {
            "type": "score",
            "instructions": "Rate it.",
            "criteria": ["a", "b", "c"][: i + 2],
        }
        for i in range(2)
    },
    "c9": {
        "type": "choice",
        "instructions": "Last?",
        "criteria": {"yes": None, "no": None},
    },
}


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision1")
    return {
        "qwen": write_package(
            root / "qwen",
            seed=3,
            temperatures={"choice": 1.1, "noul": 0.9, "score": 1.3},
        ),
        "vela": write_package(root / "vela", runtime=pkg.VELA, seed=4, presets=True),
    }


@pytest.fixture(scope="module")
def runtimes(packages):
    started = {}
    for name, root in packages.items():
        runtime = Runtime(
            ServeConfig(models=(ModelConfig(model=str(root), device="cpu"),))
        )
        runtime.start(background=False)
        started[name] = runtime
    yield started
    for runtime in started.values():
        runtime.stop()


def decide(runtime, questions, state=STATE, **options):
    body = {"state": state, "questions": questions}
    if options:
        body["options"] = options
    return asyncio.run(runtime.call("decisions", body))


def model_of(runtime):
    return runtime.lookup(None).model


@pytest.mark.parametrize("name", ["qwen", "vela"])
def test_answers_are_typed_and_deterministic(runtimes, name):
    status, body = decide(runtimes[name], QUESTIONS)
    assert status == 200
    answers = body["answers"]
    assert [answers[q]["type"] for q in QUESTIONS] == ["choice", "noul", "score"]
    assert answers["domain"]["choice"] in ("code", "math", "other")
    assert answers["difficulty"]["legend"]["2"] == '{"level":"Hard"}'
    assert body["usage"]["input_tokens"] > 0
    assert decide(runtimes[name], QUESTIONS)[1]["answers"] == answers
    assert runtimes[name].lookup(None).health.state == "ready"


def test_qwen_physical_batches_and_padding(runtimes):
    model = model_of(runtimes["qwen"])
    plan = model.plan(STATE, MANY)
    assert model.exact_batches(plan.items) == [list(range(8)), [8, 9, 10]]
    batch = segments.collate(plan.items[:3], model.tokenizer.pad_id, qwen.PAD_MULTIPLE)
    assert batch["input_ids"].shape[1] % 32 == 0


def test_qwen_physical_batches_split_only_beyond_the_token_budget():
    items = [
        RenderedItem(f"q{i}", "choice", [0] * length, [], 0, [], [])
        for i, length in enumerate((40, 40, 40, 100, 10, 10, 10, 10, 10))
    ]
    assert qwen.physical_batches(items) == [list(range(8)), [8]]
    assert qwen.physical_batches(items, 1024) == [list(range(8)), [8]]
    assert qwen.physical_batches(items, 256) == [[0, 1, 2], [3, 4], [5, 6, 7], [8]]
    with pytest.raises(ValueError, match="exceeds the forward token budget"):
        qwen.physical_batches(items, 64)


def test_qwen_exact_batches_keep_the_forward_token_budget(runtimes, monkeypatch):
    model = model_of(runtimes["qwen"])
    plan = model.plan(STATE, MANY)

    def padded(indices):
        width = max(len(plan.items[index].ids) for index in indices)
        return len(indices) * -(-width // qwen.PAD_MULTIPLE) * qwen.PAD_MULTIPLE

    budget = 3 * max(padded([index]) for index in range(len(plan.items)))
    monkeypatch.setattr(model, "forward_token_budget", lambda: budget)
    batches = model.exact_batches(plan.items)
    assert [index for batch in batches for index in batch] == list(range(11))
    assert len(batches) > 2 and all(padded(batch) <= budget for batch in batches)


def test_the_engine_sizes_gated_delta_forwards_on_gpu_only(runtimes, monkeypatch):
    model = model_of(runtimes["qwen"])
    engine = model.engine_model
    assert engine.max_forward_tokens() is None
    assert model.forward_token_budget() is None
    config = engine.spec.backbone.config
    width = config["linear_num_value_heads"] * max(
        config["linear_key_head_dim"], config["linear_value_head_dim"]
    )
    monkeypatch.setattr(engine, "device", torch.device("cuda", 0))
    assert model.forward_token_budget() == (2**30 - 1) // width
    assert model_of(runtimes["vela"]).engine_model.max_forward_tokens() is None


def test_vela_physical_batches_sort_by_type(runtimes):
    model = model_of(runtimes["vela"])
    plan = model.plan(STATE, MANY)
    kinds = [
        plan.items[index].task_type
        for batch in model.exact_batches(plan.items)
        for index in batch
    ]
    assert kinds == sorted(kinds, key=KINDS.index)
    assert [len(batch) for batch in model.exact_batches(plan.items)] == [8, 3]


def test_exact_profile_runs_the_released_physical_batches(runtimes):
    model = model_of(runtimes["vela"])
    plan = model.plan(STATE, MANY)
    profile = ExactProfile()
    profile.bind(model)
    job = Job(
        items=plan.items, deadline=None, enqueued=time.monotonic(), profile="exact"
    )
    batches = profile.plan([job], model.forward_token_budget())
    assert [
        indices for batch in batches for _, indices in batch.parts
    ] == model.exact_batches(plan.items)


def test_vela_coalesces_within_the_cpu_budget_while_exact_keeps_its_batches(runtimes):
    model = model_of(runtimes["vela"])
    budget = model.forward_token_budget()
    assert budget == vela.CPU_BATCH_TOKENS
    plan = model.plan(" ".join([STATE] * 12), MANY)
    released = model.exact_batches(plan.items)
    widest = max(len(item.ids) for item in plan.items)
    assert len(released[0]) * padded(widest) > budget
    jobs = [
        Job(items=plan.items, deadline=None, enqueued=time.monotonic(), profile=name)
        for name in ("exact", "batching", "batching", "batching")
    ]
    for batch in BatchingProfile().plan(jobs[1:], budget):
        rows = batch.items()
        assert len(rows) * padded(max(len(row.ids) for row in rows)) <= budget
    exact = ExactProfile()
    exact.bind(model)
    batches = exact.plan(jobs[:1], budget)
    assert [indices for batch in batches for _, indices in batch.parts] == released


def qwen_reference(package, items, temperatures, pad_id):
    """The bundled decoder runtime's predict() on CPU: FP32 backbone, FP32 head, T-softmax, renormalized."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    backbone = Qwen3_5TextModel.from_pretrained(
        str(package / "backbone"), dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    config = json.loads((package / "decision_config.json").read_text())
    head = load_head(
        package / "decision_head.safetensors",
        backbone.config.hidden_size,
        config["head_dim"],
    )
    batch = segments.collate(items, pad_id, qwen.PAD_MULTIPLE)
    with torch.inference_mode():
        hidden = backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).last_hidden_state
        rows = torch.arange(hidden.shape[0])
        scores = logits(
            head,
            hidden[rows[:, None], batch["candidate_positions"]],
            hidden[rows, batch["query_positions"]],
            batch["candidate_mask"],
        )
        staged = [
            (
                scores[slot, : len(item.keys)].float() / temperatures[item.task_type]
            ).softmax(-1)
            for slot, item in enumerate(items)
        ]
    host = torch.cat(staged).tolist()
    out, offset = [], 0
    for item in items:
        values = host[offset : offset + len(item.keys)]
        offset += len(item.keys)
        out.append([value / sum(values) for value in values])
    return out


@pytest.mark.reference
def test_qwen_matches_the_bundled_numerics_bit_for_bit(packages, runtimes):
    pytest.importorskip("transformers")
    model = model_of(runtimes["qwen"])
    plan = model.plan(STATE, MANY)
    for indices in model.exact_batches(plan.items):
        items = [plan.items[index] for index in indices]
        expected = qwen_reference(
            packages["qwen"], items, model.temperatures, model.tokenizer.pad_id
        )
        assert model.run(items) == expected


class VelaReference:
    """The bundled encoder runtime from Transformers and PyTorch modules."""

    def __init__(self, package):
        transformers = pytest.importorskip("transformers")
        from safetensors.torch import load_file

        config = json.loads((package / "native/encoder/config.json").read_text())
        hf_config = transformers.ModernBertConfig(
            **{key: value for key, value in config.items() if value is not None},
            attn_implementation="sdpa",
        )
        base = load_file(str(package / "native/encoder/model.safetensors"))
        self.encoders = {}
        for kind, branch in vela.BRANCH.items():
            state = dict(base)
            if branch is not None:
                renames = {
                    f"{branch}_blocks.": "layers.",
                    f"{branch}_final_norm.": "final_norm.",
                }
                for name, value in load_file(
                    str(package / f"native/{branch}_encoder.safetensors")
                ).items():
                    prefix = next(old for old in renames if name.startswith(old))
                    state[renames[prefix] + name[len(prefix) :]] = value
            encoder = transformers.ModernBertModel(hf_config).eval()
            encoder.load_state_dict(state, strict=True)
            self.encoders[kind] = encoder
        heads = load_file(str(package / "native/decision_heads.safetensors"))
        hidden = config["hidden_size"]
        self.type_embedding = heads["type_embedding.weight"]
        self.layers, self.scorers = {}, {}
        for kind in KINDS:
            self.layers[kind] = []
            for index in range(2):
                layer = nn.TransformerEncoderLayer(
                    hidden,
                    4,
                    4 * hidden,
                    0.1,
                    activation="relu",
                    batch_first=True,
                    norm_first=True,
                ).eval()
                prefix = f"heads.{kind}.{index}."
                layer.load_state_dict(
                    {
                        k[len(prefix) :]: v
                        for k, v in heads.items()
                        if k.startswith(prefix)
                    }
                )
                self.layers[kind].append(layer)
            scorer = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            ).eval()
            prefix = f"scorers.{kind}."
            scorer.load_state_dict(
                {k[len(prefix) :]: v for k, v in heads.items() if k.startswith(prefix)}
            )
            self.scorers[kind] = scorer

    def probabilities(self, items, pad_id):
        batch = vela.collate(items, pad_id)
        mask, markers = batch["attention_mask"], batch["markers"]
        output = torch.empty(markers.shape)
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.inference_mode():
                for index, kind in enumerate(KINDS):
                    rows = [
                        slot
                        for slot, item in enumerate(items)
                        if item.task_type == kind
                    ]
                    if not rows:
                        continue
                    hidden = self.encoders[kind](
                        input_ids=batch["input_ids"], attention_mask=mask.long()
                    ).last_hidden_state
                    hidden = (hidden + self.type_embedding[index])[rows]
                    for layer in self.layers[kind]:
                        hidden = layer(hidden, src_key_padding_mask=~mask[rows])
                    where = markers[rows][:, :, None].expand(-1, -1, hidden.shape[-1])
                    output[rows] = self.scorers[kind](
                        torch.gather(hidden, 1, where)
                    ).squeeze(-1)
                output = output.masked_fill(
                    ~batch["valid"], torch.finfo(torch.float32).min
                )
                return [
                    output[slot, : len(item.keys)].softmax(-1).tolist()
                    for slot, item in enumerate(items)
                ]
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)


@pytest.mark.reference
def test_vela_matches_the_bundled_numerics_bit_for_bit(packages, runtimes):
    reference = VelaReference(packages["vela"])
    model = model_of(runtimes["vela"])
    plan = model.plan(STATE, MANY)
    for indices in model.exact_batches(plan.items):
        items = [plan.items[index] for index in indices]
        assert model.run(items) == reference.probabilities(items, model.special["pad"])


def test_vela_layout(runtimes):
    model = model_of(runtimes["vela"])
    tokens, special = model.tokenizer.encode, model.special
    item = model.plan("Hello there.", {"q": QUESTIONS["domain"]}).items[0]
    expected = [
        special["bos"],
        *tokens("choice question: Which domain is this request about?"),
        special["sep"],
    ]
    markers = []
    for text in ("code: Programming", "math: Mathematics", "other"):
        markers.append(len(expected))
        expected += [special["marker"], *tokens(text), special["sep"]]
    expected += [*tokens("Hello there."), special["sep"]]
    assert item.ids == expected and item.gather == markers
    score = model.plan("x", {"q": QUESTIONS["difficulty"]}).items[0]
    assert (
        tokens("level 2: " + '{"level":"Hard"}')
        == score.ids[
            score.gather[2]
            + 1 : score.gather[2]
            + 1
            + len(tokens('level 2: {"level":"Hard"}'))
        ]
    )


def test_qwen_renders_null_choices_per_model(runtimes):
    model = model_of(runtimes["qwen"])
    item = model.plan("x", {"q": QUESTIONS["domain"]}).items[0]
    assert model.tokenizer.encode(
        '\n<option>\n{"description":null,"key":"other"}\n</option>'
    ) == (item.ids[item.gather[1] + 1 : item.gather[2] + 1])
    model.null_choice_as_key = True
    try:
        keyed = model.plan("x", {"q": QUESTIONS["domain"]}).items[0]
    finally:
        model.null_choice_as_key = False
    assert model.tokenizer.encode(
        '\n<option>\n{"description":"other","key":"other"}\n</option>'
    ) == (keyed.ids[keyed.gather[1] + 1 : keyed.gather[2] + 1])


def test_one_over_long_question_fails_every_question(runtimes):
    long = {"type": "noul", "instructions": "Is it long? " * 400}
    status, body = decide(
        runtimes["vela"], {**QUESTIONS, "long": long, "bad": {"type": "nope"}}
    )
    assert status == 200
    answers = body["answers"]
    assert {answers[q]["error"] for q in (*QUESTIONS, "long")} == {
        "max_length_exceeded"
    }
    assert (answers["bad"]["type"], answers["bad"]["error"]) == (
        None,
        "invalid_question",
    )
    assert answers["bad"]["message"]
    assert body["usage"]["input_tokens"] == 0


def test_presets_answer_like_their_questions(packages, runtimes):
    # One request each: rows of one batch may round differently on some CPUs.
    presets = pkg.presets(packages["vela"])
    answers = [
        decide(runtimes["vela"], {"q": question})
        for question in ({"preset": "hazard.weapons"}, presets["hazard.weapons"])
    ]
    assert [status for status, _ in answers] == [200, 200]
    assert answers[0][1]["answers"] == answers[1][1]["answers"]
    assert model_of(runtimes["vela"]).info.presets == tuple(sorted(presets))


def test_malformed_requests_are_rejected(runtimes):
    status, body = decide(runtimes["qwen"], QUESTIONS, state="  ")
    assert status == 400 and body["error"]["code"] == "invalid_request"


def test_model_info(runtimes):
    info = model_of(runtimes["vela"]).info
    assert info.family == "decision1" and info.question_types == KINDS
    assert info.limits["max_input_tokens"] == vela.MAX_INPUT_TOKENS
    assert (
        model_of(runtimes["qwen"]).info.limits["max_input_tokens"]
        == qwen.MAX_INPUT_TOKENS
    )


def test_only_approximate_batches_ask_for_the_reduced_copy(runtimes, monkeypatch):
    model = model_of(runtimes["vela"])
    plan = model.plan(STATE, MANY)
    batch = [plan.items[index] for index in model.exact_batches(plan.items)[0]]
    encode, asked = model.engine_model.encode, []

    def recording(encoder_batch):
        asked.append(encoder_batch.reduced)
        return encode(encoder_batch)

    monkeypatch.setattr(model.engine_model, "encode", recording)
    model.run(batch)
    exact = asked[:]
    asked.clear()
    model.run_approximate(batch)
    assert exact and not any(exact)
    assert asked and all(asked)


def test_approximate_batches_run_each_stack_over_its_own_rows(packages, runtimes):
    model = model_of(runtimes["vela"])
    plan = model.plan(STATE, MANY)
    batch = [plan.items[index] for index in model.exact_batches(plan.items)[0]]
    assert len({item.task_type for item in batch}) > 1
    exact, approximate = model.run(batch), model.run_approximate(batch)
    for row, value in zip(exact, approximate, strict=True):
        assert value == pytest.approx(row, abs=1e-4)
    shuffled = list(reversed(plan.items))
    for row, value in zip(
        model.run_approximate(shuffled),
        [model.run([i])[0] for i in shuffled],
        strict=True,
    ):
        assert row == pytest.approx(value, abs=1e-4)


def test_type_heads_read_packed_rows_by_length_only_when_padding_wastes():
    assert vela.readout_groups([40, 44, 48]) == [[0, 1, 2]]
    assert vela.readout_groups([10, 300, 30]) == [[0, 1, 2]]
    assert vela.readout_groups([30, 33, 500]) == [[0, 1], [2]]
    assert vela.readout_groups([9, 200, 10, 130, 12, 150]) == [
        [0, 2],
        [4],
        [3, 5],
        [1],
    ]


def test_coalesced_rows_of_mixed_lengths_answer_like_their_own_requests(runtimes):
    model = model_of(runtimes["vela"])
    states = (STATE, " ".join([STATE] * 12))
    items = [item for state in states for item in model.plan(state, MANY).items]
    choices = [len(item.ids) for item in items if item.task_type == "choice"]
    assert len(vela.readout_groups(choices)) > 1
    for row, value in zip(
        model.run_approximate(items),
        [model.run([item])[0] for item in items],
        strict=True,
    ):
        assert row == pytest.approx(value, abs=1e-4)


def test_batching_profile_serves_the_encoder(packages):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(packages["vela"]), device="cpu", profile="batching"
                ),
            )
        )
    )
    runtime.start(background=False)
    try:
        exact = decide(runtime, MANY, profile="exact")[1]["answers"]
        batched = decide(runtime, MANY, profile="batching")[1]["answers"]
    finally:
        runtime.stop()
    assert batched.keys() == exact.keys()
    assert all(batched[q]["type"] == exact[q]["type"] for q in exact)


def test_shared_context_profile_serves_the_decoder(packages):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(packages["qwen"]), device="cpu", profile="shared_context"
                ),
            )
        )
    )
    runtime.start(background=False)
    try:
        exact = decide(runtime, MANY, profile="exact")[1]["answers"]
        shared = decide(runtime, MANY, profile="shared_context")[1]["answers"]
    finally:
        runtime.stop()
    for question, answer in exact.items():
        if answer["type"] == "noul":
            assert shared[question]["noul"] == pytest.approx(answer["noul"], abs=1e-4)
        else:
            for key, value in answer["probabilities"].items():
                assert shared[question]["probabilities"][key] == pytest.approx(
                    value, abs=1e-4
                )
