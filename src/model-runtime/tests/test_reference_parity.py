"""The native CPU path is bit-identical to the Transformers modules the packages were scored with."""

import json

import pytest
import torch
from vllm_srun.heads.candidate import load_head, logits
from vllm_srun.text.segments import collate

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.reference

TEXTS = [
    "Context:\nhello world",
    '\n<option>\n{"a":1}\n</option>',
    " spaced  text\t\n",
    "Ünïcödé 中文 🚀 and <|endoftext|> inline",
]


def reference_scores(package, items, pad_id):
    from transformers import Qwen3Model
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config = json.loads((package / "backbone" / "config.json").read_text())
    model_class = Qwen3Model if config["model_type"] == "qwen3" else Qwen3_5TextModel
    backbone = model_class.from_pretrained(
        str(package / "backbone"),
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    decision = json.loads((package / "decision_config.json").read_text())
    head = load_head(
        package / "decision_head.safetensors",
        config["hidden_size"],
        decision["head_dim"],
    )
    batch = collate(items, pad_id)
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
    return [
        scores[index, : len(item.keys)].tolist() for index, item in enumerate(items)
    ]


def questions_with_unpadded_row(runtime):
    """Questions whose rendered length is a multiple of 8, so one batch runs SDPA without a mask."""
    for extra in range(64):
        question = {"type": "noul", "instructions": "Is this hard?" + "!" * extra}
        plan = runtime.lookup(None).model.plan("state", {"q": question})
        if len(plan.items[0].ids) % 8 == 0:
            return {"q": question}
    raise AssertionError("no unpadded length found")


@pytest.mark.parametrize(
    "runtime_name,package_name",
    [("qwen3_runtime", "qwen3_package"), ("qwen35_runtime", "qwen35_package")],
)
def test_native_backbones_match_transformers_bit_for_bit(
    request, runtime_name, package_name
):
    runtime = request.getfixturevalue(runtime_name)
    package = request.getfixturevalue(package_name)
    questions = {
        f"c{i}": {
            "type": "choice",
            "instructions": "Which domain is this about? " * (i + 1),
            "criteria": {
                "code": "Programming",
                "math": "Mathematics",
                "other": None,
                "x": "y" * i,
            },
        }
        for i in range(4)
    }
    questions["n"] = {"type": "noul", "instructions": "Reason?"}
    questions["s"] = {
        "type": "score",
        "instructions": "Hard?",
        "criteria": ["a", "b", "c", "d", "e"],
    }
    model = runtime.lookup(None).model
    plan = model.plan("Merge two sorted lists in Python. " * 3, questions)
    unpadded = model.plan("state", questions_with_unpadded_row(runtime)).items
    pad_id = model.tokenizer.pad_id
    for items in (plan.items, plan.items[:1], plan.items[2:4], unpadded):
        assert model.run(items) == reference_scores(package, items, pad_id)


def test_tokenizer_matches_transformers(qwen3_runtime, qwen3_package):
    from transformers import PreTrainedTokenizerFast

    reference = PreTrainedTokenizerFast(
        tokenizer_file=str(qwen3_package / "tokenizer.json")
    )
    for text in TEXTS:
        assert qwen3_runtime.lookup(None).model.tokenizer.encode(
            text
        ) == reference.encode(text, add_special_tokens=False)
