"""The native Decision 3.0 path is bit-identical to Transformers 5.17 as the released d3 runtime drives it.

The reference is ``Qwen3_5Model`` in BF16 with SDPA attention, the released
runtime's noncausal full-attention hook (adapted from
perplexity-ai/pplx-decider-v1.1-27b, Apache-2.0) and its matrix-product patch
embedding, read at the last position by the FP32 readout.
"""

import json
import types

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from vllm_srun.families.decision3 import package as pkg

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.reference

QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {"code": "Programming", "math": None, "other": {"x": [1, 2]}},
    },
    "reasoning": {"type": "noul", "instructions": "Does this need reasoning?"},
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", "Moderate", "Hard", "Expert"],
    },
}


def enable_noncausal(text_model):
    from transformers.masking_utils import create_recurrent_attention_mask

    def mask_inputs(module, args, kwargs):
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs["attention_mask"]
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def linear_patch_embed(self, hidden_states):
    weight = self.proj.weight
    flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    return F.linear(flat, weight.reshape(weight.shape[0], -1), self.proj.bias).view(
        -1, self.embed_dim
    )


@pytest.fixture(scope="module")
def reference(d3_package):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    model = Qwen3_5Model.from_pretrained(
        str(d3_package), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    enable_noncausal(model.language_model)
    embed = model.visual.patch_embed
    embed.forward = types.MethodType(linear_patch_embed, embed)
    return model


def reference_probabilities(model, package, items):
    decision = json.loads((package / "decision_config.json").read_text())
    readout = load_file(str(package / "readout.safetensors"))["weight"].float()
    width = max(len(item.ids) for item in items)
    ids = torch.full((len(items), width), 0, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for row, item in enumerate(items):
        ids[row, width - len(item.ids) :] = torch.tensor(item.ids)
        mask[row, width - len(item.ids) :] = 1
    inputs = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
    images = items[0].images
    if images is not None:
        config = json.loads((package / "config.json").read_text())
        inputs["pixel_values"] = torch.cat([images.pixel_rows()] * len(items))
        inputs["image_grid_thw"] = torch.tensor(
            [image.grid for _ in items for image in images.images]
        )
        inputs["mm_token_type_ids"] = (ids == config["image_token_id"]).to(torch.int32)
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state[:, -1]
        logits = hidden.float() @ readout.T
        counts = torch.tensor([len(item.keys) for item in items])[:, None]
        invalid = torch.arange(pkg.MAX_OPTIONS)[None] >= counts
        probs = (
            logits.masked_fill(invalid, float("-inf")) / decision["temperature"]
        ).softmax(-1)
    return [
        row[: len(item.keys)] for row, item in zip(probs.tolist(), items, strict=True)
    ]


def test_text_passes_match_transformers(d3_runtime, d3_package, reference):
    model = d3_runtime.lookup(None).model
    plan = model.plan(
        "Merge two sorted lists in Python and explain the cost. " * 3, QUESTIONS
    )
    short = model.plan("hi", {"q": {"type": "noul", "instructions": "Greeting?"}})
    for items in (plan.items, plan.items[:1], plan.items[1:], short.items):
        assert model.run(items) == reference_probabilities(reference, d3_package, items)


def test_image_passes_match_transformers(d3_runtime, d3_package, reference, png_url):
    model = d3_runtime.lookup(None).model
    images = [
        png_url(300, 200, seed=1),
        png_url(64, 64, seed=2),
        png_url(640, 96, seed=3),
    ]
    for count in (1, 3):
        plan = model.plan_images(
            "What does the picture show?", QUESTIONS, images[:count]
        )
        for items in (plan.items, plan.items[:1]):
            assert model.run(items) == reference_probabilities(
                reference, d3_package, items
            )
