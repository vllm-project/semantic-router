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


@pytest.fixture(scope="module", params=["d3", "d3_pruned"])
def case(request):
    """The runtime, the package and the reference model, for a package and one with a pruned backbone."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    package = request.getfixturevalue(f"{request.param}_package")
    runtime = request.getfixturevalue(f"{request.param}_runtime")
    model = Qwen3_5Model.from_pretrained(
        str(package), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    enable_noncausal(model.language_model)
    embed = model.visual.patch_embed
    embed.forward = types.MethodType(linear_patch_embed, embed)
    return runtime, package, model


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
    media = items[0].media
    if media is not None:
        config = json.loads((package / "config.json").read_text())
        inputs["pixel_values"] = torch.cat([media.pixel_rows()] * len(items))
        inputs["image_grid_thw"] = torch.tensor(
            [image.grid for _ in items for image in media.images]
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


def test_text_passes_match_transformers(case):
    runtime, package, reference = case
    model = runtime.lookup(None).model
    plan = model.plan(
        "Merge two sorted lists in Python and explain the cost. " * 3, QUESTIONS
    )
    short = model.plan("hi", {"q": {"type": "noul", "instructions": "Greeting?"}})
    for items in (plan.items, plan.items[:1], plan.items[1:], short.items):
        assert model.run(items) == reference_probabilities(reference, package, items)


def test_image_passes_match_transformers(case, png_url):
    runtime, package, reference = case
    model = runtime.lookup(None).model
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
                reference, package, items
            )


def video_reference(model, package, items):
    """The d3 runtime's video pass: the request's image and video features computed once and repeated per row."""
    config = json.loads((package / "config.json").read_text())
    decision = json.loads((package / "decision_config.json").read_text())
    readout = load_file(str(package / "readout.safetensors"))["weight"].float()
    media = items[0].media
    rows = len(items)
    width = max(len(item.ids) for item in items)
    ids = torch.full((rows, width), 0, dtype=torch.long)
    mask = torch.zeros((rows, width), dtype=torch.long)
    for row, item in enumerate(items):
        ids[row, width - len(item.ids) :] = torch.tensor(item.ids)
        mask[row, width - len(item.ids) :] = 1
    with torch.inference_mode():
        embeds = model.get_input_embeddings()(ids)
        image_grid = None
        if media.images:
            grids = torch.tensor([image.grid for image in media.images])
            features = model.get_image_features(
                media.pixel_rows(), grids, return_dict=True
            ).pooler_output
            values = torch.cat(list(features) * rows).to(embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(
                ids, inputs_embeds=embeds, image_features=values
            )
            embeds = embeds.masked_scatter(image_mask, values)
            image_grid = grids.repeat(rows, 1)
        grids = torch.tensor([clip.grid for clip in media.videos])
        features = model.get_video_features(
            torch.cat([clip.pixel_values for clip in media.videos]),
            grids,
            return_dict=True,
        ).pooler_output
        values = torch.cat(list(features) * rows).to(embeds.dtype)
        _, video_mask = model.get_placeholder_mask(
            ids, inputs_embeds=embeds, video_features=values
        )
        embeds = embeds.masked_scatter(video_mask, values)
        types_ = (ids == config["image_token_id"]).to(torch.int32) + 2 * (
            ids == config["video_token_id"]
        ).to(torch.int32)
        positions = model.compute_3d_position_ids(
            input_ids=ids,
            image_grid_thw=image_grid,
            video_grid_thw=grids.repeat(rows, 1),
            inputs_embeds=embeds,
            attention_mask=mask,
            past_key_values=None,
            mm_token_type_ids=types_,
        )
        hidden = model.language_model(
            input_ids=None,
            position_ids=positions,
            attention_mask=mask,
            past_key_values=None,
            inputs_embeds=embeds,
            use_cache=False,
        ).last_hidden_state[:, -1]
        logits = hidden.float() @ readout.T
        counts = torch.tensor([len(item.keys) for item in items])[:, None]
        invalid = torch.arange(pkg.MAX_OPTIONS)[None] >= counts
        probs = (
            logits.masked_fill(invalid, float("-inf")) / decision["temperature"]
        ).softmax(-1)
    return [
        row[: len(item.keys)] for row, item in zip(probs.tolist(), items, strict=True)
    ]


def test_video_passes_match_transformers(case, mp4_url, png_url):
    runtime, package, reference = case
    model = runtime.lookup(None).model
    clips = [
        mp4_url(96, 64, frames=12, fps=8.0, seed=1),
        mp4_url(64, 96, frames=5, seed=4),
    ]
    for images, videos in (([], clips[:1]), ([png_url(80, 60, 5)], clips)):
        plan = model.plan_images("What happens in the clip?", QUESTIONS, images, videos)
        for items in (plan.items, plan.items[:1]):
            assert model.run(items) == video_reference(reference, package, items)
