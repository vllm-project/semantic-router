"""d3-edge training stages through the Omni multimodal trainer, on one GPU.

    python -m d25.family.edge_train --init INIT --arm E1-text --rows M2T-d3/train-*.jsonl.gz \
        --dev-rows M2T-d3/dev.jsonl.gz --lr 1e-5 --warmup-ratio 0.15 --out OUT [d25.omni.train.train arguments]

``INIT`` is the composite of ``d25.family.edge build`` or a checkpoint trained from it (init kind
``edge``). The arms set the trainable groups (learning-rate scales; 0 freezes the group):

| Arm | encoder | merger | language | readout | stage |
| --- | --- | --- | --- | --- | --- |
| E0-merger | 0 | 1 | 0 | 0 | merger alignment on image rows, language model frozen |
| E1-text | 0 | 0 | 1 | 1 | full fine-tune of the language model and readout on text rows |
| E2-joint | 0 | 1 | 1 | 1 | image rows with text replay |

On one process the Omni trainer keeps FP32 weights and the AdamW state in FP32; here the backbone
runs under BF16 autocast on the accelerator while the readout and the loss stay FP32 (the mixed
precision FSDP2 applies on several GPUs). The vision patch embedding runs as a matrix product
(``d25.omni.model.patch_embed``): same weights and result, no per-image-size Conv3d search on ROCm.

Batches arrive left-padded with a padding mask; the model runs them right-padded without a mask
instead, so causal attention sees exactly each row's tokens on the fused (flash) SDPA path rather
than a masked one, and the readout pools each row's last real token. The function is the same:
positions start at 0 in every row (M-RoPE positions of image rows are computed per row as before) and
no real token attends to padding.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace

import torch

from d25.omni.model.patch_embed import linearize_patch_embed
from d25.omni.train import arms, fsdp
from d25.omni.train import collator as collator_module
from d25.omni.train.model import DUMMY_GRID

EDGE_ARMS = {
    "E0-merger": arms.Arm(
        init="edge", encoder=0.0, merger=1.0, language=0.0, readout=0.0
    ),
    "E1-text": arms.Arm(
        init="edge", encoder=0.0, merger=0.0, language=1.0, readout=1.0
    ),
    "E2-joint": arms.Arm(
        init="edge", encoder=0.0, merger=1.0, language=1.0, readout=1.0
    ),
}

_build = fsdp.build
_collate = collator_module.MultimodalCollator.__call__


def collate_unplanned(self, rows, dummy: bool = False):
    """The collator without its planned-length check: shuffled options can change a prompt's token count
    by one (BPE merges at the option boundaries), which only moves the microbatch plan by a token.
    """
    return _collate(self, [replace(row, tokens=0) for row in rows], dummy=dummy)


def _autocast_backbone(model: torch.nn.Module, device_type: str) -> None:
    forward = model.backbone.forward

    def run(*args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return forward(*args, **kwargs)

    model.backbone.forward = run


def right_padded_forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> torch.Tensor:
    """``OmniDecisionModel.forward`` for left-padded batches, run right-padded without a mask."""
    lengths = attention_mask.sum(-1)
    width = input_ids.shape[1]
    order = (
        torch.arange(width, device=input_ids.device)[None] + (width - lengths)[:, None]
    ) % width
    input_ids = input_ids.gather(1, order)
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.gather(1, order)
    anchor = None
    if pixel_values is None and self.vision_sync:
        dummy = torch.zeros(
            DUMMY_GRID[0] * DUMMY_GRID[1] * DUMMY_GRID[2],
            self.patch_dim,
            device=input_ids.device,
        )
        grid = torch.tensor([DUMMY_GRID], device=input_ids.device)
        anchor = (
            self.backbone.visual(dummy, grid_thw=grid).pooler_output.float().sum() * 0.0
        )
    hidden = self.backbone(
        input_ids=input_ids,
        attention_mask=None,
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=False,
    ).last_hidden_state
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    logits = self.readout(pooled.float())
    return logits if anchor is None else logits + anchor


OPTIONS = {"masked_forward": False}


def build(init, distributed, *args, **kwargs):
    model, counts = _build(init, distributed, *args, **kwargs)
    if not linearize_patch_embed(model.backbone):
        raise RuntimeError("vision patch embedding not found")
    if model.attention_mode != "causal":
        raise ValueError("d3-edge trains with causal attention")
    if not OPTIONS["masked_forward"]:
        model.forward = types.MethodType(right_padded_forward, model)
    if not distributed.sharded and distributed.device.type == "cuda":
        _autocast_backbone(model, "cuda")
    return model, counts


def main(argv: list[str] | None = None) -> None:
    """``--masked-forward`` keeps the Omni trainer's left-padded masked forward; the rest goes to it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--masked-forward" in argv:
        argv.remove("--masked-forward")
        OPTIONS["masked_forward"] = True
    arms.ARMS.update(EDGE_ARMS)
    fsdp.build = build
    collator_module.MultimodalCollator.__call__ = collate_unplanned
    from d25.omni.train import train

    train.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
