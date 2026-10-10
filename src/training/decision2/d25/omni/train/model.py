"""Trainable Omni decision model: ``Qwen3_5Model`` (or d3-edge's ``Qwen3VLModel``) backbone plus the
255-code readout.

The forward pass pools the last token and applies the readout in FP32. With ``vision_sync`` a batch
without images still runs the vision tower once on a 2x2-patch dummy image whose output joins the
logits with weight zero, so every rank executes (and back-propagates through) the same FSDP units
in every microbatch, whatever mix of text and image rows it received. The dummy leaves zero (not
missing) gradients on the vision parameters; the trainer drops them on steps without any image so
AdamW does not apply momentum or weight decay to an untouched tower.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from d25.omni.model import checkpoint
from d25.omni.model.attention import apply_attention_mode

DUMMY_GRID = (1, 2, 2)


class OmniDecisionModel(torch.nn.Module):
    def __init__(
        self, backbone, attention_mode: str = "causal", vision_sync: bool = True
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.text_config.hidden_size
        self.readout = torch.nn.Linear(hidden, checkpoint.NUM_CODES, bias=False)
        self.attention_mode = attention_mode
        self.vision_sync = vision_sync
        self.hook = apply_attention_mode(backbone, attention_mode)
        vision = backbone.config.vision_config
        self.patch_dim = (
            vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        )

    @classmethod
    def from_config(
        cls, config, attention_mode: str = "causal", vision_sync: bool = True
    ) -> "OmniDecisionModel":
        for part in (config, config.text_config, config.vision_config):
            part._attn_implementation = "sdpa"
        return cls(
            checkpoint.backbone_class(config)(config), attention_mode, vision_sync
        )

    @classmethod
    def from_checkpoint(
        cls,
        directory: str | Path,
        attention_mode: str | None = None,
        dtype: torch.dtype = torch.float32,
        vision_sync: bool = True,
    ) -> "OmniDecisionModel":
        decision = checkpoint.read_decision_config(directory)
        backbone, info = checkpoint.backbone_class(directory).from_pretrained(
            str(directory),
            dtype=dtype,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if info.get("missing_keys"):
            raise ValueError(
                f"{directory}: missing backbone weights {list(info['missing_keys'])[:5]}"
            )
        model = cls(
            backbone,
            attention_mode or decision.get("attention_mode", "causal"),
            vision_sync,
        )
        with torch.no_grad():
            model.readout.weight.copy_(checkpoint.load_readout(directory).to(dtype))
        return model.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor = None
        if pixel_values is None and self.vision_sync:
            dummy = torch.zeros(
                DUMMY_GRID[0] * DUMMY_GRID[1] * DUMMY_GRID[2],
                self.patch_dim,
                device=input_ids.device,
            )
            grid = torch.tensor([DUMMY_GRID], device=input_ids.device)
            anchor = (
                self.backbone.visual(dummy, grid_thw=grid).pooler_output.float().sum()
                * 0.0
            )
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        ).last_hidden_state
        logits = self.readout(hidden[:, -1].float())
        return logits if anchor is None else logits + anchor


def reset_nonpersistent_buffers(model: torch.nn.Module, device) -> None:
    """Recompute buffers absent from state dicts (rotary ``inv_freq``) after ``to_empty``."""
    for module in model.modules():
        names = getattr(module, "_non_persistent_buffers_set", set())
        if not names:
            continue
        if not hasattr(module, "config"):
            raise ValueError(
                f"cannot rebuild non-persistent buffers of {type(module).__name__}"
            )
        with torch.device("cpu"):
            fresh = type(module)(module.config)
        for name in names:
            getattr(module, name).copy_(getattr(fresh, name).to(device))


def save_checkpoint(
    directory: str | Path,
    state: dict[str, torch.Tensor],
    init: str | Path,
    decision_updates: dict[str, Any],
    provenance: dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Write a code-readout v1 checkpoint from a full state dict (``backbone.*``, ``readout.weight``).

    Returns the decision config and the per-tensor digests of the saved vision tower.
    """
    import shutil

    out, init = Path(directory), Path(init)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    config = checkpoint.load_config(init)
    expected = checkpoint.expected_backbone_shapes(config)
    writer = checkpoint.ShardWriter(tmp)
    vision_digests: dict[str, str] = {}
    readout = None
    for key in sorted(state):
        if key == "readout.weight":
            readout = state[key]
            continue
        if not key.startswith("backbone."):
            raise ValueError(f"unexpected state entry {key}")
        name = key[len("backbone.") :]
        if name not in expected:
            continue
        tensor = state[key].to(dtype) if state[key].is_floating_point() else state[key]
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {expected[name]}")
        writer.add(name, tensor)
        if name.startswith("visual."):
            vision_digests[name] = checkpoint.tensor_digest(tensor)
    shard_hashes = writer.close()
    missing = set(expected) - set(writer.digests)
    if missing or readout is None:
        raise ValueError(
            f"incomplete state: missing {sorted(missing)[:5]} readout={readout is not None}"
        )
    config.save_pretrained(str(tmp))
    checkpoint.copy_processor_files(init, tmp)
    readout_sha256 = checkpoint.save_readout(tmp, readout.float())
    encoder = {
        k: v
        for k, v in vision_digests.items()
        if checkpoint.component(k) == "vision_encoder"
    }
    merger = {
        k: v
        for k, v in vision_digests.items()
        if checkpoint.component(k) == "vision_merger"
    }
    decision = checkpoint.read_decision_config(init)
    decision.update(decision_updates)
    decision["provenance"] = {
        **provenance,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vision_encoder_digest": checkpoint.digest_of_digests(encoder),
        "vision_merger_digest": checkpoint.digest_of_digests(merger),
        "readout_sha256": readout_sha256,
        "shards_sha256": shard_hashes,
    }
    checkpoint.write_json(tmp / checkpoint.DECISION_CONFIG, decision)
    if out.exists():
        out.rmdir()
    tmp.replace(out)
    return decision, vision_digests


def init_vision_digests(init: str | Path) -> dict[str, str]:
    """Per-tensor digests of the init checkpoint's vision tower, for frozen-weight checks."""
    names = {k for k in checkpoint.weight_files(init) if k.startswith("visual.")}
    return {
        key: checkpoint.tensor_digest(t)
        for key, t in checkpoint.iter_tensors(init, names)
    }
