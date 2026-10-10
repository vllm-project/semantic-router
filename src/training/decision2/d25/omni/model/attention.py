"""Noncausal full attention for the Qwen3.5 text model.

The rule is the one in Perplexity's Apache-2.0 decider recipe: the 16 gated-attention layers see
every non-padding key (past and future), the Gated DeltaNet layers keep their causal recurrence and
padding mask. Training and inference must install the same hook.
"""

from __future__ import annotations

import torch

ATTENTION_MODES = ("causal", "noncausal_full_attention")


def enable_noncausal_full_attention(text_model: torch.nn.Module):
    """Install the mask hook on a ``Qwen3_5TextModel``; returns the hook handle."""
    from transformers.masking_utils import create_recurrent_attention_mask

    if text_model.config._attn_implementation != "sdpa":
        raise ValueError("noncausal_full_attention requires SDPA attention")

    def mask_inputs(module, args, kwargs):
        if args:
            raise ValueError("noncausal_full_attention requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("noncausal_full_attention does not support a KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(
                embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
            )
        if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
            raise ValueError("expected a 2D padding mask")
        if (
            padding.shape != embeddings.shape[:2]
            or not padding.bool().any(dim=-1).all()
        ):
            raise ValueError("padding mask must match a nonempty input")
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding
            ),
        }
        return args, kwargs

    return text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def apply_attention_mode(backbone: torch.nn.Module, mode: str):
    """Configure a ``Qwen3_5Model`` for ``mode``; returns the hook handle or None."""
    if mode not in ATTENTION_MODES:
        raise ValueError(f"unknown attention mode {mode!r}")
    if mode == "noncausal_full_attention":
        return enable_noncausal_full_attention(backbone.language_model)
    return None
