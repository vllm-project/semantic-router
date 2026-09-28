"""Exact full-window attention with bounded query workspace in portable ONNX."""

import torch
from torch import Tensor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


@torch.jit.script
def query_block_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    scale: float,
    block: int = 256,
) -> Tensor:
    """Partition queries only; every query retains the entire original key axis.

    TorchScript preserves the dynamic loop in ONNX instead of tracing the short
    export example. Maximum score workspace is B*H*256*sequence, not B*H*S*S.
    The list becomes an ONNX sequence and concatenation; no runtime Python runs.
    """
    length = query.size(2)
    keys = torch.arange(key.size(2), device=query.device)
    result = torch.jit.annotate(list[Tensor], [])
    for start in range(0, length, block):
        end = min(start + block, length)
        positions = torch.arange(start, end, device=query.device)
        allowed = (keys.unsqueeze(0) <= positions.unsqueeze(1))[None, None] & (
            mask[:, None, None, :] != 0
        )
        scores = torch.matmul(query[:, :, start:end, :], key.transpose(-2, -1))
        scores = (scores * scale).masked_fill(~allowed, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        # SDPA returns zero for rows with no visible key (e.g. left padding).
        probabilities = torch.where(
            allowed.any(-1, keepdim=True),
            probabilities,
            torch.zeros_like(probabilities),
        )
        result.append(torch.matmul(probabilities, value))
    return torch.cat(result, dim=2)


def export_attention(module, query, key, value, attention_mask, scaling, **kwargs):

    if module.training or module.sliding_window is not None:
        raise ValueError("bounded export requires full-window evaluation attention")
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    output = query_block_attention(query, key, value, attention_mask, scaling)
    return output.transpose(1, 2).contiguous(), None


def install_export_attention(encoder):

    ALL_ATTENTION_FUNCTIONS.register("vela_export_query_blocks", export_attention)
    encoder.config._attn_implementation = "vela_export_query_blocks"
