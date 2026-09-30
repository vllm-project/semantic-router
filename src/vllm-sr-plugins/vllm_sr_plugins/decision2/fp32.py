"""Opt-in float32 serving of Qwen3.5 hybrids, for numerics studies.

vLLM's chunked gated-delta kernel rejects float32 inputs, so ``--dtype float32``
fails at engine start. With ``VLLM_SR_GDN_BF16_INPUTS=1`` the plugin replaces
the ``chunk_gated_delta_rule`` custom op (``CustomOp.register_oot``) with one
that casts float32 q, k, v and beta to bfloat16 for that kernel only and returns
float32; the recurrent state stays float32 and every other layer runs in the
engine dtype. BF16 autocast in the package runtime feeds this kernel the same
bfloat16 inputs. Only the Triton path (``forward_native``, the ROCm backend) is
covered; bfloat16 engines are unaffected.
"""

from __future__ import annotations

import os

import torch

ENV = "VLLM_SR_GDN_BF16_INPUTS"
OP_NAME = "ChunkGatedDeltaRule"


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1"


def register_gdn_bf16_inputs() -> type:
    """Register the override once per process and return the registered class."""
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )

    if OP_NAME in op_registry_oot:
        return op_registry_oot[OP_NAME]

    class Bf16InputChunkGatedDeltaRule(ChunkGatedDeltaRule):
        def forward_native(
            self,
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            output_final_state,
            cu_seqlens=None,
            chunk_indices=None,
            chunk_offsets=None,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=None,
        ):
            if q.dtype != torch.float32:
                return super().forward_native(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    output_final_state,
                    cu_seqlens,
                    chunk_indices,
                    chunk_offsets,
                    use_qk_l2norm_in_kernel,
                    core_attn_out,
                )
            half = torch.bfloat16
            o, state = super().forward_native(
                q.to(half),
                k.to(half),
                v.to(half),
                g,
                beta.to(half),
                initial_state,
                output_final_state,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
                use_qk_l2norm_in_kernel,
                None,
            )
            o = o.to(q.dtype)
            if core_attn_out is not None:
                flat = o.reshape(-1)
                core_attn_out.reshape(-1)[: flat.numel()].copy_(flat)
            return o, state

    CustomOp.register_oot(name=OP_NAME)(Bf16InputChunkGatedDeltaRule)
    return Bf16InputChunkGatedDeltaRule
