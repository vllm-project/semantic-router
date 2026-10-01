"""Decision 2.0 on the text-only Qwen3.5 backbone, as a vLLM pooling model.

Served straight from a package directory (``--model <package>``, with
``--hf-config-path <package>/backbone``): the backbone weights come from
``backbone/*.safetensors`` and the head from ``decision_head.safetensors``.
There is no LM head. The backbone runs in the engine dtype with vLLM's own
Qwen3.5 layers and gated-delta kernels; the head is FP32.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsMRoPE,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase, Qwen3_5Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from ..compat import get_logger
from .contract import read_decision_config
from .head import CandidateHead
from .pooler import CandidatePooler

logger = get_logger("decision2.model")

HEAD_PREFIX = "decision_head."


class Decision2Qwen3_5ForScoring(
    nn.Module, HasInnerState, IsHybrid, SupportsLoRA, SupportsMRoPE
):
    is_pooling_model = True
    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping
    embedding_modules = {"embed_tokens": "input_embeddings"}
    hf_to_vllm_mapper = Qwen3_5ForCausalLMBase.hf_to_vllm_mapper
    allow_patterns_overrides = ["backbone/*.safetensors"]

    get_mamba_state_dtype_from_config = classmethod(
        Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config.__func__
    )
    get_mamba_state_shape_from_config = classmethod(
        Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config.__func__
    )
    get_mamba_state_copy_func = classmethod(
        Qwen3_5ForCausalLMBase.get_mamba_state_copy_func.__func__
    )
    get_mrope_input_positions = Qwen3_5ForCausalLMBase.get_mrope_input_positions

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("Qwen3.5 does not support 'all' prefix caching")
        root = Path(model_config.model)
        metadata = read_decision_config(root)
        self.config = config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.decision_head = CandidateHead(
            config.hidden_size, metadata["head_dim"], dtype=torch.float32
        )
        self.pooler = CandidatePooler(self.decision_head)
        self.secondary_weights = [
            DefaultModelLoader.Source(
                str(root),
                revision=None,
                prefix=HEAD_PREFIX,
                fall_back_to_pt=False,
                allow_patterns_overrides=["decision_head.safetensors"],
            )
        ]
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        logger.info(
            "Decision 2.0 scoring model: dtype=%s head=float32 mamba_ssm_cache_dtype=%s "
            "rope=%s",
            model_config.dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
            sorted(getattr(config, "rope_parameters", None) or {}),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        head: dict[str, torch.Tensor] = {}

        def backbone():
            # Packages store Qwen3_5TextModel keys, without the "model." prefix.
            for name, tensor in weights:
                if name.startswith(HEAD_PREFIX):
                    head[name[len(HEAD_PREFIX) :]] = tensor
                else:
                    yield f"model.{name}", tensor

        loaded = AutoWeightsLoader(self).load_weights(
            backbone(), mapper=self.hf_to_vllm_mapper
        )
        parameters = dict(self.decision_head.named_parameters())
        if set(head) != set(parameters):
            raise ValueError(
                "decision_head.safetensors does not match the candidate head: "
                f"missing {sorted(set(parameters) - set(head))}, "
                f"unexpected {sorted(set(head) - set(parameters))}"
            )
        with torch.no_grad():
            for name, tensor in head.items():
                target = parameters[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"decision head {name}: shape {tuple(tensor.shape)}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=torch.float32))
        loaded.update(HEAD_PREFIX + name for name in head)
        logger.info(
            "Decision 2.0 scoring model loaded %d parameters (%d in the head)",
            sum(p.numel() for p in self.parameters()),
            sum(p.numel() for p in self.decision_head.parameters()),
        )
        return loaded
