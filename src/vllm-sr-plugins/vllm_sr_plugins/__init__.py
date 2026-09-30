"""vLLM plugins that serve vLLM Semantic Router models without modifying vLLM.

``register`` is the ``vllm.general_plugins`` entry point. vLLM calls it in
every process (API server, engine core, workers), possibly more than once, so
it only registers lazily imported model classes and is idempotent.
"""

from __future__ import annotations

DECISION2_QWEN3_5 = "Decision2Qwen3_5ForScoring"
_MODELS = {
    DECISION2_QWEN3_5: "vllm_sr_plugins.decision2.model:Decision2Qwen3_5ForScoring",
}


def register() -> None:
    from vllm import ModelRegistry

    from .compat import get_logger

    supported = set(ModelRegistry.get_supported_archs())
    for architecture, target in _MODELS.items():
        if architecture not in supported:
            ModelRegistry.register_model(architecture, target)
    from .decision2 import fp32

    if fp32.enabled():
        fp32.register_gdn_bf16_inputs()
        get_logger("register").warning(
            "%s=1: float32 engines feed the chunked gated-delta kernel bfloat16 inputs",
            fp32.ENV,
        )
    # vLLM keys its Qwen3.5 text-model config fix-ups by architecture name: the
    # gated-delta state dtype follows the checkpoint's mamba_ssm_dtype (FP32),
    # and text-only checkpoints drop the multimodal M-RoPE sections.
    try:
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            Qwen3_5ForCausalLMConfig,
        )
    except ImportError:
        get_logger("register").warning(
            "vLLM has no Qwen3_5ForCausalLMConfig; pass --mamba-ssm-cache-dtype float32 "
            "and serve with 1-D positions"
        )
        return
    MODELS_CONFIG_MAP.setdefault(DECISION2_QWEN3_5, Qwen3_5ForCausalLMConfig)
