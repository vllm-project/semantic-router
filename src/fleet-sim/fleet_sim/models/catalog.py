"""Built-in fleet-sim model catalog."""

from __future__ import annotations

from .spec import ModelSpec

LLAMA_3_1_8B = ModelSpec(
    name="llama-3.1-8b",
    aliases=("meta-llama/Meta-Llama-3.1-8B",),
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=128256,
    max_position=131072,
)

LLAMA_3_1_70B = ModelSpec(
    name="llama-3.1-70b",
    aliases=("meta-llama/Meta-Llama-3.1-70B",),
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    hidden_size=8192,
    intermediate_size=28672,
    vocab_size=128256,
    max_position=131072,
)

LLAMA_3_1_405B = ModelSpec(
    name="llama-3.1-405b",
    aliases=("meta-llama/Meta-Llama-3.1-405B",),
    n_layers=126,
    n_heads=128,
    n_kv_heads=8,
    hidden_size=16384,
    intermediate_size=53248,
    vocab_size=128256,
    max_position=131072,
)

MISTRAL_7B = ModelSpec(
    name="mistral-7b",
    aliases=("mistralai/Mistral-7B-v0.3",),
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=32000,
    max_position=32768,
)

MIXTRAL_8X7B = ModelSpec(
    name="mixtral-8x7b",
    aliases=("mistralai/Mixtral-8x7B-v0.1",),
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=32000,
    max_position=32768,
    moe_intermediate_size=14336,
    n_experts=8,
    n_experts_topk=2,
)

QWEN3_8B = ModelSpec(
    name="qwen3-8b",
    aliases=("Qwen/Qwen3-8B",),
    n_layers=36,
    n_heads=32,
    n_kv_heads=8,
    hidden_size=4096,
    intermediate_size=12288,
    vocab_size=151936,
    max_position=40960,
)

QWEN3_32B = ModelSpec(
    name="qwen3-32b",
    aliases=("Qwen/Qwen3-32B",),
    n_layers=64,
    n_heads=40,
    n_kv_heads=8,
    hidden_size=5120,
    intermediate_size=27648,
    vocab_size=151936,
    max_position=40960,
)

QWEN3_235B_A22B = ModelSpec(
    name="qwen3-235b-a22b",
    aliases=("qwen3-235b", "Qwen/Qwen3-235B-A22B"),
    n_layers=94,
    n_heads=64,
    n_kv_heads=4,
    hidden_size=4096,
    intermediate_size=0,
    vocab_size=151936,
    max_position=40960,
    moe_intermediate_size=1536,
    n_experts=128,
    n_experts_topk=8,
)

QWEN3_30B_A3B = ModelSpec(
    name="qwen3-30b-a3b",
    aliases=("qwen3-30b", "Qwen/Qwen3-30B-A3B"),
    n_layers=48,
    n_heads=16,
    n_kv_heads=4,
    hidden_size=2048,
    intermediate_size=0,
    vocab_size=151936,
    max_position=40960,
    moe_intermediate_size=768,
    n_experts=128,
    n_experts_topk=8,
)

DEEPSEEK_V3 = ModelSpec(
    name="deepseek-v3",
    aliases=("deepseek-ai/DeepSeek-V3",),
    n_layers=61,
    n_heads=128,
    n_kv_heads=128,
    hidden_size=7168,
    intermediate_size=0,
    vocab_size=129280,
    max_position=163840,
    moe_intermediate_size=2048,
    n_experts=256,
    n_experts_topk=8,
    n_shared_experts=1,
)

_CATALOG = (
    LLAMA_3_1_8B,
    LLAMA_3_1_70B,
    LLAMA_3_1_405B,
    MISTRAL_7B,
    MIXTRAL_8X7B,
    QWEN3_8B,
    QWEN3_32B,
    QWEN3_235B_A22B,
    QWEN3_30B_A3B,
    DEEPSEEK_V3,
)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


_LOOKUP: dict[str, ModelSpec] = {}
for spec in _CATALOG:
    keys = {spec.name, spec.display_name, *spec.aliases}
    for key in keys:
        _LOOKUP[_normalize(key)] = spec


def get(name: str) -> ModelSpec:
    key = _normalize(name)
    if key not in _LOOKUP:
        raise KeyError(f"Unknown model '{name}'. Available: {list_names()}")
    return _LOOKUP[key]


def list_names() -> list[str]:
    return sorted(spec.name for spec in _CATALOG)
