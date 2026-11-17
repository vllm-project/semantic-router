"""Model-backed Qwen3 handoff probe through the C2 paged-cache path.

This uses Transformers to export source KV and decode with the mapped target
cache. It does not exercise vLLM's scheduler or report a vLLM cache hit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.kv_connector.handoff import apply_handoff
from src.kv_connector.paged_cache import extract_prefix
from src.kv_connector.runtime import MapperArtifact
from src.kv_connector.snapshot import LocalSnapshotStore, SourceSnapshot
from src.training.kv_mapper.artifact import Manifest


def _load_model(path: Path, device: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    return model.to(device).eval()


@torch.inference_mode()
def run(
    artifact_path: Path,
    source_path: Path,
    target_path: Path,
    *,
    source_device: str,
    target_device: str,
    block_size: int,
    rope_theta: float,
) -> dict[str, object]:
    if block_size <= 0:
        raise ValueError("block size must be positive")
    manifest = Manifest.from_dict(
        json.loads((artifact_path / "manifest.json").read_text())
    )
    artifact = MapperArtifact.open(artifact_path, manifest.compatibility)
    compat = artifact.manifest.compatibility
    if compat.precision != "bf16" or compat.source_tp != 1 or compat.target_tp != 1:
        raise ValueError("model probe requires a bf16 TP=1 artifact")
    source_tokenizer = AutoTokenizer.from_pretrained(source_path)
    target_tokenizer = AutoTokenizer.from_pretrained(target_path)
    if source_tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise ValueError("source and target tokenizer vocabularies differ")
    source = _load_model(source_path, source_device)
    target = _load_model(target_path, target_device)
    if len(source.model.layers) <= max(
        index
        for layers in artifact.manifest.source_layers_per_target.values()
        for sources in layers.values()
        for index in sources
    ):
        raise ValueError("source layer count differs from mapper")
    target_layers = len(artifact.manifest.source_layers_per_target["k"])
    if len(target.model.layers) != target_layers:
        raise ValueError("target layer count differs from mapper")
    prompt = (
        "In a quiet research laboratory, two language models share the same "
        "tokenizer but have different layer counts. The goal is to transfer "
        "a cache safely and continue the sentence. "
    ) * 8
    token_ids = target_tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= block_size * 2:
        raise ValueError("probe prompt is too short")
    prefix = token_ids[: block_size * 2]
    next_token = token_ids[len(prefix)]
    source_ids = torch.tensor([prefix], device=source_device)
    source_cache = source(input_ids=source_ids, use_cache=True).past_key_values
    source_layers = {
        index: (
            layer.keys[0].transpose(0, 1).contiguous(),
            layer.values[0].transpose(0, 1).contiguous(),
        )
        for index, layer in enumerate(source_cache.layers)
    }
    snapshot = SourceSnapshot(
        namespace="model-probe",
        cache_id="first-prefix",
        source_model=compat.source_model,
        source_revision=compat.source_revision,
        token_ids=tuple(prefix),
        layers=source_layers,
        rope_theta=rope_theta,
        expires_at=time.time() + 1200,
    )
    width = compat.num_kv_heads * compat.head_dim
    caches = {
        layer: torch.zeros(
            (3, 2, block_size, width), device=target_device, dtype=torch.bfloat16
        )
        for layer in range(target_layers)
    }
    with tempfile.TemporaryDirectory() as folder:
        store = LocalSnapshotStore(Path(folder))
        store.publish(snapshot)
        loaded = store.load(
            snapshot.namespace,
            snapshot.cache_id,
            source_model=compat.source_model,
            source_revision=compat.source_revision,
        )
        torch.cuda.synchronize(target_device)
        start = time.perf_counter()
        matched = apply_handoff(
            artifact,
            loaded,
            namespace=snapshot.namespace,
            cache_id=snapshot.cache_id,
            mapper_id=artifact.manifest.mapper_id,
            target_prompt_ids=[*prefix, next_token],
            target_caches=caches,
            block_ids=[2, 0],
            source_rope_theta=rope_theta,
            target_rope_theta=rope_theta,
        )
        torch.cuda.synchronize(target_device)
        handoff_seconds = time.perf_counter() - start
    if matched != len(prefix):
        raise AssertionError("handoff matched a different token count")
    target_cache = DynamicCache()
    for layer in range(target_layers):
        key, value = extract_prefix(
            caches[layer],
            [2, 0],
            len(prefix),
            heads=compat.num_kv_heads,
            head_dim=compat.head_dim,
        )
        target_cache.update(
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            layer,
        )
    continuation_id = torch.tensor([[next_token]], device=target_device)
    mapped_logits = (
        target(
            input_ids=continuation_id,
            past_key_values=target_cache,
            use_cache=True,
        )
        .logits[0, -1]
        .float()
    )
    cold_ids = torch.tensor([[*prefix, next_token]], device=target_device)
    cold_logits = target(input_ids=cold_ids, use_cache=False).logits[0, -1].float()
    if not torch.isfinite(mapped_logits).all() or not torch.isfinite(cold_logits).all():
        raise AssertionError("model continuation produced non-finite logits")
    cold_top = int(cold_logits.argmax())
    cold_probs = cold_logits.softmax(-1)
    mapped_probs = mapped_logits.softmax(-1)
    return {
        "artifact": artifact.manifest.mapper_id,
        "source_model": compat.source_model,
        "target_model": compat.target_model,
        "prefix_tokens": matched,
        "handoff_seconds": round(handoff_seconds, 3),
        "cold_top_token_id": cold_top,
        "mapped_top_token_id": int(mapped_logits.argmax()),
        "cold_top_probability": round(float(cold_probs[cold_top]), 6),
        "mapped_probability_of_cold_top": round(float(mapped_probs[cold_top]), 6),
        "logit_cosine_similarity": round(
            float(
                torch.nn.functional.cosine_similarity(
                    cold_logits[None], mapped_logits[None]
                )
            ),
            6,
        ),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=1_000_000)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.artifact,
                args.source_model,
                args.target_model,
                source_device=args.source_device,
                target_device=args.target_device,
                block_size=args.block_size,
                rope_theta=args.rope_theta,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
