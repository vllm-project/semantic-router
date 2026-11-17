"""Exercise a published mapper through the same-host GPU handoff core.

Run with an explicit artifact path on a host with two CUDA GPUs. This checks
the real artifact geometry and device transfers; it does not start vLLM or
claim an external cache hit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from src.kv_connector.handoff import apply_handoff
from src.kv_connector.paged_cache import extract_prefix
from src.kv_connector.runtime import MapperArtifact
from src.kv_connector.snapshot import LocalSnapshotStore, SourceSnapshot
from src.kv_connector.transform import qwen3_rope
from src.training.kv_mapper.artifact import Manifest

_REQUIRED_GPU_COUNT = 2


def run(
    artifact_path: Path,
    *,
    source_device: str,
    target_device: str,
    block_size: int,
    rope_theta: float,
) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < _REQUIRED_GPU_COUNT:
        raise RuntimeError("the GPU probe requires two CUDA devices")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    manifest = Manifest.from_dict(
        json.loads((artifact_path / "manifest.json").read_text())
    )
    artifact = MapperArtifact.open(artifact_path, manifest.compatibility)
    compat = artifact.manifest.compatibility
    if compat.precision != "bf16" or compat.source_tp != 1 or compat.target_tp != 1:
        raise ValueError("this probe requires a bf16 TP=1 mapper")
    source_count = (
        max(
            index
            for channels in artifact.manifest.source_layers_per_target.values()
            for sources in channels.values()
            for index in sources
        )
        + 1
    )
    target_count = len(artifact.manifest.source_layers_per_target["k"])
    tokens = block_size * 2
    positions = torch.arange(tokens, device=source_device)
    torch.manual_seed(2976)
    source_layers = {}
    for layer in range(source_count):
        key = (
            torch.randn(
                tokens,
                compat.num_kv_heads,
                compat.head_dim,
                device=source_device,
                dtype=torch.bfloat16,
            )
            / 100
        )
        value = torch.randn_like(key) / 100
        source_layers[layer] = (
            qwen3_rope(key, positions, theta=rope_theta),
            value,
        )
    snapshot = SourceSnapshot(
        namespace="gpu-probe",
        cache_id="published-artifact-probe",
        source_model=compat.source_model,
        source_revision=compat.source_revision,
        token_ids=tuple(range(tokens)),
        layers=source_layers,
        rope_theta=rope_theta,
        expires_at=time.time() + 1200,
    )
    width = compat.num_kv_heads * compat.head_dim
    caches = {
        layer: torch.zeros(
            (3, 2, block_size, width), device=target_device, dtype=torch.bfloat16
        )
        for layer in range(target_count)
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
        start = time.perf_counter()
        matched = apply_handoff(
            artifact,
            loaded,
            namespace=snapshot.namespace,
            cache_id=snapshot.cache_id,
            mapper_id=artifact.manifest.mapper_id,
            target_prompt_ids=list(range(tokens + 1)),
            target_caches=caches,
            block_ids=[2, 0],
            source_rope_theta=rope_theta,
            target_rope_theta=rope_theta,
        )
        torch.cuda.synchronize(target_device)
        elapsed = time.perf_counter() - start
    if matched != tokens:
        raise AssertionError(f"unexpected matched token count: {matched}")
    for layer, cache in caches.items():
        if not torch.isfinite(cache).all() or not cache.any():
            raise AssertionError(
                f"target layer {layer} was not populated with finite KV"
            )
    actual_key, actual_value = extract_prefix(
        caches[0],
        [2, 0],
        tokens,
        heads=compat.num_kv_heads,
        head_dim=compat.head_dim,
    )
    source_keys = {
        layer: qwen3_rope(key, torch.arange(tokens), theta=rope_theta, inverse=True)
        for layer, (key, _) in loaded.layers.items()
        if layer in artifact.manifest.source_layers_per_target["k"]["0"]
    }
    source_values = {layer: value for layer, (_, value) in loaded.layers.items()}
    expected_key = qwen3_rope(
        artifact.apply_layer(0, "k", source_keys, device="cpu", dtype=torch.bfloat16),
        torch.arange(tokens),
        theta=rope_theta,
    )
    expected_value = artifact.apply_layer(
        0, "v", source_values, device="cpu", dtype=torch.bfloat16
    )
    torch.testing.assert_close(actual_key.cpu(), expected_key, rtol=0.05, atol=0.05)
    torch.testing.assert_close(actual_value.cpu(), expected_value, rtol=0.05, atol=0.05)
    return {
        "mapper_id": artifact.manifest.mapper_id,
        "source_layers": source_count,
        "target_layers": target_count,
        "tokens": matched,
        "source_device": source_device,
        "target_device": target_device,
        "handoff_seconds": round(elapsed, 3),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=1_000_000)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.artifact,
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
