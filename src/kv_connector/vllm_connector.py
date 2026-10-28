"""Same-host vLLM KV transfer using a trusted shared snapshot directory.

The router's HTTP headers still need an adapter before routed reuse is enabled.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from src.kv_connector.handoff import apply_handoff
from src.kv_connector.paged_cache import extract_prefix
from src.kv_connector.runtime import MapperArtifact
from src.kv_connector.snapshot import LocalSnapshotStore, SourceSnapshot
from src.training.kv_mapper.artifact import CompatibilitySpec, Manifest
from src.training.kv_mapper.mapper_id import require_weight_commit

logger = logging.getLogger(__name__)
_HF_REPO_PARTS = 2
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class SaveJob:
    request_id: str
    namespace: str
    cache_id: str
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class LoadJob(SaveJob):
    mapper_id: str
    target_prompt_ids: tuple[int, ...]


@dataclass
class MapperMetadata(KVConnectorMetadata):
    saves: list[SaveJob] = field(default_factory=list)
    loads: list[LoadJob] = field(default_factory=list)


def _hint(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, dict):
        return None
    namespace, cache_id = value.get("namespace"), value.get("cache_id")
    mapper_id = value.get("mapper_id", "")
    if not all(isinstance(item, str) and item for item in (namespace, cache_id)):
        return None
    if not isinstance(mapper_id, str):
        return None
    return namespace, cache_id, mapper_id


def _layer_index(name: str) -> int | None:
    match = _LAYER.search(name)
    return int(match.group(1)) if match else None


def _new_request_hint(new_req: Any) -> tuple[str, str, str] | None:
    sampling = getattr(new_req, "sampling_params", None)
    extra_args = getattr(sampling, "extra_args", None) or {}
    return _hint(extra_args.get("kv_transfer_params"))


def _precision(dtype: Any) -> str:
    aliases = {"torch.bfloat16": "bf16", "torch.float16": "fp16"}
    value = str(dtype)
    if value not in aliases:
        raise ValueError(f"unsupported mapper serving dtype: {value}")
    return aliases[value]


def _model_name(model: Any) -> str:
    """Recover a pinned Hugging Face ID when vLLM loads its offline snapshot."""
    name = str(model.model)
    path = Path(name)
    if (
        path.parent.name == "snapshots"
        and path.name == str(model.revision)
        and path.parent.parent.name.startswith("models--")
    ):
        parts = path.parent.parent.name.removeprefix("models--").split("--")
        if len(parts) == _HF_REPO_PARTS and all(parts):
            return "/".join(parts)
    return name


def _rope_theta(hf_config: Any) -> float:
    """Accept only the unscaled Qwen3 RoPE used by the mapper."""
    scaling = getattr(hf_config, "rope_scaling", None)
    if scaling and (
        not isinstance(scaling, dict)
        or set(scaling) - {"type", "rope_type", "rope_theta"}
        or scaling.get("rope_type", scaling.get("type", "default")) != "default"
    ):
        raise ValueError("scaled RoPE is unsupported by the mapper")
    parameters = getattr(hf_config, "rope_parameters", None)
    if parameters is not None and (
        not isinstance(parameters, dict)
        or parameters.get("rope_type", parameters.get("type", "default")) != "default"
    ):
        raise ValueError("scaled RoPE is unsupported by the mapper")
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None and isinstance(parameters, dict):
        theta = parameters.get("rope_theta")
    if theta is None and isinstance(scaling, dict):
        theta = scaling.get("rope_theta")
    value = float(theta)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("RoPE theta must be positive and finite")
    for config in (parameters, scaling):
        if (
            isinstance(config, dict)
            and "rope_theta" in config
            and float(config["rope_theta"]) != value
        ):
            raise ValueError("RoPE theta differs between configuration fields")
    return value


def _deployment(
    vllm_config: Any, manifest: Manifest, extra: dict[str, Any]
) -> CompatibilitySpec:
    model = vllm_config.model_config
    hf = model.hf_config
    heads = int(hf.num_key_value_heads)
    head_dim = getattr(hf, "head_dim", None)
    if head_dim is None:
        head_dim = hf.hidden_size // hf.num_attention_heads
    return CompatibilitySpec(
        source_model=str(extra["source_model"]),
        source_revision=str(extra["source_revision"]),
        target_model=_model_name(model),
        target_revision=str(model.revision),
        variant=manifest.compatibility.variant,
        precision=_precision(model.dtype),
        source_tp=int(extra["source_tp"]),
        target_tp=int(vllm_config.parallel_config.tensor_parallel_size),
        head_order=str(extra["head_order"]),
        num_kv_heads=heads,
        head_dim=int(head_dim),
    )


class KVMapperConnector(KVConnectorBase_V1):
    """Use a block-aligned snapshot only after identity and prefix checks."""

    @property
    def requires_kv_delivery(self) -> bool:
        # A failed snapshot save is a later cache miss, not a failed inference.
        return False

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._producer = self._kv_transfer_config.is_kv_producer
        self._consumer = self._kv_transfer_config.is_kv_consumer
        self._block_size = int(
            getattr(getattr(vllm_config, "cache_config", None), "block_size", 16)
        )
        root = extra.get("snapshot_root")
        self.store = LocalSnapshotStore(Path(root)) if root else None
        self._ttl_seconds = float(extra.get("snapshot_ttl_seconds", 300))
        self._ready_loads: dict[str, tuple[str, str, str, tuple[int, ...], int]] = {}
        self._pending_loads: dict[str, tuple[str, str, str, tuple[int, ...], int]] = {}
        self._attempted: set[str] = set()
        self._invalid_blocks: set[int] = set()
        self._captured: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._target_caches: dict[int, torch.Tensor] = {}
        self.artifact: MapperArtifact | None = None
        self._source_identity: tuple[str, str] | None = None
        self._source_rope_theta: float | None = None
        self._target_rope_theta: float | None = None

        if self._producer and not self._consumer:
            try:
                model = vllm_config.model_config
                revision = require_weight_commit(str(model.revision))
                if int(vllm_config.parallel_config.tensor_parallel_size) != 1:
                    raise ValueError("source snapshot requires TP=1")
                if _precision(model.dtype) != "bf16":
                    raise ValueError("source snapshot requires bf16")
                if self._ttl_seconds <= 0:
                    raise ValueError("snapshot TTL must be positive")
                rope_theta = _rope_theta(model.hf_config)
                self._source_identity = _model_name(model), revision
                self._source_rope_theta = rope_theta
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning("Source snapshot disabled: %s", exc)
            return

        if (
            self._consumer
            and getattr(
                getattr(vllm_config, "cache_config", None),
                "enable_prefix_caching",
                True,
            )
            is not False
        ):
            logger.warning(
                "Mapper cache reuse requires consumer prefix caching to be "
                "disabled; set --no-enable-prefix-caching"
            )
            return

        artifact_path = extra.get("artifact_path")
        if getattr(vllm_config, "use_v2_model_runner", False):
            logger.warning(
                "Mapper cache reuse requires the vLLM V1 model runner for safe "
                "load-failure recomputation; set VLLM_USE_V2_MODEL_RUNNER=0"
            )
            return
        if not artifact_path:
            logger.warning("No mapper artifact configured; using normal prefill")
            return
        try:
            path = Path(artifact_path)
            manifest = Manifest.from_dict(
                json.loads((path / "manifest.json").read_text())
            )
            deployment = _deployment(vllm_config, manifest, extra)
            artifact = MapperArtifact.open(path, deployment)
            if len(artifact.manifest.source_layers_per_target["k"]) != int(
                vllm_config.model_config.hf_config.num_hidden_layers
            ):
                raise ValueError("mapper target layer count differs from running model")
            source_theta = extra.get("source_rope_theta")
            self._target_rope_theta = _rope_theta(vllm_config.model_config.hf_config)
            if source_theta is not None:
                self._source_rope_theta = float(source_theta)
                if (
                    not math.isfinite(self._source_rope_theta)
                    or self._source_rope_theta <= 0
                ):
                    raise ValueError("source RoPE theta must be positive and finite")
            self.artifact = artifact
        except (
            OSError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            SafetensorError,
        ) as exc:
            logger.warning("Mapper unavailable; using normal prefill: %s", exc)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._target_caches = {}
        for name, cache in kv_caches.items():
            index = _layer_index(name)
            if index is not None:
                if index in self._target_caches:
                    raise ValueError(f"duplicate KV cache for layer {index}")
                self._target_caches[index] = cache

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        if not (
            self._consumer
            and self.artifact
            and self.store
            and self._source_rope_theta
            and self._target_rope_theta
            and self._kv_transfer_config.kv_load_failure_policy == "recompute"
            and num_computed_tokens == 0
            and request is not None
            and not getattr(request, "num_preemptions", 0)
            and not getattr(request, "lora_request", None)
        ):
            return 0, False
        if request.request_id in self._attempted:
            return 0, False
        self._attempted.add(request.request_id)
        hint = _hint(getattr(request, "kv_transfer_params", None))
        if hint is None or hint[2] != self.artifact.manifest.mapper_id:
            return 0, False
        prompt = getattr(request, "prompt_token_ids", None)
        if not prompt or getattr(request, "mm_features", None):
            return 0, False
        namespace, cache_id, mapper_id = hint
        compat = self.artifact.manifest.compatibility
        try:
            snapshot = self.store.load(
                namespace,
                cache_id,
                source_model=compat.source_model,
                source_revision=compat.source_revision,
            )
            count = len(snapshot.token_ids)
            required_source_layers = 1 + max(
                index
                for layers in self.artifact.manifest.source_layers_per_target.values()
                for indices in layers.values()
                for index in indices
            )
            if (
                count >= len(prompt)
                or count % self._block_size
                or tuple(prompt[:count]) != snapshot.token_ids
                or len(snapshot.layers) < required_source_layers
                or snapshot.rope_theta != self._source_rope_theta
            ):
                return 0, False
        except (OSError, KeyError, TypeError, ValueError, SafetensorError) as exc:
            logger.info("No eligible source KV snapshot: %s", exc)
            return 0, False
        self._ready_loads[request.request_id] = (
            namespace,
            cache_id,
            mapper_id,
            tuple(prompt),
            count,
        )
        logger.info("External mapper prefix available: %d tokens", count)
        return count, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        del blocks
        candidate = self._ready_loads.pop(request.request_id, None)
        if candidate is not None and num_external_tokens == candidate[4]:
            self._pending_loads[request.request_id] = candidate

    def build_connector_meta(self, scheduler_output: Any) -> MapperMetadata:
        metadata = MapperMetadata()
        if self._producer and self._source_identity and self.store:
            for new_req in scheduler_output.scheduled_new_reqs:
                hint = _new_request_hint(new_req)
                prompt = new_req.prompt_token_ids
                scheduled = scheduler_output.num_scheduled_tokens.get(new_req.req_id, 0)
                if (
                    hint
                    and prompt
                    and not new_req.mm_features
                    and not new_req.lora_request
                    and new_req.num_computed_tokens == 0
                    and len(prompt) % self._block_size == 0
                    and scheduled >= len(prompt)
                ):
                    metadata.saves.append(
                        SaveJob(
                            new_req.req_id,
                            hint[0],
                            hint[1],
                            tuple(prompt),
                            tuple(new_req.block_ids[0]),
                        )
                    )
        if self._consumer:
            for new_req in scheduler_output.scheduled_new_reqs:
                candidate = self._pending_loads.pop(new_req.req_id, None)
                if candidate is None:
                    continue
                namespace, cache_id, mapper_id, prompt, count = candidate
                metadata.loads.append(
                    LoadJob(
                        new_req.req_id,
                        namespace,
                        cache_id,
                        prompt[:count],
                        tuple(new_req.block_ids[0][: count // self._block_size]),
                        mapper_id,
                        prompt,
                    )
                )
        return metadata

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        self._attempted.discard(request.request_id)
        self._ready_loads.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        return False, None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MapperMetadata)
        if not metadata.loads:
            return
        assert self.artifact is not None and self.store is not None
        compat = self.artifact.manifest.compatibility
        for job in metadata.loads:
            try:
                snapshot = self.store.load(
                    job.namespace,
                    job.cache_id,
                    source_model=compat.source_model,
                    source_revision=compat.source_revision,
                )
                if job.token_ids != snapshot.token_ids:
                    raise ValueError("source snapshot changed after scheduler lookup")
                if snapshot.rope_theta != self._source_rope_theta:
                    raise ValueError("source RoPE theta changed after scheduler lookup")
                matched = apply_handoff(
                    self.artifact,
                    snapshot,
                    namespace=job.namespace,
                    cache_id=job.cache_id,
                    mapper_id=job.mapper_id,
                    target_prompt_ids=list(job.target_prompt_ids),
                    target_caches=self._target_caches,
                    block_ids=list(job.block_ids),
                    source_rope_theta=self._source_rope_theta,
                    target_rope_theta=self._target_rope_theta,
                    cache_layout="lbnhc",
                )
                logger.info("Applied mapper KV prefix: %d tokens", matched)
            except Exception:
                logger.exception("Mapper KV load failed; vLLM must recompute")
                self._invalid_blocks.update(job.block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid = self._invalid_blocks
        self._invalid_blocks = set()
        return invalid

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:
        del attn_metadata, kwargs
        if not self._producer or not self._source_identity or not self.store:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MapperMetadata)
        index = _layer_index(layer_name)
        if index is None:
            return
        hf = self._vllm_config.model_config.hf_config
        heads = int(hf.num_key_value_heads)
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            head_dim = hf.hidden_size // hf.num_attention_heads
        for job in metadata.saves:
            try:
                key, value = extract_prefix(
                    kv_layer,
                    list(job.block_ids),
                    len(job.token_ids),
                    heads=heads,
                    head_dim=int(head_dim),
                    layout="lbnhc",
                )
                self._captured.setdefault(job.request_id, {})[index] = (
                    key.detach().cpu().contiguous(),
                    value.detach().cpu().contiguous(),
                )
            except Exception:
                logger.exception(
                    "Could not capture source KV layer %d with shape %s",
                    index,
                    tuple(kv_layer.shape),
                )
                self._captured.pop(job.request_id, None)

    def wait_for_save(self) -> None:
        if not self._producer or not self._source_identity or not self.store:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MapperMetadata)
        expected = int(self._vllm_config.model_config.hf_config.num_hidden_layers)
        for job in metadata.saves:
            layers = self._captured.pop(job.request_id, {})
            if set(layers) != set(range(expected)):
                logger.warning("Incomplete source KV capture for %s", job.request_id)
                continue
            try:
                snapshot = SourceSnapshot(
                    namespace=job.namespace,
                    cache_id=job.cache_id,
                    source_model=self._source_identity[0],
                    source_revision=self._source_identity[1],
                    token_ids=job.token_ids,
                    layers=layers,
                    rope_theta=self._source_rope_theta,
                    expires_at=time.time() + self._ttl_seconds,
                )
                self.store.publish(snapshot)
                logger.info(
                    "Published source KV snapshot: %d tokens", len(job.token_ids)
                )
            except (OSError, ValueError):
                logger.exception("Could not publish source KV snapshot")
