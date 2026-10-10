"""Fail-closed vLLM connector skeleton for mapper artifacts.

The current vLLM API passes connector hints through ``kv_transfer_params`` in
the JSON request body. HTTP hint headers need a separate trusted adapter before
the router can use this connector. Until the receive path is installed, this
connector reports zero external matches and vLLM always prefills normally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from safetensors import SafetensorError
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)

from src.kv_connector.runtime import MapperArtifact
from src.training.kv_mapper.artifact import CompatibilitySpec, Manifest

logger = logging.getLogger(__name__)


class EmptyMapperMetadata(KVConnectorMetadata):
    """C1 schedules no remote reads or writes."""


def _precision(dtype: Any) -> str:
    aliases = {"torch.bfloat16": "bf16", "torch.float16": "fp16"}
    value = str(dtype)
    if value not in aliases:
        raise ValueError(f"unsupported mapper serving dtype: {value}")
    return aliases[value]


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
        target_model=str(model.model),
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
    """Validate a mapper at startup and leave all requests on cold prefill."""

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self.artifact: MapperArtifact | None = None
        artifact_path = extra.get("artifact_path")
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

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        # Returning a hit before source availability and a complete receive
        # path are proven would make a cache miss a correctness failure.
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> EmptyMapperMetadata:
        return EmptyMapperMetadata()

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return
