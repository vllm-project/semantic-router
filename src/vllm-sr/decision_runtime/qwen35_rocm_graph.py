"""Opt-in, instance-local ROCm graphs for short Qwen physical batches.

Only the text backbone is captured. Candidate gathering, the FP32 head,
calibration, and host synchronization remain eager. Every shape is qualified
against the ordinary two-dimensional-mask forward before it may be replayed.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen35_rocm_binder import (
    QwenRocmBindingError,
    assert_qwen_rocm_graph_replay_safe,
)
from .qwen35_torch import Qwen35TorchRuntime, QwenRocmProfileBinding

_PHYSICAL_BATCH = 8
_INPUT_RANK = 2
_PADDED_TOKEN_MULTIPLE = 32
_SHA256_HEX_LENGTH = 64
_MAX_PADDED_TOKENS = 256
_MAX_GRAPHS = 2
_MAX_CAPTURE_BYTES = 512 << 20
_MAX_TOTAL_CAPTURE_BYTES = 768 << 20
_MIN_FREE_HBM_BYTES = 8 << 30
_MAX_PARITY_WARNINGS = 4
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _GraphKey:
    artifact_content_id: str
    source_sha256: tuple[str, str, str]
    model_instance: int
    device: str
    profile_sha256: str
    kernel_config_sha256: str
    guard_contract_sha256: str
    physical_batch: int
    padded_tokens: int
    attention: str
    mask_layout: tuple[tuple[str, tuple[int, ...] | None, str | None], ...]


@dataclass(slots=True)
class _GraphEntry:
    graph: Any
    hidden: Any
    static_ids: Any
    static_masks: dict[str, Any | None]
    allocated_bytes: int


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class QwenRocmBackboneGraphs:
    """Keep at most two qualified graphs owned by one loaded Qwen runtime.

    The caller holds ``lock`` through the final host transfer, so another
    request or cancelled caller cannot overwrite the static replay buffers.
    """

    def __init__(
        self,
        runtime: Qwen35TorchRuntime,
        *,
        artifact_content_id: str,
        event_recorder: Callable[[str], None] | None = None,
    ) -> None:
        import threading  # noqa: PLC0415

        from . import qwen35_rocm_binder, qwen35_torch  # noqa: PLC0415

        if (
            runtime.backend != "rocm"
            or runtime.rocm_profile_binding is None
            or not runtime.rocm_profile_binding.strict
            or not runtime.rocm_profile_binding.source_hashes_verified
            or not runtime.rocm_profile_binding.strict_cache_enforced
            or not runtime.rocm_profile_binding.unknown_key_guard_enforced
            or not isinstance(artifact_content_id, str)
            or len(artifact_content_id) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in artifact_content_id
            )
        ):
            raise ValueError(
                "ROCm graph needs a verified Qwen artifact and strict profile"
            )
        self.runtime = runtime
        self.artifact_content_id = artifact_content_id
        self.source_sha256 = (
            _source_sha256(Path(inspect.getfile(qwen35_torch))),
            _source_sha256(Path(inspect.getfile(qwen35_rocm_binder))),
            _source_sha256(Path(__file__)),
        )
        self.lock = threading.RLock()
        self._graphs: dict[_GraphKey, _GraphEntry] = {}
        self._rejected: set[_GraphKey] = set()
        self._captured_bytes = 0
        self._capture_attempts = 0
        self._parity_warnings = 0
        self._event_recorder = event_recorder

    def logits(self, batch: dict[str, Any]) -> Any:
        """Return original eager logits or qualified guarded-graph logits."""

        model = self.runtime.model
        ids = batch["input_ids"]
        if (
            len(ids.shape) != _INPUT_RANK
            or ids.shape[0] != _PHYSICAL_BATCH
            or ids.shape[1] > _MAX_PADDED_TOKENS
            or ids.shape[1] < _PADDED_TOKEN_MULTIPLE
            or ids.shape[1] % _PADDED_TOKEN_MULTIPLE
        ):
            return self._fallback(batch)
        try:
            masks = self._exact_masks(batch)
            key = self._key(batch, masks)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # An unrecognized pinned-helper mask layout never enters a graph.
            return self._fallback(batch)
        entry = self._graphs.get(key)
        if entry is not None:
            output = self._logits_from_hidden(batch, self._replay(entry, batch, masks))
            self._record_event("replay")
            return output
        if (
            key in self._rejected
            or len(self._graphs) >= _MAX_GRAPHS
            or self._capture_attempts >= _MAX_GRAPHS
            or self._captured_bytes >= _MAX_TOTAL_CAPTURE_BYTES
        ):
            return self._fallback(batch)

        ordinary = model(**batch)
        mapped = model(**{**batch, "attention_mask": masks})
        if not self._same_valid_logits_and_probabilities(batch, ordinary, mapped):
            self._reject_for_parity(key, stage="exact-mask")
            return self._fallback(batch, ordinary=ordinary)
        torch = self.runtime.torch
        if torch.cuda.mem_get_info(self.runtime.device)[0] < _MIN_FREE_HBM_BYTES:
            self._rejected.add(key)
            return self._fallback(batch, ordinary=ordinary)
        before = torch.cuda.memory_allocated(self.runtime.device)
        before_reserved = torch.cuda.memory_reserved(self.runtime.device)
        torch.cuda.reset_peak_memory_stats(self.runtime.device)
        self._capture_attempts += 1
        try:
            captured = self._capture(batch, masks)
        except QwenRocmBindingError:
            raise
        except (RuntimeError, NotImplementedError):
            self._rejected.add(key)
            return self._fallback(batch, ordinary=ordinary)
        captured.allocated_bytes = max(
            0,
            torch.cuda.memory_allocated(self.runtime.device) - before,
            torch.cuda.memory_reserved(self.runtime.device) - before_reserved,
            torch.cuda.max_memory_allocated(self.runtime.device) - before,
            torch.cuda.max_memory_reserved(self.runtime.device) - before_reserved,
        )
        if (
            captured.allocated_bytes > _MAX_CAPTURE_BYTES
            or self._captured_bytes + captured.allocated_bytes
            > _MAX_TOTAL_CAPTURE_BYTES
            or torch.cuda.mem_get_info(self.runtime.device)[0] < _MIN_FREE_HBM_BYTES
        ):
            self._rejected.add(key)
            return self._fallback(batch, ordinary=ordinary)
        candidate = self._logits_from_hidden(
            batch, self._replay(captured, batch, masks)
        )
        if not self._same_valid_logits_and_probabilities(batch, ordinary, candidate):
            self._reject_for_parity(key, stage="captured-replay")
            return self._fallback(batch, ordinary=ordinary)
        self._graphs[key] = captured
        self._captured_bytes += captured.allocated_bytes
        self._record_event("capture")
        return ordinary  # First request keeps its exact ordinary-eager result.

    def _record_event(self, event: str) -> None:
        if self._event_recorder is not None:
            self._event_recorder(event)

    def _fallback(self, batch: dict[str, Any], *, ordinary: Any = None) -> Any:
        output = self.runtime.model(**batch) if ordinary is None else ordinary
        self._record_event("fallback")
        return output

    def _reject_for_parity(self, key: _GraphKey, *, stage: str) -> None:
        """Disable one graph key and log at most four content-free warnings."""

        self._rejected.add(key)
        if self._parity_warnings < _MAX_PARITY_WARNINGS:
            _LOGGER.warning(
                "Qwen ROCm graph %s valid-logit/probability parity mismatch; "
                "using eager for B%d/T%d",
                stage,
                key.physical_batch,
                key.padded_tokens,
            )
            self._parity_warnings += 1

    def _exact_masks(self, batch: dict[str, Any]) -> dict[str, Any | None]:
        from transformers.masking_utils import (  # noqa: PLC0415
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        backbone = self.runtime.model.backbone
        ids = batch["input_ids"]
        placeholder = backbone.embed_tokens.weight.new_empty(
            (ids.shape[0], ids.shape[1], backbone.config.hidden_size)
        )
        options = {
            "config": backbone.config,
            "inputs_embeds": placeholder,
            "attention_mask": batch["attention_mask"],
            "past_key_values": None,
            "position_ids": None,
        }
        masks = {
            "full_attention": create_causal_mask(**options),
            "linear_attention": create_recurrent_attention_mask(**options),
        }
        for name, expected_shape, expected_dtype in (
            (
                "full_attention",
                (ids.shape[0], 1, ids.shape[1], ids.shape[1]),
                self.runtime.torch.bool,
            ),
            (
                "linear_attention",
                tuple(ids.shape),
                self.runtime.torch.int64,
            ),
        ):
            value = masks[name]
            if value is not None and (
                tuple(value.shape) != expected_shape
                or value.dtype != expected_dtype
                or value.device != ids.device
            ):
                raise ValueError(f"unsupported Qwen graph {name} mask layout")
        return masks

    def _key(self, batch: dict[str, Any], masks: dict[str, Any | None]) -> _GraphKey:
        binding = self.runtime.rocm_profile_binding
        if binding is None:
            raise ValueError("Qwen graph strict binding is absent")
        backbone = self.runtime.model.backbone
        layout = tuple(
            (
                name,
                None if mask is None else tuple(mask.shape),
                None if mask is None else str(mask.dtype),
            )
            for name, mask in sorted(masks.items())
        )
        return _GraphKey(
            artifact_content_id=self.artifact_content_id,
            source_sha256=self.source_sha256,
            model_instance=id(self.runtime.model),
            device=str(self.runtime.device),
            profile_sha256=binding.profile_sha256,
            kernel_config_sha256=binding.kernel_config_sha256,
            guard_contract_sha256=binding.guard_contract_sha256,
            physical_batch=batch["input_ids"].shape[0],
            padded_tokens=batch["input_ids"].shape[1],
            attention=str(backbone.config._attn_implementation),
            mask_layout=layout,
        )

    def _capture(
        self, batch: dict[str, Any], masks: dict[str, Any | None]
    ) -> _GraphEntry:
        torch = self.runtime.torch
        model = self.runtime.model
        static_ids = torch.empty_like(batch["input_ids"])
        static_masks = {
            name: None if value is None else torch.empty_like(value)
            for name, value in masks.items()
        }
        static_ids.copy_(batch["input_ids"])
        for name, value in masks.items():
            if value is not None:
                static_masks[name].copy_(value)
        side = torch.cuda.Stream(device=self.runtime.device)
        side.wait_stream(torch.cuda.current_stream(self.runtime.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                model.backbone(
                    input_ids=static_ids,
                    attention_mask=static_masks,
                    use_cache=False,
                )
        torch.cuda.current_stream(self.runtime.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden = model.backbone(
                input_ids=static_ids,
                attention_mask=static_masks,
                use_cache=False,
            ).last_hidden_state
        torch.cuda.synchronize(self.runtime.device)
        return _GraphEntry(graph, hidden, static_ids, static_masks, 0)

    def _replay(
        self, entry: _GraphEntry, batch: dict[str, Any], masks: dict[str, Any | None]
    ) -> Any:
        entry.static_ids.copy_(batch["input_ids"])
        for name, value in masks.items():
            if value is not None:
                entry.static_masks[name].copy_(value)
        binding: QwenRocmProfileBinding | None = self.runtime.rocm_profile_binding
        if binding is None:
            raise QwenRocmBindingError("Qwen graph strict binding is absent")
        assert_qwen_rocm_graph_replay_safe(
            binding,
            physical_batch_size=_PHYSICAL_BATCH,
            padded_tokens=batch["input_ids"].shape[1],
        )
        entry.graph.replay()
        return entry.hidden

    def _logits_from_hidden(self, batch: dict[str, Any], hidden: Any) -> Any:
        torch = self.runtime.torch
        indices = torch.arange(hidden.shape[0], device=hidden.device)
        candidates = hidden[indices[:, None], batch["candidate_positions"]]
        query = hidden[indices, batch["query_positions"]]
        scores = self.runtime.model.head(candidates, query).float()
        return scores.masked_fill(~batch["candidate_mask"], -float("inf"))

    def _same_valid_logits_and_probabilities(
        self, batch: dict[str, Any], expected: Any, actual: Any
    ) -> bool:
        torch = self.runtime.torch
        valid = batch["candidate_mask"]
        left = expected.float()[valid]
        right = actual.float()[valid]
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            return False
        left_prob = (expected.float() / self.runtime.temperature).softmax(-1)[valid]
        right_prob = (actual.float() / self.runtime.temperature).softmax(-1)[valid]
        return bool(
            torch.isfinite(left_prob).all()
            and torch.isfinite(right_prob).all()
            and torch.equal(left.view(torch.int32), right.view(torch.int32))
            and torch.equal(left_prob.view(torch.int32), right_prob.view(torch.int32))
        )
