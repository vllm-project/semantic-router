"""Qwen3.5's setwise inputs, calibration, and ROCm execution policy."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..model_inputs import (
    QWEN_DEFAULT_NO,
    QWEN_DEFAULT_YES,
    build_model_inputs,
)
from ..qwen35_inputs import EncodedQwenRow, encode_qwen_rows
from ..release_artifacts import select_qwen_weight_files
from ..runtime_limits import MAX_JOB_TURN_BATCHES
from ..runtime_profile import (
    BackendExecutionPolicy,
    RuntimeProfile,
    RuntimeProfileError,
)
from .base import FamilyArtifactFiles, FamilyLoadError

if TYPE_CHECKING:
    from ..artifacts import VerifiedArtifact
    from ..physical_batching import DecisionRow

_MAX_GRAPH_PREWARM_SHAPES = 2
_MAX_GRAPH_PADDED_TOKENS = 256
_GRAPH_PADDED_TOKEN_MULTIPLE = 32


class Qwen35FamilyAdapter:
    family = "qwen3.5"
    manifest_paths = frozenset({"MODEL_MANIFEST.json", "bundle-manifest.json"})
    encoded_row_type = EncodedQwenRow
    inference_workers = 1

    def artifact_files(
        self, manifest_path: str, available_paths: Collection[str]
    ) -> FamilyArtifactFiles:
        if manifest_path not in self.manifest_paths:
            raise ValueError("Qwen artifact manifest layout is unsupported")
        required = {
            "backbone/config.json",
            "decision_config.json",
            "decision_head.safetensors",
            "runtime.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        required.update(select_qwen_weight_files(available_paths))
        return FamilyArtifactFiles(
            required=frozenset(required),
            optional=frozenset(
                {
                    "chat_template.jinja",
                    "temperature.json",
                    "runtime-profile/profile.json",
                    "runtime-profile/l2norm_fwd_kernel.json",
                    "code/profile_guard.py",
                }
            ),
        )

    def batch_key(self, question_type: str) -> str:
        # Its shared setwise scorer can mix question types in one forward.
        return self.family

    def encode_rows(
        self, rows: tuple[DecisionRow, ...], tokenizer: Any, profile: RuntimeProfile
    ) -> tuple[EncodedQwenRow, ...]:
        inputs = build_model_inputs(
            rows,
            choice_null_description=profile.prompt_policy.choice_null_description,
            noul_default_false=QWEN_DEFAULT_NO,
            noul_default_true=QWEN_DEFAULT_YES,
            noul_explicit_null="preserve_json_null",
        )
        return encode_qwen_rows(inputs, tokenizer, max_length=profile.max_input_tokens)

    def parse_execution(self, value: object) -> Mapping[str, BackendExecutionPolicy]:
        execution = _mapping(value, "execution")
        if set(execution) != {"rocm"}:
            raise RuntimeProfileError(
                "execution backend or model family is unsupported"
            )
        rocm = _mapping(execution["rocm"], "execution.rocm")
        allowed = {
            "backbone_graph",
            "graph_prewarm_padded_tokens",
            "gated_delta",
            "max_physical_batch_size",
            "job_turn_batches",
        }
        if not rocm or not set(rocm) <= allowed:
            raise RuntimeProfileError(
                "execution.rocm fields do not match the runtime contract"
            )
        graph = rocm.get("backbone_graph")
        if "backbone_graph" in rocm and graph != "short_b8":
            raise RuntimeProfileError("execution.rocm.backbone_graph is unsupported")
        prewarm = rocm.get("graph_prewarm_padded_tokens", [])
        if (
            not isinstance(prewarm, list)
            or len(prewarm) > _MAX_GRAPH_PREWARM_SHAPES
            or any(
                type(tokens) is not int
                or not _GRAPH_PADDED_TOKEN_MULTIPLE
                <= tokens
                <= _MAX_GRAPH_PADDED_TOKENS
                or tokens % _GRAPH_PADDED_TOKEN_MULTIPLE
                for tokens in prewarm
            )
            or len(set(prewarm)) != len(prewarm)
            or ("graph_prewarm_padded_tokens" in rocm and graph != "short_b8")
        ):
            raise RuntimeProfileError(
                "execution.rocm.graph_prewarm_padded_tokens is unsupported"
            )
        kernel = rocm.get("gated_delta", "accelerated")
        if not isinstance(kernel, str) or kernel not in {
            "native_torch",
            "accelerated",
        }:
            raise RuntimeProfileError("execution.rocm.gated_delta is unsupported")
        maximum = rocm.get("max_physical_batch_size")
        if kernel == "native_torch":
            maximum = _positive_int(maximum, "execution.rocm.max_physical_batch_size")
        elif maximum is not None:
            raise RuntimeProfileError(
                "execution.rocm.max_physical_batch_size requires native_torch"
            )
        if graph is not None and kernel == "native_torch":
            raise RuntimeProfileError(
                "execution.rocm cannot combine short_b8 graph with native_torch"
            )
        job_turn_batches = rocm.get("job_turn_batches")
        if job_turn_batches is not None and (
            type(job_turn_batches) is not int
            or not 1 <= job_turn_batches <= MAX_JOB_TURN_BATCHES
        ):
            raise RuntimeProfileError(
                f"execution.rocm.job_turn_batches must be 1 to {MAX_JOB_TURN_BATCHES}"
            )
        return MappingProxyType(
            {
                "rocm": BackendExecutionPolicy(
                    backbone_graph=graph,
                    graph_prewarm_padded_tokens=tuple(prewarm),
                    gated_delta=kernel,
                    max_physical_batch_size=maximum,
                    job_turn_batches=job_turn_batches,
                )
            }
        )

    def load(
        self,
        artifact: VerifiedArtifact,
        profile: RuntimeProfile,
        backend: str,
        *,
        physical_batch_size: int,
        graph_event_recorder: Any,
    ) -> Any:
        from ..qwen35_torch import Qwen35TorchRuntime  # noqa: PLC0415

        manifest = getattr(artifact, "manifest", None)
        if manifest is None:
            raise FamilyLoadError("verified Decision artifact has no manifest identity")
        if manifest.path not in {"MODEL_MANIFEST.json", "bundle-manifest.json"}:
            raise FamilyLoadError("Qwen release manifest layout is unsupported")
        kernel_policy = profile.execution.get(backend, BackendExecutionPolicy())
        binder = None
        if backend == "rocm":
            from ..qwen35_rocm_binder import (  # noqa: PLC0415
                create_qwen_rocm_profile_binder,
            )

            binder = create_qwen_rocm_profile_binder()
        graph_options = (
            {
                "enable_rocm_graph": True,
                "artifact_content_id": artifact.content_id,
                "graph_event_recorder": graph_event_recorder,
                "graph_prewarm_padded_tokens": (
                    kernel_policy.graph_prewarm_padded_tokens
                ),
            }
            if profile.use_short_b8_graph(backend, physical_batch_size)
            else {}
        )
        return Qwen35TorchRuntime.load(
            artifact.data_root,
            temperature=qwen_temperature(artifact, fallback=profile.temperature),
            max_length=profile.max_input_tokens,
            backend=backend,
            gated_delta_kernel_policy=kernel_policy.gated_delta,
            native_rocm_max_physical_batch_size=kernel_policy.max_physical_batch_size,
            physical_batch_size=physical_batch_size,
            rocm_profile_binder=binder,
            expected_manifest_sha256=(
                manifest.sha256 if manifest.path == "MODEL_MANIFEST.json" else None
            ),
            **graph_options,
        )


def qwen_temperature(artifact: VerifiedArtifact, *, fallback: float | None) -> float:
    """Read calibration from the verified snapshot, not its template revision."""

    # Test assemblers may inject a synthetic artifact without a receipt. Real
    # materializations always include manifest identity and selected file data.
    if getattr(artifact, "manifest", None) is None:
        if fallback is None:
            raise FamilyLoadError("Qwen release has no calibration temperature")
        return fallback

    selected = {item.manifest_path for item in artifact.files}
    source = "temperature.json" if "temperature.json" in selected else "runtime.json"
    try:
        metadata = json.loads((artifact.data_root / source).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FamilyLoadError("Qwen calibration metadata is invalid") from error
    temperature = metadata.get("temperature") if isinstance(metadata, dict) else None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise FamilyLoadError("Qwen calibration temperature is invalid")
    try:
        value = float(temperature)
    except OverflowError as error:
        raise FamilyLoadError("Qwen calibration temperature is invalid") from error
    if not math.isfinite(value) or value <= 0:
        raise FamilyLoadError("Qwen calibration temperature is invalid")
    return value


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeProfileError(f"{field} must be an object")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeProfileError(f"{field} must be a positive integer")
    return value


ADAPTER = Qwen35FamilyAdapter()
