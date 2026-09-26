"""Vela's marker-input, homogeneous-head, and release-loading contract."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from ..model_inputs import (
    VELA_DEFAULT_NO,
    VELA_DEFAULT_YES,
    build_model_inputs,
)
from ..runtime_profile import (
    BackendExecutionPolicy,
    RuntimeProfile,
    RuntimeProfileError,
)
from ..vela_inputs import EncodedVelaRow, encode_vela_rows
from .base import FamilyArtifactFiles, FamilyLoadError

if TYPE_CHECKING:
    from ..artifacts import VerifiedArtifact
    from ..physical_batching import DecisionRow


class VelaFamilyAdapter:
    family = "vela"
    manifest_paths = frozenset({"native/MANIFEST.json"})
    encoded_row_type = EncodedVelaRow
    inference_workers = None

    def artifact_files(
        self, manifest_path: str, available_paths: Collection[str]
    ) -> FamilyArtifactFiles:
        if manifest_path not in self.manifest_paths:
            raise ValueError("Vela artifact manifest layout is unsupported")
        return FamilyArtifactFiles(
            required=frozenset(
                {
                    "INVENTORY.json",
                    "STATE_LAYOUT.json",
                    "choice_encoder.safetensors",
                    "decision_config.json",
                    "decision_heads.safetensors",
                    "encoder/config.json",
                    "encoder/model.safetensors",
                    "score_encoder.safetensors",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                }
            ),
            optional=frozenset({"tokenizer/special_tokens_map.json"}),
        )

    def batch_key(self, question_type: str) -> str:
        # Vela has a separate head for each question type.
        return f"{self.family}:{question_type}"

    def encode_rows(
        self, rows: tuple[DecisionRow, ...], tokenizer: Any, profile: RuntimeProfile
    ) -> tuple[EncodedVelaRow, ...]:
        inputs = build_model_inputs(
            rows,
            choice_null_description=profile.prompt_policy.choice_null_description,
            noul_default_false=VELA_DEFAULT_NO,
            noul_default_true=VELA_DEFAULT_YES,
            noul_explicit_null="use_default",
        )
        return encode_vela_rows(inputs, tokenizer, max_length=profile.max_input_tokens)

    def parse_execution(self, value: object) -> Mapping[str, BackendExecutionPolicy]:
        raise RuntimeProfileError("execution backend or model family is unsupported")

    def load(
        self,
        artifact: VerifiedArtifact,
        profile: RuntimeProfile,
        backend: str,
        *,
        physical_batch_size: int,
        graph_event_recorder: Any,
    ) -> Any:
        from ..vela_torch import VelaTorchRuntime  # noqa: PLC0415

        manifest = getattr(artifact, "manifest", None)
        if manifest is None:
            raise FamilyLoadError("verified Decision artifact has no manifest identity")
        if manifest.path != "native/MANIFEST.json":
            raise FamilyLoadError("Vela release manifest layout is unsupported")
        return VelaTorchRuntime.load(
            artifact.data_root,
            max_length=profile.max_input_tokens,
            backend=backend,
            expected_manifest_sha256=manifest.sha256,
        )


ADAPTER = VelaFamilyAdapter()
