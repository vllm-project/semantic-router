"""Owned Torch execution core for Qwen3.5 Decision checkpoints.

This module deliberately imports Torch, Transformers, and Safetensors only when
``Qwen35TorchRuntime.load`` is called.  Ordinary CLI/catalog processes therefore
do not import a tensor framework, and model-repository Python is never executed.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import re
import types
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from .qwen35_inputs import EncodedQwenRow
from .release_artifacts import (
    ReleaseArtifactError,
    verify_release_manifest,
)

SUPPORTED_TRANSFORMERS_VERSION = "5.17.0"
RELEASED_MAX_INPUT_TOKENS = 16_384
_ROCM_PROFILE_FORMAT = "decision-fla-l2norm-profile-v1"
_FLA_SOURCE_FILES = ("modules/l2norm.py", "ops/utils/cache.py")
_FLA_AUTOTUNE_KEY_FIELDS = 5
_FLA_L2NORM_WIDTH = 128
_FLA_MAX_NORMALIZATION_BLOCKS = 64
_FLA_NUM_STAGES = 3
_SHA256_HEX_LENGTH = 64
_GRAPH_PHYSICAL_BATCH = 8
_QWEN_LINEAR_KEY_HEADS = 16
_QWEN_LINEAR_HEAD_DIM = 128
_QWEN_SHORT_LINEAR_VALUE_HEADS = 16
INFERENCE_FILES = (
    "backbone/config.json",
    "decision_config.json",
    "decision_head.safetensors",
    "runtime.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Qwen35RuntimeError(RuntimeError):
    """The owned Qwen runtime cannot load or execute the checkpoint safely."""


@dataclass(frozen=True, slots=True)
class ValidatedQwenRocmProfile:
    """Hash-bound launch-profile data safe to pass to vLLM-SR-owned code."""

    profile_path: Path
    kernel_config_path: Path
    profile_sha256: str
    kernel_config_sha256: str
    guard_contract_sha256: str
    runtime: tuple[tuple[str, str], ...]
    fla_source_sha256: tuple[tuple[str, str], ...]
    # ``B`` is the number of rows in a physical inference batch.  It is not
    # the FLA l2norm autotune key: that key's second element is ``NB``.
    physical_batch_size_min: int
    physical_batch_size_max: int
    padded_tokens_max: int
    normalization_blocks_min: int
    normalization_blocks_max: int
    normalization_blocks_formula: str
    unknown_key_policy: str

    def normalization_blocks_for(
        self, *, physical_batch_size: int, padded_tokens: int
    ) -> int:
        """Calculate the FLA ``NB`` key without conflating it with ``B``."""

        if (
            isinstance(physical_batch_size, bool)
            or not isinstance(physical_batch_size, int)
            or not self.physical_batch_size_min
            <= physical_batch_size
            <= self.physical_batch_size_max
        ):
            raise Qwen35RuntimeError("physical batch size is outside the profile")
        if (
            isinstance(padded_tokens, bool)
            or not isinstance(padded_tokens, int)
            or not 1 <= padded_tokens <= self.padded_tokens_max
        ):
            raise Qwen35RuntimeError("padded token length is outside the profile")
        heads = 16 if "*16/" in self.normalization_blocks_formula else 32
        blocks = math.ceil(physical_batch_size * padded_tokens * heads / 65536)
        if not self.normalization_blocks_min <= blocks <= self.normalization_blocks_max:
            raise Qwen35RuntimeError("normalization-block count is outside the profile")
        return blocks


@dataclass(frozen=True, slots=True)
class QwenRocmProfileBinding:
    """Receipt returned only after an owned binder installs the strict profile."""

    profile_sha256: str
    kernel_config_sha256: str
    guard_contract_sha256: str
    runtime: tuple[tuple[str, str], ...]
    fla_source_sha256: tuple[tuple[str, str], ...]
    physical_batch_size_min: int
    physical_batch_size_max: int
    normalization_blocks_min: int
    normalization_blocks_max: int
    strict: bool
    # The binder has read the *installed* FLA sources and compared them to the
    # artifact-bound digests.  A profile declaration alone is not evidence.
    source_hashes_verified: bool
    strict_cache_enforced: bool
    unknown_key_guard_enforced: bool


class QwenRocmProfileBinder(Protocol):
    """Seam for an owned FLA binder; repository Python is never loaded here."""

    def bind(self, profile: ValidatedQwenRocmProfile) -> QwenRocmProfileBinding: ...


@dataclass(frozen=True, slots=True)
class Qwen35Prediction:
    """One ordered probability result before public response adaptation."""

    question_id: str
    type: Literal["noul", "choice", "score"]
    probabilities: tuple[float, ...]
    input_tokens: int


@dataclass(slots=True)
class Qwen35TorchRuntime:
    """One resident Qwen3.5 text backbone plus its FP32 candidate head."""

    model: Any
    tokenizer: Any
    torch: Any
    device: Any
    temperature: float
    max_length: int
    backend: Literal["cpu", "rocm", "cuda"]
    rocm_profile_binding: QwenRocmProfileBinding | None
    gated_delta_kernel_policy: Literal["native_torch", "accelerated"] = "accelerated"
    rocm_graphs: Any | None = None

    @classmethod
    def load(
        cls,
        artifact_root: Path,
        *,
        temperature: float,
        max_length: int,
        backend: Literal["cpu", "rocm", "cuda"],
        gated_delta_kernel_policy: Literal[
            "native_torch", "accelerated"
        ] = "accelerated",
        native_rocm_max_physical_batch_size: int | None = None,
        device: str | None = None,
        attention: str = "sdpa",
        physical_batch_size: int = 8,
        rocm_profile_binder: QwenRocmProfileBinder | None = None,
        expected_manifest_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        config_path: Path | None = None,
        model_name: str | None = None,
        enable_rocm_graph: bool = False,
        artifact_content_id: str | None = None,
        graph_event_recorder: Callable[[str], None] | None = None,
        graph_prewarm_padded_tokens: tuple[int, ...] = (),
    ) -> Qwen35TorchRuntime:
        """Load verified data files with the distribution-owned implementation."""

        root = Path(artifact_root).resolve(strict=True)
        if expected_manifest_sha256 is not None and expected_config_sha256 is not None:
            raise Qwen35RuntimeError("Qwen release has conflicting identities")
        release_config = (
            _verified_repository_config(
                root,
                config_path=config_path,
                expected_sha256=expected_config_sha256,
                model_name=model_name,
            )
            if expected_config_sha256 is not None
            else None
        )
        if release_config is None and (
            config_path is not None or model_name is not None
        ):
            raise Qwen35RuntimeError("Qwen release config identity is incomplete")
        rocm_profile = _validate_configuration(
            root,
            temperature=temperature,
            max_length=max_length,
            backend=backend,
            gated_delta_kernel_policy=gated_delta_kernel_policy,
            native_rocm_max_physical_batch_size=native_rocm_max_physical_batch_size,
            attention=attention,
            physical_batch_size=physical_batch_size,
            repository_config=release_config,
        )
        if type(enable_rocm_graph) is not bool:
            raise Qwen35RuntimeError("Qwen graph policy must be boolean")
        if graph_prewarm_padded_tokens and not enable_rocm_graph:
            raise Qwen35RuntimeError("Qwen graph prewarm requires a graph profile")
        if enable_rocm_graph and (
            backend != "rocm"
            or rocm_profile is None
            or physical_batch_size != _GRAPH_PHYSICAL_BATCH
            or not isinstance(artifact_content_id, str)
            or len(artifact_content_id) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in artifact_content_id
            )
        ):
            raise Qwen35RuntimeError(
                "Qwen ROCm graph requires a verified strict B8 profile"
            )
        if (
            backend == "cpu"
            and expected_manifest_sha256 is None
            and release_config is None
        ):
            raise Qwen35RuntimeError(
                "CPU Qwen requires a pinned release manifest or config digest"
            )
        if expected_manifest_sha256 is not None:
            try:
                verify_release_manifest(
                    root,
                    manifest_name="MODEL_MANIFEST.json",
                    expected_sha256=expected_manifest_sha256,
                    required_files=INFERENCE_FILES,
                    qwen_weight_layout=True,
                )
            except ReleaseArtifactError as error:
                raise Qwen35RuntimeError(
                    f"Qwen artifact integrity failed: {error}"
                ) from error
        if (
            backend == "rocm"
            and rocm_profile is not None
            and rocm_profile_binder is None
        ):
            raise Qwen35RuntimeError(
                "this Qwen release requires a vLLM-SR-owned strict ROCm profile binder"
            )
        torch = _required_module("torch")
        selected = torch.device(
            device if device is not None else ("cpu" if backend == "cpu" else "cuda:0")
        )
        _validate_device(torch, selected, backend=backend, rocm_profile=rocm_profile)
        binding = None
        if backend == "rocm" and rocm_profile is not None:
            binding = _bind_rocm_profile(rocm_profile_binder, rocm_profile)
        transformers = _required_module("transformers")
        if getattr(transformers, "__version__", None) != SUPPORTED_TRANSFORMERS_VERSION:
            raise Qwen35RuntimeError(
                "Qwen3.5 Decision runtime requires Transformers "
                f"{SUPPORTED_TRANSFORMERS_VERSION}"
            )
        safetensors = _required_module("safetensors.torch")
        body_dtype = torch.float32 if backend == "cpu" else torch.bfloat16

        try:
            modeling = importlib.import_module(
                "transformers.models.qwen3_5.modeling_qwen3_5"
            )
            load_options = {
                "dtype": body_dtype,
                "local_files_only": True,
                "trust_remote_code": False,
                "attn_implementation": attention,
            }
            if release_config is not None:
                load_options["output_loading_info"] = True
            backbone_directory = (
                root / Path(release_config.backbone_config).parent
                if release_config is not None
                else root / "backbone"
            )
            loaded = modeling.Qwen3_5TextModel.from_pretrained(
                backbone_directory, **load_options
            )
            backbone = (
                _complete_backbone(loaded) if release_config is not None else loaded
            )
            if release_config is not None:
                _validate_backbone_tensors(
                    backbone,
                    root=root,
                    weight_files=release_config.backbone_weights,
                    index_file=release_config.backbone_index,
                    safe_open=_required_module("safetensors").safe_open,
                )
            backbone.config.use_cache = False
            model_config_path = (
                release_config.model_config
                if release_config is not None
                else "decision_config.json"
            )
            metadata = _read_json(root / model_config_path)
            head_dim = metadata.get("head_dim")
            if (
                isinstance(head_dim, bool)
                or not isinstance(head_dim, int)
                or head_dim < 1
            ):
                raise Qwen35RuntimeError("Decision head dimension is invalid")
            head = _candidate_head(torch, backbone.config.hidden_size, head_dim)
            head_path = (
                release_config.decision_weights["decision_head"]
                if release_config is not None
                else "decision_head.safetensors"
            )
            head_state = safetensors.load_file(str(root / head_path), device="cpu")
            head.load_state_dict(head_state, strict=True)
            model = _decision_model(torch, backbone, head)
            model.to(selected).eval()
            if backend == "cpu" or gated_delta_kernel_policy == "native_torch":
                _install_instance_native_gated_delta_kernels(model, modeling)
            tokenizer_directory = (
                root / Path(release_config.tokenizer_json).parent
                if release_config is not None
                else root
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_directory,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Qwen35RuntimeError:
            raise
        except Exception as error:
            raise Qwen35RuntimeError(
                f"Qwen3.5 Decision checkpoint could not be loaded: "
                f"{type(error).__name__}"
            ) from error
        if any(parameter.dtype != body_dtype for parameter in backbone.parameters()):
            raise Qwen35RuntimeError(
                "Qwen3.5 backbone storage must be FP32 on CPU and BF16 on GPU"
            )
        if any(parameter.dtype != torch.float32 for parameter in head.parameters()):
            raise Qwen35RuntimeError("Decision candidate-head storage must be FP32")
        runtime = cls(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=selected,
            temperature=float(temperature),
            max_length=max_length,
            backend=backend,
            rocm_profile_binding=binding,
            gated_delta_kernel_policy=(
                "native_torch" if backend == "cpu" else gated_delta_kernel_policy
            ),
        )
        if enable_rocm_graph:
            from .qwen35_rocm_graph import QwenRocmBackboneGraphs  # noqa: PLC0415

            runtime.rocm_graphs = QwenRocmBackboneGraphs(
                runtime,
                artifact_content_id=artifact_content_id,
                event_recorder=graph_event_recorder,
                capture_on_request=not bool(graph_prewarm_padded_tokens),
            )
            if graph_prewarm_padded_tokens:
                runtime.rocm_graphs.prewarm(graph_prewarm_padded_tokens)
        return runtime

    def predict_encoded(
        self, rows: tuple[EncodedQwenRow, ...]
    ) -> tuple[Qwen35Prediction, ...]:
        """Run one complete physical batch and synchronize once for host output."""

        lock = self.rocm_graphs.lock if self.rocm_graphs is not None else nullcontext()
        with lock:
            return self._predict_encoded_locked(rows)

    def _predict_encoded_locked(
        self, rows: tuple[EncodedQwenRow, ...]
    ) -> tuple[Qwen35Prediction, ...]:
        """Keep static graph buffers owned through the host synchronization."""

        if not rows:
            raise ValueError("at least one encoded Qwen row is required")
        batch = _collate(self.torch, rows, self.tokenizer, device=self.device)
        try:
            with self.torch.inference_mode():
                autocast = (
                    nullcontext()
                    if self.backend == "cpu"
                    else self.torch.autocast(
                        device_type=self.device.type,
                        dtype=self.torch.bfloat16,
                    )
                )
                with autocast:
                    logits = (
                        self.rocm_graphs.logits(batch)
                        if self.rocm_graphs is not None
                        else self.model(**batch)
                    )
                # Keep padded candidates out of the shared softmax even if a
                # model implementation returns finite values in those slots.
                valid_logits = logits.float().masked_fill(
                    ~batch["candidate_mask"], -float("inf")
                )
                probabilities = (valid_logits / self.temperature).softmax(-1)
                # Transfer logits and probabilities together for one batch sync.
                host = self.torch.cat((valid_logits, probabilities)).tolist()
                batch_size = len(rows)
                if len(host) != 2 * batch_size:
                    raise Qwen35RuntimeError("invalid Qwen probability vector")
                for row, values in zip(rows, host[:batch_size], strict=True):
                    count = len(row.candidate_positions)
                    if len(values) < count or any(
                        not math.isfinite(value) for value in values[:count]
                    ):
                        raise Qwen35RuntimeError("non-finite Qwen candidate logits")
                host = host[batch_size:]
        except Qwen35RuntimeError:
            raise
        except Exception as error:
            raise Qwen35RuntimeError(
                f"Qwen3.5 Decision inference failed: {type(error).__name__}"
            ) from error

        output = []
        for row, probabilities in zip(rows, host, strict=True):
            count = len(row.candidate_positions)
            values = tuple(float(value) for value in probabilities[:count])
            if (
                len(values) != count
                or any(not math.isfinite(value) or value < 0.0 for value in values)
                or not math.isclose(math.fsum(values), 1.0, abs_tol=2e-5, rel_tol=0.0)
            ):
                raise Qwen35RuntimeError("invalid Qwen probability vector")
            output.append(
                Qwen35Prediction(
                    question_id=row.question_id,
                    type=row.type,  # type: ignore[arg-type]
                    probabilities=values,
                    input_tokens=row.input_tokens,
                )
            )
        return tuple(output)


def _candidate_head(torch, hidden_size: int, head_dim: int):
    nn = torch.nn
    functional = torch.nn.functional

    class CandidateHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.candidate_norm = nn.LayerNorm(hidden_size)
            self.query_norm = nn.LayerNorm(hidden_size)
            self.key = nn.Linear(hidden_size, head_dim, bias=False)
            self.query = nn.Linear(hidden_size, head_dim, bias=False)
            self.candidate_mlp = nn.Linear(hidden_size, head_dim, bias=True)
            self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
            self.scalar = nn.Linear(head_dim, 1, bias=False)

        def forward(self, candidates, query):
            with torch.autocast(device_type=candidates.device.type, enabled=False):
                candidates = self.candidate_norm(candidates.float())
                query = self.query_norm(query.float())
                bilinear = (self.key(candidates) * self.query(query)[:, None, :]).sum(
                    -1
                ) / math.sqrt(head_dim)
                interaction = self.scalar(
                    functional.gelu(
                        self.candidate_mlp(candidates)
                        + self.query_mlp(query)[:, None, :]
                    )
                ).squeeze(-1)
                return bilinear + interaction

    return CandidateHead()


def _decision_model(torch, backbone, head):
    class DecisionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(
            self,
            input_ids,
            attention_mask,
            candidate_positions,
            candidate_mask,
            query_positions,
        ):
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
            batches = torch.arange(hidden.shape[0], device=hidden.device)
            candidates = hidden[batches[:, None], candidate_positions]
            query = hidden[batches, query_positions]
            scores = self.head(candidates, query).float()
            return scores.masked_fill(~candidate_mask, -float("inf"))

    return DecisionModel()


def _install_instance_native_gated_delta_kernels(model, modeling) -> None:
    """Bind native PyTorch GatedDeltaNet functions to one model instance.

    Transformers can select installed FLA and causal-conv packages, including
    for CPU tensors. Their reference functions live in the pinned Transformers
    module; no model-repository code or module-global replacement is used.
    """

    layer_type = modeling.Qwen3_5GatedDeltaNet
    forward = inspect.unwrap(layer_type.forward)
    source_name = "modeling_qwen3_5.py"
    if (
        not isinstance(forward, types.FunctionType)
        or forward.__closure__ is not None
        or Path(forward.__code__.co_filename).name != source_name
    ):
        raise Qwen35RuntimeError("native Qwen GatedDeltaNet forward is unavailable")
    if not _native_signature_matches(
        forward,
        (
            ("self", inspect.Parameter.empty),
            ("hidden_states", inspect.Parameter.empty),
            ("cache_params", None),
            ("attention_mask", None),
            ("kwargs", Ellipsis),
        ),
    ):
        raise Qwen35RuntimeError("native Qwen GatedDeltaNet signature changed")
    names = {
        "causal_conv1d_fn": (
            ("hidden_states", inspect.Parameter.empty),
            ("weight", inspect.Parameter.empty),
            ("bias", None),
            ("activation", None),
            ("kwargs", Ellipsis),
        ),
        "causal_conv1d_update": (
            ("hidden_states", inspect.Parameter.empty),
            ("conv_state", inspect.Parameter.empty),
            ("weight", inspect.Parameter.empty),
            ("bias", None),
            ("activation", None),
        ),
        "torch_chunk_gated_delta_rule": (
            ("query", inspect.Parameter.empty),
            ("key", inspect.Parameter.empty),
            ("value", inspect.Parameter.empty),
            ("g", inspect.Parameter.empty),
            ("beta", inspect.Parameter.empty),
            ("chunk_size", 64),
            ("initial_state", None),
            ("output_final_state", False),
            ("use_qk_l2norm_in_kernel", False),
            ("kwargs", Ellipsis),
        ),
        "torch_recurrent_gated_delta_rule": (
            ("query", inspect.Parameter.empty),
            ("key", inspect.Parameter.empty),
            ("value", inspect.Parameter.empty),
            ("g", inspect.Parameter.empty),
            ("beta", inspect.Parameter.empty),
            ("initial_state", None),
            ("output_final_state", False),
            ("use_qk_l2norm_in_kernel", False),
            ("kwargs", Ellipsis),
        ),
    }
    if any(name not in forward.__code__.co_names for name in names):
        raise Qwen35RuntimeError("native Qwen GatedDeltaNet contract changed")
    namespace = dict(forward.__globals__)
    for name, signature in names.items():
        selected = inspect.unwrap(getattr(modeling, name, None))
        if (
            not isinstance(selected, types.FunctionType)
            or selected.__closure__ is not None
            or Path(selected.__code__.co_filename).name != source_name
            or not _native_signature_matches(selected, signature)
        ):
            raise Qwen35RuntimeError(f"native Qwen function is unavailable: {name}")
        namespace[name] = selected

    norm_type = modeling.Qwen3_5RMSNormGated
    norm_forward = inspect.unwrap(norm_type.forward)
    if (
        not isinstance(norm_forward, types.FunctionType)
        or norm_forward.__closure__ is not None
        or Path(norm_forward.__code__.co_filename).name != source_name
        or not _native_signature_matches(
            norm_forward,
            (
                ("self", inspect.Parameter.empty),
                ("hidden_states", inspect.Parameter.empty),
                ("gate", inspect.Parameter.empty),
            ),
        )
    ):
        raise Qwen35RuntimeError("native Qwen gated norm forward is unavailable")

    layers = [layer for layer in model.modules() if isinstance(layer, layer_type)]
    if not layers or any(
        type(layer) is not layer_type
        or hasattr(layer, "_hf_hook")
        or hasattr(layer.conv1d, "_hf_hook")
        or "forward" in layer.__dict__
        or type(layer.norm) is not norm_type
        or hasattr(layer.norm, "_hf_hook")
        or "forward" in layer.norm.__dict__
        for layer in layers
    ):
        raise Qwen35RuntimeError("Qwen GatedDeltaNet layers are modified or offloaded")
    local_forward = types.FunctionType(
        forward.__code__,
        namespace,
        forward.__name__,
        forward.__defaults__,
        forward.__closure__,
    )
    local_forward.__kwdefaults__ = forward.__kwdefaults__
    for layer in layers:
        layer.forward = types.MethodType(local_forward, layer)
        layer.norm.forward = types.MethodType(norm_forward, layer.norm)


def _native_signature_matches(
    function: types.FunctionType, expected: tuple[tuple[str, object], ...]
) -> bool:
    """Reject source changes before binding reference functions to an instance."""

    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    if len(parameters) != len(expected):
        return False
    for actual, (name, default) in zip(parameters, expected, strict=True):
        kind = (
            inspect.Parameter.VAR_KEYWORD
            if default is Ellipsis
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if (
            actual.name != name
            or actual.kind is not kind
            or (default is not Ellipsis and actual.default != default)
        ):
            return False
    return True


def _collate(torch, rows, tokenizer, *, device):
    length = ((max(row.input_tokens for row in rows) + 31) // 32) * 32
    candidates = max(len(row.candidate_positions) for row in rows)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
        raise Qwen35RuntimeError("Qwen tokenizer has no valid PAD/EOS token")
    input_ids = torch.full((len(rows), length), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    positions = torch.zeros((len(rows), candidates), dtype=torch.long)
    candidate_mask = torch.zeros((len(rows), candidates), dtype=torch.bool)
    query_positions = torch.empty(len(rows), dtype=torch.long)
    for index, row in enumerate(rows):
        size = row.input_tokens
        count = len(row.candidate_positions)
        input_ids[index, :size] = torch.tensor(row.input_ids, dtype=torch.long)
        attention_mask[index, :size] = 1
        positions[index, :count] = torch.tensor(
            row.candidate_positions, dtype=torch.long
        )
        candidate_mask[index, :count] = True
        query_positions[index] = row.query_position
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "candidate_positions": positions.to(device),
        "candidate_mask": candidate_mask.to(device),
        "query_positions": query_positions.to(device),
    }


def _verified_repository_config(
    root: Path,
    *,
    config_path: Path | None,
    expected_sha256: str,
    model_name: str | None,
) -> Any:
    """Bind the clean model layout to its selected immutable root config."""

    if (
        config_path is None
        or not isinstance(model_name, str)
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise Qwen35RuntimeError("Qwen release config identity is invalid")
    path = Path(config_path)
    try:
        if path.is_symlink() or path.resolve(strict=True) != root / "config.json":
            raise Qwen35RuntimeError("Qwen release config path is invalid")
        payload = path.read_bytes()
    except OSError as error:
        raise Qwen35RuntimeError("Qwen release config is missing") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise Qwen35RuntimeError("Qwen release config digest mismatch")
    from .artifacts import ArtifactError, parse_decision_config  # noqa: PLC0415

    try:
        config = parse_decision_config(payload, model_name=model_name, family="qwen3.5")
    except (ArtifactError, ValueError) as error:
        raise Qwen35RuntimeError("Qwen release config layout is invalid") from error
    backbone_directory = Path(config.backbone_config).parent
    if Path(config.backbone_config).name != "config.json" or any(
        Path(relative).parent != backbone_directory
        or not relative.endswith(".safetensors")
        for relative in config.backbone_weights
    ):
        raise Qwen35RuntimeError("Qwen release weight layout is invalid")
    if config.backbone_index is None:
        if (
            len(config.backbone_weights) != 1
            or Path(config.backbone_weights[0]).name != "model.safetensors"
        ):
            raise Qwen35RuntimeError("Qwen release weight layout is invalid")
    elif Path(config.backbone_index).parent != backbone_directory:
        raise Qwen35RuntimeError("Qwen release weight index layout is invalid")
    if config.backbone_index is not None:
        index = _read_json(root / config.backbone_index)
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str) or not key or not isinstance(value, str)
                for key, value in weight_map.items()
            )
            or set(weight_map.values())
            != {Path(relative).name for relative in config.backbone_weights}
        ):
            raise Qwen35RuntimeError("Qwen release weight index is invalid")
    return config


def _complete_backbone(loaded: Any) -> Any:
    """Reject a checkpoint that Transformers partially initialized."""

    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if (
        not isinstance(loaded, tuple)
        or len(loaded) != 2  # noqa: PLR2004 - model and loading information
        or loaded[0] is None
        or not isinstance(loaded[1], dict)
        or not set(fields) <= loaded[1].keys()
        or any(loaded[1][field] for field in fields)
    ):
        raise Qwen35RuntimeError(
            "Qwen backbone tensors do not match the model architecture"
        )
    return loaded[0]


def _validate_backbone_tensors(
    backbone: Any,
    *,
    root: Path,
    weight_files: tuple[str, ...],
    index_file: str | None = None,
    safe_open: Callable[..., Any],
) -> None:
    """Compare shard metadata to the actual owned model, without reloading tensors."""

    expected = {
        name: tuple(tensor.shape) for name, tensor in backbone.state_dict().items()
    }
    if not expected:
        raise Qwen35RuntimeError("Qwen backbone has no model tensors")
    observed: dict[str, tuple[int, ...]] = {}
    locations: dict[str, str] = {}
    try:
        for relative in weight_files:
            with safe_open(
                str(root / relative), framework="pt", device="cpu"
            ) as reader:
                # The safetensors reader exposes keys() but is not an iterator.
                tensor_names = reader.keys()
                for name in tensor_names:
                    if name in observed:
                        raise Qwen35RuntimeError(
                            "Qwen backbone shard contains duplicate tensors"
                        )
                    observed[name] = tuple(reader.get_slice(name).get_shape())
                    locations[name] = Path(relative).name
    except Qwen35RuntimeError:
        raise
    except Exception as error:
        raise Qwen35RuntimeError(
            "Qwen backbone tensor metadata could not be verified"
        ) from error
    if observed != expected:
        raise Qwen35RuntimeError(
            "Qwen backbone tensors do not match the model architecture"
        )
    if index_file is not None:
        index = _read_json(root / index_file)
        if index.get("weight_map") != locations:
            raise Qwen35RuntimeError(
                "Qwen weight index does not match the checkpoint tensors"
            )


def _validate_configuration(
    root: Path,
    *,
    temperature: float,
    max_length: int,
    backend: str,
    gated_delta_kernel_policy: str = "accelerated",
    native_rocm_max_physical_batch_size: int | None = None,
    attention: str,
    physical_batch_size: int,
    repository_config: Any | None = None,
) -> ValidatedQwenRocmProfile | None:
    if backend not in {"cpu", "rocm", "cuda"}:
        raise Qwen35RuntimeError("Qwen Torch backend must be cpu, rocm, or cuda")
    if not isinstance(
        gated_delta_kernel_policy, str
    ) or gated_delta_kernel_policy not in {
        "native_torch",
        "accelerated",
    }:
        raise Qwen35RuntimeError("Qwen gated-delta kernel policy is unsupported")
    if attention != "sdpa":
        raise Qwen35RuntimeError("Qwen Decision releases require SDPA attention")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise Qwen35RuntimeError("Qwen calibration temperature is invalid")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or not 1 <= max_length <= RELEASED_MAX_INPUT_TOKENS
    ):
        raise Qwen35RuntimeError("Qwen max input length is invalid")
    if (
        isinstance(physical_batch_size, bool)
        or not isinstance(physical_batch_size, int)
        or physical_batch_size < 1
    ):
        raise Qwen35RuntimeError("Qwen physical batch size is invalid")
    if backend == "rocm" and gated_delta_kernel_policy == "native_torch":
        maximum = native_rocm_max_physical_batch_size
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise Qwen35RuntimeError(
                "Qwen native Torch ROCm requires a qualified physical-batch maximum"
            )
        if physical_batch_size > maximum:
            raise Qwen35RuntimeError(
                "Qwen native Torch ROCm physical batch size "
                f"{physical_batch_size} exceeds qualified maximum {maximum}"
            )
    required_files = (
        INFERENCE_FILES if repository_config is None else repository_config.files
    )
    for relative in required_files:
        if not (root / relative).is_file():
            raise Qwen35RuntimeError(f"verified Qwen artifact is missing {relative}")
    # Prompt rendering belongs to this runtime; the checkpoint's prompt_version
    # is author metadata and does not define the executable model structure.
    _read_json(
        root
        / (
            repository_config.model_config
            if repository_config is not None
            else "decision_config.json"
        )
    )
    if (
        repository_config is None
        and backend == "rocm"
        and gated_delta_kernel_policy == "accelerated"
    ):
        profile = _validated_rocm_profile(root)
        if profile is not None and (
            physical_batch_size > profile.physical_batch_size_max
            or max_length > profile.padded_tokens_max
        ):
            raise Qwen35RuntimeError(
                "Qwen launch exceeds the verified ROCm kernel envelope"
            )
        return profile
    return None


def _validate_device(
    torch,
    device,
    *,
    backend: str,
    rocm_profile: ValidatedQwenRocmProfile | None,
) -> None:
    if backend == "cpu":
        if device.type != "cpu":
            raise Qwen35RuntimeError("CPU Qwen backend requires a CPU device")
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Qwen35RuntimeError("Qwen Torch runtime requires an available GPU")
    is_rocm = torch.version.hip is not None
    if backend == "rocm" and not is_rocm:
        raise Qwen35RuntimeError("ROCm backend requires a HIP-enabled Torch build")
    if backend == "cuda" and is_rocm:
        raise Qwen35RuntimeError("CUDA backend requires a non-HIP Torch build")
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise Qwen35RuntimeError("Qwen Torch runtime requires BF16 GPU support")
    if backend == "rocm" and rocm_profile is not None:
        expected = dict(rocm_profile.runtime)
        if (
            str(torch.__version__) != expected["torch"]
            or torch.version.hip != expected["hip"]
        ):
            raise Qwen35RuntimeError(
                "Torch/HIP runtime does not match the bound Qwen ROCm profile"
            )
        properties = torch.cuda.get_device_properties(device)
        architecture = getattr(properties, "gcnArchName", "").split(":")[0]
        if architecture != expected["gpu_arch"]:
            raise Qwen35RuntimeError(
                "GPU architecture does not match the bound Qwen ROCm profile"
            )


def _bind_rocm_profile(
    binder: QwenRocmProfileBinder | None,
    profile: ValidatedQwenRocmProfile,
) -> QwenRocmProfileBinding:
    if binder is None:  # pragma: no cover - guarded before dependency import
        raise Qwen35RuntimeError("strict ROCm profile binder is unavailable")
    try:
        receipt = binder.bind(profile)
    except Exception as error:
        raise Qwen35RuntimeError(
            f"owned ROCm profile binding failed: {type(error).__name__}"
        ) from error
    if (
        not isinstance(receipt, QwenRocmProfileBinding)
        or not receipt.strict
        or receipt.profile_sha256 != profile.profile_sha256
        or receipt.kernel_config_sha256 != profile.kernel_config_sha256
        or receipt.guard_contract_sha256 != profile.guard_contract_sha256
        or receipt.runtime != profile.runtime
        or receipt.fla_source_sha256 != profile.fla_source_sha256
        or receipt.physical_batch_size_min != profile.physical_batch_size_min
        or receipt.physical_batch_size_max != profile.physical_batch_size_max
        or receipt.normalization_blocks_min != profile.normalization_blocks_min
        or receipt.normalization_blocks_max != profile.normalization_blocks_max
        or not receipt.source_hashes_verified
        or not receipt.strict_cache_enforced
        or not receipt.unknown_key_guard_enforced
    ):
        raise Qwen35RuntimeError(
            "owned ROCm profile binder returned an invalid receipt"
        )
    return receipt


def _validated_rocm_profile(root: Path) -> ValidatedQwenRocmProfile | None:
    runtime = _read_json(root / "runtime.json")
    specification = runtime.get("normalization_profile")
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise Qwen35RuntimeError("invalid Qwen ROCm profile specification")
    if (
        specification.get("kind") != _ROCM_PROFILE_FORMAT
        or specification.get("validated_arch") != "gfx942"
    ):
        raise Qwen35RuntimeError("unsupported Qwen ROCm profile specification")

    profile_path = _safe_profile_path(root, specification.get("profile_file"))
    expected_profile_sha = _sha256_value(specification.get("profile_sha256"))
    if _file_sha256(profile_path) != expected_profile_sha:
        raise Qwen35RuntimeError("Qwen ROCm profile hash mismatch")
    profile = _read_json(profile_path)
    if (
        profile.get("format") != _ROCM_PROFILE_FORMAT
        or profile.get("cache_mode") != "strict"
    ):
        raise Qwen35RuntimeError("unsupported Qwen ROCm launch profile")
    # ``status`` is author-supplied evaluation provenance, not permission to
    # execute an installed kernel. The runtime validates the complete launch
    # envelope, artifact-bound hashes, installed FLA source, and live GPU below.

    # Select the kernel envelope from the verified backbone geometry. Model IDs
    # and profile model_family labels are author metadata, not tensor contracts.
    backbone_config = _read_json(root / "backbone/config.json")
    supported = _validate_supported_profile(profile.get("supported"), backbone_config)
    fla_source_sha256 = _validate_fla_source_hashes(profile.get("fla_source_sha256"))

    expected_runtime_keys = {"torch", "hip", "triton", "fla", "gpu_arch"}
    profile_runtime = profile.get("runtime")
    if (
        not isinstance(profile_runtime, dict)
        or set(profile_runtime) != expected_runtime_keys
        or any(
            not isinstance(value, str) or not value
            for value in profile_runtime.values()
        )
        or profile_runtime.get("gpu_arch") != "gfx942"
    ):
        raise Qwen35RuntimeError("invalid Qwen ROCm profile runtime")

    files = profile.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise Qwen35RuntimeError("invalid Qwen ROCm profile file set")
    kernel = files[0]
    if kernel.get("file") != "l2norm_fwd_kernel.json":
        raise Qwen35RuntimeError("unexpected Qwen ROCm kernel profile")
    kernel_path = _safe_profile_path(profile_path.parent, kernel.get("file"))
    expected_kernel_sha = _sha256_value(kernel.get("sha256"))
    if _file_sha256(kernel_path) != expected_kernel_sha:
        raise Qwen35RuntimeError("Qwen ROCm kernel-profile hash mismatch")
    _validate_kernel_profile(
        _read_json(kernel_path),
        minimum=supported["NB_min"],
        maximum=supported["NB_max"],
    )

    guard_path = _safe_profile_path(root, specification.get("guard_file"))
    guard_sha = _sha256_value(specification.get("guard_sha256"))
    if _file_sha256(guard_path) != guard_sha:
        raise Qwen35RuntimeError("Qwen ROCm guard-contract hash mismatch")

    return ValidatedQwenRocmProfile(
        profile_path=profile_path,
        kernel_config_path=kernel_path,
        profile_sha256=expected_profile_sha,
        kernel_config_sha256=expected_kernel_sha,
        guard_contract_sha256=guard_sha,
        runtime=tuple(sorted(profile_runtime.items())),
        fla_source_sha256=fla_source_sha256,
        physical_batch_size_min=supported["batch_size_min"],
        physical_batch_size_max=supported["batch_size_max"],
        padded_tokens_max=supported["padded_tokens_max"],
        normalization_blocks_min=supported["NB_min"],
        normalization_blocks_max=supported["NB_max"],
        normalization_blocks_formula=supported["NB_formula"],
        unknown_key_policy=supported["unknown_key"],
    )


def _validate_supported_profile(
    value: object, backbone_config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Qwen35RuntimeError("invalid Qwen ROCm supported envelope")
    if (
        backbone_config.get("model_type") != "qwen3_5_text"
        or backbone_config.get("linear_num_key_heads") != _QWEN_LINEAR_KEY_HEADS
        or backbone_config.get("linear_key_head_dim") != _QWEN_LINEAR_HEAD_DIM
        or backbone_config.get("linear_value_head_dim") != _QWEN_LINEAR_HEAD_DIM
        or backbone_config.get("linear_num_value_heads") not in {16, 32}
    ):
        raise Qwen35RuntimeError("unsupported Qwen ROCm backbone geometry")
    common = {
        "head_dimension": 128,
        "body_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "rstd_dtype": "float32",
        "batch_size_min": 1,
        "batch_size_max": 8,
        "padded_tokens_max": RELEASED_MAX_INPUT_TOKENS,
        "NB_min": 1,
        "unknown_key": "raise before kernel/autotune",
    }
    if backbone_config["linear_num_value_heads"] == _QWEN_SHORT_LINEAR_VALUE_HEADS:
        expected = {
            **common,
            "key_heads": 16,
            "NB_max": 32,
            "NB_formula": "ceil(B*padded_tokens*16/65536)",
        }
    else:
        expected = {
            **common,
            "configured_key_heads": 16,
            "normalized_qk_heads_after_repeat": 32,
            "value_heads": 32,
            "NB_max": 64,
            "NB_formula": "ceil(B*padded_tokens*32/65536)",
        }
    if value != expected:
        raise Qwen35RuntimeError("unsupported Qwen ROCm execution envelope")
    return value


def _validate_fla_source_hashes(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != set(_FLA_SOURCE_FILES):
        raise Qwen35RuntimeError("invalid Qwen ROCm FLA source identity")
    validated = tuple((path, _sha256_value(value[path])) for path in _FLA_SOURCE_FILES)
    return validated


def _validate_kernel_profile(
    config: dict[str, Any], *, minimum: int, maximum: int
) -> None:
    entries = config.get("autotune_entries")
    if config.get("default_config") is not None or not isinstance(entries, dict):
        raise Qwen35RuntimeError("Qwen ROCm kernel profile permits an unsafe fallback")
    batch_sizes = set()
    for digest, entry in entries.items():
        if not isinstance(digest, str) or not isinstance(entry, dict):
            raise Qwen35RuntimeError("invalid Qwen ROCm kernel-profile entry")
        key = entry.get("autotune_key")
        encoded = json.dumps(key, sort_keys=True, separators=(",", ":"))
        # The upstream cache-key contract uses MD5 only as a key identifier.
        actual_digest = hashlib.md5(encoded.encode(), usedforsecurity=False).hexdigest()
        if (
            digest != actual_digest
            or not isinstance(key, list)
            or len(key) != _FLA_AUTOTUNE_KEY_FIELDS
            or key[0] != _FLA_L2NORM_WIDTH
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or not 1 <= key[1] <= _FLA_MAX_NORMALIZATION_BLOCKS
            or key[2:] != ["torch.bfloat16", "torch.bfloat16", "torch.float32"]
        ):
            raise Qwen35RuntimeError("invalid Qwen ROCm kernel-profile key")
        launch = entry.get("config")
        if not isinstance(launch, dict):
            raise Qwen35RuntimeError("invalid Qwen ROCm kernel launch configuration")
        if (
            launch.get("kwargs") not in ({"BT": 8}, {"BT": 16}, {"BT": 32}, {"BT": 64})
            or launch.get("num_warps") not in {1, 2, 4, 8, 16}
            or launch.get("num_stages") != _FLA_NUM_STAGES
            or launch.get("num_ctas") != 1
            or any(
                launch.get(field) is not None
                for field in ("maxnreg", "pre_hook", "ir_override")
            )
        ):
            raise Qwen35RuntimeError("invalid Qwen ROCm kernel launch configuration")
        batch_sizes.add(key[1])
    if batch_sizes != set(range(minimum, maximum + 1)):
        raise Qwen35RuntimeError("incomplete Qwen ROCm kernel-profile coverage")


def _safe_profile_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Qwen35RuntimeError("unsafe Qwen ROCm profile path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise Qwen35RuntimeError("unsafe Qwen ROCm profile path")
    try:
        path = (root / Path(*relative.parts)).resolve(strict=True)
    except OSError as error:
        raise Qwen35RuntimeError("Qwen ROCm profile file is missing") from error
    resolved_root = root.resolve(strict=True)
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise Qwen35RuntimeError("unsafe Qwen ROCm profile path")
    return path


def _sha256_value(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Qwen35RuntimeError("invalid Qwen ROCm profile hash")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise Qwen35RuntimeError("Qwen ROCm profile file could not be read") from error
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen35RuntimeError(
            f"invalid verified Qwen metadata: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise Qwen35RuntimeError(f"invalid verified Qwen metadata: {path.name}")
    return value


def _required_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise Qwen35RuntimeError(
            f"Qwen3.5 Decision runtime dependency {name!r} is unavailable"
        ) from error
