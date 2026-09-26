"""Owned Torch execution core for Kai/Lex Vela Decision checkpoints.

The implementation is inference-only and consumes verified model data.  It
does not import Python from the Hugging Face repository.  Heavy dependencies
are loaded only by ``VelaTorchRuntime.load`` so normal CLI paths stay light.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Literal

from .release_artifacts import ReleaseArtifactError, verify_release_manifest
from .vela_inputs import MIN_VELA_INPUT_TOKENS, EncodedVelaRow

SUPPORTED_TRANSFORMERS_VERSION = "4.57.6"
REQUIRED_HEAD_STRUCTURE = {
    "head_layers": 2,
    "head_heads": 12,
}
KINDS = ("choice", "noul", "score")
_BATCH_TENSOR_RANK = 2
_ENCODER_MAX_POSITIONS = 32768
INFERENCE_FILES = (
    "INVENTORY.json",
    "STATE_LAYOUT.json",
    "decision_config.json",
    "encoder/config.json",
    "encoder/model.safetensors",
    "decision_heads.safetensors",
    "choice_encoder.safetensors",
    "score_encoder.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


class VelaRuntimeError(RuntimeError):
    """The owned Vela runtime cannot load or execute the checkpoint safely."""


@dataclass(frozen=True, slots=True)
class VelaPrediction:
    """One ordered probability result before public response adaptation."""

    question_id: str
    type: Literal["noul", "choice", "score"]
    probabilities: tuple[float, ...]
    input_tokens: int


@dataclass(frozen=True, slots=True)
class ValidatedVelaArtifact:
    """Release data verified before importing any accelerator dependency."""

    config: dict[str, Any]
    inventory: dict[str, dict[str, tuple[int, ...]]]


@dataclass(slots=True)
class VelaTorchRuntime:
    """One resident three-path Vela Decision model."""

    model: Any
    tokenizer: Any
    torch: Any
    device: Any
    max_length: int
    backend: Literal["cpu", "rocm", "cuda"]

    @classmethod
    def load(
        cls,
        artifact_root: Path,
        *,
        max_length: int,
        backend: Literal["cpu", "rocm", "cuda"],
        device: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> VelaTorchRuntime:
        """Load the exact native data layout without executing bundled code."""

        root = Path(artifact_root).resolve(strict=True)
        if backend == "cpu" and expected_manifest_sha256 is None:
            raise VelaRuntimeError("CPU Vela requires a pinned release manifest digest")
        artifact = _validate_artifact(root, max_length=max_length, backend=backend)
        if expected_manifest_sha256 is not None:
            try:
                verify_release_manifest(
                    root,
                    manifest_name="MANIFEST.json",
                    expected_sha256=expected_manifest_sha256,
                    required_files=INFERENCE_FILES,
                )
            except ReleaseArtifactError as error:
                raise VelaRuntimeError(
                    f"Vela artifact integrity failed: {error}"
                ) from error
        torch = _required_module("torch")
        transformers = _required_module("transformers")
        safetensors = _required_module("safetensors.torch")
        if getattr(transformers, "__version__", None) != SUPPORTED_TRANSFORMERS_VERSION:
            raise VelaRuntimeError(
                f"Vela Decision runtime requires Transformers "
                f"{SUPPORTED_TRANSFORMERS_VERSION}"
            )
        selected = torch.device(
            device if device is not None else ("cpu" if backend == "cpu" else "cuda:0")
        )
        _validate_device(torch, selected, backend=backend)
        try:
            encoder_config = transformers.AutoConfig.from_pretrained(
                root / "encoder",
                local_files_only=True,
                trust_remote_code=False,
            )
            _validate_encoder_config(encoder_config)
            encoder = transformers.AutoModel.from_config(
                encoder_config,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )
            base_state = safetensors.load_file(
                str(root / "encoder/model.safetensors"), device="cpu"
            )
            _validate_tensor_inventory(
                torch,
                base_state,
                artifact.inventory["encoder_shapes"],
                label="Vela shared encoder",
            )
            encoder.load_state_dict(base_state, strict=True)
            if backend == "rocm":
                _install_rocm_layout_guard(torch, encoder)
            model = _vela_model(torch, encoder, artifact.config["head"])
            persisted = {}
            for filename, inventory_key in (
                ("decision_heads.safetensors", "head_shapes"),
                ("choice_encoder.safetensors", "choice_suffix_shapes"),
                ("score_encoder.safetensors", "score_suffix_shapes"),
            ):
                state = safetensors.load_file(str(root / filename), device="cpu")
                _validate_tensor_inventory(
                    torch,
                    state,
                    artifact.inventory[inventory_key],
                    label=f"Vela {inventory_key}",
                )
                overlap = set(persisted).intersection(state)
                if overlap:
                    raise VelaRuntimeError("Vela state files are not disjoint")
                persisted.update(state)
            state = model.state_dict()
            _validate_model_inventory(torch, state, artifact.inventory)
            unknown = set(persisted) - set(state)
            if unknown:
                raise VelaRuntimeError("Vela checkpoint contains unknown state tensors")
            state.update(persisted)
            model.load_state_dict(state, strict=True)
            model.to(selected).eval()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                root / "tokenizer",
                local_files_only=True,
                trust_remote_code=False,
            )
        except VelaRuntimeError:
            raise
        except Exception as error:
            raise VelaRuntimeError(
                f"Vela Decision checkpoint could not be loaded: {type(error).__name__}"
            ) from error
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise VelaRuntimeError("Vela checkpoint storage must be FP32")
        return cls(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=selected,
            max_length=max_length,
            backend=backend,
        )

    def predict_encoded(
        self, rows: tuple[EncodedVelaRow, ...]
    ) -> tuple[VelaPrediction, ...]:
        """Run one homogeneous physical batch and preserve row identity."""

        if not rows:
            raise ValueError("at least one encoded Vela row is required")
        kind = _homogeneous_kind(rows)
        batch = _collate(
            self.torch, rows, self.tokenizer, device=self.device, kind=kind
        )
        try:
            with self.torch.inference_mode():
                logits = self.model(batch)
                valid_logits = []
                probabilities = []
                for row, values in zip(rows, logits, strict=True):
                    valid_values = values[: len(row.marker_positions)]
                    valid_logits.append(valid_values)
                    probabilities.append(valid_values.softmax(-1))
                # Transfer raw valid logits and probabilities together. Checking
                # each row's GPU tensor as a Python bool would synchronize once
                # per row; the combined transfer synchronizes once per batch.
                host = self.torch.cat(valid_logits + probabilities).tolist()
                candidate_count = sum(len(row.marker_positions) for row in rows)
                if len(host) != 2 * candidate_count:
                    raise VelaRuntimeError("invalid Vela probability vector")
                if any(not math.isfinite(value) for value in host[:candidate_count]):
                    raise VelaRuntimeError("non-finite Vela candidate logits")
                host = host[candidate_count:]
        except VelaRuntimeError:
            raise
        except Exception as error:
            raise VelaRuntimeError(
                f"Vela Decision inference failed: {type(error).__name__}"
            ) from error

        output = []
        offset = 0
        for row in rows:
            count = len(row.marker_positions)
            values = tuple(float(value) for value in host[offset : offset + count])
            offset += count
            if (
                len(values) != count
                or any(not math.isfinite(value) or value < 0.0 for value in values)
                or not math.isclose(math.fsum(values), 1.0, abs_tol=2e-5, rel_tol=0.0)
            ):
                raise VelaRuntimeError("invalid Vela probability vector")
            output.append(
                VelaPrediction(
                    question_id=row.question_id,
                    type=row.type,  # type: ignore[arg-type]
                    probabilities=values,
                    input_tokens=row.input_tokens,
                )
            )
        return tuple(output)


def _vela_model(torch, encoder, head_config):
    nn = torch.nn

    class VelaDecisionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            hidden = encoder.config.hidden_size
            self.type_embedding = nn.Embedding(3, hidden)
            self.heads = nn.ModuleDict(
                {
                    kind: nn.ModuleList(
                        [
                            nn.TransformerEncoderLayer(
                                hidden,
                                REQUIRED_HEAD_STRUCTURE["head_heads"],
                                4 * hidden,
                                0.1,
                                activation="relu",
                                batch_first=True,
                                norm_first=True,
                            )
                            for _ in range(REQUIRED_HEAD_STRUCTURE["head_layers"])
                        ]
                    )
                    for kind in KINDS
                }
            )
            self.scorers = nn.ModuleDict(
                {
                    kind: nn.Sequential(
                        nn.LayerNorm(hidden),
                        nn.Linear(hidden, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, 1),
                    )
                    for kind in KINDS
                }
            )
            self.choice_blocks = nn.ModuleList(
                [copy.deepcopy(layer) for layer in encoder.layers]
            )
            self.choice_final_norm = copy.deepcopy(encoder.final_norm)
            self.score_blocks = nn.ModuleList(
                [copy.deepcopy(layer) for layer in encoder.layers]
            )
            self.score_final_norm = copy.deepcopy(encoder.final_norm)

        @staticmethod
        def _path(hidden, layers, final_norm, layer_kwargs):
            for layer in layers:
                hidden = layer(hidden, **layer_kwargs)[0]
            return final_norm(hidden)

        def forward(self, batch):
            kind_ids = batch["kind_ids"]
            kind = batch["kind"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            if (
                kind not in KINDS
                or input_ids.ndim != _BATCH_TENSOR_RANK
                or attention_mask.shape != input_ids.shape
                or input_ids.shape[0] != kind_ids.numel()
            ):
                raise VelaRuntimeError("invalid Vela tensor batch")
            self.encoder._maybe_set_compile()
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
            global_mask, local_mask = self.encoder._update_attention_mask(
                attention_mask, output_attentions=False
            )
            layer_kwargs = {
                "attention_mask": global_mask,
                "sliding_window_mask": local_mask,
                "position_ids": position_ids,
                "cu_seqlens": None,
                "max_seqlen": None,
                "output_attentions": False,
            }
            embedded = self.encoder.embeddings(input_ids=input_ids, inputs_embeds=None)
            if kind == "choice":
                hidden = self._path(
                    embedded, self.choice_blocks, self.choice_final_norm, layer_kwargs
                )
            elif kind == "noul":
                hidden = self._path(
                    embedded, self.encoder.layers, self.encoder.final_norm, layer_kwargs
                )
            else:
                hidden = self._path(
                    embedded, self.score_blocks, self.score_final_norm, layer_kwargs
                )
            pad = ~attention_mask.bool()
            hidden = hidden + self.type_embedding(kind_ids)[:, None, :].to(hidden.dtype)
            for layer in self.heads[kind]:
                hidden = layer(hidden, src_key_padding_mask=pad)
            positions = batch["marker_positions"]
            markers = torch.gather(
                hidden,
                1,
                positions[:, :, None].expand(-1, -1, hidden.shape[-1]),
            )
            logits = self.scorers[kind](markers).squeeze(-1).float()
            return logits.masked_fill(
                ~batch["valid_candidates"], torch.finfo(torch.float32).min
            )

    return VelaDecisionModel()


def _homogeneous_kind(rows: tuple[EncodedVelaRow, ...]) -> str:
    kind = rows[0].type
    if kind not in KINDS or any(row.type != kind for row in rows[1:]):
        raise ValueError("a Vela physical batch must contain exactly one question type")
    return kind


def _collate(torch, rows, tokenizer, *, device, kind):
    length = max(row.input_tokens for row in rows)
    candidates = max(len(row.marker_positions) for row in rows)
    pad = tokenizer.pad_token_id
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
        raise VelaRuntimeError("Vela tokenizer has no valid PAD token")
    input_ids = torch.full((len(rows), length), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), length), dtype=torch.bool)
    positions = torch.zeros((len(rows), candidates), dtype=torch.long)
    valid = torch.zeros((len(rows), candidates), dtype=torch.bool)
    for index, row in enumerate(rows):
        size = row.input_tokens
        count = len(row.marker_positions)
        input_ids[index, :size] = torch.tensor(row.input_ids, dtype=torch.long)
        attention_mask[index, :size] = True
        positions[index, :count] = torch.tensor(row.marker_positions, dtype=torch.long)
        valid[index, :count] = True
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "marker_positions": positions.to(device),
        "valid_candidates": valid.to(device),
        "kind": kind,
        "kind_ids": torch.tensor(
            [KINDS.index(row.type) for row in rows],
            dtype=torch.long,
            device=device,
        ),
    }


def _install_rocm_layout_guard(torch, encoder) -> None:
    modeling = importlib.import_module(
        "transformers.models.modernbert.modeling_modernbert"
    )
    if not isinstance(encoder, modeling.ModernBertModel):
        raise VelaRuntimeError("Vela encoder must be native ModernBERT")

    def contiguous_forward(
        attention,
        hidden_states,
        *,
        attention_mask,
        sliding_window_mask,
        position_ids,
    ):
        batch_size = hidden_states.shape[0]
        qkv = attention.Wqkv(hidden_states).view(
            batch_size, -1, 3, attention.num_heads, attention.head_dim
        )
        cos, sin = attention.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = modeling.apply_rotary_pos_emb(query, key, cos, sin)
        mask = (
            attention_mask
            if attention.local_attention == (-1, -1)
            else sliding_window_mask
        )
        output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=mask,
            dropout_p=attention.attention_dropout if attention.training else 0.0,
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, attention.all_head_size)
        )
        return (attention.out_drop(attention.Wo(output)),)

    def guarded_forward(self, hidden_states, output_attentions=False, **kwargs):
        if (
            torch.version.hip is None
            or hidden_states.device.type != "cuda"
            or not torch.backends.cuda.mem_efficient_sdp_enabled()
            or self.config._attn_implementation != "sdpa"
            or output_attentions
        ):
            return self._vllm_sr_original_attention_forward(
                hidden_states, output_attentions=output_attentions, **kwargs
            )
        return contiguous_forward(
            self,
            hidden_states,
            attention_mask=kwargs["attention_mask"],
            sliding_window_mask=kwargs["sliding_window_mask"],
            position_ids=kwargs["position_ids"],
        )

    for layer in encoder.layers:
        attention = layer.attn
        if not isinstance(attention, modeling.ModernBertAttention):
            raise VelaRuntimeError("unexpected Vela attention module")
        if hasattr(attention, "_vllm_sr_original_attention_forward"):
            raise VelaRuntimeError("Vela attention guard is already installed")
        attention._vllm_sr_original_attention_forward = attention.forward
        attention.forward = MethodType(guarded_forward, attention)


def _validate_artifact(
    root: Path, *, max_length: int, backend: str
) -> ValidatedVelaArtifact:
    if backend not in {"cpu", "rocm", "cuda"}:
        raise VelaRuntimeError("Vela Torch backend must be cpu, rocm, or cuda")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < MIN_VELA_INPUT_TOKENS
    ):
        raise VelaRuntimeError("Vela max input length is invalid")
    config = _read_json(root / "decision_config.json")
    # Provenance labels do not define the execution contract. Compatibility
    # depends on tensor layout and fields used to interpret those tensors.
    head = config.get("head")
    if not isinstance(head, dict) or any(
        head.get(key) != expected for key, expected in REQUIRED_HEAD_STRUCTURE.items()
    ):
        raise VelaRuntimeError("Vela checkpoint head structure mismatch")
    if config.get("type_order") != list(KINDS):
        raise VelaRuntimeError("Vela checkpoint type order mismatch")
    packing = config.get("packing")
    if not isinstance(packing, dict) or packing.get("state_truncation") != "error":
        raise VelaRuntimeError("Vela checkpoint packing contract mismatch")
    packed_max_length = packing.get("max_length")
    if (
        isinstance(packed_max_length, bool)
        or not isinstance(packed_max_length, int)
        or packed_max_length < MIN_VELA_INPUT_TOKENS
    ):
        raise VelaRuntimeError("Vela checkpoint packing contract mismatch")
    if max_length > packed_max_length:
        raise VelaRuntimeError("Vela max input length exceeds the release packing")
    if max_length > _ENCODER_MAX_POSITIONS:
        raise VelaRuntimeError("Vela max input length exceeds encoder positions")
    for relative in (
        "encoder/config.json",
        "encoder/model.safetensors",
        "decision_heads.safetensors",
        "choice_encoder.safetensors",
        "score_encoder.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "INVENTORY.json",
        "STATE_LAYOUT.json",
    ):
        if not (root / relative).is_file():
            raise VelaRuntimeError(f"verified Vela artifact is missing {relative}")
    # This sidecar is manifest-verified provenance, not an execution input.
    _read_json(root / "STATE_LAYOUT.json")
    inventory = _validate_inventory(_read_json(root / "INVENTORY.json"))
    return ValidatedVelaArtifact(config=config, inventory=inventory)


def _validate_inventory(value: dict[str, Any]) -> dict[str, dict[str, tuple[int, ...]]]:
    groups = (
        "encoder_shapes",
        "choice_suffix_shapes",
        "score_suffix_shapes",
        "head_shapes",
    )
    parsed: dict[str, dict[str, tuple[int, ...]]] = {}
    all_names: set[str] = set()
    for name in groups:
        group = value.get(name)
        if not isinstance(group, dict) or not group:
            raise VelaRuntimeError(f"Vela tensor-inventory group is invalid: {name}")
        normalized: dict[str, tuple[int, ...]] = {}
        for tensor_name, shape in group.items():
            if (
                not isinstance(tensor_name, str)
                or not tensor_name
                or not isinstance(shape, list)
                or not shape
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 1
                    for dimension in shape
                )
            ):
                raise VelaRuntimeError("Vela tensor-inventory shape is invalid")
            normalized[tensor_name] = tuple(shape)
        names = set(normalized)
        if all_names.intersection(names):
            raise VelaRuntimeError("Vela tensor inventories are not disjoint")
        all_names.update(names)
        parsed[name] = normalized
    _validate_inventory_names(parsed)
    return parsed


def _validate_inventory_names(inventory: dict[str, dict[str, tuple[int, ...]]]) -> None:
    allowed_prefixes = {
        "encoder_shapes": ("embeddings.", "layers.", "final_norm."),
        "choice_suffix_shapes": ("choice_blocks.", "choice_final_norm."),
        "score_suffix_shapes": ("score_blocks.", "score_final_norm."),
        "head_shapes": ("type_embedding.", "heads.", "scorers."),
    }
    for group, prefixes in allowed_prefixes.items():
        if any(not name.startswith(prefixes) for name in inventory[group]):
            raise VelaRuntimeError(f"Vela tensor-inventory names are invalid: {group}")


def _validate_tensor_inventory(torch, state, expected, *, label: str) -> None:
    if set(state) != set(expected):
        raise VelaRuntimeError(
            f"{label} tensor keys do not match the release inventory"
        )
    for name, tensor in state.items():
        if tuple(tensor.shape) != expected[name]:
            raise VelaRuntimeError(f"{label} tensor shape mismatch: {name}")
        if tensor.dtype != torch.float32:
            raise VelaRuntimeError(f"{label} tensor dtype must be FP32: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise VelaRuntimeError(f"{label} tensor contains non-finite values: {name}")


def _validate_model_inventory(torch, state, inventory) -> None:
    expected = {
        **{
            f"encoder.{name}": shape
            for name, shape in inventory["encoder_shapes"].items()
        },
        **inventory["choice_suffix_shapes"],
        **inventory["score_suffix_shapes"],
        **inventory["head_shapes"],
    }
    _validate_tensor_inventory(torch, state, expected, label="Vela model")


def _validate_encoder_config(config) -> None:
    required = {
        "model_type": "modernbert",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "max_position_embeddings": _ENCODER_MAX_POSITIONS,
    }
    for key, value in required.items():
        if getattr(config, key, None) != value:
            raise VelaRuntimeError(f"Vela encoder configuration mismatch: {key}")


def _validate_device(torch, device, *, backend: str) -> None:
    if backend == "cpu":
        if device.type != "cpu":
            raise VelaRuntimeError("CPU Vela backend requires a CPU device")
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise VelaRuntimeError("Vela Torch runtime requires an available GPU")
    is_rocm = torch.version.hip is not None
    if backend == "rocm" and not is_rocm:
        raise VelaRuntimeError("ROCm backend requires a HIP-enabled Torch build")
    if backend == "cuda" and is_rocm:
        raise VelaRuntimeError("CUDA backend requires a non-HIP Torch build")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VelaRuntimeError(
            f"invalid verified Vela metadata: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise VelaRuntimeError(f"invalid verified Vela metadata: {path.name}")
    return value


def _required_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise VelaRuntimeError(
            f"Vela Decision runtime dependency {name!r} is unavailable"
        ) from error
