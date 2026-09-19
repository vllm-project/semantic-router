"""Full-encoder task models initialized from an exact encoder checkpoint.

No task checkpoint, remote code, inferred RoPE setting or teacher is loaded by
``from_base``. Resume is a separate path with a complete checkpoint manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel

from .newbase_data import file_digest
from .representation_contract import (
    read_representation_contract,
    set_vela_representation_contract,
    vela_representation_contract,
)
from .representation_outputs import (
    forward_with_raw_hidden_states,
    masked_mean,
    select_hidden_state,
    truncate_and_normalize,
)

_SHA256_LENGTH = 64
_HEAD_DIVISOR = 2
_WEIGHT_SUM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ExitSpec:
    layers: tuple[int, ...] = (3, 6, 11, 22)
    dimensions: tuple[int, ...] = (768, 512, 256, 128, 64)
    layer_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.4)
    dimension_weights: tuple[float, ...] = (2 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6)
    exit_weights: tuple[tuple[float, ...], ...] | None = None

    def validate(self, layer_count: int, hidden_size: int) -> None:
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("Exits must use unique ascending layers")
        if (
            not self.dimensions
            or tuple(sorted(set(self.dimensions), reverse=True)) != self.dimensions
        ):
            raise ValueError("Exit dimensions must be unique and descending")
        if self.layers[0] < 1 or self.layers[-1] != layer_count:
            raise ValueError("Exits must include the exact full encoder depth")
        if self.dimensions[-1] < _HEAD_DIVISOR or self.dimensions[0] != hidden_size:
            raise ValueError("Exits must include the exact full hidden width")
        for indices, weights in (
            (self.layers, self.layer_weights),
            (self.dimensions, self.dimension_weights),
        ):
            if len(indices) != len(weights) or any(
                not 0 < weight <= 1 for weight in weights
            ):
                raise ValueError("Every exit requires a positive finite weight")
            if abs(sum(weights) - 1.0) > _WEIGHT_SUM_TOLERANCE:
                raise ValueError("Exit weights must sum to one on each axis")
        if self.exit_weights is not None:
            if len(self.exit_weights) != len(self.layers) or any(
                len(row) != len(self.dimensions) for row in self.exit_weights
            ):
                raise ValueError(
                    "Explicit exit weights require the complete exit matrix"
                )
            weights = [weight for row in self.exit_weights for weight in row]
            if (
                any(not 0 < weight <= 1 for weight in weights)
                or abs(sum(weights) - 1.0) > _WEIGHT_SUM_TOLERANCE
            ):
                raise ValueError(
                    "Explicit exit weights must be positive and sum to one"
                )

    def weighted(self):
        if self.exit_weights is not None:
            for layer, row in zip(self.layers, self.exit_weights, strict=True):
                for dimension, weight in zip(self.dimensions, row, strict=True):
                    yield (layer, dimension), weight
            return
        for layer, layer_weight in zip(self.layers, self.layer_weights, strict=True):
            for dimension, dimension_weight in zip(
                self.dimensions, self.dimension_weights, strict=True
            ):
                yield (layer, dimension), layer_weight * dimension_weight

    def to_dict(self) -> dict:
        result = {
            name: list(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "exit_weights"
        }
        if self.exit_weights is not None:
            result["exit_weights"] = [list(row) for row in self.exit_weights]
        return result

    @classmethod
    def from_dict(cls, value: dict) -> ExitSpec:
        required = set(cls.__dataclass_fields__) - {"exit_weights"}
        if not required <= set(value) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Incomplete or unknown exit contract")
        args = {name: tuple(value[name]) for name in required}
        if "exit_weights" in value:
            args["exit_weights"] = tuple(tuple(row) for row in value["exit_weights"])
        return cls(**args)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Bind names, shapes, dtypes and actual values, including all task heads."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(
            json.dumps(
                [name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")
            ).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_files(directory: Path, expected: dict[str, str]) -> None:
    """Only explicit relative files may participate in an identity lock."""
    if not expected:
        raise ValueError("An empty artifact identity is not a lock")
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact paths must stay inside their directory")
        if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH:
            raise ValueError("Artifact requires a complete SHA256")
        if file_digest(directory / relative) != digest:
            raise ValueError(f"Artifact bytes changed: {name}")


def _load_encoder(directory: Path):
    config = AutoConfig.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "modernbert":
        raise ValueError("This task requires a native ModernBERT Base")
    # reference_compile is a supported HF execution option, not model geometry.
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    encoder, info = AutoModel.from_pretrained(
        directory,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    unexpected = [
        key
        for key in info["unexpected_keys"]
        if not key.startswith(("head.", "decoder."))
    ]
    if (
        info["missing_keys"]
        or info["mismatched_keys"]
        or info["error_msgs"]
        or unexpected
    ):
        raise ValueError("Base encoder was not completely and exactly loaded")
    return encoder, info


class NewBaseTask(nn.Module):
    """All selected exits from one full forward; FP32 parameters and readout."""

    def __init__(self, encoder, task: str, exits: ExitSpec, lineage: dict):
        super().__init__()
        if task not in {"embedding", "reranker"}:
            raise ValueError("Unknown new-Base task")
        exits.validate(len(encoder.layers), encoder.config.hidden_size)
        if encoder.config.model_type != "modernbert":
            raise ValueError("Unsupported encoder architecture")
        self.encoder, self.task, self.exits, self.lineage = (
            encoder,
            task,
            exits,
            lineage,
        )
        self.contract = set_vela_representation_contract(encoder.config, task)
        self.layer_heads = nn.ModuleDict()
        if task == "reranker":
            for layer in exits.layers:
                heads = nn.ModuleDict()
                for dimension in exits.dimensions:
                    head = nn.Sequential(
                        nn.Linear(dimension, dimension // _HEAD_DIVISOR),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(dimension // _HEAD_DIVISOR, 1),
                    )
                    for module in head:
                        if isinstance(module, nn.Linear):
                            nn.init.normal_(module.weight, std=0.02)
                            nn.init.zeros_(module.bias)
                    heads[str(dimension)] = head
                self.layer_heads[str(layer)] = heads
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.assert_parameters()

    def assert_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise ValueError(
                    f"Every encoder/head parameter must train in FP32: {name}"
                )

    @classmethod
    def from_base(
        cls,
        directory: Path,
        task: str,
        exits: ExitSpec,
        *,
        expected_files: dict[str, str],
        provenance: dict | None = None,
    ):
        if (
            not {"config.json", "tokenizer.json", "model.safetensors"}
            <= expected_files.keys()
        ):
            raise ValueError("Base lock must bind native weights, config and tokenizer")
        verify_files(directory, expected_files)
        config = json.loads((directory / "config.json").read_bytes())
        if (
            "representation_contract" in config
            or (directory / "classification_heads.safetensors").exists()
            or (directory / "classification_heads.pt").exists()
        ):
            raise ValueError("A task checkpoint cannot initialize a fresh Base task")
        encoder, info = _load_encoder(directory)
        original = load_file(str(directory / "model.safetensors"))
        for name, value in encoder.state_dict().items():
            key = name if name in original else "model." + name
            if key not in original or not torch.equal(
                value.cpu(), original[key].float()
            ):
                raise ValueError(
                    f"Loaded encoder differs from the locked checkpoint: {name}"
                )
        lineage = {
            "base_files": expected_files,
            "provenance": provenance or {},
            "initial_encoder_state_sha256": state_digest(encoder.state_dict()),
            "unused_mlm_keys": sorted(info["unexpected_keys"]),
            "initialization": "exact_encoder_fresh_task",
        }
        verify_files(directory, expected_files)
        return cls(encoder, task, exits, lineage)

    @classmethod
    def from_task(
        cls,
        directory: Path,
        task: str,
        exits: ExitSpec,
        *,
        expected_files: dict[str, str],
        provenance: dict,
        heads_file: str = "classification_heads.safetensors",
    ):
        """Continue a locked task artifact, explicitly distinct from fresh Base.

        Only this representation contract is supported. All encoder tensors and
        every reranker head must match; this is neither a partial head warm start
        nor optimizer resume. Source lineage is preserved as caller provenance.
        """
        required = {"config.json", "tokenizer.json", "model.safetensors"}
        if task == "reranker":
            required |= {heads_file, "matryoshka_config.json"}
        if not required <= expected_files.keys() or not provenance:
            raise ValueError(
                "Continued task requires complete files and explicit lineage"
            )
        verify_files(directory, expected_files)
        config = json.loads((directory / "config.json").read_bytes())
        contract = read_representation_contract(config, task)
        if contract != vela_representation_contract(task):
            raise ValueError("Continued task representation differs from the trainer")
        # Discarded temporary head initialization must not alter training RNG.
        with torch.random.fork_rng(devices=[]):
            encoder, _ = _load_encoder(directory)
            result = cls(encoder, task, exits, {})
        original = load_file(str(directory / "model.safetensors"))
        if set(original) != set(encoder.state_dict()) or any(
            value.dtype != original[name].dtype
            or not torch.equal(value.cpu(), original[name])
            for name, value in encoder.state_dict().items()
        ):
            raise ValueError("Continued encoder differs from its exact native tensors")
        if task == "reranker":
            metadata = json.loads((directory / "matryoshka_config.json").read_bytes())
            if any(
                metadata.get(name) != expected
                for name, expected in {
                    "layer_indices": list(exits.layers),
                    "dim_indices": list(exits.dimensions),
                    "hidden_size": encoder.config.hidden_size,
                    "num_layers": len(encoder.layers),
                    "pooling_strategy": "cls",
                    "representation_contract": contract,
                }.items()
            ):
                raise ValueError("Continued head layout or representation differs")
            state = load_file(str(directory / heads_file))
            expected = result.layer_heads.state_dict()
            if set(state) != set(expected) or any(
                value.dtype != expected[name].dtype
                or value.shape != expected[name].shape
                or not torch.isfinite(value).all()
                for name, value in state.items()
            ):
                raise ValueError("Continued task requires every exact FP32 head")
            result.layer_heads.load_state_dict(state, strict=True)
        result.lineage = {
            "initialization": "continued_task",
            "source_files": expected_files,
            "provenance": provenance,
            "initial_encoder_state_sha256": state_digest(encoder.state_dict()),
            "initial_task_state_sha256": state_digest(result.state_dict()),
        }
        verify_files(directory, expected_files)
        return result

    def forward(self, input_ids, attention_mask) -> dict[tuple[int, int], torch.Tensor]:
        if (
            input_ids.shape != attention_mask.shape
            or not attention_mask.any(dim=1).all()
        ):
            raise ValueError("Inputs require matching, nonempty attention masks")
        if input_ids.shape[1] > self.encoder.config.max_position_embeddings:
            raise ValueError("Whole input exceeds the actual Base context capacity")
        if self.task == "reranker" and not attention_mask[:, 0].all():
            raise ValueError(
                "CLS reranking requires a real first token; left padding is unsupported"
            )
        outputs = forward_with_raw_hidden_states(
            self.encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        result = {}
        for layer in self.exits.layers:
            states = select_hidden_state(
                self.encoder, outputs, layer, self.contract, task=self.task
            )
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                pooled = (
                    masked_mean(states, attention_mask)
                    if self.task == "embedding"
                    else states[:, 0].float()
                )
                for dimension in self.exits.dimensions:
                    result[layer, dimension] = (
                        truncate_and_normalize(pooled, dimension)
                        if self.task == "embedding"
                        else self.layer_heads[str(layer)][str(dimension)](
                            pooled[:, :dimension]
                        ).squeeze(-1)
                    )
        return result

    def save(self, directory: Path, tokenizer=None) -> dict:
        """Write a new immutable standard HF artifact and its task provenance."""
        self.assert_parameters()
        directory.mkdir(parents=True, exist_ok=False)
        self.encoder.save_pretrained(directory, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)
        if self.task == "reranker":
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in self.layer_heads.state_dict().items()
                },
                str(directory / "classification_heads.safetensors"),
            )
            metadata = {
                "layer_indices": list(self.exits.layers),
                "dim_indices": list(self.exits.dimensions),
                "hidden_size": self.encoder.config.hidden_size,
                "num_layers": len(self.encoder.layers),
                "pooling_strategy": "cls",
                "representation_contract": self.contract,
            }
            (directory / "matryoshka_config.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        else:
            pooling = directory / "1_Pooling"
            pooling.mkdir()
            (pooling / "config.json").write_text(
                json.dumps(
                    {
                        "word_embedding_dimension": self.encoder.config.hidden_size,
                        "pooling_mode_cls_token": False,
                        "pooling_mode_mean_tokens": True,
                        "pooling_mode_max_tokens": False,
                        "pooling_mode_mean_sqrt_len_tokens": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            # Normalize has no parameters or configuration file. The module
            # makes default SentenceTransformer.encode match the full exit.
            (directory / "2_Normalize").mkdir()
            (directory / "modules.json").write_text(
                json.dumps(
                    [
                        {
                            "idx": 0,
                            "name": "0",
                            "path": "",
                            "type": "sentence_transformers.models.Transformer",
                        },
                        {
                            "idx": 1,
                            "name": "1",
                            "path": "1_Pooling",
                            "type": "sentence_transformers.models.Pooling",
                        },
                        {
                            "idx": 2,
                            "name": "2",
                            "path": "2_Normalize",
                            "type": "sentence_transformers.models.Normalize",
                        },
                    ],
                    indent=2,
                )
                + "\n"
            )
        files = {
            str(path.relative_to(directory)): file_digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        receipt = {
            "version": 1,
            "task": self.task,
            "exits": self.exits.to_dict(),
            "lineage": self.lineage,
            "state_sha256": state_digest(self.state_dict()),
            "files": files,
        }
        (directory / "newbase_checkpoint.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        return receipt

    @classmethod
    def resume(cls, directory: Path):
        receipt = json.loads((directory / "newbase_checkpoint.json").read_bytes())
        verify_files(directory, receipt["files"])
        # Constructor initialization is isolated: resume never consumes the stream RNG.
        with torch.random.fork_rng(devices=[]):
            encoder, _ = _load_encoder(directory)
            if read_representation_contract(
                encoder.config, receipt["task"]
            ) != vela_representation_contract(receipt["task"]):
                raise ValueError(
                    "A resumed task needs an explicit representation contract"
                )
            result = cls(
                encoder,
                receipt["task"],
                ExitSpec.from_dict(receipt["exits"]),
                receipt["lineage"],
            )
        if result.task == "reranker":
            state = load_file(str(directory / "classification_heads.safetensors"))
            expected = result.layer_heads.state_dict()
            if set(state) != set(expected) or any(
                value.dtype != expected[name].dtype
                or value.shape != expected[name].shape
                for name, value in state.items()
            ):
                raise ValueError(
                    "Every saved task head requires its exact shape and dtype"
                )
            result.layer_heads.load_state_dict(state, strict=True)
        if state_digest(result.state_dict()) != receipt["state_sha256"]:
            raise ValueError("Reloaded task state differs from the saved checkpoint")
        verify_files(directory, receipt["files"])
        return result
