"""Qwen3.5 text backbone with a shared, dynamic-option decision readout.

Option endpoint vectors preserve local candidate context; the final query
vector sees all candidates. The head is explicitly FP32 inside BF16 autocast.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .data import MAX_OPTIONS, canonical
from .lora import LORA_FORMAT, select_target_modules, verify_adapter_config
from .source import verify_source

PROMPT_VERSION = "decision2-segmented-options-global-query-v1"
ARCHITECTURE = "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp"


def _payload(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def segments(row: dict[str, Any]) -> tuple[str, list[str], str]:
    prefix = (
        f"Context:\n{_payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n{_payload(row['instructions'])}\nOptions:"
    )
    options = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in row["options"]
    ]
    suffix = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
    return prefix, options, suffix


def encode(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    prefix, options, suffix = segments(row)
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    endpoints = []
    for option in options:
        part = tokenizer.encode(option, add_special_tokens=False)
        if not part:
            raise ValueError(f"{row['id']}: empty tokenized option")
        ids.extend(part)
        endpoints.append(len(ids) - 1)
    tail = tokenizer.encode(suffix, add_special_tokens=False)
    if not tail:
        raise ValueError(f"{row['id']}: empty tokenized query")
    ids.extend(tail)
    if len(ids) > max_length:
        raise ValueError(
            f"{row['id']}: {len(ids)} tokens exceeds max_length={max_length}; no truncation"
        )
    prompt = prefix + "".join(options) + suffix
    return {
        "id": row["id"],
        "ids": ids,
        "candidate_positions": endpoints,
        "query_position": len(ids) - 1,
        "label": row["label"],
        "keys": [option["key"] for option in row["options"]],
        "task_type": row["task_type"],
        "family": row["family"],
        "teacher_probs": (
            [row["teacher_probs"][option["key"]] for option in row["options"]]
            if "teacher_probs" in row
            else None
        ),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(canonical(ids).encode("utf-8")).hexdigest(),
    }


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    length = math.ceil(max(len(item["ids"]) for item in items) / 8) * 8
    width = max(len(item["keys"]) for item in items)
    input_ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    positions = torch.zeros((len(items), width), dtype=torch.long)
    candidate_mask = torch.zeros((len(items), width), dtype=torch.bool)
    teacher_probs = torch.zeros((len(items), width), dtype=torch.float32)
    replay_mask = torch.zeros(len(items), dtype=torch.bool)
    for index, item in enumerate(items):
        count = len(item["keys"])
        endpoints = item["candidate_positions"]
        if not 2 <= count <= MAX_OPTIONS or len(endpoints) != count:
            raise ValueError(f"{item['id']}: candidate count mismatch")
        if len(set(endpoints)) != count or not all(
            0 <= p < item["query_position"] < len(item["ids"]) for p in endpoints
        ):
            raise ValueError(f"{item['id']}: invalid option endpoints")
        if not 0 <= item["label"] < count:
            raise ValueError(f"{item['id']}: invalid label")
        input_ids[index, : len(item["ids"])] = torch.tensor(
            item["ids"], dtype=torch.long
        )
        attention_mask[index, : len(item["ids"])] = 1
        positions[index, :count] = torch.tensor(endpoints, dtype=torch.long)
        candidate_mask[index, :count] = True
        if item["teacher_probs"] is not None:
            teacher_probs[index, :count] = torch.tensor(
                item["teacher_probs"], dtype=torch.float32
            )
            replay_mask[index] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "candidate_positions": positions,
        "candidate_mask": candidate_mask,
        "query_positions": torch.tensor(
            [item["query_position"] for item in items], dtype=torch.long
        ),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "teacher_probs": teacher_probs,
        "replay_mask": replay_mask,
    }


class CandidateHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = nn.Linear(head_dim, 1, bias=False)
        nn.init.normal_(self.scalar.weight, mean=0.0, std=0.01)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            candidate = self.candidate_norm(candidates.float())
            global_query = self.query_norm(query.float())
            bilinear = (self.key(candidate) * self.query(global_query)[:, None, :]).sum(
                -1
            ) / math.sqrt(self.head_dim)
            nonlinear = self.scalar(
                F.gelu(
                    self.candidate_mlp(candidate)
                    + self.query_mlp(global_query)[:, None, :]
                )
            ).squeeze(-1)
            return bilinear + nonlinear


class DecisionModel(nn.Module):
    def __init__(
        self, backbone: nn.Module, head: CandidateHead, metadata: dict[str, Any]
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.metadata = metadata

    @classmethod
    def from_base(
        cls,
        path: str | Path,
        revision: str,
        head_dim: int = 256,
        *,
        source_stage: str = "base",
    ) -> tuple[DecisionModel, Any]:
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        if source_stage not in ("base", "posttrained"):
            raise ValueError("Qwen source stage must be base or posttrained")

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        full, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
            output_loading_info=True,
            low_cpu_mem_usage=True,
        )
        if any(
            info.get(name) for name in ("missing_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError(f"Incomplete Qwen3.5 base loading: {info}")
        backbone = full.model.language_model
        backbone.config.use_cache = False
        head = CandidateHead(backbone.config.hidden_size, head_dim)
        metadata = {
            "architecture": ARCHITECTURE,
            "prompt_version": PROMPT_VERSION,
            "base_revision": revision,
            "source_stage": source_stage,
            "initialization": f"{source_stage}-random-head",
            "head_dim": head_dim,
            "max_options": MAX_OPTIONS,
            "text_parameter_count": sum(p.numel() for p in backbone.parameters()),
            "parameter_dtype": "float32",
            "autocast_dtype": "bfloat16",
            "head_compute_dtype": "float32",
            "attention": "sdpa",
        }
        return cls(backbone, head, metadata), tokenizer

    @classmethod
    def from_decision1(
        cls, path: str | Path, head_dim: int = 256
    ) -> tuple[DecisionModel, Any]:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        path = Path(path)
        old = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
        if (
            old.get("architecture")
            != "contextual-candidate-endpoint-plus-global-query-shared-bilinear-mlp"
            or old.get("prompt_version")
            != "structured-segmented-candidate-endpoints-global-query-v2"
            or old.get("head_dim") != head_dim
        ):
            raise ValueError(
                "Decision 1.0 architecture, prompt or head dimension is incompatible"
            )
        backbone = Qwen3_5TextModel.from_pretrained(
            path / "backbone",
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        backbone.config.use_cache = False
        head = CandidateHead(backbone.config.hidden_size, head_dim)
        head.load_state_dict(
            load_file(str(path / "decision_head.safetensors")), strict=True
        )
        metadata = {
            "architecture": ARCHITECTURE,
            "prompt_version": PROMPT_VERSION,
            "base_revision": old.get("base_revision", "unknown"),
            "initialization": "decision1-text-backbone-and-head",
            "source_prompt_version": old["prompt_version"],
            "head_dim": head_dim,
            "max_options": MAX_OPTIONS,
            "text_parameter_count": sum(p.numel() for p in backbone.parameters()),
            "parameter_dtype": "float32",
            "autocast_dtype": "bfloat16",
            "head_compute_dtype": "float32",
            "attention": "sdpa",
        }
        return cls(backbone, head, metadata), AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        source_path: str | Path | None = None,
        trainable_adapter: bool = False,
    ) -> tuple[DecisionModel, Any]:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        path = Path(path)
        metadata = json.loads(
            (path / "decision_config.json").read_text(encoding="utf-8")
        )
        if (
            metadata.get("prompt_version") != PROMPT_VERSION
            or metadata.get("architecture") != ARCHITECTURE
        ):
            raise ValueError("Checkpoint is not a Decision 2.0 dynamic-option model")
        if metadata.get("checkpoint_format") == LORA_FORMAT:
            if source_path is None:
                raise ValueError(
                    "LoRA checkpoint requires --source-path for its immutable base or Decision 1.0 source"
                )
            source_path = Path(source_path)
            contract = metadata.get("lora")
            if not isinstance(contract, dict):
                raise ValueError(
                    "LoRA checkpoint is missing its source and adapter contract"
                )
            verify_source(source_path, contract.get("source_fingerprint"))
            verify_adapter_config(path / "adapter", contract)
            source_kind = contract.get("source_kind")
            if source_kind in ("base", "posttrained"):
                if not contract.get("base_revision"):
                    raise ValueError(
                        "LoRA Qwen initialization has no immutable revision"
                    )
                source_model, _ = cls.from_base(
                    source_path,
                    contract["base_revision"],
                    metadata["head_dim"],
                    source_stage=source_kind,
                )
            elif source_kind == "decision1":
                source_model, _ = cls.from_decision1(source_path, metadata["head_dim"])
            elif source_kind == "decision2":
                source_model, _ = cls.from_checkpoint(source_path)
                if source_model.metadata.get("checkpoint_format") == LORA_FORMAT:
                    raise ValueError("Nested LoRA source checkpoints are unsupported")
            else:
                raise ValueError("LoRA checkpoint has an unsupported source kind")
            if contract.get("target_modules") != select_target_modules(
                source_model.backbone
            ):
                raise ValueError(
                    "LoRA checkpoint target modules differ from the source Qwen3.5 text backbone"
                )
            source_modules = dict(source_model.backbone.named_modules())
            source_dimensions = {
                name: [
                    source_modules[name].in_features,
                    source_modules[name].out_features,
                ]
                for name in contract["target_modules"]
            }
            if contract.get("target_dimensions") != source_dimensions:
                raise ValueError(
                    "LoRA checkpoint projection dimensions differ from its pinned source"
                )
            from peft import PeftModel

            source_model.backbone = PeftModel.from_pretrained(
                source_model.backbone,
                path / "adapter",
                is_trainable=trainable_adapter,
                local_files_only=True,
            )
            source_model.head.load_state_dict(
                load_file(str(path / "decision_head.safetensors")), strict=True
            )
            source_model.metadata = metadata
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            return source_model, tokenizer
        if metadata.get("checkpoint_format") not in (None, "full"):
            raise ValueError("Unknown Decision 2.0 checkpoint format")
        backbone = Qwen3_5TextModel.from_pretrained(
            path / "backbone",
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        backbone.config.use_cache = False
        head = CandidateHead(backbone.config.hidden_size, metadata["head_dim"])
        head.load_state_dict(
            load_file(str(path / "decision_head.safetensors")), strict=True
        )
        return cls(backbone, head, metadata), AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
        query_positions: torch.Tensor,
        **unused: Any,
    ) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        candidates = hidden[batch[:, None], candidate_positions]
        query = hidden[batch, query_positions]
        scores = self.head(candidates, query)
        return scores.masked_fill(~candidate_mask, -float("inf"))

    def save(self, path: str | Path, tokenizer: Any) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        if self.metadata.get("checkpoint_format") == LORA_FORMAT:
            self.backbone.save_pretrained(path / "adapter", safe_serialization=True)
            adapter_config = path / "adapter" / "adapter_config.json"
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            # PEFT may capture an absolute local initialization path here. The
            # independent source fingerprint is authoritative and portable.
            config["base_model_name_or_path"] = None
            adapter_config.write_text(
                json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            readme = path / "adapter" / "README.md"
            if readme.exists():
                readme.write_text(
                    "# Decision 2.0 PEFT adapter\n\n"
                    "Load with the exact source model identified by the parent "
                    "`decision_config.json` fingerprint.\n",
                    encoding="utf-8",
                )
            verify_adapter_config(path / "adapter", self.metadata["lora"])
        else:
            self.backbone.save_pretrained(
                path / "backbone", safe_serialization=True, max_shard_size="4GB"
            )
        save_file(
            {
                name: tensor.detach().float().cpu().contiguous()
                for name, tensor in self.head.state_dict().items()
            },
            str(path / "decision_head.safetensors"),
        )
        tokenizer.save_pretrained(path)
        (path / "decision_config.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def merge_lora(self, identity: dict[str, Any]) -> None:
        """Materialize adapter weights into a normal full-backbone checkpoint."""
        if self.metadata.get("checkpoint_format") != LORA_FORMAT:
            raise ValueError("Only a loaded LoRA checkpoint can be merged")
        origin = {
            "source": self.metadata["lora"]["source_fingerprint"],
            "adapter": identity,
            "configuration": self.metadata["lora"],
        }
        self.backbone = self.backbone.merge_and_unload(safe_merge=True)
        self.metadata = {
            **self.metadata,
            "checkpoint_format": "full",
            "initialization": "merged-peft-lora",
            "training_mode": "materialized-lora",
            "lora_origin": origin,
        }
        del self.metadata["lora"]
