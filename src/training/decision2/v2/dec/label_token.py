"""LM-head label-token readout for decoder-track readout contrasts (decoder M10).

The prompt is the shared segmented prompt with every option rendered behind a
one-token label at the start of its line, and an answer cue at the end. The
option logits are the tied LM head's logits of those label tokens at the last
prompt position, from one forward pass:

    logit_i = h_last . E[label_i]        (E = the input embedding = the LM head)

Option ``i``'s label token is read from the prompt itself: the encoder puts the
label's position in ``candidate_positions[i]``, so the shared ``collate``,
``evaluate``, preflight and benchmark adapters run unchanged and the model
gathers ``input_ids[candidate_positions]``. A stock generate engine serves the
same readout from the same token ids: ``max_tokens=1``, ``allowed_token_ids`` =
the row's label ids and per-token logprobs, renormalised over those labels
(``label_token_ids`` in every encoded row).

Labels: Noul ``false`` / ``true`` -> ``no`` / ``yes``; Score levels ``0``..``9``
-> the digit itself; Choice (and Score above ten levels) -> the fixed alphabet
A..Z then the two-letter capitals that the tokenizer keeps as one token, in
lexical order, cut at 255. The alphabet's SHA-256 is bound in the checkpoint.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import string
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.model.data import MAX_OPTIONS, canonical, file_sha256
from training.model.decision_model import DecisionModel, _payload
from training.model.lora import (
    LORA_FORMAT,
    select_target_modules,
    verify_adapter_config,
)
from training.model.source import verify_source

LABEL_PROMPT_VERSION = "decision2-label-token-v1"
LABEL_ARCHITECTURE = "qwen3.5-text-label-token-tied-lm-head-v1"
LABEL_READOUT = "label_token"
LABEL_VERSION = "dec-label-token/1"
NOUL_LABELS = {"false": "no", "true": "yes"}
OPTION_HEAD = "\n<option>\n"
LABEL_SEPARATOR = ". "
OPTION_TAIL = "\n</option>"
ANSWER_CUE = (
    "\n\nSelect the single option best supported by the context and instructions."
    "\nAnswer with its label only.\nAnswer:\n"
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
)


def _single(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def choice_alphabet(tokenizer: Any) -> list[str]:
    """A..Z, then two-letter capitals that encode as one token; at most 255."""
    cached = getattr(tokenizer, "_dec_label_alphabet", None)
    if cached is not None:
        return cached
    letters = list(string.ascii_uppercase)
    pairs = ["".join(pair) for pair in itertools.product(letters, repeat=2)]
    alphabet = [s for s in letters + pairs if _single(tokenizer, s) is not None]
    alphabet = alphabet[:MAX_OPTIONS]
    if alphabet[:26] != letters:
        raise ValueError("Tokenizer does not keep every capital letter as one token")
    for word in (*NOUL_LABELS.values(), *(str(d) for d in range(10))):
        if _single(tokenizer, word) is None:
            raise ValueError(f"Label {word!r} is not one token")
    try:
        tokenizer._dec_label_alphabet = alphabet
    except AttributeError:
        pass
    return alphabet


def alphabet_sha256(alphabet: list[str]) -> str:
    return hashlib.sha256(canonical(alphabet).encode("utf-8")).hexdigest()


def labels_for(row: dict[str, Any], alphabet: list[str]) -> list[str]:
    keys = [option["key"] for option in row["options"]]
    if row["task_type"] == "noul":
        if set(keys) != set(NOUL_LABELS):
            raise ValueError(f"{row['id']}: noul keys must be false/true")
        return [NOUL_LABELS[key] for key in keys]
    if row["task_type"] == "score" and len(keys) <= 10:
        if set(keys) != {str(level) for level in range(len(keys))}:
            raise ValueError(f"{row['id']}: score keys must enumerate levels")
        return keys
    if len(keys) > len(alphabet):
        raise ValueError(f"{row['id']}: {len(keys)} options exceed the label alphabet")
    return alphabet[: len(keys)]


def encode_label(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    """The label-token prompt; ``candidate_positions[i]`` is option i's label token."""
    alphabet = choice_alphabet(tokenizer)
    labels = labels_for(row, alphabet)
    prefix = (
        f"Context:\n{_payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n{_payload(row['instructions'])}\nOptions:"
    )
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    head = tokenizer.encode(OPTION_HEAD, add_special_tokens=False)
    positions: list[int] = []
    label_ids: list[int] = []
    text = [prefix]
    for label, option in zip(labels, row["options"]):
        label_id = _single(tokenizer, label)
        if label_id is None:
            raise ValueError(f"{row['id']}: label {label!r} is not one token")
        body = (
            LABEL_SEPARATOR
            + canonical({"key": option["key"], "description": option["description"]})
            + OPTION_TAIL
        )
        ids.extend(head)
        positions.append(len(ids))
        label_ids.append(label_id)
        ids.append(label_id)
        ids.extend(tokenizer.encode(body, add_special_tokens=False))
        text.append(OPTION_HEAD + label + body)
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"{row['id']}: repeated label token")
    ids.extend(tokenizer.encode(ANSWER_CUE, add_special_tokens=False))
    text.append(ANSWER_CUE)
    if len(ids) > max_length:
        raise ValueError(
            f"{row['id']}: {len(ids)} tokens exceeds max_length={max_length}; no truncation"
        )
    prompt = "".join(text)
    return {
        "id": row["id"],
        "ids": ids,
        "candidate_positions": positions,
        "query_position": len(ids) - 1,
        "label": row["label"],
        "keys": [option["key"] for option in row["options"]],
        "labels": labels,
        "label_token_ids": label_ids,
        "score_level_indices": (
            [int(option["key"]) for option in row["options"]]
            if row["task_type"] == "score"
            else list(range(len(row["options"])))
        ),
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


def label_logits(
    query: torch.Tensor, label_ids: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """FP32 LM-head logits of each row's label tokens: [B, H] x [B, W] -> [B, W]."""
    with torch.autocast(device_type=query.device.type, enabled=False):
        return torch.einsum("bh,bwh->bw", query.float(), weight[label_ids].float())


class LabelTokenModel(nn.Module):
    """A Qwen3.5 text backbone read through its tied LM head at the answer cue."""

    def __init__(self, backbone: nn.Module, metadata: dict[str, Any]):
        super().__init__()
        self.backbone = backbone
        # No trainable readout: the label logits come from the tied embedding.
        self.head = nn.Module()
        self.metadata = metadata
        self.ordinal_score = None
        self.layer_mix = None

    @classmethod
    def wrap(cls, model: Any, tokenizer: Any) -> LabelTokenModel:
        """Keep a shared-architecture DecisionModel's backbone; drop its head."""
        backbone = model.backbone
        config = getattr(backbone, "config", None)
        if getattr(config, "tie_word_embeddings", None) is not True:
            raise ValueError("The label-token readout needs a tied LM head")
        if getattr(config, "model_type", None) != "qwen3_5_text":
            raise ValueError("The label-token readout is defined for Qwen3.5 text")
        alphabet = choice_alphabet(tokenizer)
        metadata = {
            key: value
            for key, value in model.metadata.items()
            if key not in ("head_variant", "head_dim", "architecture", "prompt_version")
        }
        metadata.update(
            {
                "architecture": LABEL_ARCHITECTURE,
                "prompt_version": LABEL_PROMPT_VERSION,
                "readout": LABEL_READOUT,
                "label_token": {
                    "version": LABEL_VERSION,
                    "lm_head": "tied-input-embeddings",
                    "noul_labels": NOUL_LABELS,
                    "score_labels": "level digits 0-9 (letters above ten levels)",
                    "choice_alphabet_size": len(alphabet),
                    "choice_alphabet_sha256": alphabet_sha256(alphabet),
                    "answer_cue": ANSWER_CUE,
                },
                "readout_compute_dtype": "float32",
            }
        )
        return cls(backbone, metadata)

    def residual_parameters(self) -> list[nn.Parameter]:
        return []

    def lm_head_weight(self) -> torch.Tensor:
        return self.backbone.get_input_embeddings().weight

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
        query = hidden[batch, query_positions]
        label_ids = input_ids.gather(1, candidate_positions)
        logits = label_logits(query, label_ids, self.lm_head_weight())
        return logits.masked_fill(~candidate_mask, -float("inf"))

    def save(self, path: str | Path, tokenizer: Any) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        if self.metadata.get("checkpoint_format") == LORA_FORMAT:
            self.backbone.save_pretrained(path / "adapter", safe_serialization=True)
            adapter_config = path / "adapter" / "adapter_config.json"
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = None
            adapter_config.write_text(
                json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            readme = path / "adapter" / "README.md"
            if readme.exists():
                readme.write_text(
                    "# Decision 2.0 PEFT adapter (label-token readout)\n\n"
                    "Load with the exact source model identified by the parent "
                    "`decision_config.json` fingerprint.\n",
                    encoding="utf-8",
                )
            verify_adapter_config(path / "adapter", self.metadata["lora"])
        else:
            self.backbone.save_pretrained(
                path / "backbone", safe_serialization=True, max_shard_size="4GB"
            )
        tokenizer.save_pretrained(path)
        (path / "decision_config.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def merge_lora(self) -> None:
        """Materialize the adapter into a full backbone (same answers up to FP32 rounding)."""
        if self.metadata.get("checkpoint_format") != LORA_FORMAT:
            raise ValueError("Only a loaded LoRA label-token model can be merged")
        origin = {
            "source": self.metadata["lora"]["source_fingerprint"],
            "configuration": {
                k: v
                for k, v in self.metadata["lora"].items()
                if k != "source_fingerprint"
            },
        }
        self.backbone = self.backbone.merge_and_unload(safe_merge=True)
        self.metadata = {
            **self.metadata,
            "checkpoint_format": "full",
            "initialization": "merged-peft-lora",
            "training_mode": "materialized-lora",
            "lora_origin": origin,
            "full_training_source": {
                "kind": f"merged-lora-{self.metadata['lora']['source_kind']}",
                "base_revision": self.metadata["lora"].get("base_revision"),
                "source_fingerprint": self.metadata["lora"]["source_fingerprint"],
            },
        }
        del self.metadata["lora"]


def is_label_checkpoint(path: str | Path) -> bool:
    metadata = json.loads(
        (Path(path) / "decision_config.json").read_text(encoding="utf-8")
    )
    return metadata.get("readout") == LABEL_READOUT


def _source_model(contract: dict[str, Any], source_path: Path) -> tuple[Any, Any]:
    kind = contract.get("source_kind")
    if kind in ("base", "posttrained"):
        if not contract.get("base_revision"):
            raise ValueError("LoRA Qwen initialization has no immutable revision")
        return DecisionModel.from_base(
            source_path, contract["base_revision"], source_stage=kind
        )
    if kind == "decision1":
        return DecisionModel.from_decision1(source_path)
    raise ValueError(
        "Label-token LoRA checkpoints start from a Qwen base or Decision 1.0"
    )


def load_label_checkpoint(
    path: str | Path,
    source_path: str | Path | None,
    *,
    trainable_adapter: bool = False,
) -> tuple[LabelTokenModel, Any]:
    from transformers import AutoTokenizer

    path = Path(path)
    metadata = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    if (
        metadata.get("readout") != LABEL_READOUT
        or metadata.get("architecture") != LABEL_ARCHITECTURE
        or metadata.get("prompt_version") != LABEL_PROMPT_VERSION
        or (metadata.get("label_token") or {}).get("version") != LABEL_VERSION
    ):
        raise ValueError("Not a decoder-track label-token checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    alphabet = choice_alphabet(tokenizer)
    if alphabet_sha256(alphabet) != metadata["label_token"]["choice_alphabet_sha256"]:
        raise ValueError("Tokenizer label alphabet differs from the checkpoint")
    if metadata.get("checkpoint_format") == LORA_FORMAT:
        contract = metadata.get("lora")
        if not isinstance(contract, dict) or source_path is None:
            raise ValueError("A label-token LoRA checkpoint needs its --source-path")
        source_path = Path(source_path)
        verify_source(source_path, contract.get("source_fingerprint"))
        verify_adapter_config(path / "adapter", contract)
        source, _ = _source_model(contract, source_path)
        if contract.get("target_modules") != select_target_modules(source.backbone):
            raise ValueError("Adapter target modules differ from the pinned source")
        modules = dict(source.backbone.named_modules())
        dimensions = {
            name: [modules[name].in_features, modules[name].out_features]
            for name in contract["target_modules"]
        }
        if contract.get("target_dimensions") != dimensions:
            raise ValueError(
                "Adapter projection dimensions differ from the pinned source"
            )
        from peft import PeftModel

        backbone = PeftModel.from_pretrained(
            source.backbone,
            path / "adapter",
            is_trainable=trainable_adapter,
            local_files_only=True,
        )
    elif metadata.get("checkpoint_format") == "full":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        backbone = Qwen3_5TextModel.from_pretrained(
            path / "backbone",
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        backbone.config.use_cache = False
    else:
        raise ValueError("Unknown label-token checkpoint format")
    if getattr(backbone.config, "tie_word_embeddings", None) is not True:
        raise ValueError("The label-token readout needs a tied LM head")
    return LabelTokenModel(backbone, metadata), tokenizer


def label_fingerprint(
    path: str | Path, source_path: str | Path | None
) -> dict[str, Any]:
    """Inference identity: config, tokenizer, weights (and a LoRA source's files)."""
    path = Path(path)
    metadata = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    if metadata.get("readout") != LABEL_READOUT:
        raise ValueError("Not a label-token checkpoint")
    files = [path / "decision_config.json"] + [
        path / name for name in TOKENIZER_FILES if (path / name).is_file()
    ]
    for folder in ("backbone", "adapter"):
        if (path / folder).is_dir():
            files.extend(
                f
                for f in (path / folder).rglob("*")
                if f.is_file()
                and f.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
            )
    hashes = {str(f.relative_to(path)): file_sha256(f) for f in sorted(files)}
    if metadata.get("checkpoint_format") == LORA_FORMAT:
        if source_path is None:
            raise ValueError("A label-token LoRA fingerprint needs --source-path")
        contract = metadata["lora"]
        verify_adapter_config(path / "adapter", contract)
        source = verify_source(Path(source_path), contract.get("source_fingerprint"))
        hashes = {
            **{f"checkpoint/{k}": v for k, v in hashes.items()},
            **{f"source/{k}": v for k, v in source["files_sha256"].items()},
        }
    elif not any(
        k.startswith("backbone/") and k.endswith(".safetensors") for k in hashes
    ):
        raise ValueError("Full label-token checkpoint is missing backbone weights")
    return {
        "model_sha256": hashlib.sha256(canonical(hashes).encode("utf-8")).hexdigest(),
        "files_sha256": hashes,
    }
