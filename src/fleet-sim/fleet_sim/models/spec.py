"""Model architecture spec and Hugging Face config helpers."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _fetch_hf_config(model_id: str, token: str | None = None) -> dict[str, Any]:
    """Fetch a Hugging Face ``config.json`` for ``model_id``."""

    quoted_model_id = urllib.parse.quote(model_id, safe="/")
    url = f"https://huggingface.co/{quoted_model_id}/raw/main/config.json"
    headers = {"User-Agent": "vllm-sr-sim/0.1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _humanize_name(name: str) -> str:
    """Turn a catalog slug or HF repo id into a human-readable display name."""

    base = name.rsplit("/", maxsplit=1)[-1]
    if any(ch.isupper() for ch in base):
        return base

    parts = []
    for raw in base.replace("_", "-").split("-"):
        token = raw.strip()
        lower = token.lower()
        if lower == "deepseek":
            parts.append("DeepSeek")
        elif lower.startswith("qwen"):
            parts.append(lower[0].upper() + lower[1:])
        elif lower in {"llama", "mistral", "mixtral"}:
            parts.append(lower.capitalize())
        elif lower.startswith("v") and lower[1:].isdigit():
            parts.append("V" + lower[1:])
        elif any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            parts.append(token.upper())
        else:
            parts.append(token.capitalize())
    return "-".join(parts)


@dataclass(frozen=True)
class ModelSpec:
    """Self-contained model architecture description used by fleet-sim."""

    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position: int
    moe_intermediate_size: int = 0
    n_experts: int = 0
    n_experts_topk: int = 0
    n_shared_experts: int = 0
    aliases: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return _humanize_name(self.name)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0 or self.moe_intermediate_size > 0

    @property
    def kv_bytes_per_token(self) -> int:
        return int(self.kv_bytes_per_token_dtype(dtype_bytes=2.0))

    def kv_bytes_per_token_dtype(self, dtype_bytes: float = 2.0) -> float:
        return 2.0 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes

    def _attention_param_count(self) -> int:
        return self.n_layers * (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.n_kv_heads * self.head_dim
        )

    def _embedding_param_count(self) -> int:
        return self.vocab_size * self.hidden_size

    def _ffn_param_count(self) -> int:
        if self.is_moe:
            return self.n_layers * (
                self.n_experts * 3 * self.hidden_size * self.moe_intermediate_size
                + self.n_shared_experts * 3 * self.hidden_size * self.intermediate_size
            )
        return self.n_layers * 3 * self.hidden_size * self.intermediate_size

    def param_count(self) -> int:
        return (
            self._embedding_param_count()
            + self._attention_param_count()
            + self._ffn_param_count()
        )

    def active_param_count(self) -> int:
        if not self.is_moe:
            return self.param_count()
        active_ffn = self.n_layers * (
            self.n_experts_topk * 3 * self.hidden_size * self.moe_intermediate_size
            + self.n_shared_experts * 3 * self.hidden_size * self.intermediate_size
        )
        return (
            self._embedding_param_count() + self._attention_param_count() + active_ffn
        )

    def param_bytes(self, dtype_bytes: float = 2.0) -> float:
        return self.param_count() * dtype_bytes

    def param_bytes_per_gpu(self, tp: int = 1, dtype_bytes: float = 2.0) -> float:
        if tp <= 0:
            raise ValueError("tp must be >= 1")
        return self.param_bytes(dtype_bytes=dtype_bytes) / tp

    @classmethod
    def from_hf_config(
        cls,
        config: Mapping[str, Any] | str | Path,
        name: str | None = None,
        token: str | None = None,
    ) -> ModelSpec:
        cfg: dict[str, Any]
        resolved_name = name

        if isinstance(config, Mapping):
            cfg = dict(config)
        else:
            path = Path(config)
            if path.exists():
                cfg = json.loads(path.read_text())
                if resolved_name is None:
                    resolved_name = cfg.get("_name_or_path") or path.stem
            else:
                cfg = _fetch_hf_config(str(config), token=token)

        if resolved_name is None:
            resolved_name = cfg.get("_name_or_path") or "hf-model"

        n_heads = int(cfg["num_attention_heads"])
        n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
        n_experts = int(cfg.get("num_experts", cfg.get("num_local_experts", 0)) or 0)
        moe_intermediate_size = int(
            cfg.get(
                "moe_intermediate_size",
                cfg.get("intermediate_size", 0) if n_experts else 0,
            )
            or 0
        )
        n_experts_topk = int(cfg.get("num_experts_per_tok", 0) or 0)

        return cls(
            name=resolved_name,
            n_layers=int(cfg["num_hidden_layers"]),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg.get("intermediate_size", 0) or 0),
            vocab_size=int(cfg["vocab_size"]),
            max_position=int(cfg.get("max_position_embeddings", 0) or 0),
            moe_intermediate_size=moe_intermediate_size,
            n_experts=n_experts,
            n_experts_topk=n_experts_topk,
            n_shared_experts=int(cfg.get("num_shared_experts", 0) or 0),
        )

    @classmethod
    def from_hf_repo(
        cls,
        model_id: str,
        name: str | None = None,
        token: str | None = None,
    ) -> ModelSpec:
        cfg = _fetch_hf_config(model_id, token=token)
        return cls.from_hf_config(cfg, name=name or model_id)
