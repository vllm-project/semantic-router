"""Official bidirectional encoder with Kai-style candidate markers.

A question, every candidate description and the state are encoded jointly in
one bidirectional pass: `[BOS] {type} question: {text} [SEP] ([MARK] {candidate}
[SEP])... {state} [SEP]`, the same text layout as Decision 1.0 Kai. Each
candidate marker's final state is scored against the BOS state by the shared
FP32 `CandidateHead` used by the official-Qwen 0.6B control, then softmaxed
over the offered candidates. Candidate IDs never enter the tokens except as
the System One Choice text already includes them.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from .common import MAX_INPUT_TOKENS, file_sha256, write_json

ARCHITECTURE = "dev2-06b-bidirectional-marker-shared-candidate-head-v1"
KINDS = ("choice", "noul", "score")
DEFAULT_NO = "No. The statement or question is not satisfied."
DEFAULT_YES = "Yes. The statement or question is satisfied."

SOURCES: dict[str, dict[str, Any]] = {
    "eurobert-610m": {
        "repo": "EuroBERT/EuroBERT-610m",
        "revision": "d9af784ed20db6c2096e335ec6a67dd4a219924c",
        "license": "apache-2.0",
        "trust_remote_code": True,
        "weights": "model.safetensors",
        "code": ["configuration_eurobert.py", "modeling_eurobert.py"],
        "files": {
            "config.json": "1af8c7280d5a5c9b605c837a55e3c0b8265bba8174ef44f4b60208058d96f403",
            "configuration_eurobert.py": "8737378a5cea9e6c7be0e077138d6c725fb5909ad122c979adcdcc1a005a51b3",
            "modeling_eurobert.py": "c800961405876db2ead5d9e8bdb62d90314282a9697c00a03cddfdd5e49339fd",
            "model.safetensors": "50e41b32655cbe62c63dac675b1a2a6625632ed1991243eed5d641d2f6952791",
            "special_tokens_map.json": "6d66005b83cd0646e882e57e09c5a3e3e570ef53dd1f6d4315a65943fa81f2f7",
            "tokenizer.json": "98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069",
            "tokenizer_config.json": "0a6c6b1edf6b4f71604cf97f47521276e5a1f6c6d6b6243cd727a87e0efdc763",
        },
        "tokens": {
            "bos": "<|begin_of_text|>",
            "sep": "<|end_of_text|>",
            "marker": "<|mask|>",
            "pad": "<|end_of_text|>",
        },
    },
    "mmbert-base": {
        "repo": "jhu-clsp/mmBERT-base",
        "revision": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
        "license": "mit",
        "trust_remote_code": False,
        "weights": "pytorch_model.bin",
        "code": [],
        "files": {
            "config.json": "47b40fd2e1df8299426dd5f4bb18c28f028cfafcb51b73645f83e596d187eb37",
            "pytorch_model.bin": "8ea64ec1ea4eb8fca0fc14b69a2ae571de6bfbc25fd214bb932dd4aba6a3a04e",
            "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
            "tokenizer.json": "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
            "tokenizer_config.json": "1d2f82c1341a79748e00efe82e67690f99d00b3c2a894f2b23128fd9d3519da3",
        },
        "tokens": {"bos": "<bos>", "sep": "<eos>", "marker": "<mask>", "pad": "<pad>"},
    },
    "qwen3-0.6b-base": {
        "repo": "Qwen/Qwen3-0.6B-Base",
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "license": "apache-2.0",
        "trust_remote_code": False,
        "weights": "model.safetensors",
        "code": [],
        "files": {
            "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
            "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
            "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
            "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
        },
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"],
        "tokens": {
            "bos": "<|im_start|>",
            "sep": "<|im_end|>",
            "marker": "<|box_start|>",
            "pad": "<|endoftext|>",
        },
        "bidirectional": True,
    },
}


def view(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[float]]:
    """Kai `native.policy.packing.view` candidate semantics (targets not read)."""
    question = record["question"]
    kind = str(question["type"]).lower()
    if kind not in KINDS:
        raise ValueError("Unsupported question type")
    for value, name in (
        (record.get("state_text"), "state_text"),
        (question.get("text"), "question.text"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if kind == "noul":
        options = [
            {"id": "no", "text": question.get("false_criterion", DEFAULT_NO)},
            {"id": "yes", "text": question.get("true_criterion", DEFAULT_YES)},
        ]
    else:
        options = question["options" if kind == "choice" else "levels"]
    if not isinstance(options, list) or not 2 <= len(options) <= 255:
        raise ValueError("Require 2..255 complete candidate descriptions")
    for option in options:
        if (
            not isinstance(option.get("id"), str)
            or not isinstance(option.get("text"), str)
            or not option["text"].strip()
        ):
            raise ValueError("Every candidate needs an ID and nonempty text")
    if len({o["id"] for o in options}) != len(options):
        raise ValueError("Duplicate candidate IDs")
    values = (
        [float(o["value"]) for o in options]
        if kind == "score"
        else [float(i) for i in range(len(options))]
    )
    if kind == "score" and any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("Score level values must be strictly increasing")
    return kind, options, values


class MarkerPacker:
    """Kai's complete-input marker layout over another tokenizer; never truncates."""

    def __init__(
        self,
        tokenizer: Any,
        token_ids: dict[str, int],
        max_length: int = MAX_INPUT_TOKENS,
    ):
        if set(token_ids) != {"bos", "sep", "marker", "pad"} or any(
            not isinstance(v, int) for v in token_ids.values()
        ):
            raise ValueError("Explicit BOS/SEP/marker/pad token IDs are required")
        self.tokenizer = tokenizer
        self.ids = dict(token_ids)
        self.max_length = max_length

    def tokens(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("Empty token span")
        return list(ids)

    def encode(self, record: dict[str, Any]) -> dict[str, Any]:
        kind, options, values = view(record)
        ids = (
            [self.ids["bos"]]
            + self.tokens(f"{kind} question: {record['question']['text']}")
            + [self.ids["sep"]]
        )
        positions, spans = [], []
        for index, option in enumerate(options):
            text = (
                f"level {index}: {option['text']}"
                if kind == "score"
                else option["text"]
            )
            positions.append(len(ids))
            ids.append(self.ids["marker"])
            ids.extend(self.tokens(text))
            spans.append((positions[-1], len(ids)))
            ids.append(self.ids["sep"])
        state = self.tokens(record["state_text"])
        if len(ids) + len(state) + 1 > self.max_length:
            raise ValueError(
                f"Record {record.get('id')} exceeds {self.max_length} tokens; no implicit truncation"
            )
        ids.extend(state + [self.ids["sep"]])
        return {
            "ids": ids,
            "positions": positions,
            "spans": spans,
            "kind": kind,
            "values": values,
            "candidate_ids": [o["id"] for o in options],
            "input_tokens": len(ids),
            "id": record.get("id"),
        }

    def collate(
        self, encoded: list[dict[str, Any]], device: Any = "cpu"
    ) -> dict[str, Any]:
        import torch

        if not encoded:
            raise ValueError("Empty batch")
        length = max(len(e["ids"]) for e in encoded)
        width = max(len(e["positions"]) for e in encoded)
        ids = torch.full((len(encoded), length), self.ids["pad"], dtype=torch.long)
        mask = torch.zeros((len(encoded), length), dtype=torch.long)
        positions = torch.zeros((len(encoded), width), dtype=torch.long)
        valid = torch.zeros((len(encoded), width), dtype=torch.bool)
        for i, e in enumerate(encoded):
            ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
            mask[i, : len(e["ids"])] = 1
            positions[i, : len(e["positions"])] = torch.tensor(e["positions"])
            valid[i, : len(e["positions"])] = True
        kinds = torch.tensor(
            [KINDS.index(e["kind"]) for e in encoded], dtype=torch.long
        )
        span_mask = torch.zeros((len(encoded), width, length), dtype=torch.bool)
        for i, e in enumerate(encoded):
            for j, (start, end) in enumerate(e["spans"]):
                span_mask[i, j, start:end] = True
        return {
            "span_mask": span_mask.to(device),
            "input_ids": ids.to(device),
            "attention_mask": mask.to(device),
            "marker_positions": positions.to(device),
            "valid_candidates": valid.to(device),
            "kind_ids": kinds.to(device),
        }


def pool_candidates(hidden: Any, batch: dict[str, Any], mode: str) -> Any:
    """Candidate vectors: the marker state, or the mean over marker and description."""
    import torch

    if mode == "span-mean":
        span = batch["span_mask"].to(hidden.dtype)
        total = torch.einsum("bkl,bld->bkd", span, hidden)
        return total / span.sum(-1, keepdim=True).clamp(min=1)
    positions = batch["marker_positions"]
    return torch.gather(
        hidden, 1, positions[:, :, None].expand(-1, -1, hidden.shape[-1])
    )


def _torch_module() -> Any:
    import torch
    from torch import nn

    from training.model.decision_model import CandidateHead

    class OrdinalScore(nn.Module):
        """One latent position shared by every Score level count.

        Level k of K sits at c_k = k/(K-1) on [0, 1]; the request's position z
        comes from the global state. The Score logit adds -beta*(z - c_k)^2, with
        beta starting at zero so the zero-step output equals the plain head.
        """

        def __init__(self, hidden_size: int):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.position = nn.Linear(hidden_size, 1)
            self.beta = nn.Parameter(torch.zeros(()))

        def forward(self, query: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=query.device.type, enabled=False):
                z = torch.sigmoid(self.position(self.norm(query.float()))).squeeze(-1)
                count = valid.sum(-1).clamp(min=2).float()
                index = torch.arange(valid.shape[1], device=valid.device).float()
                centre = index[None, :] / (count[:, None] - 1)
                return -self.beta * (z[:, None] - centre).square()

    class EncoderDecision(nn.Module):
        def __init__(
            self,
            backbone: nn.Module,
            head_dim: int = 256,
            *,
            bidirectional: bool = False,
            ordinal_score: bool = False,
            candidate_pool: str = "marker",
            query_pool: str = "first",
        ):
            super().__init__()
            if candidate_pool not in ("marker", "span-mean"):
                raise ValueError("candidate_pool must be marker or span-mean")
            if query_pool not in ("first", "tokens-mean"):
                raise ValueError("query_pool must be first or tokens-mean")
            self.candidate_pool = candidate_pool
            self.query_pool = query_pool
            self.backbone = backbone
            self.head = CandidateHead(backbone.config.hidden_size, head_dim)
            self.bidirectional = bidirectional
            self.ordinal = (
                OrdinalScore(backbone.config.hidden_size) if ordinal_score else None
            )
            if bidirectional:
                for module in backbone.modules():
                    if hasattr(module, "is_causal"):
                        module.is_causal = False

        def forward(self, batch: dict[str, Any]) -> torch.Tensor:
            mask = batch["attention_mask"]
            if self.bidirectional:
                # A prepared 4D key-padding mask replaces the decoder's causal mask.
                length = mask.shape[1]
                mask = mask.bool()[:, None, None, :].expand(-1, 1, length, length)
            hidden = self.backbone(
                input_ids=batch["input_ids"], attention_mask=mask, return_dict=True
            ).last_hidden_state
            markers = self.candidates(hidden, batch)
            query = self.query(hidden, batch)
            logits = self.head(markers, query).float()
            if self.ordinal is not None:
                is_score = (batch["kind_ids"] == 2).float()[:, None]
                ordinal = self.ordinal(query, batch["valid_candidates"])
                logits = logits + is_score * ordinal
            return logits.masked_fill(
                ~batch["valid_candidates"], torch.finfo(torch.float32).min
            )

    def candidates(
        self: Any, hidden: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        return pool_candidates(hidden, batch, self.candidate_pool)

    def query(self: Any, hidden: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        if self.query_pool == "tokens-mean":
            # Mean over every real token except the first, which can act as an attention sink.
            keep = batch["attention_mask"].to(hidden.dtype).clone()
            keep[:, 0] = 0
            return (keep[:, :, None] * hidden).sum(1) / keep.sum(1, keepdim=True)
        return hidden[:, 0]

    EncoderDecision.candidates = candidates
    EncoderDecision.query = query
    return EncoderDecision


def verify_source(path: str | Path, source: str) -> dict[str, Any]:
    """Exact pinned official files and HF local-dir revision metadata."""
    spec = SOURCES[source]
    root = Path(path).resolve(strict=True)
    for name, expected in spec["files"].items():
        item = root / name
        if item.is_symlink() or not item.is_file() or file_sha256(item) != expected:
            raise ValueError(f"{source}: pinned file differs: {name}")
    metadata = root / ".cache/huggingface/download" / f"{spec['weights']}.metadata"
    if (
        not metadata.is_file()
        or metadata.read_text().splitlines()[0].strip() != spec["revision"]
    ):
        raise ValueError(
            f"{source}: HF local-dir revision metadata missing or different"
        )
    return {
        "source": source,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "license": spec["license"],
    }


def tokenizer_and_ids(
    path: str | Path, tokens: dict[str, str]
) -> tuple[Any, dict[str, int]]:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(path) / "tokenizer.json")
    )
    ids = {}
    for role, text in tokens.items():
        value = tokenizer.convert_tokens_to_ids(text)
        if (
            not isinstance(value, int)
            or value < 0
            or tokenizer.convert_ids_to_tokens(value) != text
        ):
            raise ValueError(f"Special token {text!r} is not a single vocabulary item")
        ids[role] = value
    return tokenizer, ids


def from_official(
    path: str | Path,
    source: str,
    *,
    head_dim: int = 256,
    seed: int = 20260928,
    ordinal_score: bool = False,
    candidate_pool: str = "marker",
    query_pool: str = "first",
) -> tuple[Any, MarkerPacker, dict[str, Any]]:
    import torch
    from transformers import AutoModel

    identity = verify_source(path, source)
    spec = SOURCES[source]
    tokenizer, ids = tokenizer_and_ids(path, spec["tokens"])
    backbone, info = AutoModel.from_pretrained(
        str(path),
        trust_remote_code=spec["trust_remote_code"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    unexpected = [
        k
        for k in info.get("unexpected_keys", [])
        if not k.startswith(("lm_head", "head.", "decoder"))
    ]
    if info.get("missing_keys") or info.get("mismatched_keys") or unexpected:
        raise RuntimeError(f"Incomplete official encoder load: {info}")
    torch.manual_seed(seed)
    bidirectional = bool(spec.get("bidirectional", False))
    model = _torch_module()(
        backbone,
        head_dim,
        bidirectional=bidirectional,
        ordinal_score=ordinal_score,
        candidate_pool=candidate_pool,
        query_pool=query_pool,
    )
    packer = MarkerPacker(tokenizer, ids)
    metadata = {
        "architecture": ARCHITECTURE,
        "source": identity,
        "head_dim": head_dim,
        "head_init_seed": seed,
        "token_ids": ids,
        "tokens": spec["tokens"],
        "max_input_tokens": packer.max_length,
        "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
        "head_parameters": sum(p.numel() for p in model.head.parameters())
        + (sum(p.numel() for p in model.ordinal.parameters()) if ordinal_score else 0),
        "discarded_pretraining_head_keys": sorted(info.get("unexpected_keys", [])),
        "attention": (
            "bidirectional (4D key-padding mask, causal flag cleared)"
            if bidirectional
            else "native bidirectional encoder"
        ),
        "score_readout": "latent-ordinal-v1" if ordinal_score else "candidate-head",
        "candidate_pool": candidate_pool,
        "query_pool": query_pool,
    }
    metadata["loaded_parameters"] = (
        metadata["backbone_parameters"] + metadata["head_parameters"]
    )
    return model, packer, metadata


def save(
    model: Any,
    packer: MarkerPacker,
    metadata: dict[str, Any],
    source_path: str | Path,
    output: str | Path,
    provenance: dict[str, Any],
) -> str:
    """Standalone export: backbone, head, tokenizer, config and a file manifest."""
    from safetensors.torch import save_file

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name(output.name + ".pending")
    stage.mkdir(parents=True)
    spec = SOURCES[metadata["source"]["source"]]
    model.backbone.save_pretrained(stage / "backbone", safe_serialization=True)
    for name in spec["code"]:
        shutil.copyfile(Path(source_path) / name, stage / "backbone" / name)
    head = {
        k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()
    }
    save_file(head, str(stage / "head.safetensors"))
    if model.ordinal is not None:
        ordinal = {
            k: v.detach().cpu().contiguous()
            for k, v in model.ordinal.state_dict().items()
        }
        save_file(ordinal, str(stage / "ordinal.safetensors"))
    (stage / "tokenizer").mkdir()
    for name in spec.get(
        "tokenizer_files",
        ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    ):
        shutil.copyfile(Path(source_path) / name, stage / "tokenizer" / name)
    write_json(stage / "decision_config.json", {**metadata, "provenance": provenance})
    files = {
        item.relative_to(stage).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(stage.rglob("*"))
        if item.is_file()
    }
    manifest_sha = write_json(
        stage / "MANIFEST.json", {"schema": "dev2-06b-encoder-files/1", "files": files}
    )
    stage.rename(output)
    return manifest_sha


def load(
    path: str | Path, manifest_sha256: str, device: str = "cuda:0"
) -> tuple[Any, MarkerPacker, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel

    root = Path(path).resolve(strict=True)
    if file_sha256(root / "MANIFEST.json") != manifest_sha256:
        raise ValueError("Export manifest identity mismatch")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if present != set(manifest["files"]) | {"MANIFEST.json"}:
        raise ValueError("Export file roster differs from its manifest")
    for name, ref in manifest["files"].items():
        if file_sha256(root / name) != ref["sha256"]:
            raise ValueError(f"Export file differs: {name}")
    metadata = json.loads((root / "decision_config.json").read_text())
    if metadata["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported export architecture")
    spec = SOURCES[metadata["source"]["source"]]
    backbone = AutoModel.from_pretrained(
        str(root / "backbone"),
        trust_remote_code=spec["trust_remote_code"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model = _torch_module()(
        backbone,
        metadata["head_dim"],
        bidirectional=bool(spec.get("bidirectional", False)),
        ordinal_score=metadata.get("score_readout") == "latent-ordinal-v1",
        candidate_pool=metadata.get("candidate_pool", "marker"),
        query_pool=metadata.get("query_pool", "first"),
    )
    model.head.load_state_dict(load_file(str(root / "head.safetensors")), strict=True)
    if model.ordinal is not None:
        model.ordinal.load_state_dict(
            load_file(str(root / "ordinal.safetensors")), strict=True
        )
    tokenizer, ids = tokenizer_and_ids(root / "tokenizer", metadata["tokens"])
    if ids != metadata["token_ids"]:
        raise ValueError("Tokenizer special IDs changed")
    model.to(device).eval()
    return model, MarkerPacker(tokenizer, ids, metadata["max_input_tokens"]), metadata


def probabilities(
    model: Any,
    packer: MarkerPacker,
    records: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int = 8,
) -> list[list[float]]:
    """FP32 inference; records are grouped by type as in the Kai SystemOne default."""
    import torch

    encoded = [packer.encode(r) for r in records]
    order = sorted(range(len(records)), key=lambda i: encoded[i]["kind"])
    output: list[list[float] | None] = [None] * len(records)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            logits = model(packer.collate([encoded[i] for i in chunk], device))
            probs = logits.float().softmax(-1).cpu().tolist()
            for i, row in zip(chunk, probs):
                output[i] = row[: len(encoded[i]["positions"])]
    model.train(was_training)
    if any(v is None or not all(math.isfinite(x) for x in v) for v in output):
        raise FloatingPointError("Nonfinite or missing probabilities")
    return output  # type: ignore[return-value]
