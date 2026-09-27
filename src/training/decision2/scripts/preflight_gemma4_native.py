"""Pinned, label-free Gemma 4 source and typed-decoder feasibility check.

``inspect`` reads safetensors headers without loading tensor bodies. ``zero-step``
loads the unmodified official model on one explicitly selected GPU and checks
that Choice, Noul and Score admit a finite dynamic-option readout. Its random
head has no trained decision ability: these outputs must never be scored or
described as model quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from training.model.decision_model import CandidateHead, encode
from training.model.infer import question_to_row

SOURCE_ID = "google/gemma-4-26B-A4B-it"
SOURCE_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
CONFIG_SHA256 = "ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3"
PROBE_VERSION = "gemma4-native-zero-step-v1"


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def local_revision(snapshot: Path) -> str:
    root = snapshot / ".cache/huggingface/download"
    revisions = (
        {
            path.read_text(encoding="utf-8").splitlines()[0].strip()
            for path in root.rglob("*.metadata")
            if path.read_text(encoding="utf-8").splitlines()
        }
        if root.is_dir()
        else set()
    )
    if revisions != {SOURCE_REVISION}:
        raise ValueError(
            "Snapshot files do not all attest the pinned official revision"
        )
    return SOURCE_REVISION


def _group(name: str) -> str:
    if name.startswith("model.language_model."):
        return "language_model"
    if name.startswith("model.vision_tower."):
        return "vision_tower"
    if name.startswith("model.embed_vision."):
        return "vision_projection"
    return "other"


def tensor_inventory(snapshot: Path) -> dict[str, Any]:
    """Count stored elements, grouped by model submodule, from pinned headers."""
    from safetensors import safe_open

    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Official shard index has no weight_map")
    shards = set(weight_map.values())
    if any(Path(name).name != name for name in shards):
        raise ValueError("Shard index contains a non-local path")
    groups: dict[str, int] = {}
    observed: set[str] = set()
    for name in sorted(shards):
        with safe_open(str(snapshot / name), framework="pt", device="cpu") as shard:
            for key in sorted(shard.keys()):
                if key in observed or weight_map.get(key) != name:
                    raise ValueError("Shard tensor names differ from the pinned index")
                item = shard.get_slice(key)
                count = math.prod(item.get_shape())
                group = _group(key)
                groups[group] = groups.get(group, 0) + count
                if count < 1:
                    raise ValueError("Empty stored tensor")
                observed.add(key)
    if observed != set(weight_map):
        raise ValueError("Shard index references missing tensors")
    if not groups.get("language_model") or not groups.get("vision_tower"):
        raise ValueError("Official Gemma model is missing a text or vision submodule")
    return {
        "stored_tensor_count": len(observed),
        "stored_parameter_count": sum(groups.values()),
        "stored_parameter_groups": groups,
        "index_declared_parameters": index.get("metadata", {}).get("total_parameters"),
        "index_declared_bytes": index.get("metadata", {}).get("total_size"),
        "shard_names": sorted(shards),
        "shard_sha256": {name: sha256_file(snapshot / name) for name in sorted(shards)},
        "index_sha256": sha256_file(index_path),
    }


def synthetic_questions() -> list[dict[str, Any]]:
    """Three tiny schema probes, not assessment items or training data."""
    state = "The badge is blue. A blue badge allows entry after 09:00."
    questions = [
        {
            "type": "choice",
            "instructions": "Select the entry rule that applies at 10:00.",
            "criteria": {"a": "Entry allowed", "b": "Entry blocked"},
        },
        {
            "type": "noul",
            "instructions": "Does this badge allow entry at 10:00?",
            "criteria": {"false": "No", "true": "Yes"},
        },
        {
            "type": "score",
            "instructions": "Rate how strongly the rule supports entry at 10:00.",
            "criteria": ["Not supported", "Weakly supported", "Clearly supported"],
        },
    ]
    return [
        {"id": f"schema-{i}", "state": state, "questions": {"q": q}}
        for i, q in enumerate(questions)
    ]


def encode_probes(tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    if tokenizer.bos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Gemma tokenizer lacks BOS or PAD")
    result = []
    for item in synthetic_questions():
        row = question_to_row(item, "q", item["questions"]["q"])
        encoded = encode(row, tokenizer, max_length - 1)
        ids = [tokenizer.bos_token_id, *encoded["ids"]]
        if len(ids) > max_length:
            raise ValueError("Gemma schema probe would be truncated")
        result.append(
            {
                "task_type": row["task_type"],
                "ids": ids,
                "candidate_positions": [p + 1 for p in encoded["candidate_positions"]],
                "query_position": encoded["query_position"] + 1,
                "option_count": len(row["options"]),
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return result


def inspect(snapshot: Path, *, require_weights: bool = True) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    snapshot = snapshot.resolve(strict=True)
    if sha256_file(snapshot / "config.json") != CONFIG_SHA256:
        raise ValueError("Gemma official config changed from the pinned revision")
    revision = local_revision(snapshot)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if (
        config.model_type != "gemma4"
        or config.architectures != ["Gemma4ForConditionalGeneration"]
        or config.text_config.model_type != "gemma4_text"
        or config.text_config.num_hidden_layers != 30
        or config.text_config.num_experts != 128
        or config.text_config.top_k_experts != 8
    ):
        raise ValueError("Pinned Gemma architecture differs from expected model")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    cap = min(8192, config.text_config.max_position_embeddings)
    probes = encode_probes(tokenizer, cap)
    result = {
        "source_id": SOURCE_ID,
        "source_revision": revision,
        "config_sha256": CONFIG_SHA256,
        "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
        "model_type": config.model_type,
        "text_hidden_size": config.text_config.hidden_size,
        "text_layers": config.text_config.num_hidden_layers,
        "max_admitted_tokens": cap,
        "advertised_context_tokens": config.text_config.max_position_embeddings,
        "synthetic_probe_token_counts": {
            item["task_type"]: len(item["ids"]) for item in probes
        },
        "synthetic_probe_token_hashes": {
            item["task_type"]: item["token_ids_sha256"] for item in probes
        },
        "truncation_count": 0,
        "model_quality_evaluated": False,
    }
    if require_weights:
        result["weights"] = tensor_inventory(snapshot)
    return result


def verify_loaded_state(
    model: Any, snapshot: Path, inventory: dict[str, Any]
) -> dict[str, int | bool]:
    """Account for tied LM head and persisted buffers in official shards."""
    from safetensors import safe_open

    weight_map = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    state = model.state_dict()
    expected = set(weight_map) | {"lm_head.weight"}
    if set(state) != expected:
        raise RuntimeError(
            "Loaded state keys differ from official index plus tied LM head"
        )
    text = model.model.language_model
    if model.lm_head.weight.data_ptr() != text.embed_tokens.weight.data_ptr():
        raise RuntimeError("Omitted official LM head is not tied to token embeddings")
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(
            str(snapshot / shard_name), framework="pt", device="cpu"
        ) as shard:
            for key in sorted(shard.keys()):
                if tuple(state[key].shape) != tuple(shard.get_slice(key).get_shape()):
                    raise RuntimeError(f"Loaded official tensor shape differs: {key}")
    buffers = dict(model.named_buffers())
    stored_buffer_count = sum(
        state[key].numel() for key in weight_map if key in buffers
    )
    loaded_total_count = sum(parameter.numel() for parameter in model.parameters())
    loaded_text_count = sum(parameter.numel() for parameter in text.parameters())
    text_stored_buffers = sum(
        state[key].numel()
        for key in weight_map
        if key.startswith("model.language_model.") and key in buffers
    )
    if (
        loaded_total_count + stored_buffer_count != inventory["stored_parameter_count"]
        or loaded_text_count + text_stored_buffers
        != inventory["stored_parameter_groups"]["language_model"]
    ):
        raise RuntimeError(
            "Loaded parameters plus stored buffers differ from official shards"
        )
    return {
        "loaded_text_parameters": loaded_text_count,
        "loaded_total_parameters": loaded_total_count,
        "stored_buffer_elements": stored_buffer_count,
        "stored_text_buffer_elements": text_stored_buffers,
        "lm_head_tied": True,
    }


def zero_step(snapshot: Path, *, device: str, result: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    if device != "cuda:0" or not torch.cuda.is_available():
        raise ValueError("One isolated CUDA/ROCm-visible GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    probes = encode_probes(tokenizer, result["max_admitted_tokens"])
    torch.manual_seed(264)
    model, load_info = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": device},
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if any(
        load_info.get(field)
        for field in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError("Official Gemma package did not load without omissions")
    model.eval()
    text = model.model.language_model
    text.config.use_cache = False
    loaded = verify_loaded_state(model, snapshot, result["weights"])
    torch.manual_seed(264)
    head = CandidateHead(text.config.hidden_size).to(device)
    head.eval()
    outputs = {}
    with torch.inference_mode():
        for item in probes:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            direct = text(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
            wrapped = model.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
            maximum_drift = (direct.float() - wrapped.float()).abs().max().item()
            if not math.isfinite(maximum_drift) or maximum_drift > 1e-4:
                raise RuntimeError(
                    "Gemma text-only source path is not wrapper-identical"
                )
            positions = torch.tensor(item["candidate_positions"], device=device)
            query = direct[:, item["query_position"]]
            logits = head(direct[:, positions], query).float()[0]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Gemma decision readout produced nonfinite logits")
            values = logits.cpu().tolist()
            outputs[item["task_type"]] = {
                "candidate_count": item["option_count"],
                "input_tokens": len(item["ids"]),
                "finite_logits": True,
                "logits": values,
                "wrapper_hidden_max_abs_drift": maximum_drift,
            }
    torch.cuda.synchronize()
    return {
        **result,
        "probe_version": PROBE_VERSION,
        **loaded,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "typed_outputs": outputs,
        "random_untrained_head": True,
        "model_quality_evaluated": False,
    }


def compare_zero_step(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Admit repeated zero-step mechanics without interpreting accuracy."""
    required = (
        "source_id",
        "source_revision",
        "config_sha256",
        "tokenizer_json_sha256",
        "weights",
        "loaded_text_parameters",
        "loaded_total_parameters",
        "synthetic_probe_token_hashes",
        "probe_version",
    )
    if any(first.get(field) != second.get(field) for field in required):
        raise ValueError("Independent zero-step receipts differ in source identity")
    if (
        first.get("probe_version") != PROBE_VERSION
        or first.get("random_untrained_head") is not True
        or second.get("random_untrained_head") is not True
        or first.get("model_quality_evaluated") is not False
        or second.get("model_quality_evaluated") is not False
    ):
        raise ValueError("Receipts do not describe this untrained source probe")
    left = first.get("typed_outputs")
    right = second.get("typed_outputs")
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or set(left) != {"choice", "noul", "score"}
        or set(right) != set(left)
    ):
        raise ValueError("Independent zero-step receipts lack all three task types")
    drift = 0.0
    for kind in sorted(left):
        a, b = left[kind], right[kind]
        if a.get("candidate_count") != b.get("candidate_count") or a.get(
            "input_tokens"
        ) != b.get("input_tokens"):
            raise ValueError(f"{kind}: candidate or input lengths changed")
        if a.get("finite_logits") is not True or b.get("finite_logits") is not True:
            raise ValueError(f"{kind}: nonfinite model output")
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value > 1e-4
            or value < 0
            for value in (
                a.get("wrapper_hidden_max_abs_drift"),
                b.get("wrapper_hidden_max_abs_drift"),
            )
        ):
            raise ValueError(f"{kind}: wrapper and direct text path differ")
        x, y = a.get("logits"), b.get("logits")
        if (
            not isinstance(x, list)
            or not isinstance(y, list)
            or len(x) != a["candidate_count"]
            or len(y) != len(x)
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in [*x, *y]
            )
        ):
            raise ValueError(f"{kind}: malformed logits")
        if x.index(max(x)) != y.index(max(y)):
            raise ValueError(f"{kind}: selected category changed between processes")
        drift = max(drift, *(abs(v1 - v2) for v1, v2 in zip(x, y)))
    if drift > 1e-3:
        raise ValueError("Independent zero-step logit drift exceeds 1e-3")
    return {
        "probe_version": PROBE_VERSION,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "task_types_admitted": ["choice", "noul", "score"],
        "selected_categories_unchanged": True,
        "max_logit_abs_drift": drift,
        "max_allowed_logit_abs_drift": 1e-3,
        "model_quality_evaluated": False,
        "status": "zero_step_mechanics_passed",
    }


def private_output(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.is_absolute():
        raise ValueError("Receipt must be a new absolute path")
    parent = path.parent.resolve(strict=True)
    if (
        stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
    ):
        raise ValueError("Receipt directory must be private and owned by the runner")
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with os.fdopen(
        os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600), "wb"
    ) as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "zero-step", "compare"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "compare":
        if args.snapshot is not None or args.first is None or args.second is None:
            parser.error("compare requires --first and --second, without --snapshot")
        result = compare_zero_step(
            json.loads(args.first.read_text(encoding="utf-8")),
            json.loads(args.second.read_text(encoding="utf-8")),
        )
    else:
        if args.snapshot is None or args.first is not None or args.second is not None:
            parser.error("inspect/zero-step require only --snapshot")
        snapshot = args.snapshot.resolve(strict=True)
        result = inspect(snapshot)
        if args.mode == "zero-step":
            result = zero_step(snapshot, device=args.device, result=result)
    private_output(args.output, result)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "source_revision": SOURCE_REVISION,
                "receipt_sha256": sha256_file(args.output),
                "model_quality_evaluated": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
