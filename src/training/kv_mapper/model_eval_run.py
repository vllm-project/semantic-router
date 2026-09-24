#!/usr/bin/env python3
"""Held-out HellaSwag teacher-forced evaluation of a fitted Qwen3 mapper."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import read_artifact
from src.training.kv_mapper.eval import build_report, write_report
from src.training.kv_mapper.hooks import attach_pre_rope_hooks, remove_hooks


def _load_model(name: str, revision: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    return model


def _examples(revision: str, count: int, seed: int):
    from datasets import load_dataset

    data = load_dataset(
        "Rowan/hellaswag", split="validation", revision=revision, streaming=True
    )
    examples = []
    rng = random.Random(seed)
    seen = 0
    for row_index, row in enumerate(data):
        if row["label"] in ("", None):
            continue
        row = dict(row)
        row["_eval_id"] = f"validation:{row_index}"
        seen += 1
        if len(examples) < count:
            examples.append(row)
        else:
            slot = rng.randrange(seen)
            if slot < count:
                examples[slot] = row
    if len(examples) != count:
        raise ValueError(f"only {len(examples)} labeled HellaSwag rows available")
    return sorted(examples, key=lambda row: int(row["ind"]))


def _cache_pairs(cache):
    return [(layer.keys, layer.values) for layer in cache.layers]


def _rotate_keys(model, keys: torch.Tensor) -> torch.Tensor:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    # keys are (seq, heads, dim) before RoPE; DynamicCache uses (1, heads, seq, dim).
    key = keys.transpose(0, 1).unsqueeze(0)
    positions = torch.arange(key.shape[-2], device=key.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(key, positions)
    return apply_rotary_pos_emb(key, key, cos, sin)[1]


def _mapped_pairs(source: dict, target, manifest, weights, dtype):
    selections = manifest.source_layers_per_target["k"]
    n_layers = len(selections)
    result = []
    for layer in range(n_layers):
        indices = selections[str(layer)]
        channels = []
        for channel in ("k", "v"):
            features = torch.cat(
                [
                    source[i][channel][0].reshape(source[i][channel].shape[1], -1)
                    for i in indices
                ],
                dim=-1,
            ).to(target.device, dtype=torch.float32)
            mapped = features @ weights[f"target.{layer}.{channel}.W"]
            mapped += weights[f"target.{layer}.{channel}.b"]
            mapped = mapped.reshape(
                features.shape[0],
                manifest.compatibility.num_kv_heads,
                manifest.compatibility.head_dim,
            ).to(dtype)
            channels.append(mapped)
        key = _rotate_keys(target, channels[0])
        value = channels[1].transpose(0, 1).unsqueeze(0)
        result.append((key, value, channels))
    return result


def _raw_pairs(source: dict, target, n_target: int, dtype):
    pairs = []
    n_source = len(source)
    for layer in range(n_target):
        source_layer = round(layer * (n_source - 1) / max(n_target - 1, 1))
        slot = source[source_layer]
        key = slot["k"][0].to(target.device, dtype=dtype)
        value = slot["v"][0].to(target.device, dtype=dtype)
        pairs.append((_rotate_keys(target, key), value.transpose(0, 1).unsqueeze(0)))
    return pairs


def _continuation_score(
    model, prefix_pairs, last_context_id: int, ending_ids: list[int]
) -> float:
    from transformers.cache_utils import DynamicCache, DynamicLayer

    cache = DynamicCache()
    cache.layers = [DynamicLayer.from_tensors(k, v) for k, v in prefix_pairs]
    cache.num_hidden_layers = len(prefix_pairs)
    ids = torch.tensor(
        [[last_context_id] + ending_ids[:-1]], device=model.device, dtype=torch.long
    )
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    answers = torch.tensor(ending_ids, device=model.device, dtype=torch.long)
    log_probs = out.logits[0].float().log_softmax(dim=-1)
    return float(log_probs.gather(1, answers[:, None]).mean().item())


def _relative_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float((pred.float() - true.float()).norm().div(true.float().norm() + 1e-8))


@torch.inference_mode()
def run(args) -> dict:
    from transformers import AutoTokenizer

    manifest, tensors = read_artifact(args.artifact)
    compat = manifest.compatibility
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[compat.precision]
    source = _load_model(
        compat.source_model, compat.source_revision, args.source_device, dtype
    )
    target = _load_model(
        compat.target_model, compat.target_revision, args.target_device, dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        compat.target_model, revision=compat.target_revision
    )
    source_tokenizer = AutoTokenizer.from_pretrained(
        compat.source_model, revision=compat.source_revision
    )
    if tokenizer.get_vocab() != source_tokenizer.get_vocab():
        raise ValueError("source and target tokenizers differ")
    if len(source.model.layers) <= max(
        max(x) for x in manifest.source_layers_per_target["k"].values()
    ) or len(target.model.layers) != len(manifest.source_layers_per_target["k"]):
        raise ValueError("artifact layer selection does not match loaded models")
    device = target.device
    weights = {
        name: torch.from_numpy(value).to(device) for name, value in tensors.items()
    }
    del tensors
    examples = _examples(args.dataset_revision, args.count, args.seed)
    arms = {name: [] for name in ("cold", "mapped", "raw_source", "zero")}
    kv_errors = {name: [] for name in ("mapped", "raw_source", "zero")}
    item_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item_index, row in enumerate(examples):
        item_id = row["_eval_id"]
        context_ids = tokenizer.encode(row["ctx"], add_special_tokens=False)
        if len(context_ids) < 2:
            raise ValueError(f"context too short for {item_id}")
        endings = [
            tokenizer.encode(" " + text, add_special_tokens=False)
            for text in row["endings"]
        ]
        if any(not ending for ending in endings):
            raise ValueError(f"empty ending for {item_id}")
        prefix = context_ids[:-1]
        source_ids = torch.tensor([prefix], device=source.device)
        source_slots, source_handles = attach_pre_rope_hooks(
            source, compat.num_kv_heads, compat.head_dim
        )
        try:
            source(input_ids=source_ids, use_cache=False)
        finally:
            remove_hooks(source_handles)
        target_slots, target_handles = attach_pre_rope_hooks(
            target, compat.num_kv_heads, compat.head_dim
        )
        try:
            cold_out = target(
                input_ids=torch.tensor([prefix], device=device), use_cache=True
            )
        finally:
            remove_hooks(target_handles)
        cold = _cache_pairs(cold_out.past_key_values)
        if item_index == 0:
            for layer, (cached_k, cached_v) in enumerate(cold):
                hooked_k = target_slots[layer]["k"][0]
                hooked_v = target_slots[layer]["v"].transpose(1, 2)
                if not torch.allclose(
                    _rotate_keys(target, hooked_k), cached_k, atol=0.01, rtol=0.01
                ) or not torch.equal(hooked_v, cached_v):
                    raise ValueError(
                        f"hooked target KV differs from cache at layer {layer}"
                    )
        mapped = _mapped_pairs(source_slots, target, manifest, weights, dtype)
        raw = _raw_pairs(source_slots, target, len(cold), dtype)
        zero = [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in cold]
        pairs = {
            "cold": cold,
            "mapped": [(k, v) for k, v, _ in mapped],
            "raw_source": raw,
            "zero": zero,
        }
        for name in kv_errors:
            err_k, err_v = [], []
            for layer in range(len(cold)):
                true_k = target_slots[layer]["k"].transpose(1, 2)
                true_v = target_slots[layer]["v"].transpose(1, 2)
                if name == "mapped":
                    pred_k = mapped[layer][2][0].transpose(0, 1).unsqueeze(0)
                    pred_v = mapped[layer][2][1].transpose(0, 1).unsqueeze(0)
                elif name == "raw_source":
                    source_layer = round(
                        layer * (len(source_slots) - 1) / max(len(cold) - 1, 1)
                    )
                    pred_k = source_slots[source_layer]["k"].transpose(1, 2).to(device)
                    pred_v = source_slots[source_layer]["v"].transpose(1, 2).to(device)
                else:
                    pred_k, pred_v = torch.zeros_like(true_k), torch.zeros_like(true_v)
                err_k.append(_relative_error(pred_k, true_k))
                err_v.append(_relative_error(pred_v, true_v))
            kv_errors[name].append(
                {
                    "id": item_id,
                    "key_rel_err": float(np.mean(err_k)),
                    "value_rel_err": float(np.mean(err_v)),
                }
            )
        scores = {}
        for name, cache_pairs in pairs.items():
            scores[name] = [
                _continuation_score(target, cache_pairs, context_ids[-1], ending)
                for ending in endings
            ]
            gold = int(row["label"])
            arms[name].append({"id": item_id, "score": scores[name][gold]})
        item_rows.append(
            {
                "id": item_id,
                "hellaswag_ind": row["ind"],
                "label": gold,
                "scores": scores,
                "context_token_count": len(context_ids),
            }
        )
        (args.output_dir / "items.json").write_text(
            json.dumps(
                {
                    "metric": "gold_ending_mean_log_probability",
                    "reference": "cold",
                    "arms": arms,
                    "items": item_rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {"completed": item_index + 1, "count": len(examples), "id": item_id}
            ),
            flush=True,
        )
    accuracy_arms = {
        name: [
            {
                "id": item["id"],
                "score": float(int(np.argmax(item["scores"][name]) == item["label"])),
            }
            for item in item_rows
        ]
        for name in arms
    }
    reports = {
        "gold_log_probability": build_report("gold_ending_mean_log_probability", arms),
        "accuracy": build_report("hellaswag_accuracy", accuracy_arms),
        "kv_relative_error": {
            name: {
                channel: float(np.mean([item[channel] for item in rows]))
                for channel in ("key_rel_err", "value_rel_err")
            }
            for name, rows in kv_errors.items()
        },
        "dataset": "Rowan/hellaswag",
        "dataset_revision": args.dataset_revision,
        "sampling_seed": args.seed,
        "artifact": manifest.mapper_id,
    }
    write_report(args.output_dir / "report.json", reports)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
