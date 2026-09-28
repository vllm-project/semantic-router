"""One-GPU probe of an official ~27B start: native behaviour, cost, frozen features.

Stage ``native`` loads the official multimodal checkpoint in BF16, records two
greedy generations (the start's native interface) and decision-path latency on
a fixed gold-free SELECT roster. Stage ``features`` loads the text decoder at
the native Decision adapter precision (FP32 parameters, BF16 autocast), checks
cached-feature/native-forward parity, and stores candidate-endpoint and query
hidden states at 1/2 depth, 3/4 depth and the final normed output for TRAIN,
SELECT, CAL, typed DEV and CSS pilot. Labels are stored only for the training
partitions; benchmark prompts carry no gold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from training.model.data import canonical, file_sha256
from training.model.decision_model import CandidateHead, DecisionModel, encode
from training.model.gemma4 import encode_gemma
from training.model.infer import load_prompts, question_to_row

PROBE_VERSION = "decision2-27b-backbone-probe/1"
DEPTHS = (("half", 0.5), ("three_quarter", 0.75))
NATIVE_PROMPTS = (
    "Which number is larger, 17 or 71? Reply with the number only.",
    "A shop allows returns within 30 days with a receipt. Dana bought a lamp 12 days ago and has the receipt. Can she return it? Answer yes or no.",
)


def family_of(source: Path) -> str:
    model_type = json.loads((source / "config.json").read_text(encoding="utf-8"))[
        "model_type"
    ]
    if model_type == "qwen3_5":
        return "qwen"
    if model_type == "gemma4":
        return "gemma"
    raise ValueError(f"Unsupported start model_type {model_type}")


def encoder_for(family: str):
    return encode if family == "qwen" else encode_gemma


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def roster(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(("27b-probe/" + row["id"]).encode()).hexdigest(),
    )[:count]


def layer_indices(layers: int) -> dict[str, int]:
    return {name: int(layers * fraction + 0.5) - 1 for name, fraction in DEPTHS}


def load_full(source: Path, family: str, dtype: torch.dtype, device: str | None):
    kwargs = dict(
        dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if device:
        kwargs["device_map"] = {"": device}
    if family == "qwen":
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        from transformers import Gemma4ForConditionalGeneration as cls
    model, info = cls.from_pretrained(source, **kwargs)
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete official load: {info}")
    return model


def text_decoder(full: Any) -> Any:
    text = full.model.language_model
    text.config.use_cache = False
    return text


def forward_rows(
    text, items, device, *, hooks: dict[str, int] | None = None, autocast: bool
):
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, index in (hooks or {}).items():

        def hook(_module, _inputs, output, name=name):
            captured[name] = output[0] if isinstance(output, tuple) else output

        handles.append(text.layers[index].register_forward_hook(hook))
    try:
        for item in items:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast
            ):
                hidden = text(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                ).last_hidden_state
            torch.cuda.synchronize()
            yield item, hidden, dict(captured), (time.perf_counter() - started) * 1000
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()


def latency_summary(values: list[float], tokens: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "mean_ms": statistics.fmean(values),
        "tokens_mean": statistics.fmean(tokens),
        "tokens_per_second": sum(tokens) / (sum(values) / 1000),
    }


def native_stage(args, family: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    full = load_full(args.source, family, torch.bfloat16, device).eval()
    load_seconds = time.perf_counter() - started
    after_load = torch.cuda.memory_allocated()
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    generations = []
    for prompt in NATIVE_PROMPTS:
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"Question: {prompt}\nAnswer:"
        ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=not tokenizer.chat_template
        ).input_ids.to(device)
        with torch.inference_mode():
            output = full.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=48,
                do_sample=False,
            )
        continuation = tokenizer.decode(
            output[0, ids.shape[1] :], skip_special_tokens=False
        )
        generations.append(
            {
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chat_template_used": bool(tokenizer.chat_template),
                "new_tokens": int(output.shape[1] - ids.shape[1]),
                "continuation": continuation[:240],
            }
        )
    text_model = text_decoder(full)
    torch.manual_seed(20260928)
    head = CandidateHead(text_model.config.hidden_size).to(device).float().eval()
    rows = roster(load_rows(args.select), args.latency_rows)
    items = [encoder_for(family)(row, tokenizer, 8192) for row in rows]
    latencies, tokens = [], []
    for item, hidden, _, ms in forward_rows(text_model, items, device, autocast=False):
        with torch.inference_mode():
            head(
                hidden[:, item["candidate_positions"]].float(),
                hidden[:, item["query_position"]].float(),
            )
        latencies.append(ms)
        tokens.append(len(item["ids"]))
    result = {
        "precision": "BF16 parameters and compute (serving-style), full official checkpoint resident",
        "load_seconds": load_seconds,
        "resident_bytes_after_load": after_load,
        "peak_bytes": torch.cuda.max_memory_allocated(),
        "loaded_total_parameters": sum(p.numel() for p in full.parameters()),
        "loaded_text_decoder_parameters": sum(
            p.numel() for p in text_model.parameters()
        ),
        "native_generations": generations,
        "decision_forward_latency": latency_summary(latencies[1:], tokens[1:]),
    }
    del full, text_model, head
    gc.collect()
    torch.cuda.empty_cache()
    return result


def feature_items(family: str, tokenizer, args) -> dict[str, list[dict[str, Any]]]:
    encoder = encoder_for(family)
    splits: dict[str, list[dict[str, Any]]] = {}
    for role, path in (
        ("train", args.train),
        ("select", args.select),
        ("cal", args.cal),
    ):
        splits[role] = []
        for row in load_rows(path):
            item = encoder(row, tokenizer, args.train_limit)
            item["language"] = row.get("language")
            splits[role].append(item)
    for role, path in (("dev", args.dev), ("css_pilot", args.css_pilot)):
        splits[role] = []
        for prompt in load_prompts(path):
            for question_id, question in prompt["questions"].items():
                row = question_to_row(prompt, question_id, question)
                try:
                    item = encoder(row, tokenizer, args.benchmark_limit)
                except ValueError as exc:
                    if "exceeds max_length" not in str(exc) and "truncated" not in str(
                        exc
                    ):
                        raise
                    item = {
                        "id": row["id"],
                        "over_budget": True,
                        "keys": [o["key"] for o in row["options"]],
                        "task_type": row["task_type"],
                    }
                item["label"] = None
                item["family"] = None
                item["prompt_id"] = prompt["id"]
                item["question_id"] = question_id
                splits[role].append(item)
    return splits


def save_split(
    path: Path,
    role: str,
    rows: list[dict[str, Any]],
    tensors: dict[str, list[torch.Tensor]],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    payload = {
        name: torch.cat(parts).contiguous() for name, parts in tensors.items() if parts
    }
    offsets, total, query_index, live = [0], 0, [], 0
    for row in rows:
        if row.get("over_budget"):
            query_index.append(-1)
        else:
            total += len(row["keys"])
            query_index.append(live)
            live += 1
        offsets.append(total)
    payload["candidate_offsets"] = torch.tensor(offsets, dtype=torch.long)
    save_file(payload, str(path / f"{role}.features.safetensors"))
    meta = [
        {
            k: row.get(k)
            for k in (
                "id",
                "prompt_id",
                "question_id",
                "keys",
                "task_type",
                "family",
                "label",
                "language",
                "over_budget",
                "prompt_sha256",
                "token_ids_sha256",
            )
        }
        | {"tokens": len(row["ids"]) if "ids" in row else None, "query_index": index}
        for row, index in zip(rows, query_index)
    ]
    (path / f"{role}.rows.json").write_text(
        json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "rows": len(rows),
        "over_budget": sum(bool(row.get("over_budget")) for row in rows),
        "candidates": total,
        "features_sha256": file_sha256(path / f"{role}.features.safetensors"),
        "rows_sha256": file_sha256(path / f"{role}.rows.json"),
    }


def features_stage(args, family: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if family == "qwen":
        model, tokenizer = DecisionModel.from_base(
            args.source, args.revision, source_stage=args.source_stage
        )
        text = model.backbone
        del model
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
        full = load_full(args.source, family, torch.float32, None)
        text = text_decoder(full)
        del full
    gc.collect()
    text = text.float().to(device).eval()
    load_seconds = time.perf_counter() - started
    resident = torch.cuda.memory_allocated()
    hooks = layer_indices(len(text.layers))
    splits = feature_items(family, tokenizer, args)
    torch.manual_seed(20260928)
    head = CandidateHead(text.config.hidden_size).to(device).float().eval()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    parity_rows = splits["select"][: args.parity_rows]
    native_logits = []
    for item, hidden, _, _ in forward_rows(text, parity_rows, device, autocast=True):
        with torch.inference_mode():
            native_logits.append(
                head(
                    hidden[:, item["candidate_positions"]],
                    hidden[:, item["query_position"]],
                )
                .float()[0]
                .cpu()
            )
    summaries, timing = {}, {}
    for role, rows in splits.items():
        tensors = {
            f"{kind}_{layer}": []
            for kind in ("cand", "query")
            for layer in ("final", *hooks)
        }
        latencies, tokens = [], []
        live = [row for row in rows if not row.get("over_budget")]
        for item, hidden, captured, ms in forward_rows(
            text, live, device, hooks=hooks, autocast=True
        ):
            positions = item["candidate_positions"]
            layers = {"final": hidden, **captured}
            for layer, value in layers.items():
                tensors[f"cand_{layer}"].append(value[0, positions].float().cpu())
                tensors[f"query_{layer}"].append(
                    value[0, item["query_position"]][None].float().cpu()
                )
            latencies.append(ms)
            tokens.append(len(item["ids"]))
        summaries[role] = save_split(out, role, rows, tensors)
        timing[role] = latency_summary(latencies, tokens)
        print(
            json.dumps(
                {
                    "role": role,
                    **summaries[role],
                    "tokens_per_second": timing[role]["tokens_per_second"],
                }
            ),
            flush=True,
        )
    from safetensors.torch import load_file, save_file

    save_file(
        {
            k: v.detach().float().cpu().contiguous()
            for k, v in head.state_dict().items()
        },
        str(out / "parity_head.safetensors"),
    )
    cached = load_file(str(out / "select.features.safetensors"))
    offsets = cached["candidate_offsets"].tolist()
    worst, changed = 0.0, 0
    for index, reference in enumerate(native_logits):
        cand = cached["cand_final"][offsets[index] : offsets[index + 1]].to(device)
        query = cached["query_final"][index : index + 1].to(device)
        with torch.inference_mode():
            logits = head(cand[None], query).float()[0].cpu()
        p, q = logits.softmax(-1), reference.softmax(-1)
        worst = max(worst, (p - q).abs().max().item())
        changed += int(p.argmax() != q.argmax())
    parity = {
        "rows": len(native_logits),
        "argmax_changes": changed,
        "max_abs_probability_diff": worst,
        "tolerance": 1e-4,
    }
    parity["passed"] = changed == 0 and math.isfinite(worst) and worst <= 1e-4
    return {
        "precision": "FP32 parameters, BF16 autocast, FP32 head (native Decision adapter precision)",
        "load_seconds": load_seconds,
        "resident_bytes_after_load": resident,
        "peak_bytes": torch.cuda.max_memory_allocated(),
        "loaded_decision_path_parameters": sum(p.numel() for p in text.parameters()),
        "hidden_size": text.config.hidden_size,
        "decoder_layers": len(text.layers),
        "hooked_layer_indices": hooks,
        "train_limit": args.train_limit,
        "benchmark_limit": args.benchmark_limit,
        "splits": summaries,
        "forward_timing": timing,
        "cached_feature_parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--source-stage", choices=("base", "posttrained"), required=True
    )
    parser.add_argument("--stages", default="native,features")
    for name in ("train", "select", "cal", "dev", "css_pilot"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--benchmark-limit", type=int, default=4096)
    parser.add_argument("--latency-rows", type=int, default=65)
    parser.add_argument("--parity-rows", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    family = family_of(args.source)
    receipt: dict[str, Any] = {
        "schema_version": PROBE_VERSION,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "source_stage": args.source_stage,
        "family": family,
        "config_sha256": file_sha256(args.source / "config.json"),
        "tokenizer_json_sha256": file_sha256(args.source / "tokenizer.json"),
        "inputs_sha256": {
            name: file_sha256(getattr(args, name))
            for name in ("train", "select", "cal", "dev", "css_pilot")
        },
        "device": torch.cuda.get_device_name(0),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "code_sha256": file_sha256(Path(__file__)),
    }
    stages = args.stages.split(",")
    if "native" in stages:
        receipt["native"] = native_stage(args, family)
        print(
            json.dumps(
                {
                    "native": {
                        k: v
                        for k, v in receipt["native"].items()
                        if k != "native_generations"
                    }
                }
            ),
            flush=True,
        )
    if "features" in stages:
        receipt["features"] = features_stage(args, family)
    fd = os.open(args.receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    parity = receipt.get("features", {}).get("cached_feature_parity", {"passed": True})
    print(
        json.dumps({"receipt_sha256": file_sha256(args.receipt), "parity": parity}),
        flush=True,
    )
    raise SystemExit(0 if parity["passed"] else 2)


if __name__ == "__main__":
    main()
