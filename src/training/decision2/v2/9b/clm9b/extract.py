"""Frozen-backbone feature extraction for joint and disaggregated readouts.

One process loads one pinned source, verifies its bytes, and writes, per named
input set:

* joint features: the unchanged native prompt (state, question, every option,
  query suffix) in one causal pass; vectors at each option endpoint and at the
  final query token;
* disaggregated features: the state text (context + question) and every
  distinct candidate text, each encoded alone; last-token and mean pooling.

Layers are the residual outputs of decoder layers 16 and 24 and the final-norm
output (32). No backward pass, optimizer or label enters the forward. Labels
are copied only from labelled TRAIN/SELECT/CAL partitions; gold-free prompt
sets carry none. Inputs longer than ``--max-length`` are recorded as invalid,
never truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import pins
from .render import (
    DISAGGREGATED_RENDER_VERSION,
    candidate_texts,
    score_level_order,
    state_text,
    text_sha256,
)

EXTRACT_VERSION = "decision2-9b-frozen-features-v2"
FINAL_LAYER = 32


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_inputs(
    spec: str, kind: str
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Parse NAME=PATH[:ROLE] and return flattened rows plus input identity."""
    from training.model.data import load_partition
    from training.model.infer import load_prompts, question_to_row

    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise ValueError(f"Bad input spec {spec!r}")
    if kind == "partition":
        path_text, _, role = rest.rpartition(":")
        if not path_text or role not in ("train", "select", "cal"):
            raise ValueError("Partition spec must be NAME=PATH:train|select|cal")
        path = Path(path_text)
        rows = load_partition(path, role)
        for row in rows:
            row["_label"] = row["label"]
        identity = {"path_sha256": pins.file_sha256(path), "role": role, "kind": kind}
    elif kind == "prompts":
        path = Path(rest)
        items = load_prompts(path)
        rows = []
        for item in items:
            for question_id, question in item["questions"].items():
                row = question_to_row(item, question_id, question)
                row["_label"] = None
                row["_item_id"] = item["id"]
                row["_question_id"] = question_id
                rows.append(row)
        identity = {
            "path_sha256": pins.file_sha256(path),
            "items": len(items),
            "kind": kind,
        }
    elif kind == "rows":
        path = Path(rest)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            missing = {"id", "state", "instructions", "options", "task_type"} - set(row)
            if missing:
                raise ValueError(f"{path}: row lacks {sorted(missing)}")
            row.setdefault("family", "unlabelled-rows")
            row["_label"] = None
            row["label"] = 0
        identity = {"path_sha256": pins.file_sha256(path), "kind": kind}
    else:
        raise ValueError(kind)
    return name, rows, identity


def plan_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[dict[str, Any]], list[list[int]], dict[str, list[int]]]:
    """Tokenize joint prompts and disaggregated texts without any truncation."""
    from training.model.decision_model import encode

    records, joint_ids, texts = [], [], {}
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "index": index,
            "id": row["id"],
            "task_type": row["task_type"],
            "family": row.get("family"),
            "label": row["_label"],
            "keys": [option["key"] for option in row["options"]],
            "group_id": row.get("group_id"),
            "language": row.get("language"),
            "input_sha256": row.get("input_sha256"),
            "item_id": row.get("_item_id"),
            "question_id": row.get("_question_id"),
        }
        if row["task_type"] == "score":
            record["score_order"] = score_level_order(row)
        try:
            encoded = encode(row, tokenizer, max_length)
            record.update(
                j_valid=True,
                j_tokens=len(encoded["ids"]),
                j_candidate_positions=encoded["candidate_positions"],
                j_query_position=encoded["query_position"],
                prompt_sha256=encoded["prompt_sha256"],
                token_ids_sha256=encoded["token_ids_sha256"],
                j_batch_index=len(joint_ids),
            )
            joint_ids.append(encoded["ids"])
        except ValueError as exc:
            if "exceeds max_length" not in str(exc):
                raise
            record.update(j_valid=False, j_error="max_length_exceeded")
        state = state_text(row)
        candidates = candidate_texts(row)
        shas, tokens, valid = [], [], True
        for text in [state, *candidates]:
            sha = text_sha256(text)
            if sha not in texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    raise ValueError(f"{row['id']}: empty tokenized disaggregated text")
                texts[sha] = ids
            shas.append(sha)
            tokens.append(len(texts[sha]))
            valid = valid and len(texts[sha]) <= max_length
        record.update(
            d_valid=valid,
            d_state_sha=shas[0],
            d_state_tokens=tokens[0],
            d_candidate_shas=shas[1:],
            d_candidate_tokens=tokens[1:],
        )
        records.append(record)
    return records, joint_ids, texts


def token_batches(lengths: list[int], budget: int, max_rows: int) -> list[list[int]]:
    """Length-sorted batches whose padded size stays within ``budget`` tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches, current, longest = [], [], 0
    for index in order:
        padded = (-(-max(longest, lengths[index]) // 8)) * 8
        if current and (
            padded * (len(current) + 1) > budget or len(current) >= max_rows
        ):
            batches.append(current)
            current, longest = [], 0
            padded = (-(-lengths[index] // 8)) * 8
        current.append(index)
        longest = max(longest, lengths[index])
    if current:
        batches.append(current)
    return batches


class LayerTap:
    """Capture decoder-layer residual outputs during one forward call."""

    def __init__(self, backbone: Any, layers: Iterable[int]):
        self.outputs: dict[int, Any] = {}
        self.handles = []
        for layer in layers:
            if layer == FINAL_LAYER:
                continue
            module = backbone.layers[layer - 1]
            self.handles.append(module.register_forward_hook(self._hook(layer)))

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self.outputs[layer] = output[0] if isinstance(output, tuple) else output

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def forward_layers(
    backbone: Any,
    tap: LayerTap,
    ids: list[list[int]],
    pad_id: int,
    device: Any,
    layers: tuple[int, ...],
):
    import torch

    width = (-(-max(len(row) for row in ids) // 8)) * 8
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for row, values in enumerate(ids):
        input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row, : len(values)] = 1
    input_ids, mask = input_ids.to(device), mask.to(device)
    tap.outputs.clear()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        final = backbone(
            input_ids=input_ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
    hidden = {
        layer: (final if layer == FINAL_LAYER else tap.outputs[layer])
        for layer in layers
    }
    return {layer: value.float() for layer, value in hidden.items()}, mask


def load_backbone(source: str, root: Path):
    import torch

    loader = pins.SOURCES[source]["loader"]
    if loader == "qwen3_5_conditional_generation":
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        full, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            root,
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
            output_loading_info=True,
            low_cpu_mem_usage=True,
        )
        if any(
            info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError(f"Incomplete source load: {info}")
        backbone = full.model.language_model
        del full
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    elif loader == "qwen3_5_text_backbone_dir":
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        backbone, info = Qwen3_5TextModel.from_pretrained(
            root / "backbone",
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if any(
            info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError(f"Incomplete source load: {info}")
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    else:
        raise ValueError(loader)
    backbone.config.use_cache = False
    backbone.requires_grad_(False)
    parameters = sum(p.numel() for p in backbone.parameters())
    if parameters != pins.SOURCES[source]["text_parameters"]:
        raise RuntimeError(f"{source}: loaded {parameters} text parameters")
    return backbone.eval(), tokenizer, parameters


def extract_joint(
    backbone, tap, records, joint_ids, pad_id, device, layers, budget, max_rows, log
):
    import torch

    total = sum(len(r["j_candidate_positions"]) + 1 for r in records if r["j_valid"])
    store = {
        layer: torch.empty((total, backbone.config.hidden_size), dtype=torch.float32)
        for layer in layers
    }
    offset = 0
    for record in records:
        if record["j_valid"]:
            record["j_offset"] = offset
            record["j_count"] = len(record["j_candidate_positions"]) + 1
            offset += record["j_count"]
    by_batch = {r["j_batch_index"]: r for r in records if r["j_valid"]}
    lengths = [len(ids) for ids in joint_ids]
    started = time.perf_counter()
    for number, batch in enumerate(token_batches(lengths, budget, max_rows)):
        hidden, _ = forward_layers(
            backbone, tap, [joint_ids[i] for i in batch], pad_id, device, layers
        )
        for row, index in enumerate(batch):
            record = by_batch[index]
            positions = torch.tensor(
                [*record["j_candidate_positions"], record["j_query_position"]],
                device=device,
            )
            for layer in layers:
                vectors = hidden[layer][row].index_select(0, positions)
                if not torch.isfinite(vectors).all():
                    raise RuntimeError(f"{record['id']}: nonfinite joint layer {layer}")
                store[layer][
                    record["j_offset"] : record["j_offset"] + record["j_count"]
                ] = vectors.cpu()
        if number % 50 == 0:
            log(
                {
                    "phase": "joint",
                    "batch": number,
                    "rows": len(batch),
                    "seconds": time.perf_counter() - started,
                }
            )
    return store


def extract_texts(
    backbone, tap, texts, pad_id, device, layers, budget, max_rows, max_length, log
):
    import torch

    order = [sha for sha, ids in texts.items() if len(ids) <= max_length]
    index = {sha: position for position, sha in enumerate(order)}
    width = backbone.config.hidden_size
    store = {
        f"{pool}_L{layer}": torch.empty((len(order), width), dtype=torch.float32)
        for pool in ("last", "mean")
        for layer in layers
    }
    lengths = [len(texts[sha]) for sha in order]
    started = time.perf_counter()
    for number, batch in enumerate(token_batches(lengths, budget, max_rows)):
        ids = [texts[order[i]] for i in batch]
        hidden, mask = forward_layers(backbone, tap, ids, pad_id, device, layers)
        keep = mask.bool()[..., None]
        counts = mask.sum(1, keepdim=True).float()
        last = torch.tensor([len(values) - 1 for values in ids], device=device)
        rows = torch.arange(len(batch), device=device)
        for layer in layers:
            value = hidden[layer]
            pooled_last = value[rows, last]
            # Padded positions may be non-finite under fully masked attention rows.
            pooled_mean = (
                torch.where(keep, value, torch.zeros_like(value)).sum(1) / counts
            )
            if not (
                torch.isfinite(pooled_last).all() and torch.isfinite(pooled_mean).all()
            ):
                raise RuntimeError(f"nonfinite disaggregated layer {layer}")
            store[f"last_L{layer}"][batch] = pooled_last.cpu()
            store[f"mean_L{layer}"][batch] = pooled_mean.cpu()
        if number % 100 == 0:
            log(
                {
                    "phase": "texts",
                    "batch": number,
                    "rows": len(batch),
                    "seconds": time.perf_counter() - started,
                }
            )
    return store, index


PARITY_MIN_COS = 0.999


def _compare(stats, layer, observed, stored) -> None:
    import torch

    entry = stats.setdefault(str(layer), {"max_abs": 0.0, "min_cos": 1.0})
    entry["max_abs"] = max(entry["max_abs"], float((observed - stored).abs().max()))
    width = observed.shape[-1]
    cosine = torch.nn.functional.cosine_similarity(
        observed.reshape(-1, width), stored.reshape(-1, width), dim=-1
    )
    entry["min_cos"] = min(entry["min_cos"], float(cosine.min()))


def parity_probe(
    backbone,
    tap,
    records,
    joint_ids,
    texts,
    text_index,
    joint_store,
    text_store,
    pad_id,
    device,
    layers,
    sample,
):
    """Fixed-sample checks of the stored features.

    ``joint_repeat`` reruns single joint prompts (the stored shape) and must
    match; ``joint_batched_diagnostic`` places the same prompts in one padded
    batch to measure batch-shape sensitivity; ``texts`` compares batched text
    features with single-text forwards. Gated: joint_repeat and texts.
    """
    import torch

    valid = [r for r in records if r["j_valid"]]
    step = max(1, len(valid) // max(1, sample))
    chosen = valid[::step][:sample]
    report: dict[str, Any] = {
        "sample_rows": len(chosen),
        "joint_repeat": {},
        "joint_batched_diagnostic": {},
        "texts": {},
    }

    def stored_joint(record, layer):
        return joint_store[layer][
            record["j_offset"] : record["j_offset"] + record["j_count"]
        ]

    def gather(hidden, row, record, layer):
        positions = torch.tensor(
            [*record["j_candidate_positions"], record["j_query_position"]],
            device=device,
        )
        return hidden[layer][row].index_select(0, positions).cpu()

    for record in chosen:
        hidden, _ = forward_layers(
            backbone, tap, [joint_ids[record["j_batch_index"]]], pad_id, device, layers
        )
        for layer in layers:
            _compare(
                report["joint_repeat"],
                layer,
                gather(hidden, 0, record, layer),
                stored_joint(record, layer),
            )
    if chosen:
        hidden, _ = forward_layers(
            backbone,
            tap,
            [joint_ids[r["j_batch_index"]] for r in chosen],
            pad_id,
            device,
            layers,
        )
        for row, record in enumerate(chosen):
            for layer in layers:
                _compare(
                    report["joint_batched_diagnostic"],
                    layer,
                    gather(hidden, row, record, layer),
                    stored_joint(record, layer),
                )
    for record in chosen:
        for sha in [record["d_state_sha"], *record["d_candidate_shas"][:2]]:
            if sha not in text_index:
                continue
            ids = texts[sha]
            hidden, _ = forward_layers(backbone, tap, [ids], pad_id, device, layers)
            for layer in layers:
                _compare(
                    report["texts"],
                    layer,
                    hidden[layer][0, len(ids) - 1].cpu(),
                    text_store[f"last_L{layer}"][text_index[sha]],
                )
    report["gate_min_cos"] = PARITY_MIN_COS
    report["passed"] = all(
        entry["min_cos"] >= PARITY_MIN_COS
        for key in ("joint_repeat", "texts")
        for entry in report[key].values()
    )
    return report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(pins.SOURCES))
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument(
        "--partition", action="append", default=[], help="NAME=PATH:ROLE"
    )
    parser.add_argument(
        "--prompts", action="append", default=[], help="NAME=PATH gold-free prompts"
    )
    parser.add_argument(
        "--rows",
        action="append",
        default=[],
        help="NAME=PATH unlabelled flattened rows",
    )
    parser.add_argument(
        "--expect-data", action="append", default=[], help="NAME=SHA256 input pin"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument(
        "--max-batch-rows", type=int, default=32, help="disaggregated text batches"
    )
    parser.add_argument(
        "--joint-batch-rows", type=int, default=1, help="joint prompts per forward"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="technical preflight: first N rows per input",
    )
    parser.add_argument("--parity-sample", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import __version__ as transformers_version

    if args.output.exists():
        raise FileExistsError(args.output)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall = time.perf_counter()
    source_files = pins.verify_source(args.source, args.source_path)
    expected = dict(spec.split("=", 1) for spec in args.expect_data)
    inputs = []
    for kind, specs in (
        ("partition", args.partition),
        ("prompts", args.prompts),
        ("rows", args.rows),
    ):
        for spec in specs:
            name, rows, identity = load_inputs(spec, kind)
            if name in expected and expected[name] != identity["path_sha256"]:
                raise ValueError(f"{name}: input digest differs from --expect-data")
            if name in pins.DATA and pins.DATA[name] != identity["path_sha256"]:
                raise ValueError(
                    f"{name}: input digest differs from the pinned partition"
                )
            if kind != "rows" and name not in expected:
                raise ValueError(f"{name}: labelled or panel input needs --expect-data")
            if args.limit:
                rows = rows[: args.limit]
            inputs.append((name, rows, identity))
    names = [name for name, _, _ in inputs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate input names")
    labelled = {
        name: rows for name, rows, identity in inputs if identity["kind"] == "partition"
    }
    if labelled and not args.limit:
        from training.model.data import check_partition_isolation

        check_partition_isolation(labelled)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible accelerator is required")
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True)
    events = (args.output / "events.jsonl").open("x", encoding="utf-8")

    def log(event: dict[str, Any]) -> None:
        events.write(
            json.dumps({"utc": time.strftime("%H:%M:%S", time.gmtime()), **event})
            + "\n"
        )
        events.flush()
        print(json.dumps(event), flush=True)

    load_started = time.perf_counter()
    backbone, tokenizer, parameters = load_backbone(args.source, args.source_path)
    backbone = backbone.to(device)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    log(
        {
            "phase": "loaded",
            "seconds": time.perf_counter() - load_started,
            "text_parameters": parameters,
        }
    )
    tap = LayerTap(backbone, pins.LAYERS)
    manifest: dict[str, Any] = {
        "version": EXTRACT_VERSION,
        "render_version": DISAGGREGATED_RENDER_VERSION,
        "source": args.source,
        "source_identity": {
            k: pins.SOURCES[args.source][k] for k in ("repo", "revision", "license")
        },
        "source_files_sha256": source_files,
        "text_parameters": parameters,
        "layers": {
            "16": "decoder layer 16 residual output",
            "24": "decoder layer 24 residual output",
            "32": "final-norm output",
        },
        "pooling": {
            "joint": "vectors at each native option endpoint and at the final query token",
            "disaggregated": "last token and attention-mask mean over each separately encoded text",
        },
        "precision": "FP32 parameters, BF16 autocast forward, FP32 stored vectors",
        "max_length": args.max_length,
        "truncation": "none; over-length inputs are marked invalid",
        "batching": {
            "joint_prompts_per_forward": args.joint_batch_rows,
            "text_token_budget": args.token_budget,
            "text_max_rows": max(args.max_batch_rows, 64),
            "padding": "right",
        },
        "limit": args.limit,
        "code_commit": args.code_commit,
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers_version,
            "device_name": torch.cuda.get_device_name(0),
            "python": sys.version.split()[0],
        },
        "started_utc": started_utc,
        "inputs": {},
    }
    for name, rows, identity in inputs:
        phase = time.perf_counter()
        records, joint_ids, texts = plan_rows(rows, tokenizer, args.max_length)
        joint = extract_joint(
            backbone,
            tap,
            records,
            joint_ids,
            pad_id,
            device,
            pins.LAYERS,
            args.token_budget,
            args.joint_batch_rows,
            log,
        )
        text_store, text_index = extract_texts(
            backbone,
            tap,
            texts,
            pad_id,
            device,
            pins.LAYERS,
            args.token_budget,
            max(args.max_batch_rows, 64),
            args.max_length,
            log,
        )
        parity = parity_probe(
            backbone,
            tap,
            records,
            joint_ids,
            texts,
            text_index,
            joint,
            text_store,
            pad_id,
            device,
            pins.LAYERS,
            args.parity_sample,
        )
        folder = args.output / name
        folder.mkdir()
        save_file(
            {f"L{layer}": value.contiguous() for layer, value in joint.items()},
            str(folder / "joint.safetensors"),
        )
        save_file(
            {key: value.contiguous() for key, value in text_store.items()},
            str(folder / "texts.safetensors"),
        )
        text_rows = [
            {"index": text_index[sha], "sha256": sha, "tokens": len(texts[sha])}
            for sha in sorted(text_index, key=text_index.get)
        ]
        for record in records:
            record.pop("j_batch_index", None)
            record["d_state_index"] = text_index.get(record["d_state_sha"])
            record["d_candidate_indices"] = [
                text_index.get(sha) for sha in record["d_candidate_shas"]
            ]
        write_jsonl(folder / "rows.jsonl", records)
        write_jsonl(folder / "texts.jsonl", text_rows)
        files = {path.name: pins.file_sha256(path) for path in sorted(folder.iterdir())}
        manifest["inputs"][name] = {
            **identity,
            "rows": len(records),
            "joint_valid": sum(r["j_valid"] for r in records),
            "disaggregated_valid": sum(r["d_valid"] for r in records),
            "joint_vectors": int(next(iter(joint.values())).shape[0]),
            "joint_tokens": sum(r.get("j_tokens", 0) for r in records),
            "unique_texts": len(text_rows),
            "unique_text_tokens": sum(t["tokens"] for t in text_rows),
            "disaggregated_tokens_without_cache": sum(
                r["d_state_tokens"] + sum(r["d_candidate_tokens"]) for r in records
            ),
            "parity": parity,
            "seconds": time.perf_counter() - phase,
            "files_sha256": files,
        }
        log(
            {
                "phase": "input-done",
                "name": name,
                **{
                    k: v
                    for k, v in manifest["inputs"][name].items()
                    if k != "files_sha256"
                },
            }
        )
    tap.close()
    manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    manifest["wall_seconds"] = time.perf_counter() - wall
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    events.close()
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["parity_passed"] = all(
        entry["parity"]["passed"] for entry in manifest["inputs"].values()
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not manifest["parity_passed"]:
        raise SystemExit("Feature parity gate failed; see manifest parity reports")
    print(
        json.dumps(
            {"manifest": str(manifest_path), "sha256": pins.file_sha256(manifest_path)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
