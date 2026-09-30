"""Native FP32 development scoring from a frozen admitted corpus only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .newbase_batches import TokenBatch, forward_complete
from .newbase_data import FrozenCorpus, file_digest
from .newbase_evaluation import ndcg, pair_accuracy, spearman
from .newbase_model import ExitSpec, NewBaseTask, _load_encoder, verify_files
from .representation_contract import (
    read_representation_contract,
    vela_representation_contract,
)


def tokenizer_semantics_digest(tokenizer) -> str:
    """Bind vocabulary/templates/normalization, excluding mutable batch options."""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    for field in ("padding", "truncation"):
        backend.pop(field, None)
    return hashlib.sha256(
        json.dumps(backend, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exact_fixture_batches(row, tokenizer, model, components):
    """Replay an explicitly frozen legacy integer fixture without text round trips."""
    fixture = row["exact_token_inputs"]
    content = {key: value for key, value in fixture.items() if key != "sha256"}
    actual = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ids = _candidate_ids(row)
    if (
        actual != fixture["sha256"]
        or row.get("evaluation_kind") != "controlled_pair"
        or fixture["task"] != model.task
        or fixture["tokenizer_semantics_sha256"]
        != tokenizer_semantics_digest(tokenizer)
        or set(fixture["candidates"]) != set(ids)
    ):
        raise ValueError("Exact-token fixture identity/task/candidate contract differs")
    query = tokenizer(
        components[row["query_component_id"]]["text"],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    if query != fixture["query"]:
        raise ValueError("Exact fixture query differs from its complete source text")
    arrays = [fixture["query"], *[fixture["candidates"][key] for key in ids]]
    if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
        raise ValueError("Exact fixtures require explicit right padding")
    for values in arrays:
        if (
            not isinstance(values, list)
            or not values
            or len(values) > model.encoder.config.max_position_embeddings
            or any(
                type(value) is not int
                or not 0 <= value < model.encoder.config.vocab_size
                for value in values
            )
        ):
            raise ValueError("Exact fixture has invalid token IDs or exceeds capacity")
    return (
        TokenBatch((tuple(arrays[0]),), tokenizer.pad_token_id),
        TokenBatch(
            tuple(tuple(values) for values in arrays[1:]), tokenizer.pad_token_id
        ),
    )


def load_published(directory: Path, task: str, expected_files: dict[str, str]):
    """A baseline reader; this path is never used by the training initializer."""
    required = {"model.safetensors", "config.json", "tokenizer.json"}
    if task == "reranker":
        heads_name = (
            "classification_heads.safetensors"
            if (directory / "classification_heads.safetensors").exists()
            else "classification_heads.pt"
        )
        required.add(heads_name)
        required.add("matryoshka_config.json")
    if not required <= expected_files.keys():
        raise ValueError("Published baseline lock omits required model files")
    verify_files(directory, expected_files)
    with torch.random.fork_rng(devices=[]):
        encoder, _ = _load_encoder(directory)
        if read_representation_contract(
            encoder.config, task
        ) != vela_representation_contract(task):
            raise ValueError(
                "The selected current baseline must declare the exact supported representation"
            )
        model = NewBaseTask(
            encoder, task, ExitSpec(), {"baseline_only": True, "files": expected_files}
        )
    if task == "reranker":
        metadata = json.loads((directory / "matryoshka_config.json").read_bytes())
        if (
            metadata["layer_indices"] != list(model.exits.layers)
            or metadata["dim_indices"] != list(model.exits.dimensions)
            or metadata.get("pooling_strategy") != "cls"
        ):
            raise ValueError("Published baseline exit metadata changed")
        state = (
            load_file(str(directory / heads_name))
            if heads_name.endswith(".safetensors")
            else torch.load(
                directory / heads_name, map_location="cpu", weights_only=True
            )
        )
        expected = model.layer_heads.state_dict()
        if set(state) != set(expected) or any(
            value.dtype != expected[name].dtype or value.shape != expected[name].shape
            for name, value in state.items()
        ):
            raise ValueError("Published task heads do not match their complete schema")
        model.layer_heads.load_state_dict(state, strict=True)
    verify_files(directory, expected_files)
    return model.eval()


def _candidate_ids(row):
    ids = row.get("candidate_component_ids")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Every DEV query needs its complete frozen candidate list")
    if row.get("evaluation_scope") != "first_stage_top100" and not set(
        row["positive_component_ids"]
    ) <= set(ids):
        raise ValueError("DEV candidate pool omitted a known relevant item")
    return ids


@torch.inference_mode()
def score_corpus(
    model,
    tokenizer,
    corpus: FrozenCorpus,
    *,
    device,
    token_budget: int,
    component_batch_size: int = 16,
    evaluation_config: dict | None = None,
) -> tuple[dict, list[dict]]:
    evaluation_config = evaluation_config or {}
    if not corpus.records:
        raise ValueError("Evaluation requires a nonempty explicit corpus")
    if component_batch_size <= 0:
        raise ValueError("Component batch size must be positive")
    model.assert_parameters()
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maximum = model.encoder.config.max_position_embeddings
    component_ids = set()
    for row in corpus.records:
        if "exact_token_inputs" in row:
            continue
        if "pair_component_ids" in row:
            if model.task != "embedding":
                raise ValueError(
                    "Semantic similarity is not a judged reranking DEV source"
                )
            component_ids.update(row["pair_component_ids"])
        else:
            component_ids.add(row["query_component_id"])
            component_ids.update(_candidate_ids(row))
    if not component_ids <= corpus.components.keys():
        raise ValueError("Missing complete DEV text component")
    embeddings = {}
    if model.task == "embedding":
        ordered = sorted(component_ids)
        for start in range(0, len(ordered), component_batch_size):
            identities = ordered[start : start + component_batch_size]
            batch = TokenBatch.encode(
                tokenizer,
                [corpus.components[key]["text"] for key in identities],
                None,
                maximum,
            )
            values = forward_complete(model, batch, device, token_budget, amp=False)
            for index, identity in enumerate(identities):
                embeddings[identity] = {
                    key: vector[index].cpu() for key, vector in values.items()
                }
    numeric = []
    buckets = {key: defaultdict(list) for key, _ in model.exits.weighted()}
    semantic = {key: defaultdict(list) for key, _ in model.exits.weighted()}
    for row in corpus.records:
        if "pair_component_ids" in row:
            left, right = row["pair_component_ids"]
            for key in buckets:
                value = float(embeddings[left][key] @ embeddings[right][key])
                semantic[key][row["source"]].append((value, row["label"]))
                numeric.append(
                    {"id": row["id"], "exit": f"{key[0]}x{key[1]}", "score": value}
                )
            continue
        ids = _candidate_ids(row)
        if "exact_token_inputs" in row:
            query, documents = exact_fixture_batches(
                row, tokenizer, model, corpus.components
            )
            values = forward_complete(model, documents, device, token_budget, amp=False)
            if model.task == "embedding":
                encoded_query = forward_complete(
                    model, query, device, token_budget, amp=False
                )
                values = {
                    key: vectors @ encoded_query[key][0]
                    for key, vectors in values.items()
                }
        elif model.task == "embedding":
            values = {
                key: torch.stack([embeddings[item][key] for item in ids])
                @ embeddings[row["query_component_id"]][key]
                for key in buckets
            }
        else:
            left = [corpus.components[row["query_component_id"]]["text"]] * len(ids)
            right = [corpus.components[key]["text"] for key in ids]
            batch = TokenBatch.encode(tokenizer, left, right, maximum)
            values = forward_complete(model, batch, device, token_budget, amp=False)
        for key, vector in values.items():
            scores = {
                identity: float(value)
                for identity, value in zip(ids, vector.cpu(), strict=True)
            }
            positives = row["positive_component_ids"]
            if row.get("evaluation_kind") == "controlled_pair":
                negative = row["judged_negative_component_ids"]
                if (
                    len(positives) != 1
                    or len(negative) != 1
                    or set(ids) != set(positives + negative)
                ):
                    raise ValueError(
                        "A controlled DEV pair requires one positive and one negative"
                    )
                metric = pair_accuracy(scores[positives[0]], scores[negative[0]])

            else:
                metric = ndcg(
                    scores,
                    dict.fromkeys(positives, 1.0),
                    allow_unretrieved=row.get("evaluation_scope")
                    == "first_stage_top100",
                )
            buckets[key][row["source"]].append(metric)
            language_source = evaluation_config.get("language_source")
            if row["source"] == language_source:
                buckets[key][language_source + "_language:" + row["language"]].append(
                    metric
                )
            if (
                row.get("evaluation_kind") == evaluation_config.get("length_kind")
                and "length_bucket" in row
            ):
                buckets[key]["length:" + str(row["length_bucket"])].append(metric)
            for name, conditions in evaluation_config.get("cohorts", {}).items():
                if name == row["source"]:
                    raise ValueError(
                        "An aggregate cohort cannot duplicate a source name"
                    )
                if conditions and all(
                    row.get(field) == value for field, value in conditions.items()
                ):
                    buckets[key][name].append(metric)
            numeric.append(
                {
                    "id": row["id"],
                    "exit": f"{key[0]}x{key[1]}",
                    "scores": scores,
                    "metric": metric,
                }
            )
    exits, support = {}, {}
    for key, metrics in buckets.items():
        name = f"{key[0]}x{key[1]}"
        entry = {
            field: sum(values) / len(values)
            for field, values in metrics.items()
            if ":" not in field
        }
        language_source = evaluation_config.get("language_source")
        if language_source:
            entry[language_source + "_languages"] = {
                field.split(":", 1)[1]: sum(values) / len(values)
                for field, values in metrics.items()
                if field.startswith(language_source + "_language:")
            }
        entry["lengths"] = {
            field.split(":", 1)[1]: sum(values) / len(values)
            for field, values in metrics.items()
            if field.startswith("length:")
        }
        for source, pairs in semantic[key].items():
            scores, labels = map(list, zip(*pairs, strict=True))
            spec = evaluation_config.get("semantic_metrics", {}).get(source)
            if spec is None or spec["kind"] not in {"spearman", "binary_accuracy"}:
                raise ValueError("Semantic pair evaluation needs an explicit metric")
            if spec["kind"] == "spearman":
                entry[source] = spearman(scores, labels)
            else:
                threshold = spec["threshold"]
                if (
                    not isinstance(threshold, (int, float))
                    or not -1 <= threshold <= 1
                    or any(label not in (0, 1) for label in labels)
                ):
                    raise ValueError(
                        "Binary similarity needs a finite cosine threshold and binary labels"
                    )
                entry[source] = sum(
                    (score >= threshold) == bool(label) for score, label in pairs
                ) / len(pairs)
        exits[name], support[name] = entry, {
            field: len(values) for field, values in metrics.items()
        }
    report = {
        "precision": "float32",
        "autocast": False,
        "allow_tf32": False,
        "development_manifest_sha256": corpus.manifest_sha256,
        "task": model.task,
        "exits": exits,
        "support": support,
        "evaluated_records": len(corpus.records),
    }
    return report, numeric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--known-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-lock", type=Path)
    parser.add_argument("--task", choices=("embedding", "reranker"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    corpus = FrozenCorpus.load(args.known_dev, args.split)
    if args.published_lock:
        lock = json.loads(args.published_lock.read_bytes())
        model = load_published(args.model, args.task, lock["files"])
        identity = {"published_lock_sha256": file_digest(args.published_lock)}
    else:
        model = NewBaseTask.resume(args.model)
        identity = {
            "checkpoint_sha256": file_digest(args.model / "newbase_checkpoint.json")
        }
    if model.task != args.task:
        raise ValueError("Model task differs from the requested evaluator")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    device = torch.device(args.device)
    model.to(device)
    report, numeric = score_corpus(
        model,
        tokenizer,
        corpus,
        device=device,
        token_budget=args.token_budget,
        evaluation_config=(
            json.loads(args.evaluation_config.read_bytes())
            if args.evaluation_config
            else None
        ),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(
        json.dumps({**identity, **report}, indent=2) + "\n"
    )
    with (args.output / "numeric.jsonl").open("w", encoding="utf-8") as stream:
        for row in numeric:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
