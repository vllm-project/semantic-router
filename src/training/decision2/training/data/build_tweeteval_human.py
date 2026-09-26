"""Build a private human-labelled Decision 2.0 TRAIN candidate from TweetEval.

Only ``train_text.txt``, ``train_labels.txt`` and ``mapping.txt`` are opened
from the pinned TweetEval checkout. CSS prompt files are gold-free exclusions;
no CSS gold or TweetEval validation/test labels are read.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from transfer import build as transfer

from training.data import build_pilot as pilot
from training.model.data import check_partition_isolation, load_partition

TWEETEVAL_REVISION = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
BUCKETS = {
    "stance/abortion": 300,
    "stance/atheism": 300,
    "stance/climate": 300,
    "stance/feminist": 300,
    "stance/hillary": 300,
    "hate": 500,
    "irony": 400,
    "offensive": 400,
    "emotion": 400,
    "sentiment": 400,
}
TARGETS = {
    "abortion": "legal access to abortion",
    "atheism": "atheism",
    "climate": "the claim that climate change is a real concern",
    "feminist": "the feminist movement",
    "hillary": "Hillary Clinton",
}
OPTIONS = {
    "stance": ("none", "against", "favor"),
    "hate": ("not-hate", "hate"),
    "irony": ("non_irony", "irony"),
    "offensive": ("not-offensive", "offensive"),
    "emotion": ("anger", "joy", "optimism", "sadness"),
    "sentiment": ("negative", "neutral", "positive"),
}
DESCRIPTIONS = {
    "none": "No clear stance",
    "against": "Against the target",
    "favor": "In favor of the target",
    "not-hate": "Not hateful",
    "hate": "Hateful",
    "non_irony": "Not ironic",
    "irony": "Ironic",
    "not-offensive": "Not offensive",
    "offensive": "Offensive",
    "anger": "Anger",
    "joy": "Joy",
    "optimism": "Optimism",
    "sadness": "Sadness",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}
INSTRUCTIONS = {
    "hate": "Does this post express hate toward a group?",
    "irony": "Is this post ironic?",
    "offensive": "Is this post offensive?",
    "emotion": "Which emotion is most clearly expressed in this post?",
    "sentiment": "What is the overall sentiment of this post?",
}
RIGHTS = {
    "status": "private_training_candidate_original_task_rights_and_twitter_terms_require_review",
    "license": "TweetEval says it is released without restrictions, but defers to each underlying task and Twitter restrictions",
    "evidence": "https://github.com/cardiffnlp/tweeteval#license",
    "attribution": "Barbieri et al., TweetEval, Findings of EMNLP 2020; original SemEval task authors listed in the TweetEval README",
    "release_policy": "Do not redistribute raw training text in the model repository.",
}


def bucket_task(bucket: str) -> str:
    return "stance" if bucket.startswith("stance/") else bucket


def context_hash(state: Any) -> str:
    return transfer.normalized_context_sha256(
        state if isinstance(state, str) else pilot.canonical(state)
    )


def context_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "state": row["state"],
        "instructions": "",
        "options": [],
        "task_type": "context",
    }


def source_rows(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if transfer.git_head(root) != TWEETEVAL_REVISION:
        raise ValueError("TweetEval source revision differs from frozen revision")
    transfer.require_clean_tracked_files(root)
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, Any] = {}
    for bucket in BUCKETS:
        directory = root / "datasets" / bucket
        paths = {
            name: directory / name for name in ("train_text.txt", "train_labels.txt")
        }
        paths["mapping.txt"] = (
            root / "datasets" / "stance" / "mapping.txt"
            if bucket.startswith("stance/")
            else directory / "mapping.txt"
        )
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"{bucket}: train-only source files missing")
        mapping = {}
        for line in paths["mapping.txt"].read_text(encoding="utf-8").splitlines():
            index, label = line.split("\t", 1)
            mapping[index] = label
        task = bucket_task(bucket)
        if tuple(mapping[str(i)] for i in range(len(mapping))) != OPTIONS[task]:
            raise ValueError(f"{bucket}: label map changed")
        texts = paths["train_text.txt"].read_text(encoding="utf-8").splitlines()
        labels = paths["train_labels.txt"].read_text(encoding="utf-8").splitlines()
        if not texts or len(texts) != len(labels):
            raise ValueError(f"{bucket}: text/label cardinality mismatch")
        if any(label not in mapping for label in labels):
            raise ValueError(f"{bucket}: out-of-map label")
        rows = []
        empty_count = 0
        for index, (state, label_id) in enumerate(zip(texts, labels, strict=True)):
            if not state.strip():
                empty_count += 1
                continue
            rows.append(
                {
                    "source_id": f"{bucket}/{index}",
                    "state": state,
                    "gold_key": mapping[label_id],
                    "bucket": bucket,
                    "context_sha256": context_hash(state),
                }
            )
        by_bucket[bucket] = rows
        receipts[bucket] = {
            "source_rows": len(texts),
            "usable_nonempty_rows": len(rows),
            "empty_text_rows": empty_count,
            "class_counts": dict(
                sorted(collections.Counter(row["gold_key"] for row in rows).items())
            ),
            "files": {name: pilot.sha_file(path) for name, path in paths.items()},
        }
    return by_bucket, receipts


def read_gold_free_prompts(
    panel_dir: Path, dev_prompts: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = panel_dir / "css-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["data_revision"] != transfer.DATA_REVISION
        or manifest["replication_revision"] != transfer.REPLICATION_REVISION
    ):
        raise ValueError("CSS prompt source revision changed")
    rows, receipts = [], {"manifest_sha256": pilot.sha_file(manifest_path)}
    for role in ("pilot_prompts", "evaluation_prompts"):
        info = manifest["outputs"][role]
        path = panel_dir / info["file"]
        if pilot.sha_file(path) != info["sha256"]:
            raise ValueError(f"{role}: prompt SHA changed")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                rows.append({"id": item["id"], "state": item["state"]})
        receipts[role] = {"sha256": info["sha256"], "rows": info["n"]}
    if len(rows) != sum(
        receipts[role]["rows"] for role in ("pilot_prompts", "evaluation_prompts")
    ):
        raise ValueError("CSS prompt row count changed")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("CSS prompt IDs repeat")
    with dev_prompts.open(encoding="utf-8") as stream:
        dev = [json.loads(line) for line in stream]
    rows += [{"id": row["id"], "state": row["state"]} for row in dev]
    receipts["synthetic_dev"] = {
        "sha256": pilot.sha_file(dev_prompts),
        "rows": len(dev),
    }
    return rows, receipts


def reference_hashes(
    paths: list[Path],
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    fields = {
        key: set()
        for key in (
            "id",
            "group_id",
            "input_sha256",
            "raw_context",
            "normalized_context",
        )
    }
    receipts = []
    for path in paths:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for key in ("id", "group_id", "input_sha256"):
                    if row.get(key):
                        fields[key].add(str(row[key]))
                state = (
                    row["state"]
                    if isinstance(row["state"], str)
                    else pilot.canonical(row["state"])
                )
                fields["raw_context"].add(pilot.sha_bytes(state.encode("utf-8")))
                fields["normalized_context"].add(context_hash(state))
                count += 1
        receipts.append(
            {"file": path.name, "sha256": pilot.sha_file(path), "rows": count}
        )
    return fields, receipts


def make_train_row(item: dict[str, Any], seed: str) -> dict[str, Any]:
    task, bucket = bucket_task(item["bucket"]), item["bucket"]
    order = list(OPTIONS[task])
    pilot.rng_for(seed, bucket, int(item["source_id"].rsplit("/", 1)[1])).shuffle(order)
    options = [{"key": key, "description": DESCRIPTIONS[key]} for key in order]
    target = TARGETS[bucket.split("/", 1)[1]] if task == "stance" else None
    instructions = (
        f"What stance does this post take toward {target}?"
        if target
        else INSTRUCTIONS[task]
    )
    row = {
        "id": f"human_tweeteval/{item['source_id']}",
        "state": item["state"],
        "instructions": instructions,
        "options": options,
        "label": order.index(item["gold_key"]),
        "task_type": "choice",
        "family": f"human_tweeteval_{task}",
        "group_id": f"human_tweeteval_context_{item['context_sha256']}",
        "language": "en",
        "split": "train",
        "source": f"tweeteval_train:{bucket}",
        "evaluation_role": "train",
        "render_template": "tweeteval_human_choice_v1",
        "audit_metadata": {
            "upstream_partition": "train",
            "source_row": item["source_id"],
            "original_label": item["gold_key"],
            "normalized_context_sha256": item["context_sha256"],
            "sampling": "class-balanced deterministic hash without replacement",
        },
    }
    row["input_sha256"] = pilot.input_sha256(row)
    pilot.validate_train_row(row)
    return row


def balanced_sample(
    rows: list[dict[str, Any]], quota: int, seed: str, bucket: str
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_class[row["gold_key"]].append(row)
    for label, group in by_class.items():
        group.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}\0{bucket}\0{label}\0{item['source_id']}".encode()
            ).hexdigest()
        )
    selected = []
    labels = list(OPTIONS[bucket_task(bucket)])
    while len(selected) < quota:
        progressed = False
        for label in labels:
            if by_class[label] and len(selected) < quota:
                selected.append(by_class[label].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(
                f"{bucket}: too few nonoverlapping train examples for {quota}"
            )
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source, source_receipts = source_rows(args.tweet_root)
    prompts, prompt_receipts = read_gold_free_prompts(args.panel_dir, args.dev_prompts)
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    prompt_ids = {row["id"] for row in prompts}
    prompt_norm = {context_hash(row["state"]) for row in prompts}
    prompt_raw = {
        pilot.sha_bytes(
            (
                row["state"]
                if isinstance(row["state"], str)
                else pilot.canonical(row["state"])
            ).encode()
        )
        for row in prompts
    }
    reference_paths = sorted(args.train_root.rglob("*.train.jsonl")) + [
        args.legacy_v1_train
    ]
    reference, reference_receipts = reference_hashes(reference_paths)
    holdouts = [*select, *cal]
    holdout_ids = {row["id"] for row in holdouts}
    holdout_norm = {context_hash(row["state"]) for row in holdouts}
    holdout_raw = {
        pilot.sha_bytes(
            (
                row["state"]
                if isinstance(row["state"], str)
                else pilot.canonical(row["state"])
            ).encode()
        )
        for row in holdouts
    }
    excluded = collections.Counter()
    available: dict[str, list[dict[str, Any]]] = {}
    seen_context = set()
    for bucket, items in source.items():
        kept = []
        for item in items:
            source_id, norm = item["source_id"], item["context_sha256"]
            raw = pilot.sha_bytes(item["state"].encode())
            if (
                source_id in prompt_ids
                or f"human_tweeteval/{source_id}" in reference["id"]
                or source_id in holdout_ids
            ):
                excluded["id"] += 1
            elif raw in prompt_raw or norm in prompt_norm:
                excluded["css_or_dev_context"] += 1
            elif raw in holdout_raw or norm in holdout_norm:
                excluded["select_or_cal_context"] += 1
            elif (
                raw in reference["raw_context"]
                or norm in reference["normalized_context"]
            ):
                excluded["existing_train_context"] += 1
            elif norm in seen_context:
                excluded["repeated_source_context"] += 1
            else:
                kept.append(item)
                seen_context.add(norm)
        available[bucket] = kept
    # Near-match quarantine is based on gold-free benchmark prompts and audit-only
    # SELECT/CAL contexts. Resampling preserves each source quota and class policy.
    quarantined = set()
    near_receipts = []
    for _ in range(5):
        picked = [
            item
            for bucket in BUCKETS
            for item in balanced_sample(
                [
                    item
                    for item in available[bucket]
                    if item["source_id"] not in quarantined
                ],
                BUCKETS[bucket],
                args.seed,
                bucket,
            )
        ]
        rows = [make_train_row(item, args.seed) for item in picked]
        near_panel = pilot.near_duplicates(
            [context_row(row) for row in rows],
            [context_row(row) for row in prompts],
            collect_left_ids=True,
        )
        near_holdout = pilot.near_duplicates(
            [context_row(row) for row in rows],
            [context_row(row) for row in holdouts],
            collect_left_ids=True,
        )
        near_receipts.append(
            {
                "css_and_dev": {
                    key: value for key, value in near_panel.items() if key != "left_ids"
                },
                "select_cal": {
                    key: value
                    for key, value in near_holdout.items()
                    if key != "left_ids"
                },
            }
        )
        bad = set(near_panel["left_ids"]) | set(near_holdout["left_ids"])
        if not bad:
            break
        quarantined.update(
            row["audit_metadata"]["source_row"] for row in rows if row["id"] in bad
        )
    else:
        raise ValueError("Near-context exclusions did not converge")
    if len(rows) != sum(BUCKETS.values()):
        raise AssertionError("Human training quota changed")
    if len({row["group_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated normalized training context")
    for row in rows:
        state = row["state"]
        raw, norm = pilot.sha_bytes(state.encode()), context_hash(state)
        if (
            row["id"] in prompt_ids | holdout_ids | reference["id"]
            or row["group_id"] in reference["group_id"]
            or row["input_sha256"] in reference["input_sha256"]
            or raw in prompt_raw | holdout_raw | reference["raw_context"]
            or norm in prompt_norm | holdout_norm | reference["normalized_context"]
        ):
            raise ValueError(f"{row['id']}: exact protected/reference overlap")
    check_partition_isolation({"train": rows, "select": select, "cal": cal})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    lengths = [pilot.count_tokens(row, tokenizer) for row in rows]
    if max(lengths) > args.max_row_tokens:
        raise ValueError("Human TRAIN row exceeds token cap")
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}\0output\0{row['id']}".encode()
        ).hexdigest()
    )
    output = pilot.jsonl_bytes(rows)
    args.output_dir.mkdir(parents=True, mode=0o700)
    path = args.output_dir / "tweeteval_human_3600.train.jsonl"
    pilot._atomic_write(path, output)
    manifest = {
        "schema_version": "decision2-tweeteval-human-train/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "source_revision": TWEETEVAL_REVISION,
        "source_url": "https://github.com/cardiffnlp/tweeteval",
        "source_files": source_receipts,
        "rights": RIGHTS,
        "prompt_exclusion_receipts": prompt_receipts,
        "reference_train_receipts": reference_receipts,
        "select_sha256": pilot.sha_file(args.select_file),
        "cal_sha256": pilot.sha_file(args.cal_file),
        "exclusions": {
            **dict(sorted(excluded.items())),
            "near_context_source_rows": len(quarantined),
        },
        "near_context_audit": near_receipts,
        "sample": {
            "rows": len(rows),
            "bucket_quotas": BUCKETS,
            "bucket_class_counts": {
                bucket: dict(
                    sorted(
                        collections.Counter(
                            row["options"][row["label"]]["key"]
                            for row in rows
                            if row["source"] == f"tweeteval_train:{bucket}"
                        ).items()
                    )
                )
                for bucket in BUCKETS
            },
        },
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths),
            "maximum": max(lengths),
            "cap": args.max_row_tokens,
        },
        "output": {
            "file": path.name,
            "sha256": pilot.sha_bytes(output),
            "rows": len(rows),
        },
        "evaluation_interpretation": {
            "css_stance_pilot": "same benchmark family but cross-target: TRAIN covers five non-Trump targets; pilot is Trump",
            "css_implicit_hate_pilot": "related-task supervision from binary hate and irony, not six-way taxonomy labels",
            "css_discourse_pilot": "unseen-task; TweetEval contains no discourse-act labels",
            "css_emotion_evaluation": "related-task emotion supervision; source and label inventory differ; not strictly zero-shot by skill",
            "other_css_evaluation": "unseen-task except possible broad semantic overlap; CSS final contexts/IDs excluded",
        },
        "limitations": [
            "Only TweetEval TRAIN files were read; source texts remain private.",
            "TweetEval delegates underlying dataset and Twitter restrictions; check original task terms before public redistribution of rows.",
            "Class balancing changes source priors and may worsen probability calibration.",
            "Near-duplicate matching is approximate and cannot prove semantic independence.",
            "No CSS evaluation labels or TweetEval validation/test labels entered this build.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "tweeteval_human.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweet-root", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--legacy-v1-train", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-tweeteval-human-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args)
    print(
        json.dumps(
            {
                "rows": result["output"]["rows"],
                "sha256": result["output"]["sha256"],
                "total_tokens": result["token_audit"]["total"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
