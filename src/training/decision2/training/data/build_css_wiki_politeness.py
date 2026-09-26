"""Build a private, train-only CSS politeness source pool from ConvoKit.

Only ConvoKit source text and labels enter the candidate pool. CSS pilot and
evaluation gold rows are projected to exclusion IDs and hashes immediately;
the test labels and test text are never used to make a training example.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from transfer import build as transfer

from training.data import build_pilot as pilot
from training.data.build_css_flute import read_panel_exclusions
from training.model.data import check_partition_isolation, load_partition

SOURCE_URL = (
    "https://zissou.infosci.cornell.edu/convokit/datasets/"
    "wikipedia-politeness-corpus/wikipedia-politeness-corpus.zip"
)
SOURCE_SHA256 = "90ec6c2c6e05d064a805d2e4be4a8d442f370b31f0025798e8eaf62d0014ba48"
MEMBER = "wikipedia-politeness-corpus/utterances.jsonl"
SOURCE = "convokit_wikipedia_politeness"
FAMILY = "css_wiki_politeness_three_class"
INSTRUCTIONS = "Classify the politeness of this Wikipedia Talk Page request."
OPTIONS = [
    {"key": "1", "description": "Polite"},
    {"key": "0", "description": "Neutral"},
    {"key": "-1", "description": "Impolite"},
]
RIGHTS = {
    "license": "CC BY 4.0",
    "attribution": "Danescu-Niculescu-Mizil et al., A Computational Approach to Politeness with Application to Social Factors, ACL 2013; ConvoKit Developers",
    "evidence": "https://convokit.cornell.edu/documentation/wiki_politeness.html",
    "status": "official_corpus_license_documented_private_training_only",
}


def source_context(item: dict[str, Any]) -> str:
    """Match SALT's single-utterance `speaker: text` context construction."""
    if not isinstance(item.get("user"), str) or not isinstance(item.get("text"), str):
        raise ValueError("ConvoKit politeness utterance has unexpected user/text type")
    return f"{item['user']}: {item['text']}"


def read_source(source_zip: Path) -> list[dict[str, Any]]:
    if pilot.sha_file(source_zip) != SOURCE_SHA256:
        raise ValueError("ConvoKit politeness source archive changed")
    with zipfile.ZipFile(source_zip) as archive, archive.open(MEMBER) as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 4353 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("ConvoKit politeness source cardinality/IDs changed")
    return rows


def short_row(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source["id"],
        "state": source_context(source),
        "instructions": INSTRUCTIONS,
        "options": OPTIONS,
        "task_type": "choice",
    }


def make_train_row(source: dict[str, Any]) -> dict[str, Any]:
    context = source_context(source)
    answer = str(source["meta"]["Binary"])
    if answer not in {"1", "0", "-1"}:
        raise ValueError("ConvoKit politeness label changed")
    raw = pilot.sha_bytes(context.encode("utf-8"))
    normalized = transfer.normalized_context_sha256(context)
    row = {
        "id": f"css_train/wiki_politeness/{source['id']}",
        "state": context,
        "instructions": INSTRUCTIONS,
        "options": OPTIONS,
        "label": next(i for i, option in enumerate(OPTIONS) if option["key"] == answer),
        "task_type": "choice",
        "family": FAMILY,
        "group_id": f"css_wikipolite_context_{normalized}",
        "language": "en",
        "split": "train",
        "source": SOURCE,
        "evaluation_role": "train",
        "render_template": "convokit_wikipedia_politeness_three_class_v1",
        "audit_metadata": {
            "task": "wiki_politeness",
            "source_id": source["id"],
            "source_context_sha256": raw,
            "normalized_context_sha256": normalized,
            "upstream_partition": "licensed full corpus minus quarantined CSS tests",
        },
    }
    row["input_sha256"] = pilot.input_sha256(row)
    pilot.validate_train_row(row)
    return row


def candidate_pool(
    source_rows: list[dict[str, Any]],
    excluded: dict[str, set[Any]],
    holdouts: list[dict[str, Any]],
    expected_panel_raw: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holdout_raw = {
        pilot.sha_bytes(
            (
                state
                if isinstance(state := row["state"], str)
                else pilot.canonical(state)
            ).encode()
        )
        for row in holdouts
    }
    holdout_norm = {
        transfer.normalized_context_sha256(
            state if isinstance(state := row["state"], str) else pilot.canonical(state)
        )
        for row in holdouts
    }
    quarantine: list[dict[str, str]] = []
    test_source: list[dict[str, Any]] = []
    retained: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    counts: collections.Counter[str] = collections.Counter()
    for item in source_rows:
        context = source_context(item)
        raw = pilot.sha_bytes(context.encode("utf-8"))
        norm = transfer.normalized_context_sha256(context)
        reason = None
        if raw in excluded["raw_context_sha256"]:
            reason = "css_panel_raw_context"
        elif norm in excluded["normalized_context_sha256"]:
            reason = "css_panel_normalized_context"
        elif raw in holdout_raw or norm in holdout_norm:
            reason = "independent_select_cal_context"
        if reason:
            counts[reason] += 1
            quarantine.append(
                {"source_id": item["id"], "context_sha256": norm, "reason": reason}
            )
            if reason.startswith("css_panel"):
                test_source.append(short_row(item))
            continue
        # The source label is read only after test and holdout exclusion.
        retained[norm].append(item)
    mapped = {pilot.sha_bytes(row["state"].encode("utf-8")) for row in test_source}
    if not expected_panel_raw or not expected_panel_raw <= mapped:
        raise ValueError(
            "Could not reproduce every CSS politeness test context from source"
        )
    candidates = []
    for norm, duplicates in sorted(retained.items()):
        labels = {str(item["meta"]["Binary"]) for item in duplicates}
        if len(labels) != 1:
            counts["conflicting_same_context"] += len(duplicates)
            quarantine.extend(
                {
                    "source_id": item["id"],
                    "context_sha256": norm,
                    "reason": "conflicting_same_context",
                }
                for item in duplicates
            )
            continue
        chosen = min(duplicates, key=lambda item: item["id"])
        for item in duplicates:
            if item is not chosen:
                counts["duplicate_same_context"] += 1
                quarantine.append(
                    {
                        "source_id": item["id"],
                        "context_sha256": norm,
                        "reason": "duplicate_same_context",
                    }
                )
        candidates.append(chosen)
    near = pilot.near_duplicates(
        [short_row(item) for item in candidates], test_source, collect_left_ids=True
    )
    near_ids = set(near.pop("left_ids"))
    cleaned = []
    for item in candidates:
        if item["id"] in near_ids:
            counts["near_css_test_context"] += 1
            quarantine.append(
                {
                    "source_id": item["id"],
                    "context_sha256": transfer.normalized_context_sha256(
                        source_context(item)
                    ),
                    "reason": "near_css_test_context",
                }
            )
        else:
            cleaned.append(make_train_row(item))
    if not cleaned:
        raise ValueError("No politeness source candidates remain")
    if len({row["group_id"] for row in cleaned}) != len(cleaned):
        raise AssertionError("Candidate pool still has repeated normalized contexts")
    stats = {
        "source_rows": len(source_rows),
        "test_source_rows": len(test_source),
        "mapped_unique_panel_contexts": len(expected_panel_raw),
        "candidate_rows": len(cleaned),
        "excluded_counts": dict(sorted(counts.items())),
        "class_counts": dict(
            sorted(
                collections.Counter(
                    row["options"][row["label"]]["key"] for row in cleaned
                ).items()
            )
        ),
        "near_duplicate_audit": near,
        "quarantine": sorted(
            quarantine, key=lambda row: (row["source_id"], row["reason"])
        ),
    }
    if sum(stats["excluded_counts"].values()) + len(cleaned) != len(source_rows):
        raise AssertionError("Source accounting does not reconcile")
    return cleaned, stats


def sample_class_stratified(
    rows: list[dict[str, Any]], count: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not 0 < count <= len(rows):
        raise ValueError("Invalid politeness sample size")
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_class[row["options"][row["label"]]["key"]].append(row)
    quotas = {key: count * len(value) // len(rows) for key, value in by_class.items()}
    remainder = count - sum(quotas.values())
    order = sorted(
        by_class, key=lambda key: (-(count * len(by_class[key]) % len(rows)), key)
    )
    for key in order[:remainder]:
        quotas[key] += 1
    selected = []
    for key, group in sorted(by_class.items()):
        group.sort(
            key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest()
        )
        selected.extend(group[: quotas[key]])
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}\0output\0{row['id']}".encode()
        ).hexdigest()
    )
    return selected, dict(sorted(quotas.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    excluded, panel_receipt = read_panel_exclusions(args.panel_dir.resolve())
    panel_gold_file = (
        args.panel_dir.resolve() / panel_receipt["panels"]["evaluation"]["file"]
    )
    expected_panel_raw = {
        item["source_context_sha256"]
        for line in panel_gold_file.open(encoding="utf-8")
        if (item := json.loads(line))["task"] == "wiki_politeness"
    }
    select = load_partition(args.select_file.resolve(), "select")
    cal = load_partition(args.cal_file.resolve(), "cal")
    source_rows = read_source(args.source_zip.resolve())
    pool, candidate_audit = candidate_pool(
        source_rows, excluded, [*select, *cal], expected_panel_raw
    )
    sampled, quotas = sample_class_stratified(pool, args.sample_count, args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in pool}
    if max(lengths.values()) > args.max_row_tokens:
        raise ValueError("Politeness pool has row over token cap")
    cross = {}
    for name, rows in (("pool", pool), ("sample", sampled)):
        check_partition_isolation({"train": rows, "select": select, "cal": cal})
        audits = {
            "select": pilot.overlap_audit(rows, select),
            "cal": pilot.overlap_audit(rows, cal),
        }
        if any(
            pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]
            for audit in audits.values()
        ):
            raise ValueError(f"Politeness {name} overlaps SELECT/CAL")
        cross[name] = audits
    output = {
        "wiki_politeness_pool.train.jsonl": pilot.jsonl_bytes(pool),
        "wiki_politeness_1k.train.jsonl": pilot.jsonl_bytes(sampled),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    output_dir.mkdir(parents=True, mode=0o700)
    for name, content in output.items():
        pilot._atomic_write(output_dir / name, content)
    manifest = {
        "schema_version": "decision2-css-wiki-politeness-train/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "source": {
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "member": MEMBER,
            "split": "full licensed corpus minus all CSS panel contexts and SELECT/CAL; derived train complement",
        },
        "rights": RIGHTS,
        "panel_exclusions": panel_receipt,
        "candidate_audit": candidate_audit,
        "sample": {"rows": len(sampled), "seed": args.seed, "class_quotas": quotas},
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "max_row_tokens": args.max_row_tokens,
            "pool_total": sum(lengths.values()),
            "pool_maximum": max(lengths.values()),
            "sample_total": sum(lengths[row["id"]] for row in sampled),
            "sample_maximum": max(lengths[row["id"]] for row in sampled),
        },
        "cross_partition_audit": cross,
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(content),
                "bytes": len(content),
                "rows": len(content.splitlines()),
            }
            for name, content in output.items()
        },
        "evaluation_interpretation": "CSS wiki_politeness is same-task supervised if this pool enters model training.",
        "limitations": [
            "The corpus does not supply an official train split; this is a test-quarantined complement.",
            "SALT speaker IDs in this archive are anonymized, so speaker-disjointness cannot be claimed.",
            "Near-duplicate detection is approximate and cannot prove semantic independence.",
        ],
    }
    pilot._atomic_write(
        output_dir / "wiki_politeness.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", default="decision2-css-wiki-politeness-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "pool_rows": manifest["candidate_audit"]["candidate_rows"],
                "sample_sha256": manifest["outputs"]["wiki_politeness_1k.train.jsonl"][
                    "sha256"
                ],
                "sample_tokens": manifest["token_audit"]["sample_total"],
            }
        )
    )


if __name__ == "__main__":
    main()
