"""Frozen, source-disjoint ContractNLI long-input development screen.

Only ``length-*`` reads the public documents without labels. ``freeze`` opens
development labels after the separate overlap receipt is present, writes raw
text and gold only under a caller-owned private directory, and fails closed on
the preregistered population gates. No training or sealed release set is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from inference.run import file_digest

ARCHIVE_SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
DEV_SHA256 = "310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e"
RIGHTS_CLEAN_SHA256 = {
    "rights_clean.train.jsonl": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select.jsonl": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal.jsonl": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "rights_clean.manifest.json": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
}
SEED = "decision2-small06-contractnli-long-v1"
CLASSES = ("entailed", "contradicted", "not_mentioned")
ORIGINAL_LABELS = {
    "Entailment": "entailed",
    "Contradiction": "contradicted",
    "NotMentioned": "not_mentioned",
}
CRITERIA = {
    "entailed": "The agreement entails the hypothesis.",
    "contradicted": "The agreement contradicts the hypothesis.",
    "not_mentioned": "The agreement neither entails nor contradicts the hypothesis.",
}


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_archive(path: Path) -> dict[str, Any]:
    if file_digest(path) != ARCHIVE_SHA256:
        raise ValueError("Official ContractNLI archive SHA-256 differs")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("contract-nli/dev.json")
        if hashlib.sha256(raw).hexdigest() != DEV_SHA256:
            raise ValueError("ContractNLI development split differs")
        corpus = json.loads(raw)
    if not isinstance(corpus, dict) or len(corpus.get("documents", [])) != 61:
        raise ValueError("Unexpected ContractNLI development inventory")
    if len(corpus.get("labels", {})) != 17:
        raise ValueError("Unexpected ContractNLI hypothesis inventory")
    return corpus


def item_id(document: dict[str, Any], hypothesis_id: str) -> str:
    return f"contractnli-dev:{document['id']}:{hypothesis_id}"


def question(hypothesis: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": (
            "Under this entire agreement, is the following hypothesis entailed, "
            f"contradicted, or not mentioned? {hypothesis}"
        ),
        "criteria": CRITERIA,
    }


def source_items(corpus: dict[str, Any]):
    """Yield text and questions, deliberately omitting annotation sets."""
    for document in corpus["documents"]:
        for hypothesis_id, label in corpus["labels"].items():
            yield item_id(document, hypothesis_id), document["text"], question(
                label["hypothesis"]
            )


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for part in value.values():
            yield from _strings(part)
    elif isinstance(value, list):
        for part in value:
            yield from _strings(part)


def audit_overlap(
    archive: Path,
    roots: list[Path],
    rights_clean: Path,
    output: Path,
) -> dict[str, Any]:
    """Check source IDs, URLs, normalized documents and 200-char fragments.

    The scanner intentionally accesses neither ContractNLI annotations nor
    candidate training labels. Paths and any matched raw records stay private.
    """
    if output.exists():
        raise FileExistsError(output)
    verified = {}
    for name, expected in RIGHTS_CLEAN_SHA256.items():
        path = rights_clean / name
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"Frozen rights-clean source differs: {name}")
        verified[name] = actual
    rights_manifest = (rights_clean / "rights_clean.manifest.json").read_text()
    if "contractnli" in rights_manifest.lower().replace("-", "").replace("_", ""):
        raise ValueError("Rights-clean ledger mentions ContractNLI")
    corpus = read_archive(archive)
    documents = {str(doc["id"]): normalized(doc["text"]) for doc in corpus["documents"]}
    urls = {
        normalized(str(doc["url"])) for doc in corpus["documents"] if doc.get("url")
    }
    shingles = {
        document[index : index + 200]
        for document in documents.values()
        for index in range(0, max(1, len(document) - 200), 100)
    }
    paths = sorted({path for root in roots for path in root.rglob("*.jsonl")})
    hits: list[tuple[str, int, str]] = []
    errors: list[tuple[str, int]] = []
    rows = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    errors.append((str(path), line_number))
                    continue
                if not isinstance(row, dict):
                    continue
                rows += 1
                for field in (
                    "source",
                    "source_id",
                    "dataset",
                    "dataset_id",
                    "group_id",
                    "url",
                    "document_url",
                    "original_id",
                ):
                    value = row.get(field)
                    if not isinstance(value, str):
                        continue
                    source = value.casefold().replace("-", "").replace("_", "")
                    if "contractnli" in source or normalized(value) in urls:
                        hits.append((str(path), line_number, "source"))
                for field in (
                    "state",
                    "text",
                    "document",
                    "context",
                    "passage",
                    "input",
                    "prompt",
                    "instructions",
                ):
                    if field not in row:
                        continue
                    for raw in _strings(row[field]):
                        if len(raw) < 500:
                            continue
                        text = normalized(raw)
                        if len(text) < 250:
                            continue
                        if any(
                            len(text) > 0.75 * len(doc) and (text in doc or doc in text)
                            for doc in documents.values()
                        ):
                            hits.append((str(path), line_number, "fulltext"))
                        if sum(piece in text for piece in shingles) >= 3:
                            hits.append((str(path), line_number, "shingle"))
    report = {
        "source_archive_sha256": ARCHIVE_SHA256,
        "dev_document_count": len(documents),
        "rights_clean_sha256": verified,
        "audited_files": len(paths),
        "audited_rows": rows,
        "suspected_hits": len(hits),
        "hit_files": sorted({path for path, _, _ in hits}),
        "hit_classes": dict(Counter(kind for _, _, kind in hits)),
        "parse_errors": len(errors),
        "error_files": sorted({path for path, _ in errors}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.chmod(0o600)
    return report


def write_lengths(
    archive: Path,
    model_path: Path,
    output: Path,
    adapter: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    corpus = read_archive(archive)
    if adapter == "gliner":
        from gliner2.classification.schema import ClassificationSchema
        from inference import gliner25

        gliner25.verify_release(model_path, gliner25.MULTI_REVISION, "multilingual")
        native = gliner25.load_native(model_path, "cpu")
        if sum(p.numel() for p in native.model.parameters()) != 287_355_159:
            raise ValueError("Loaded multilingual GLiNER parameter count differs")
        if gliner25.native_context_limit(native) != 4096:
            raise ValueError("Multilingual GLiNER native window differs")

        def count(text: str, task: dict[str, Any]) -> tuple[int, int]:
            native_text, aliases = gliner25.prepare_question(text, task)
            schema = ClassificationSchema().single(
                "decision",
                gliner25.schema_labels(task, aliases),
                instruction=gliner25._schema_safe(task["instructions"]),
            )
            compiled = native.compile_schema(schema)
            length = len(
                native.model.processor.transform_record(
                    native_text, compiled.build()
                ).input_ids
            )
            return length, length

        identity = {
            "model_id": gliner25.MULTI_MODEL_ID,
            "model_revision": gliner25.MULTI_REVISION,
            "model_sha256": gliner25.MULTI_MODEL_FILES["model.safetensors"],
            "adapter": gliner25.PROFILES["multilingual"]["adapter_version"],
            "loaded_parameters": 287_355_159,
        }
    elif adapter == "qwen":
        from training.qwen3_reranker06 import pilot
        from transformers import AutoTokenizer

        pilot.verify_source(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

        def count(text: str, task: dict[str, Any]) -> tuple[int, int]:
            prompts, _ = pilot.candidate_texts(text, task)
            lengths = [
                len(tokenizer.encode(prompt, add_special_tokens=False))
                for prompt in prompts
            ]
            return min(lengths), max(lengths)

        identity = {
            "model_id": pilot.MODEL_ID,
            "model_revision": pilot.MODEL_REVISION,
            "model_sha256": pilot.MODEL_FILES["model.safetensors"],
            "adapter": pilot.ADAPTER,
            "previously_measured_loaded_parameters": 595_776_512,
        }
    else:
        raise ValueError(adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("x", encoding="utf-8") as stream:
        for key, text, task in source_items(corpus):
            low, high = count(text, task)
            if low < 1 or high < low:
                raise ValueError("Invalid native token count")
            stream.write(
                json.dumps(
                    {"id": key, "minimum_tokens": low, "maximum_tokens": high},
                    sort_keys=True,
                )
                + "\n"
            )
            n += 1
    return {
        "adapter": adapter,
        "source": identity,
        "rows": n,
        "lengths_sha256": file_digest(output),
        "archive_sha256": ARCHIVE_SHA256,
        "dev_sha256": DEV_SHA256,
    }


def load_lengths(path: Path) -> dict[str, tuple[int, int]]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["id"]
            low, high = row["minimum_tokens"], row["maximum_tokens"]
            if key in rows or type(low) is not int or type(high) is not int:
                raise ValueError("Duplicate or malformed length receipt")
            rows[key] = (low, high)
    if len(rows) != 61 * 17:
        raise ValueError("Incomplete native length receipt")
    return rows


def select(
    corpus: dict[str, Any],
    gliner_lengths: dict[str, tuple[int, int]],
    qwen_lengths: dict[str, tuple[int, int]],
) -> tuple[list[tuple[str, dict[str, Any], str, str]], dict[str, Any]]:
    expected = {key for key, _, _ in source_items(corpus)}
    if set(gliner_lengths) != expected or set(qwen_lengths) != expected:
        raise ValueError("Length receipt IDs do not match official development split")
    rows = []
    considered = sorted(
        corpus["documents"],
        key=lambda doc: sha_text(f"{SEED}:{doc['id']}"),
    )[:96]
    eligible = 0
    for document in considered:
        by_class: dict[str, list[str]] = {name: [] for name in CLASSES}
        annotations = document["annotation_sets"][0]["annotations"]
        for hypothesis_id in corpus["labels"]:
            key = item_id(document, hypothesis_id)
            g_min, g_max = gliner_lengths[key]
            q_min, q_max = qwen_lengths[key]
            if not (1024 < g_min <= g_max <= 4096 and 1024 < q_min <= q_max <= 4096):
                continue
            eligible += 1
            label = ORIGINAL_LABELS[annotations[hypothesis_id]["choice"]]
            by_class[label].append(hypothesis_id)
        for label in CLASSES:
            ids = sorted(by_class[label], key=lambda hid: sha_text(f"{SEED}:{hid}"))
            if ids:
                key = item_id(document, ids[0])
                rows.append((key, document, ids[0], label))
    counts = Counter(label for _, _, _, label in rows)
    distinct = {str(doc["id"]) for _, doc, _, _ in rows}
    long_docs = {
        str(doc["id"])
        for key, doc, _, _ in rows
        if min(gliner_lengths[key][0], qwen_lengths[key][0]) > 2048
    }
    summary = {
        "candidate_documents": len(considered),
        "candidate_questions": len(considered) * 17,
        "eligible_questions_before_class_cap": eligible,
        "selected_questions": len(rows),
        "selected_documents": len(distinct),
        "selected_class_counts": {name: counts[name] for name in CLASSES},
        "selected_long_documents": len(long_docs),
        "population_pass": (
            len(distinct) >= 40
            and len(long_docs) >= 20
            and all(counts[name] >= 20 for name in CLASSES)
        ),
    }
    return rows, summary


def freeze(
    archive: Path,
    audit_receipt: Path,
    gliner: Path,
    qwen: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(audit_receipt.read_text(encoding="utf-8"))
    if (
        audit.get("source_archive_sha256") != ARCHIVE_SHA256
        or audit.get("suspected_hits") != 0
        or audit.get("parse_errors") != 0
        or audit.get("audited_rows", 0) < 8_155
        or audit.get("rights_clean_sha256") != RIGHTS_CLEAN_SHA256
    ):
        raise ValueError("Source overlap audit missing, incomplete or positive")
    corpus = read_archive(archive)
    g, q = load_lengths(gliner), load_lengths(qwen)
    selected, summary = select(corpus, g, q)
    output.mkdir(parents=True, mode=0o700)
    receipt = {
        **summary,
        "archive_sha256": ARCHIVE_SHA256,
        "development_split_sha256": DEV_SHA256,
        "overlap_receipt_sha256": file_digest(audit_receipt),
        "gliner_lengths_sha256": file_digest(gliner),
        "qwen_lengths_sha256": file_digest(qwen),
        "seed": SEED,
    }
    if summary["population_pass"]:
        prompts = output / "prompts.jsonl"
        gold = output / "gold.jsonl"
        with prompts.open("x", encoding="utf-8") as p, gold.open(
            "x", encoding="utf-8"
        ) as g_file:
            for key, document, hypothesis_id, label in selected:
                p.write(
                    json.dumps(
                        {
                            "id": key,
                            "state": document["text"],
                            "questions": {
                                "decision": question(
                                    corpus["labels"][hypothesis_id]["hypothesis"]
                                )
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                g_file.write(json.dumps({"id": key, "label": label}) + "\n")
        prompts.chmod(0o600)
        gold.chmod(0o600)
        receipt["prompts_sha256"] = file_digest(prompts)
        receipt["gold_sha256"] = file_digest(gold)
    (output / "population.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "population.receipt.json").chmod(0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--archive", type=Path, required=True)
    audit.add_argument("--scan-root", type=Path, action="append", required=True)
    audit.add_argument("--rights-clean", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    for name in ("length-gliner", "length-qwen"):
        command = sub.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
    frozen = sub.add_parser("freeze")
    for name in ("archive", "audit-receipt", "gliner", "qwen", "output"):
        frozen.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        report = audit_overlap(
            args.archive, args.scan_root, args.rights_clean, args.output
        )
    elif args.command == "freeze":
        report = freeze(
            args.archive, args.audit_receipt, args.gliner, args.qwen, args.output
        )
    else:
        report = write_lengths(
            args.archive,
            args.model,
            args.output,
            "gliner" if args.command == "length-gliner" else "qwen",
        )
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
