"""CPU-only source screen for NIST TREC DL 2023 passage Score.

Publisher queries, judgments, duplicate classes, and MS MARCO passage text
stay in a private experiment directory. No model is run and no evaluation
answer or source text is written to the aggregate receipt. This is a
source-disjoint diagnostic hypothesis, never a training-data builder.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable

from training.data import audit_vitaminc_score as protected_audit

QRELS_SHA256 = "5a98a95d5714b00c5066593719e5eb7ca93dd13588d0facf7f426a4c9d6b19b8"
QUERIES_SHA256 = "56b763823a2a9027b136f42dc5d9c429112198b40c075b1c5df63672e8e3cb4b"
DUPES_SHA256 = "57ddba8918693c8532e220932f4cd9b21ad9c384341fc28ff8b9eba5df3fe32d"
CORPUS_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar"
)
CORPUS_BYTES = 21_768_192_000
PID = re.compile(r"^msmarco_passage_(\d\d)_(\d+)$")
GRADE_DESCRIPTIONS = (
    "The passage does not address the question.",
    "The passage is on the topic but does not answer the question.",
    "The passage answers the question well, with some omissions.",
    "The passage gives a complete answer to the question.",
)


def sha_file(path: Path) -> str:
    return protected_audit.sha_file(path)


def require_sha(path: Path, expected: str) -> None:
    if not path.is_file() or sha_file(path) != expected:
        raise ValueError(f"Missing or changed publisher file: {path.name}")


def parse_qrels(path: Path) -> list[tuple[str, str, int]]:
    require_sha(path, QRELS_SHA256)
    rows = []
    keys = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 4 or fields[1] != "0" or not PID.fullmatch(fields[2]):
                raise ValueError("Unexpected NIST qrels format")
            grade = int(fields[3])
            key = (fields[0], fields[2])
            if grade not in range(4) or key in keys:
                raise ValueError("Invalid or repeated NIST judgment")
            keys.add(key)
            rows.append((fields[0], fields[2], grade))
    if len(rows) != 22_327 or len({row[0] for row in rows}) != 82:
        raise ValueError("NIST qrels count changed")
    return rows


def parse_queries(path: Path) -> dict[str, str]:
    require_sha(path, QUERIES_SHA256)
    queries = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            qid, separator, query = line.rstrip("\n").partition("\t")
            if not separator or not query or qid in queries:
                raise ValueError("Invalid or repeated MS MARCO query")
            queries[qid] = query
    if len(queries) != 700:
        raise ValueError("MS MARCO query count changed")
    return queries


def duplicate_representatives(
    path: Path, passage_ids: set[str]
) -> tuple[dict[str, str], int]:
    require_sha(path, DUPES_SHA256)
    mapping = {}
    scanned = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            fields = line.split(maxsplit=2)
            if len(fields) < 2:
                raise ValueError("Invalid NIST duplicate-class line")
            scanned += 1
            pid = fields[1].decode("ascii")
            if pid in passage_ids:
                if pid in mapping:
                    raise ValueError("Duplicate passage appears in two classes")
                mapping[pid] = fields[0].decode("ascii")
    if mapping.keys() != passage_ids:
        raise ValueError("NIST duplicate classes do not cover every judgment")
    return mapping, scanned


def duplicate_summary(
    rows: list[tuple[str, str, int]], mapping: dict[str, str]
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for qid, pid, grade in rows:
        groups[(qid, mapping[pid])].append(grade)
    conflicts = [grades for grades in groups.values() if len(set(grades)) != 1]
    return {
        "distinct_query_groups": len({qid for qid, _ in groups}),
        "distinct_query_duplicate_class_groups": len(groups),
        "propagated_rows": sum(len(grades) - 1 for grades in groups.values()),
        "conflicting_grade_groups": len(conflicts),
        "conflicting_grade_rows": sum(len(grades) for grades in conflicts),
        "class_group_grade_counts": dict(
            sorted(
                collections.Counter(
                    grades[0] for grades in groups.values() if len(set(grades)) == 1
                ).items()
            )
        ),
    }


def get_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError("Publisher archive did not honor byte range")
        content_range = response.headers.get("Content-Range", "")
        if content_range != f"bytes {start}-{end}/{CORPUS_BYTES}":
            raise ValueError("Publisher archive length or range changed")
        data = response.read()
    if len(data) != end - start + 1:
        raise ValueError("Short publisher archive range")
    return data


def index_tar(
    get_bytes: Callable[[int, int], bytes], archive_length: int = CORPUS_BYTES
) -> list[dict[str, Any]]:
    offset = 0
    members = []
    while offset + 512 <= archive_length:
        header = get_bytes(offset, offset + 511)
        if len(header) != 512:
            raise ValueError("Short TAR header")
        if header == bytes(512):
            break
        name = header[:100].split(bytes([0]))[0].decode("ascii")
        size = int(header[124:136].split(bytes([0]))[0].strip() or b"0", 8)
        if header[257:263] not in (b"ustar\x00", b"ustar "):
            raise ValueError("Unexpected TAR format")
        if size and not re.fullmatch(
            r"msmarco_v2_passage/msmarco_passage_\d\d.gz", name
        ):
            raise ValueError("Unexpected TAR member name")
        if size:
            members.append({"name": name, "offset": offset + 512, "size": size})
        offset += 512 + ((size + 511) // 512) * 512
    if len(members) != 70 or len({item["name"] for item in members}) != 70:
        raise ValueError("Publisher TAR must contain 70 passage members")
    return members


def fetch_members(index: list[dict[str, Any]], shards: set[str], out: Path) -> None:
    if out.exists():
        raise ValueError("Refusing to overwrite member directory")
    out.mkdir(mode=0o700)
    found = set()
    for member in index:
        shard = member["name"].rsplit("_", 1)[1].removesuffix(".gz")
        if shard not in shards:
            continue
        found.add(shard)
        target = out / f"msmarco_passage_{shard}.gz"
        request = urllib.request.Request(
            CORPUS_URL,
            headers={
                "Range": f"bytes={member['offset']}-{member['offset'] + member['size'] - 1}"
            },
        )
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(
            request, timeout=120
        ) as response:
            if response.status != 206 or response.headers.get("Content-Range") != (
                f"bytes {member['offset']}-{member['offset'] + member['size'] - 1}/{CORPUS_BYTES}"
            ):
                raise ValueError("Publisher member range changed")
            while chunk := response.read(1 << 20):
                stream.write(chunk)
        if target.stat().st_size != member["size"]:
            raise ValueError("Short publisher member")
    if found != shards:
        raise ValueError("Requested shard absent from publisher TAR")


def join_passages(
    member_dir: Path, passage_ids: set[str]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    by_shard: dict[str, set[str]] = collections.defaultdict(set)
    for pid in passage_ids:
        match = PID.fullmatch(pid)
        if not match:
            raise ValueError("Invalid passage ID")
        by_shard[match.group(1)].add(pid)
    passages = {}
    docids = {}
    member_hashes = {}
    for shard, targets in sorted(by_shard.items()):
        path = member_dir / f"msmarco_passage_{shard}.gz"
        if not path.is_file():
            raise ValueError(f"Missing official passage member {shard}")
        member_hashes[shard] = sha_file(path)
        with gzip.open(path, "rb") as stream:
            offset = 0
            for line in stream:
                row = json.loads(line)
                expected = f"msmarco_passage_{shard}_{offset}"
                if row.get("pid") != expected:
                    raise ValueError(
                        "Passage ID byte offset does not match publisher member"
                    )
                if expected in targets:
                    if not isinstance(row.get("passage"), str) or not row["passage"]:
                        raise ValueError("Empty passage text")
                    passages[expected] = row["passage"]
                    docids[expected] = str(row.get("docid", ""))
                offset += len(line)
        if not targets <= passages.keys():
            raise ValueError("Judged passage not found in official member")
    return passages, docids, member_hashes


def native_rows(
    rows: list[tuple[str, str, int]], queries: dict[str, str], passages: dict[str, str]
) -> list[dict[str, Any]]:
    rendered = []
    for qid, pid, grade in rows:
        if pid not in passages:
            continue
        if qid not in queries:
            raise ValueError("Judged query missing publisher text")
        rendered.append(
            {
                "id": hashlib.sha256(f"{qid}:{pid}".encode()).hexdigest(),
                "group_id": qid,
                "state": f"Question: {queries[qid]}\n\nPassage: {passages[pid]}",
                "instructions": "Rate how completely the passage answers the question. A related passage that does not answer is level 1.",
                "task_type": "score",
                "family": "passage_answer_relevance",
                "options": [
                    {"key": str(index), "description": description}
                    for index, description in enumerate(GRADE_DESCRIPTIONS)
                ],
                "label": grade,
            }
        )
    return rendered


def audit(
    qrels: Path,
    queries: Path,
    dupes: Path,
    member_dir: Path | None,
    protected_inventory: Path | None,
    tokenizer_dir: Path | None,
    max_length: int,
) -> dict[str, Any]:
    rows = parse_qrels(qrels)
    query_text = parse_queries(queries)
    mapping, scanned = duplicate_representatives(dupes, {pid for _, pid, _ in rows})
    summary = {
        "source": "NIST TREC DL 2023 passage qrels + MS MARCO v2 passage text",
        "qrels_sha256": QRELS_SHA256,
        "queries_sha256": QUERIES_SHA256,
        "duplicate_classes_sha256": DUPES_SHA256,
        "judged_rows": len(rows),
        "query_groups": len({qid for qid, _, _ in rows}),
        "grade_counts": dict(
            sorted(collections.Counter(grade for _, _, grade in rows).items())
        ),
        "query_text_groups_with_non_ascii": sum(
            any(ord(char) > 127 for char in query_text[qid])
            for qid in {qid for qid, _, _ in rows}
        ),
        "duplicate_class_file_rows": scanned,
        "duplicate_groups": duplicate_summary(rows, mapping),
        "rights": "MS MARCO dataset: noncommercial research only; no underlying document IP grant; keep passage text private",
        "role": "independent public-source Score diagnostic, never TRAIN or a sealed final test",
    }
    blockers = []
    if summary["duplicate_groups"]["conflicting_grade_groups"]:
        blockers.append("CONFLICTING_GRADES_WITHIN_NIST_DUPLICATE_CLASS")
    if member_dir is None:
        blockers.append("EXACT_PASSAGE_TEXT_NOT_JOINED")
        native = []
    else:
        available = {
            f"msmarco_passage_{match.group(1)}_{match.group(2)}"
            for _, pid, _ in rows
            if (match := PID.fullmatch(pid))
            and (member_dir / f"msmarco_passage_{match.group(1)}.gz").is_file()
        }
        if not available:
            raise ValueError("No official judged passage members available")
        passages, docids, hashes = join_passages(member_dir, available)
        native = native_rows(rows, query_text, passages)
        summary["passage_member_sha256"] = hashes
        summary["joined_rows"] = len(native)
        summary["joined_query_groups"] = len({row["group_id"] for row in native})
        summary["joined_grade_counts"] = dict(
            sorted(collections.Counter(row["label"] for row in native).items())
        )
        summary["joined_passages_with_non_ascii"] = sum(
            any(ord(char) > 127 for char in passage) for passage in passages.values()
        )
        lengths = [len(passage) for passage in passages.values()]
        summary["joined_passage_characters"] = {
            "p50": protected_audit.percentile(lengths, 0.5),
            "p90": protected_audit.percentile(lengths, 0.9),
            "p99": protected_audit.percentile(lengths, 0.99),
            "max": max(lengths),
        }
        summary["joined_distinct_documents"] = len({docids[pid] for pid in passages})
        if len(native) != len(rows):
            blockers.append("PARTIAL_PASSAGE_TEXT_JOIN")
    if protected_inventory is None:
        blockers.append("NO_PINNED_PROTECTED_INVENTORY")
    elif native:
        protected, inventory = protected_audit.protected_texts(protected_inventory)
        overlap_input = [
            {
                "claim": query_text[row["group_id"]],
                "evidence": row["state"].split("\n\nPassage: ", 1)[1],
                "page": row["group_id"],
            }
            for row in native
        ]
        summary["protected_inventory"] = inventory
        summary["protected_overlap"] = protected_audit.overlap_screen(
            overlap_input, protected
        )
        if summary["protected_overlap"]["suspected_page_groups_total"]:
            blockers.append("PROTECTED_PROMPT_OVERLAP_REQUIRES_REVIEW")
        if summary["protected_overlap"][
            "protected_snippets_over_800_chars_not_near_scanned"
        ]:
            blockers.append("LONG_PROTECTED_PROMPTS_NEED_SEPARATE_NEAR_SCAN")
    if tokenizer_dir is None:
        blockers.append("NO_PINNED_NATIVE_TOKENIZER")
    elif native:
        from transformers import AutoTokenizer
        from training.model.decision_model import encode

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), local_files_only=True, trust_remote_code=False
        )
        lengths = []
        over_cap = 0
        for row in native:
            try:
                lengths.append(len(encode(row, tokenizer, max_length)["ids"]))
            except ValueError as exc:
                if "exceeds max_length" not in str(exc):
                    raise
                over_cap += 1
        summary["native_length"] = {
            "tokenizer_json_sha256": sha_file(tokenizer_dir / "tokenizer.json"),
            "max_length": max_length,
            "over_cap": over_cap,
            "p50": protected_audit.percentile(lengths, 0.5) if lengths else None,
            "p90": protected_audit.percentile(lengths, 0.9) if lengths else None,
            "p99": protected_audit.percentile(lengths, 0.99) if lengths else None,
            "max": max(lengths) if lengths else None,
        }
        if over_cap:
            blockers.append("NATIVE_REQUEST_OVER_CAP")
    blockers.append("SOURCE_DISJOINT_EVAL_CUSTODY_NOT_FROZEN")
    summary["decision"] = "HOLD"
    summary["blockers"] = blockers
    summary["gpu_hours"] = 0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index_parser = sub.add_parser("index", help="Index publisher TAR by HTTP ranges")
    index_parser.add_argument("--out", type=Path, required=True)
    fetch_parser = sub.add_parser("fetch", help="Fetch exact publisher gzip members")
    fetch_parser.add_argument("--index", type=Path, required=True)
    fetch_parser.add_argument("--shard", action="append", required=True)
    fetch_parser.add_argument("--out-dir", type=Path, required=True)
    audit_parser = sub.add_parser("audit", help="Write aggregate private HOLD receipt")
    audit_parser.add_argument("--qrels", type=Path, required=True)
    audit_parser.add_argument("--queries", type=Path, required=True)
    audit_parser.add_argument("--dupes", type=Path, required=True)
    audit_parser.add_argument("--member-dir", type=Path)
    audit_parser.add_argument("--protected-inventory", type=Path)
    audit_parser.add_argument("--tokenizer-dir", type=Path)
    audit_parser.add_argument("--max-length", type=int, default=1024)
    audit_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "index":
        result = {
            "corpus_url": CORPUS_URL,
            "corpus_bytes": CORPUS_BYTES,
            "members": index_tar(lambda a, b: get_range(CORPUS_URL, a, b)),
        }
    elif args.command == "fetch":
        payload = json.loads(args.index.read_text())
        if (
            payload.get("corpus_url") != CORPUS_URL
            or payload.get("corpus_bytes") != CORPUS_BYTES
        ):
            raise ValueError("Publisher TAR index changed")
        shards = set(args.shard)
        if any(not re.fullmatch(r"\d\d", shard) for shard in shards):
            raise ValueError("Shards must be two digits")
        fetch_members(payload["members"], shards, args.out_dir)
        return
    else:
        result = audit(
            args.qrels,
            args.queries,
            args.dupes,
            args.member_dir,
            args.protected_inventory,
            args.tokenizer_dir,
            args.max_length,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
