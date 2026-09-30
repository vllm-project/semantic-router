"""M4b soft-target files on the human-gold rows S of F1's TRAIN file (CPU, deterministic).

S is selected from row metadata only (``RULE``):

- A7 rows carry ``audit_metadata.a7`` (sub-arms A7g / A7i / A7o / A7p) and are never in S
  (A7i is typed-format intent rows and keeps gold only);
- A6h rows (all Score) come from the human-rated sources in ``A6H_SOURCES``; A6g is
  ``decision2_verifiable_v2_a6``; every other row is A0s-strict;
- the A0s-strict human rows are GoEmotions and SNLI, plus the ``legacy:stage3_replay``
  rows that keep an ``upstream_label`` (BANKING77 / CLINC150 natural intents);
- S = A0s-strict human + A6h. Everything else keeps gold only.

The pool counts are checked against ``MIXTURES.json`` (rows per source, the A6 and A7
parts). Each teacher is a list of source files, each bound to one pool of S by the
SOURCES spec; records outside S are ignored (and counted). A record used for S must
match the TRAIN row's ``input_sha256`` and option-key set, with finite non-negative
probabilities summing to 1 +- 1e-6; a missing or repeated S row fails the build.
Outputs, in TRAIN order: ``teacher-<name>.jsonl`` (``{id, input_sha256, teacher_probs}``,
the probabilities as stored), ``S.ids.txt`` and ``MANIFEST.json`` (rule, input hashes and
revisions, output hashes, counts, native tokens with ``--source`` and each teacher's
agreement with gold). Nothing in the output depends on the output path or the clock.

    python3 -m v2.27b.m4b.build_teachers --sources teacher-sources-m4b.json \
        --output-dir DIR [--source BASE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "decision2-27b-m4b-teachers/1"
SUM_TOLERANCE = 1e-6
A6H_SOURCES = (
    "argq30k_train",
    "jglue_jsts_v1.3_train",
    "klue_sts_train",
    "saf_de_train",
    "saf_en_train",
)
A6G_SOURCES = ("decision2_verifiable_v2_a6",)
A0S_HUMAN_SOURCES = ("google_goemotions_official_train", "legacy:snli")
REPLAY_SOURCE = "legacy:stage3_replay"
S_POOLS = ("A0s-strict", "A6h")
RULE = {
    "A7": "audit_metadata has key 'a7' (never in S)",
    "A6h": f"not A7 and source in {list(A6H_SOURCES)}",
    "A6g": f"not A7 and source in {list(A6G_SOURCES)}",
    "A0s-strict": "every other row",
    "A0s-strict human": f"A0s-strict and (source in {list(A0S_HUMAN_SOURCES)} or "
    f"(source == {REPLAY_SOURCE!r} and the row has 'upstream_label'))",
    "S": "A0s-strict human rows + A6h rows; every other row keeps gold only",
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pool_of(row: dict[str, Any]) -> str:
    if "a7" in (row.get("audit_metadata") or {}):
        return "A7"
    if row["source"] in A6H_SOURCES:
        return "A6h"
    if row["source"] in A6G_SOURCES:
        return "A6g"
    return "A0s-strict"


def in_s(row: dict[str, Any]) -> bool:
    pool = pool_of(row)
    if pool == "A6h":
        return True
    return pool == "A0s-strict" and (
        row["source"] in A0S_HUMAN_SOURCES
        or (row["source"] == REPLAY_SOURCE and "upstream_label" in row)
    )


def check_mixture(
    rows: list[dict[str, Any]], mixtures: dict[str, Any], name: str
) -> dict:
    """TRAIN counts per source and the A6 / A7 part sizes equal the builder's record."""
    report = mixtures["mixtures"][name]
    by_source = Counter(row["source"] for row in rows)
    if dict(by_source) != report["by_source"]:
        raise ValueError("TRAIN rows per source differ from MIXTURES.json")
    pools = Counter(pool_of(row) for row in rows)
    parts = report.get("parts", {})
    expected = {
        "A6": parts.get("A6", {}).get("rows"),
        "A7": parts.get("A7", {}).get("rows"),
    }
    observed = {"A6": pools["A6g"] + pools["A6h"], "A7": pools["A7"]}
    for part, value in expected.items():
        if value is not None and value != observed[part]:
            raise ValueError(
                f"{part}: {observed[part]} rows by the pool rule, {value} in MIXTURES.json"
            )
    return {"rows_by_pool": dict(sorted(pools.items())), "parts_checked": expected}


def unique_argmax(probs: dict[str, float]) -> str | None:
    best = max(probs.values())
    winners = [k for k, v in probs.items() if abs(v - best) <= 1e-12]
    return winners[0] if len(winners) == 1 else None


def validate(record: dict[str, Any], row: dict[str, Any], where: str) -> None:
    if record.get("input_sha256") != row["input_sha256"]:
        raise ValueError(f"{where}: teacher input hash differs from TRAIN")
    probs = record.get("teacher_probs")
    keys = [option["key"] for option in row["options"]]
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"{where}: teacher keys differ from the row's options")
    values = list(probs.values())
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
        )
        or abs(sum(values) - 1.0) > SUM_TOLERANCE
    ):
        raise ValueError(f"{where}: invalid teacher distribution")


def join_teacher(
    name: str,
    files: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    s_pool: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """S id -> teacher record, and per-file receipts."""
    chosen: dict[str, dict[str, Any]] = {}
    receipts = []
    for spec in files:
        path = Path(spec["path"])
        digest = sha_file(path)
        if digest != spec["sha256"]:
            raise ValueError(f"{name}: {path} is {digest}, expected {spec['sha256']}")
        if spec["pool"] not in S_POOLS:
            raise ValueError(f"{name}: {path} names pool {spec['pool']!r}")
        counts: Counter[str] = Counter()
        seen: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                where = f"{name}: {path}:{number}"
                row_id = record.get("id")
                if row_id in seen:
                    raise ValueError(f"{where}: repeated teacher id")
                seen.add(row_id)
                counts["records"] += 1
                if row_id not in rows:
                    counts["not_in_train"] += 1
                    continue
                if row_id not in s_pool:
                    counts["in_train_outside_s"] += 1
                    continue
                if s_pool[row_id] != spec["pool"]:
                    raise ValueError(
                        f"{where}: S row of pool {s_pool[row_id]} in a {spec['pool']} file"
                    )
                if row_id in chosen:
                    raise ValueError(f"{where}: S row already covered by another file")
                validate(record, rows[row_id], where)
                chosen[row_id] = {
                    "id": row_id,
                    "input_sha256": record["input_sha256"],
                    "teacher_probs": record["teacher_probs"],
                }
                counts["used"] += 1
        receipts.append(
            {
                **{k: spec[k] for k in sorted(spec) if k != "sha256"},
                "sha256": digest,
                "counts": dict(sorted(counts.items())),
            }
        )
    missing = [row_id for row_id in s_pool if row_id not in chosen]
    if missing:
        raise ValueError(f"{name}: {len(missing)} S rows have no teacher record")
    return chosen, receipts


def agreement(
    chosen: dict[str, dict[str, Any]], order: list[dict[str, Any]]
) -> dict[str, Any]:
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in order:
        gold = row["options"][row["label"]]["key"]
        agree = unique_argmax(chosen[row["id"]]["teacher_probs"]) == gold
        for key in (
            "all",
            f"source:{row['source']}",
            f"source:{row['source']}|{row['task_type']}",
        ):
            cells[key][0] += agree
            cells[key][1] += 1
    return {
        k: {"agree": a, "n": n, "rate": a / n} for k, (a, n) in sorted(cells.items())
    }


def token_counts(
    order: list[dict[str, Any]], source: Path, mixtures: dict[str, Any], name: str
) -> dict[str, Any]:
    """Native tokens of S (the mixture builder's ``len(encode(row)["ids"])``)."""
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    cells: Counter[str] = Counter()
    for row in order:
        n = len(encode(row, tokenizer, 1 << 30)["ids"])
        for key in (
            "all",
            f"pool:{pool_of(row)}",
            f"source:{row['source']}",
            f"type:{row['task_type']}",
            f"pool_type:{pool_of(row)}|{row['task_type']}",
        ):
            cells[key] += n
    breakdown = mixtures["mixtures"][name]["breakdown"]["source"]
    whole = Counter(row["source"] for row in order)
    checked = {}
    for src, n_rows in sorted(whole.items()):
        if breakdown[src]["rows"] == n_rows:
            if breakdown[src]["tokens"] != cells[f"source:{src}"]:
                raise ValueError(f"{src}: S tokens differ from MIXTURES.json")
            checked[src] = breakdown[src]["tokens"]
    return {
        "encoder": "pinned BEST368 training.model.decision_model.encode",
        "tokenizer_json_sha256": sha_file(source / "tokenizer.json"),
        "tokens": dict(sorted(cells.items())),
        "sources_matching_mixtures": checked,
    }


def build(spec: dict[str, Any], output: Path, source: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    train_path = Path(spec["train"]["path"])
    if sha_file(train_path) != spec["train"]["sha256"]:
        raise ValueError("TRAIN differs from its pinned SHA-256")
    mixtures_path = Path(spec["mixtures"]["path"])
    if sha_file(mixtures_path) != spec["mixtures"]["sha256"]:
        raise ValueError("MIXTURES.json differs from its pinned SHA-256")
    mixtures = json.loads(mixtures_path.read_text(encoding="utf-8"))
    order = read_jsonl(train_path)
    rows = {row["id"]: row for row in order}
    if len(rows) != len(order):
        raise ValueError("TRAIN has repeated ids")
    pools = check_mixture(order, mixtures, spec["mixtures"]["name"])
    selected = [row for row in order if in_s(row)]
    if not selected:
        raise ValueError("S is empty")
    s_pool = {row["id"]: pool_of(row) for row in selected}
    expected = spec.get("expect_s_rows")
    if expected is not None and len(selected) != expected:
        raise ValueError(f"S has {len(selected)} rows, the spec expects {expected}")

    teachers, texts = {}, {}
    for name, teacher in sorted(spec["teachers"].items()):
        chosen, receipts = join_teacher(name, teacher["files"], rows, s_pool)
        texts[name] = "".join(canonical(chosen[row["id"]]) + "\n" for row in selected)
        provenance = teacher.get("provenance")
        if provenance is not None:
            digest = sha_file(Path(provenance["path"]))
            if digest != provenance["sha256"]:
                raise ValueError(
                    f"{name}: provenance {digest} != {provenance['sha256']}"
                )
        teachers[name] = {
            "model": teacher.get("model"),
            "files": receipts,
            "provenance": provenance,
            "agreement_with_gold": agreement(chosen, selected),
        }

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        pool = pool_of(row)
        counts["by_pool"][pool] += 1
        counts["by_source"][row["source"]] += 1
        counts["by_type"][row["task_type"]] += 1
        counts["by_pool_type"][f"{pool}|{row['task_type']}"] += 1
        counts["by_pool_source"][f"{pool}|{row['source']}"] += 1
    ids_text = "".join(row["id"] + "\n" for row in selected)
    files = {"S.ids.txt": ids_text}
    files.update({f"teacher-{name}.jsonl": text for name, text in texts.items()})
    manifest = {
        "schema": SCHEMA,
        "rule": RULE,
        "code_sha256": {"build_teachers.py": sha_file(Path(__file__))},
        "inputs": {
            "train": {**spec["train"], "rows": len(order)},
            "mixtures": spec["mixtures"],
        },
        "train_pools": pools,
        "s": {
            "rows": len(selected),
            "ids_sha256": hashlib.sha256(ids_text.encode("utf-8")).hexdigest(),
            **{k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
        },
        "teachers": teachers,
        "tokens": (
            token_counts(selected, source, mixtures, spec["mixtures"]["name"])
            if source is not None
            else None
        ),
        "outputs": {
            name: {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "lines": text.count("\n"),
            }
            for name, text in sorted(files.items())
        },
    }
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    for name, text in files.items():
        (pending / name).write_text(text, encoding="utf-8")
    (pending / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in pending.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    os.replace(pending, output)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sources", type=Path, required=True, help="SOURCES spec JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="base snapshot (tokenizer) for tokens"
    )
    args = parser.parse_args(argv)
    spec = json.loads(args.sources.read_text(encoding="utf-8"))
    manifest = build(spec, args.output_dir, args.source)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "s_rows": manifest["s"]["rows"],
                "s_by_pool": manifest["s"]["by_pool"],
                "outputs": manifest["outputs"],
                "tokens_all": (manifest["tokens"] or {}).get("tokens", {}).get("all"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
