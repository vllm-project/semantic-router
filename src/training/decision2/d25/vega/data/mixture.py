"""Build training mixtures (M1, M2, ...) from converted, decontaminated row sets.

    python -m d25.vega.data.mixture --mix M1 --index DATA_ROOT/decontam/<index> --holdouts holdouts.json --out DIR

Steps: load candidate rows per part -> drop decontamination hits (suite exact / 13-gram), holdouts.json
hits and prompts over the token limit -> per-part selection (quota per source, whole groups, rank by
seeded hash) -> exact de-duplication (state, question, ordered options) -> group-disjoint dev split ->
deterministic shuffle -> shards + manifest (counts, sha256, token stats, decontamination report).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df
from d25.vega.data.holdouts import Holdouts, selected
from d25.vega.data.tokens import load_lengths
from d25.vega.data.util import (
    DATA_ROOT,
    normalize_tokens,
    rank,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

PPLX_MANIFEST = Path(__file__).with_name("pplx_v11_source_rows.json")
MAX_PROMPT_TOKENS = 8160
SHARD_ROWS = 50_000
DEV_ROWS = 2_000

D20_CORE = [
    "xlr2-A0s-strict",
    "IB1-train",
    "IB2-train",
    "IB3-train",
    "IB4-train",
    "PN1-train",
]
D20_FILL = [
    "xlr2-H1",
    "xlr2-H3",
    "xlr2-H5",
    "xlr2-H6",
    "xlr2-E11",
    "xlr2-G2",
    "xlr2-G6",
    "xlr2-G4h",
    "xlr2-V1-A1",
    "xlr2-V1-A2",
    "xlr2-V1-A3",
    "xlr2-V1-A4v2h",
    "xlr2-V1-A5",
    "xlr2-V1-A6g",
    "xlr2-V1-A6h",
    "xlr2-A7g",
    "xlr2-A7h",
    "xlr2-A7i",
    "xlr2-A7k",
    "xlr2-A7m",
    "xlr2-A7o",
    "xlr2-A7p",
    "xlr2-A7q",
    "xlr2-A7r",
    "xlr2-A7s",
    "xlr2-H7",
    "xlr2-H8",
    "HS1-train",
]

# Decision 2.0 origin sources that fall under holdouts.json row slices (holdouts v1 "decision20_corpora_note").
# Each row is matched to its upstream train record (question / sentence / text / claim / tweet) and the slice
# is computed on the exact upstream key; rows that cannot be matched are dropped (superset of the slice).
D20_SLICED = {
    "gsm8k_train": "gsm8k",
    "winogrande_xl_train": "winogrande",
    "hover_train": "hover",
    "hover_train_v1.1": "hover",
    "isarcasmeval_train": "isarcasm-en",
    "isarcasmeval_train_en": "isarcasm-en",
    "when2call_train_pref": "when2call",
    "dec10:banking77_train": "banking77",
}
D20_SLICED_FAMILIES = {
    "d20/stage4_replay_banking_train": "banking77",
    "d20/banking_train": "banking77",
}


def squash(text: str) -> str:
    return " ".join(str(text).split())


class UpstreamKeys:
    """Normalised text -> exact upstream slice key, per holdouts slice name, from the pinned upstream files."""

    def __init__(self, holdouts: Holdouts):
        raw = DATA_ROOT / "raw"
        self.maps: dict[str, dict[str, str]] = defaultdict(dict)
        self.slices = {
            e["slice"]["name"]: e["slice"]
            for e, _ in holdouts.entries
            if e.get("scope") == "rows"
        }
        if (raw / "gsm8k/train.jsonl").exists():
            for r in read_jsonl(raw / "gsm8k/train.jsonl"):
                self.maps["gsm8k"][squash(r["question"])] = r["question"]
        m2 = raw / "m2"
        try:
            import csv

            import pyarrow.parquet as pq

            for path in sorted(m2.glob("winogrande/winogrande_xl/train-*.parquet")):
                for s in (
                    pq.read_table(path, columns=["sentence"])
                    .column("sentence")
                    .to_pylist()
                ):
                    self.maps["winogrande"][squash(s)] = s
            if (m2 / "urls/train.csv").exists():
                with open(
                    m2 / "urls/train.csv", newline="", encoding="utf-8"
                ) as stream:
                    for r in csv.DictReader(stream):
                        self.maps["banking77"][squash(r["text"])] = r["text"]
            if (m2 / "urls/hover_train_release_v1.1.json").exists():
                for r in json.loads(
                    (m2 / "urls/hover_train_release_v1.1.json").read_text()
                ):
                    self.maps["hover"][squash(r["claim"])] = str(r["uid"])
            if (m2 / "urls/train.En.csv").exists():
                with open(
                    m2 / "urls/train.En.csv", newline="", encoding="utf-8"
                ) as stream:
                    for r in csv.DictReader(stream):
                        self.maps["isarcasm-en"][squash(r["tweet"])] = r["tweet"]
                        if r.get("rephrase"):
                            self.maps["isarcasm-en"].setdefault(
                                squash(r["rephrase"]), r["tweet"]
                            )
            for path in sorted(m2.glob("when2call/train/*.jsonl")):
                for r in read_jsonl(path):
                    user = next(
                        (m["content"] for m in r["messages"] if m["role"] == "user"),
                        None,
                    )
                    if user:
                        self.maps["when2call"][squash(user)] = user
        except FileNotFoundError:
            pass

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.maps.items()}

    @staticmethod
    def candidates(row: dict[str, Any]) -> list[str]:
        out = []
        state = row.get("state")

        def walk(value: Any) -> None:
            if isinstance(value, str):
                out.append(value)
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)

        walk(state)
        instructions = row["question"].get("instructions") or ""
        out.append(instructions)
        for marker in ("Claim:", "Question:", "claim:", "question:"):
            if marker in instructions:
                out.append(instructions.split(marker, 1)[1].strip().strip('"'))
        return out

    def reason(self, name: str, row: dict[str, Any]) -> str | None:
        table = self.maps.get(name)
        spec = self.slices.get(name)
        if not table or not spec:
            return f"{name}:no_key"
        for cand in self.candidates(row):
            key = table.get(squash(cand))
            if key is not None:
                return (
                    f"{name}:slice"
                    if selected(spec["name"], key, int(spec["mod"]), int(spec["keep"]))
                    else None
                )
        return f"{name}:no_key"


def rows_files(names: list[str], rowset: str) -> list[str]:
    return [str(DATA_ROOT / f"rows/{rowset}/{n}.jsonl.gz") for n in names]


MIXES: dict[str, dict[str, Any]] = {
    "M1": {
        "description": "pplx-like: tasksource filtered-full at Perplexity v1.1's per-source counts (530,103 rows incl. "
        "13 procedural configs) + ~96k Decision 2.0 original-style rows (A0s-strict, IB1-IB4, PN1, "
        "fill from XL r2 pools and HS1). Gold/soft targets as provided.",
        "seed": 20261010,
        "parts": [
            {
                "name": "tasksource",
                "rowset": "tasksource",
                "files": "rows/tasksource/part-*.jsonl.gz",
                "quota": "pplx",
            },
            {"name": "d20-core", "rowset": "d20", "files": D20_CORE, "quota": "all"},
            {
                "name": "d20-fill",
                "rowset": "d20",
                "files": D20_FILL,
                "quota": {"total_with": ["d20-core"], "rows": 95_930},
                "alloc": "sqrt",
            },
        ],
    },
    "M2": {
        "description": "broad: tasksource filtered-full at 1.35x Perplexity's per-source counts (capped by availability) + "
        "Decision 2.0 corpora (A0s-strict, IB1-IB4, PN1 in full; XL r2 pools + HS1 sqrt-allocated to 300k "
        "2.0 rows; sliced 2.0 rows recovered with exact upstream keys) + knowledge MCQ + in-distribution "
        "TRAIN splits rendered in suite style + tagged augmentation rows. Gold/soft targets as provided.",
        "seed": 20261011,
        "parts": [
            {
                "name": "tasksource",
                "rowset": "tasksource",
                "files": "rows/tasksource/part-*.jsonl.gz",
                "quota": "pplx_x",
                "factor": 1.35,
            },
            {"name": "d20-core", "rowset": "d20", "files": D20_CORE, "quota": "all"},
            {
                "name": "d20-fill",
                "rowset": "d20",
                "files": D20_FILL,
                "quota": {"total_with": ["d20-core"], "rows": 300_000},
                "alloc": "sqrt",
            },
            {
                "name": "knowledge",
                "rowset": "knowledge",
                "files": "rows/knowledge/*.jsonl.gz",
                "quota": "all",
            },
            {
                "name": "indist",
                "rowset": "indist",
                "files": "rows/indist/*.jsonl.gz",
                "quota": "all",
            },
            {
                "name": "aug",
                "rowset": "aug",
                "files": "rows/aug/*.jsonl.gz",
                "quota": "all",
            },
        ],
    },
}


def content_key(row: dict[str, Any]) -> str:
    question = row["question"]
    _, texts = df.options(question)
    payload = json.dumps(
        [
            df.describe(row.get("state")),
            question["type"],
            question.get("instructions"),
            texts,
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def d20_holdout_reason(row: dict[str, Any], upstream: UpstreamKeys) -> str | None:
    meta = row["meta"]
    name = D20_SLICED.get(meta.get("orig_source") or "") or D20_SLICED_FAMILIES.get(
        row["family"]
    )
    if name is None:
        return None
    return upstream.reason(name, row)


def load_flags(
    index_dir: Path, rowsets: set[str]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    dropped: dict[str, list[str]] = {}
    for rowset in sorted(rowsets):
        path = index_dir / f"flags-{rowset}.jsonl.gz"
        for flag in read_jsonl(path):
            if flag["drop"]:
                dropped[flag["id"]] = flag["reasons"] + (
                    [f"bench:{flag['bench']}"] if flag.get("bench") else []
                )
    return dropped, json.loads((index_dir / "rule.json").read_text())


def pick_groups(
    rows: list[dict[str, Any]], quota: int, seed: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["meta"].get("group") or row["id"]].append(row)
    chosen: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda g: rank(g, seed)):
        if len(chosen) >= quota:
            break
        chosen.extend(groups[group])
    return chosen


def allocate(sizes: dict[str, int], total: int, mode: str) -> dict[str, int]:
    weights = {
        k: (math.sqrt(v) if mode == "sqrt" else float(v))
        for k, v in sizes.items()
        if v > 0
    }
    alloc = {k: 0 for k in sizes}
    remaining = total
    pending = dict(weights)
    while remaining > 0 and pending:
        scale = remaining / sum(pending.values())
        capped = {
            k: min(sizes[k] - alloc[k], int(round(w * scale)))
            for k, w in pending.items()
        }
        given = sum(capped.values())
        for k, v in capped.items():
            alloc[k] += v
        remaining -= given
        pending = {k: w for k, w in pending.items() if alloc[k] < sizes[k]}
        if given == 0:
            break
    return alloc


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "rows": len(values),
        "total": int(sum(values)),
        "mean": round(statistics.fmean(values), 1),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def licence_bucket(meta: dict[str, Any]) -> str:
    text = (meta.get("licence") or "unknown").lower()
    for key in (
        "apache",
        "mit",
        "cc0",
        "cc-by-sa",
        "cc by-sa",
        "cc-by",
        "cc by",
        "bsd",
        "gpl",
        "odc",
        "afl",
        "mpl",
        "d20-permissive",
    ):
        if key in text:
            return {"cc by-sa": "cc-by-sa", "cc by": "cc-by"}.get(key, key)
    return text[:40]


def composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {
        name: Counter()
        for name in (
            "source",
            "family",
            "type",
            "orig_kind",
            "licence",
            "licence_raw",
            "dataset",
            "part",
            "aug",
        )
    }
    soft = 0
    noul_pos = []
    for row in rows:
        meta = row["meta"]
        by["source"][row["source"]] += 1
        by["family"][row["family"]] += 1
        by["type"][row["question"]["type"]] += 1
        by["orig_kind"][meta.get("orig_kind") or row["question"]["type"]] += 1
        by["licence"][licence_bucket(meta)] += 1
        by["licence_raw"][meta.get("licence") or "unknown"] += 1
        by["dataset"][meta.get("dataset") or "?"] += 1
        by["part"][meta.get("part") or "?"] += 1
        by["aug"][meta.get("aug") or "none"] += 1
        soft += bool(meta.get("soft"))
        if row["question"]["type"] == "noul":
            noul_pos.append(row["target"][1])
    out = {k: dict(v.most_common()) for k, v in by.items()}
    out["soft_target_rows"] = soft
    out["noul_mean_p_true"] = round(statistics.fmean(noul_pos), 4) if noul_pos else None
    out["noul_share_true"] = (
        round(sum(p >= 0.5 for p in noul_pos) / len(noul_pos), 4) if noul_pos else None
    )
    return out


def build(
    mix: str, index_dir: Path, holdouts_path: Path, out: Path, code_tag: str
) -> dict[str, Any]:
    started = time.time()
    spec = MIXES[mix]
    seed = spec["seed"]
    holdouts = Holdouts(holdouts_path)
    upstream = UpstreamKeys(holdouts)
    print("upstream slice keys:", upstream.counts(), flush=True)
    rowsets = {part["rowset"] for part in spec["parts"]}
    dropped, rule = load_flags(index_dir, rowsets)
    pplx = json.loads(PPLX_MANIFEST.read_text())
    filters: dict[str, Counter] = defaultdict(Counter)
    decon_by_source: dict[str, Counter] = defaultdict(Counter)
    decon_by_bench: Counter = Counter()
    part_rows: dict[str, list[dict[str, Any]]] = {}
    for part in spec["parts"]:
        if isinstance(part["files"], str):
            files = sorted(
                p
                for p in glob.glob(str(DATA_ROOT / part["files"]))
                if not p.endswith(".ntok.jsonl.gz")
            )
        else:
            files = rows_files(part["files"], part["rowset"])
        lengths = load_lengths(files)
        kept = []
        for path in files:
            for row in read_jsonl(path):
                f = filters[part["name"]]
                f["candidates"] += 1
                decon_by_source[row["source"]]["checked"] += 1
                reasons = dropped.get(row["id"])
                if reasons:
                    f["decontam"] += 1
                    decon_by_source[row["source"]]["dropped"] += 1
                    for reason in reasons:
                        if reason.startswith("bench:"):
                            decon_by_bench[reason[6:]] += 1
                        else:
                            decon_by_source[row["source"]][reason] += 1
                    continue
                reason = holdouts.check(row)
                if reason is None and row["source"].startswith("d20:"):
                    reason = d20_holdout_reason(row, upstream)
                if reason:
                    f["holdouts"] += 1
                    holdouts.counts[reason] += 1
                    continue
                n = lengths.get(row["id"], -1)
                if n < 0 or n > MAX_PROMPT_TOKENS:
                    f["overlong_or_unrenderable"] += 1
                    continue
                row["meta"]["n_tokens"] = n
                row["meta"]["part"] = part["name"]
                kept.append(row)
        part_rows[part["name"]] = kept
        filters[part["name"]]["eligible"] = len(kept)
        print(part["name"], dict(filters[part["name"]]), flush=True)
    selected_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for part in spec["parts"]:
        rows = part_rows[part["name"]]
        quota = part["quota"]
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_source[row["source"]].append(row)
        chosen: list[dict[str, Any]] = []
        if quota == "all":
            chosen = rows
        elif quota in ("pplx", "pplx_x"):
            short = {}
            factor = float(part.get("factor", 1.0)) if quota == "pplx_x" else 1.0
            for source, base in pplx.items():
                target = int(math.ceil(base * factor))
                pool = by_source.get(f"tasksource:{source}", [])
                got = pick_groups(pool, target, f"{seed}:{source}")
                if len(got) < target:
                    short[source] = target - len(got)
                chosen.extend(got)
            selection[part["name"]] = {
                "target": sum(int(math.ceil(v * factor)) for v in pplx.values()),
                "factor": factor,
                "shortfall_by_source": short,
                "shortfall_total": sum(short.values()),
            }
        elif isinstance(quota, dict):
            have = sum(
                len(selection.get(name, {}).get("_rows", []))
                for name in quota.get("total_with", [])
            )
            total = max(0, quota["rows"] - have)
            alloc = allocate(
                {s: len(v) for s, v in by_source.items()},
                total,
                part.get("alloc", "proportional"),
            )
            for source, n in alloc.items():
                chosen.extend(pick_groups(by_source[source], n, f"{seed}:{source}"))
            selection[part["name"]] = {"target": total, "alloc": alloc}
        selection.setdefault(part["name"], {})["_rows"] = chosen
        selected_rows.extend(chosen)
    for value in selection.values():
        value["rows"] = len(value.pop("_rows"))
    # exact de-duplication across parts (first occurrence wins)
    seen: set[str] = set()
    seen_ids: set[str] = set()
    unique = []
    duplicates = Counter()
    for row in selected_rows:
        key = content_key(row)
        if key in seen:
            duplicates[row["meta"]["part"]] += 1
            continue
        if row["id"] in seen_ids:
            duplicates[f"{row['meta']['part']}:duplicate_id"] += 1
            continue
        seen.add(key)
        seen_ids.add(row["id"])
        unique.append(row)
    # group-disjoint dev split
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        groups[row["meta"].get("group") or row["id"]].append(row)
    dev: list[dict[str, Any]] = []
    dev_groups: set[str] = set()
    for group in sorted(groups, key=lambda g: rank(g, f"{seed}:dev")):
        if len(dev) >= DEV_ROWS:
            break
        if len(groups[group]) > 50:
            continue
        dev.extend(groups[group])
        dev_groups.add(group)
    train = [
        row
        for row in unique
        if (row["meta"].get("group") or row["id"]) not in dev_groups
    ]
    train.sort(key=lambda r: rank(r["id"], seed))
    dev.sort(key=lambda r: rank(r["id"], seed))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("train-*.jsonl.gz"):
        old.unlink()
    shards = max(1, math.ceil(len(train) / SHARD_ROWS))
    files: dict[str, Any] = {}
    for i in range(shards):
        name = f"train-{i:05d}-of-{shards:05d}.jsonl.gz"
        chunk = train[i * SHARD_ROWS : (i + 1) * SHARD_ROWS]
        write_jsonl(out / name, chunk)
        files[name] = {
            "rows": len(chunk),
            "sha256": sha256_file(out / name),
            "bytes": (out / name).stat().st_size,
        }
    write_jsonl(out / "dev.jsonl.gz", dev)
    files["dev.jsonl.gz"] = {
        "rows": len(dev),
        "sha256": sha256_file(out / "dev.jsonl.gz"),
        "bytes": (out / "dev.jsonl.gz").stat().st_size,
    }
    index_meta = json.loads((index_dir / "meta.json").read_text())
    manifest = {
        "mix": mix,
        "corpus": "v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "description": spec["description"],
        "format": {
            "row_contract": "d25.vega.common.decision_format (one question per row; target over options() order; "
            "noul = [p_false, p_true])",
            "format_id": df.FORMAT_ID,
            "prompt_for_token_stats": "d25-vega",
            "tokenizer": "Qwen/Qwen3.8-27B@1d4bf0f2 (local copy)",
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
        },
        "target_rule": "gold or source soft targets as provided (tasksource soft votes/ratings, procedural exact posteriors, "
        "Decision 2.0 gold one-hot); no teacher targets",
        "seed": seed,
        "rows": {"train": len(train), "dev": len(dev)},
        "files": files,
        "selection": selection,
        "filters": {k: dict(v) for k, v in filters.items()},
        "duplicates_removed": dict(duplicates),
        "dev_rule": f"~{DEV_ROWS} rows, whole groups (<=50 rows), seeded hash order; disjoint from train",
        "composition": {"train": composition(train), "dev": composition(dev)},
        "tokens": {
            "train": stats([r["meta"]["n_tokens"] for r in train]),
            "dev": stats([r["meta"]["n_tokens"] for r in dev]),
            "train_by_part": {
                p: stats(
                    [r["meta"]["n_tokens"] for r in train if r["meta"]["part"] == p]
                )
                for p in sorted({r["meta"]["part"] for r in train})
            },
        },
        "decontamination": {
            "index": str(index_dir),
            "suite_files": index_meta["suite_files"],
            "suite_items": index_meta["stats"]["items"],
            "benchmarks": index_meta["benchmarks"],
            "index_stats": index_meta["stats"],
            "rule": {k: v for k, v in rule.items() if k != "grid"},
            "by_source": {
                k: dict(v)
                for k, v in sorted(decon_by_source.items())
                if v.get("dropped")
            },
            "by_benchmark": dict(decon_by_bench.most_common()),
            "checked_rows": sum(v["checked"] for v in decon_by_source.values()),
            "dropped_rows": sum(v.get("dropped", 0) for v in decon_by_source.values()),
        },
        "holdouts": holdouts.report(),
        "inputs": {
            "tasksource": "tasksource/tasksource-jev-typed-decisions@5c3e4ebb (filtered-full)",
            "procedural": "tasksource/procedural-typed-decisions@609513a3",
            "decision20": "vllm-sr/decision-2.0-training-data@47855751 (XL r2 mx-xl-full-r2 + IB1-IB4 + HS1 + PN1; HR2 excluded)",
            "pplx_manifest": "perplexity-ai/pplx-decider-v1.1-27b training/data-manifest.json source_rows",
        },
        "row_set_reports": {
            rs: json.loads((DATA_ROOT / f"rows/{rs}/report.json").read_text())
            for rs in sorted(rowsets)
            if (DATA_ROOT / f"rows/{rs}/report.json").exists()
        },
        "code": {
            "tag": code_tag,
            "modules": {
                p.name: sha256_file(p)
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
        "seconds": round(time.time() - started, 1),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def refilter(
    src: Path, flags: Path, holdouts_path: Path, out: Path, code_tag: str, revision: str
) -> dict[str, Any]:
    """Drop rows of a built mixture that a newer decontamination index or holdouts file flags; everything else
    (shard names, row order, ids, targets) is unchanged, so teacher labels keyed by id stay valid.
    """
    base = json.loads((src / "manifest.json").read_text())
    dropped = {f["id"]: f for f in read_jsonl(flags) if f["drop"]}
    holdouts = Holdouts(holdouts_path)
    upstream = UpstreamKeys(holdouts)
    removed: Counter = Counter()
    by_bench: Counter = Counter()
    out.mkdir(parents=True, exist_ok=True)
    files, kept_rows = {}, {"train": [], "dev": []}
    for name in base["files"]:
        rows = []
        for row in read_jsonl(src / name):
            flag = dropped.get(row["id"])
            reason = None
            if flag:
                reason = "decontam:" + "+".join(flag["reasons"])
                by_bench[flag.get("bench") or "?"] += 1
            else:
                reason = holdouts.check(row)
                if reason is None and row["source"].startswith("d20:"):
                    reason = d20_holdout_reason(row, upstream)
                if reason:
                    reason = "holdouts:" + reason
            if reason:
                removed[f"{row['meta'].get('part')}|{reason}"] += 1
                continue
            rows.append(row)
        write_jsonl(out / name, rows)
        files[name] = {
            "rows": len(rows),
            "sha256": sha256_file(out / name),
            "bytes": (out / name).stat().st_size,
        }
        kept_rows["dev" if name == "dev.jsonl.gz" else "train"].extend(rows)
    manifest = dict(base)
    manifest.update(
        {
            "revision": revision,
            "refiltered_from": {
                "files": {k: v["sha256"] for k, v in base["files"].items()},
                "rows": base["rows"],
            },
            "rows": {"train": len(kept_rows["train"]), "dev": len(kept_rows["dev"])},
            "files": files,
            "refilter": {
                "flags": str(flags),
                "holdouts": holdouts.report(),
                "removed": dict(removed.most_common()),
                "removed_by_benchmark": dict(by_bench.most_common()),
                "code_tag": code_tag,
            },
            "composition": {
                "train": composition(kept_rows["train"]),
                "dev": composition(kept_rows["dev"]),
            },
            "tokens": {
                "train": stats([r["meta"]["n_tokens"] for r in kept_rows["train"]]),
                "dev": stats([r["meta"]["n_tokens"] for r in kept_rows["dev"]]),
            },
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=sorted(MIXES))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--holdouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-tag", default=os.environ.get("D25_CODE_TAG", "unknown"))
    parser.add_argument(
        "--refilter-src",
        type=Path,
        help="existing mixture dir to refilter instead of building",
    )
    parser.add_argument(
        "--flags", type=Path, help="decontam flags for the refiltered mixture rows"
    )
    parser.add_argument("--revision", default="v1.1")
    args = parser.parse_args()
    if args.refilter_src:
        manifest = refilter(
            args.refilter_src,
            args.flags,
            args.holdouts,
            args.out,
            args.code_tag,
            args.revision,
        )
        print(
            json.dumps({k: manifest[k] for k in ("rows", "refilter")}, indent=1)[:4000]
        )
        return
    manifest = build(args.mix, args.index, args.holdouts, args.out, args.code_tag)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "rows",
                    "selection",
                    "filters",
                    "duplicates_removed",
                    "tokens",
                )
            },
            indent=1,
        )[:6000]
    )
    print(json.dumps(manifest["holdouts"], indent=1)[:2000])


if __name__ == "__main__":
    main()
