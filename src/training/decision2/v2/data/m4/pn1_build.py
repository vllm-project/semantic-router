"""PN1 CPU build steps (prereg ``v2/data/records/m4-pn1-prereg-2026-09-29.md``).

    python3 -m v2.data.m4.pn1_build candidates --source-dir SRC --work-dir WORK [--workers N]
    python3 -m v2.data.m4.pn1_build judgeset --work-dir WORK --gen-dir DIR [--gen-dir DIR ...]
    python3 -m v2.data.m4.pn1_build judgeset-extend --work-dir WORK --judge-dir DIR ... [--dry-run]
    python3 -m v2.data.m4.pn1_build probe-items --work-dir WORK
    python3 -m v2.data.m4.pn1_build probe-eval --judge-dir DIR --output FILE
    python3 -m v2.data.m4.pn1_build judgeset-rejudge --work-dir WORK [--prompt-version v2]
    python3 -m v2.data.m4.pn1_build finalize --work-dir WORK --source-dir SRC --judge-dir DIR ... \
        --gpu-time-dir DIR --output-root ROOT [--judgeset judgeset2] \
        [--drop-groups FILE --expect-selection SHA256]

``candidates`` builds name swaps, twin seeds, hop and near pairs (``pn1_source``).
``judgeset`` turns generations into twin rows, stratifies the pool and picks the
rows the judge labels: per language, stratum and label the first
``JUDGE_FACTOR`` x planned rows in seed-hash order. ``finalize`` applies the
judge filters, draws dev first, blocks TRAIN candidates touching a dev
neighbourhood or near-duplicating a dev sentence, balances TRAIN and writes the
outputs. ``judgeset-extend`` (amendment 2) adds unjudged rows per language,
stratum and label, in seed-hash order, until the rows kept so far plus the
expected keeps at the measured keep rates reach 1.25 x the planned units.
Amendment 3: ``probe-items`` / ``probe-eval`` run the control probe and
``judgeset-rejudge`` judges the whole pool again with the v2 prompts.

Balance (amendment 3): within language x group (natural = hop + near, swap =
name + twin) x same-multiset flag, yes and no rows are matched 1:1 (greedy
nearest neighbour on bigram Jaccard, multiset Jaccard and length ratio,
caliper 0.03 on each; no rows in ``sha256("pn1-seed:" + group key)`` order);
unmatched rows are dropped. A group's pairs are apportioned over its two
strata in proportion to their pair counts (largest remainder), first pairs
first. Dev draws matched pairs in ``sha256("pn1-dev:" + no-row group key)``
order, with stratum units and family-pair quotas proportional to the planned
TRAIN selection. The bigram bins stay as reported strata only.

``--drop-groups`` recomputes the same selection, removes the listed groups and
re-matches the remaining rows of each (language, split), keeping every matched
pair, so the result is a subset of the audited rows (an empty list reproduces
the selection).
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import multiprocessing
import os
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from training.model.data import (
    INPUT_FIELDS,
    canonical,
    digest,
    file_sha256,
    validate_row,
)
from training.model.infer import question_to_row
from v2.data.m4 import pn1_source as source
from v2.data.m4.pn1_text import (
    FAMILIES,
    GROUP_OF,
    INSTRUCTIONS,
    LANGS,
    LAYOUTS,
    PROMPTS,
    extra_metrics,
    grams,
    near_duplicate,
    norm,
    pair_metrics,
    parse_twin,
    stratum,
    twin_checks,
)
from v2.data.sources.common import noul_options, sha

SCHEMA = "dev2-m4-pn1-build/1"
PREREG = "v2/data/records/m4-pn1-prereg-2026-09-29.md"
REGISTRY = "v2/data/records/license-registry-m4.json"
TRAIN_TARGETS = {
    "ja": 4400,
    "es": 3200,
    "ko": 2800,
    "zh": 2800,
    "de": 2000,
    "fr": 2000,
    "ru": 1400,
    "ar": 1400,
}
DEV_QUOTAS = {
    "de": 300,
    "es": 300,
    "fr": 300,
    "ja": 300,
    "ko": 300,
    "zh": 300,
    "ru": 100,
    "ar": 100,
}
SWAP_MIN_SHARE = {
    lang: 0.5 if lang in ("ja", "ko", "zh", "es") else 0.4 for lang in LANGS
}
JUDGE_FACTOR = 1.6
MIX_TOLERANCE = 0.10
DEV_PASSES = 3
LONG_INPUT_CHARS = 4000
FEATURES = ("bigram_jaccard", "multiset_jaccard", "same_multiset", "length_ratio")
FEATURES_EXTENDED = FEATURES + (
    "word_jaccard",
    "edit_distance",
    "containment_ab",
    "containment_ba",
)
MATCH_COORDINATES = ("bigram_jaccard", "multiset_jaccard", "length_ratio")
CALIPER = 0.03


# --- small helpers ----------------------------------------------------------------------------


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl_bytes(records: Iterable[Any]) -> bytes:
    return "".join(canonical(record) + "\n" for record in records).encode("utf-8")


def write_new(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return file_sha256(path)


def write_json(path: Path, value: Any) -> str:
    return write_new(
        path,
        (
            json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode(),
    )


def code_identity() -> dict[str, str]:
    for parent in Path(__file__).resolve().parents:
        receipt = parent / ".dev2-mirror.json"
        if receipt.is_file():
            data = json.loads(receipt.read_text(encoding="utf-8"))
            return {"commit": data["commit"], "tree": data["tree"], "origin": "mirror"}
    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "tree": tree, "origin": "git (not a mirror)"}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "tree": "unknown", "origin": "unknown"}


def verify_source(root: Path, names: Sequence[str]) -> dict[str, Any]:
    """The download manifest, with every used file re-hashed against it."""
    manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    files = source.export_files()
    for name in names:
        rel = files[name]
        entry = manifest["files"][rel]
        if file_sha256(root / rel) != entry["sha256"]:
            raise ValueError(f"{rel}: SHA-256 differs from source-manifest.json")
    return manifest


def seed_key(record: dict[str, Any]) -> tuple[str, str]:
    return sha(f"pn1-seed:{record['group_key']}"), record["cid"]


def dev_key(record: dict[str, Any]) -> tuple[str, str]:
    return sha(f"pn1-dev:{record['group_key']}"), record["cid"]


def group_id(group_key: str) -> str:
    return f"m4pn1:tatoeba:{sha(group_key)[:24]}"


def source_key(export_date: str, record: dict[str, Any]) -> str:
    return f"tatoeba-{export_date}" + ("-edited" if any(record["edited"]) else "")


def row_id(export_date: str, record: dict[str, Any]) -> str:
    return f"m4pn1-{record['family']}-{sha(source_key(export_date, record) + '|' + record['cid'])[:24]}"


def pick(rid: str, salt: str, n: int) -> int:
    return int(sha(f"pn1-{salt}:{rid}"), 16) % n


def group_targets(
    lang: str, total: int, swap_share: float | None = None
) -> dict[str, int]:
    share = SWAP_MIN_SHARE[lang] if swap_share is None else swap_share
    swap = (
        2 * math.ceil(share * total / 2 - 1e-9)
        if swap_share is None
        else 2 * round(share * total / 2)
    )
    return {"natural": total - swap, "swap": swap}


def natural_units(lang: str) -> int:
    return (
        group_targets(lang, TRAIN_TARGETS[lang])["natural"]
        + group_targets(lang, DEV_QUOTAS[lang])["natural"]
    ) // 2


def allocate(
    total: int, weights: dict[Any, float], caps: dict[Any, int]
) -> dict[Any, int]:
    """Largest-remainder apportionment of ``total`` in proportion to ``weights``, capped."""
    alloc = dict.fromkeys(weights, 0)
    while True:
        room = {
            k: caps.get(k, 0) - alloc[k]
            for k in weights
            if weights[k] > 0 and caps.get(k, 0) > alloc[k]
        }
        left = total - sum(alloc.values())
        if left <= 0 or not room:
            return alloc
        mass = sum(weights[k] for k in room)
        exact = {k: left * weights[k] / mass for k in room}
        grant = {k: min(room[k], math.floor(exact[k])) for k in room}
        given = sum(grant.values())
        for k in sorted(
            room, key=lambda k: (-(exact[k] - math.floor(exact[k])), str(k))
        ):
            if given >= left:
                break
            if grant[k] < room[k]:
                grant[k] += 1
                given += 1
        for k, value in grant.items():
            alloc[k] += value


# --- candidates -------------------------------------------------------------------------------

_CORPUS: source.Corpus | None = None


def _build_language(lang: str) -> dict[str, Any]:
    corpus = _CORPUS
    started = time.perf_counter()
    texts = corpus.texts[lang]
    names = source.name_candidates(lang, texts, corpus)
    used = {sid for record in names for sid in record["ids"]}
    seeds = source.twin_seeds(lang, texts, used, source.PLANNED_SEEDS[lang], corpus)
    used |= {seed["seed"] for seed in seeds}
    hops, hop_stats = source.hop_pairs(
        lang, texts, corpus.graph, corpus.lang_of, used, natural_units(lang), corpus
    )
    used |= {sid for record in hops for sid in record["ids"]}
    edges = source.length_edges(hops)
    quotas = source.near_quotas(hops, edges)
    nears, near_stats = source.near_pairs(
        lang,
        texts,
        corpus.graph,
        used,
        corpus.english_norms,
        quotas,
        edges,
        corpus=corpus,
    )
    kinds = collections.Counter(record["kind"] for record in names)
    return {
        "language": lang,
        "records": names + hops + nears,
        "seeds": seeds,
        "stats": {
            "sentences": len(texts),
            "name_candidates": dict(sorted(kinds.items())),
            "twin_seeds": len(seeds),
            "twin_seeds_planned": source.PLANNED_SEEDS[lang],
            "twin_eligible_note": "length rule, not a name candidate, sha256('pn1-twin:'+id) order",
            "hop": hop_stats,
            "near": near_stats,
            "length_edges": edges,
            "seconds": round(time.perf_counter() - started, 1),
        },
    }


def cmd_candidates(args: argparse.Namespace) -> int:
    global _CORPUS
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.perf_counter()
    manifest = verify_source(args.source_dir, list(source.export_files()))
    _CORPUS = source.Corpus.load(args.source_dir)
    loaded = time.perf_counter() - started
    with multiprocessing.get_context("fork").Pool(
        min(args.workers, len(LANGS))
    ) as pool:
        results = pool.map(_build_language, LANGS, chunksize=1)
    records = sorted(
        (r for result in results for r in result["records"]),
        key=lambda r: (r["language"], r["family"], r["cid"]),
    )
    seeds = sorted(
        (s for result in results for s in result["seeds"]),
        key=lambda s: (s["language"], s["rank"]),
    )
    receipt = {
        "schema": SCHEMA + "/candidates",
        "created_utc": utc(),
        "code": code_identity(),
        "export_date": manifest["export_date"],
        "source_manifest_sha256": file_sha256(args.source_dir / "source-manifest.json"),
        "corpus": {
            "sentences": {lang: len(table) for lang, table in _CORPUS.texts.items()},
            "links_undirected": _CORPUS.graph.edges,
            "cc0_ids": len(_CORPUS.cc0),
            "load_seconds": round(loaded, 1),
        },
        "per_language": {result["language"]: result["stats"] for result in results},
        "counts": {
            "records": dict(
                sorted(
                    collections.Counter(
                        f"{r['language']}|{r['family']}|{r['label']}" for r in records
                    ).items()
                )
            ),
            "seeds": dict(
                sorted(collections.Counter(s["language"] for s in seeds).items())
            ),
        },
        "files_sha256": {
            "candidates.jsonl": write_new(
                work / "candidates.jsonl", jsonl_bytes(records)
            ),
            "seeds.jsonl": write_new(work / "seeds.jsonl", jsonl_bytes(seeds)),
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    write_json(work / "candidates.receipt.json", receipt)
    print(
        json.dumps(
            {"records": len(records), "seeds": len(seeds), **receipt["files_sha256"]}
        )
    )
    return 0


# --- judge set --------------------------------------------------------------------------------


def load_generations(
    gen_dirs: Sequence[Path],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    outputs: dict[int, dict[str, Any]] = {}
    receipts = []
    for directory in gen_dirs:
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        path = directory / "generations.jsonl"
        if file_sha256(path) != receipt["outputs_sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from its receipt")
        for record in read_jsonl(path):
            if record["seed"] in outputs:
                raise ValueError(f"seed {record['seed']} generated twice")
            outputs[record["seed"]] = record
        receipts.append(
            {
                "dir": directory.name,
                **{
                    k: receipt[k]
                    for k in ("model", "outputs_sha256", "counts", "timing")
                },
            }
        )
    return outputs, receipts


def twin_records(
    seeds: list[dict[str, Any]], outputs: dict[int, dict[str, Any]], editor: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, receipts = [], []
    for seed in seeds:
        sid, lang, text = seed["seed"], seed["language"], seed["text"]
        output = outputs.get(sid)
        status: dict[str, Any] = {"seed": sid, "language": lang}
        if output is None:
            receipts.append({**status, "status": "not_generated"})
            continue
        parsed = parse_twin(output["raw"])
        status.update(new_tokens=output["new_tokens"], finished=output["finished"])
        if parsed is None:
            receipts.append({**status, "status": "unparsable"})
            continue
        reasons = twin_checks(text, parsed, lang)
        receipts.append({**status, "status": "parsed", "reasons": reasons})
        for kind, reason in reasons.items():
            if reason is not None:
                continue
            records.append(
                source.candidate(
                    None,
                    family="pn-twin",
                    lang=lang,
                    cid=f"twin:{sid}:{kind}",
                    group_key=f"seed:{sid}",
                    ids=[sid],
                    texts=[text, parsed[kind]],
                    edited=[False, True],
                    label=1 if kind == "same" else 0,
                    extra={
                        "kind": kind,
                        "editor": editor,
                        "attribution": seed["attribution"],
                    },
                )
            )
    return records, receipts


def with_strata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record["stratum"] = stratum(record["family"], record["metrics"])
        record["group"] = GROUP_OF[record["family"]]
    return records


def cells_of(
    records: Iterable[dict[str, Any]], key
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    cells: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        cells[(record["stratum"], record["label"])].append(record)
    for cell in cells.values():
        cell.sort(key=key)
    return cells


def balance(
    records: list[dict[str, Any]], targets: dict[str, int], key=seed_key
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per group: units over strata by largest remainder on min(yes, no); first rows in key order."""
    cells = cells_of(records, key)
    strata = sorted({s for s, _ in cells})
    chosen: list[dict[str, Any]] = []
    units: dict[str, int] = {}
    for group in ("natural", "swap"):
        avail = {
            s: min(len(cells.get((s, 0), [])), len(cells.get((s, 1), [])))
            for s in strata
            if s.startswith(group + "|")
        }
        for s, n in allocate(targets[group] // 2, avail, avail).items():
            units[s] = n
            chosen += cells[(s, 1)][:n] + cells[(s, 0)][:n]
    return chosen, units


def plan_units(records: list[dict[str, Any]], lang: str) -> dict[str, int]:
    targets = group_targets(lang, TRAIN_TARGETS[lang])
    dev = group_targets(lang, DEV_QUOTAS[lang])
    _, units = balance(records, {g: targets[g] + dev[g] for g in targets})
    return units


def judge_selection(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen, info = [], {}
    by_lang = collections.defaultdict(list)
    for record in records:
        by_lang[record["language"]].append(record)
    for lang in LANGS:
        pool = by_lang.get(lang, [])
        units = plan_units(pool, lang)
        cells = cells_of(pool, seed_key)
        planned = {}
        for (s, label), cell in sorted(cells.items()):
            need = units.get(s, 0)
            take = min(len(cell), math.ceil(JUDGE_FACTOR * need) + (2 if need else 0))
            for index, record in enumerate(cell[:take]):
                record["rank"] = round(index / max(need, 1), 6)
                chosen.append(record)
            planned[f"{s}|y{label}"] = {
                "available": len(cell),
                "planned_units": need,
                "judged": take,
            }
        info[lang] = planned
    return chosen, info


def judge_items(record: dict[str, Any], version: str = "v1") -> list[dict[str, Any]]:
    lang, texts = record["language"], record["texts"]
    label_prompt_, fluency_prompt_ = PROMPTS[version]
    order = "ab" if pick(record["cid"], "judge-order", 2) == 0 else "ba"
    a, b = texts if order == "ab" else texts[::-1]
    items = [
        {
            "item": f"{record['cid']}#label",
            "cid": record["cid"],
            "kind": "label",
            "language": lang,
            "order": order,
            "rank": record["rank"],
            "prompt": label_prompt_(lang, a, b),
        }
    ]
    for index, edited in enumerate(record["edited"]):
        if edited:
            items.append(
                {
                    "item": f"{record['cid']}#fluency",
                    "cid": record["cid"],
                    "kind": "fluency",
                    "language": lang,
                    "rank": record["rank"],
                    "prompt": fluency_prompt_(lang, texts[index]),
                }
            )
    return items


def assign_gpus(counts: dict[str, int], gpus: Sequence[str]) -> dict[str, str]:
    load = dict.fromkeys(gpus, 0)
    out = {}
    for lang in sorted(counts, key=lambda lang: (-counts[lang], lang)):
        gpu = min(gpus, key=lambda g: (load[g], g))
        out[lang] = gpu
        load[gpu] += counts[lang]
    return out


def cmd_judgeset(args: argparse.Namespace) -> int:
    work = args.work_dir
    out = work / "judgeset"
    out.mkdir(mode=0o700)
    candidates = read_jsonl(work / "candidates.jsonl")
    seeds = read_jsonl(work / "seeds.jsonl")
    outputs, gen_receipts = load_generations(args.gen_dir)
    models = {
        r["model"]["repo_id"] + "@" + r["model"]["revision"] for r in gen_receipts
    }
    if len(models) != 1:
        raise ValueError(f"generations come from {sorted(models)}")
    repo, revision = next(iter(models)).split("@")
    editor = f"{repo.split('/')[-1].lower()}@{revision}"
    twins, twin_status = twin_records(seeds, outputs, editor)
    pool = with_strata(candidates + twins)
    judged, plan = judge_selection(pool)
    judged_ids = {record["cid"] for record in judged}
    items = [
        item
        for record in sorted(judged, key=lambda r: (r["rank"], r["cid"]))
        for item in judge_items(record)
    ]
    per_lang = collections.Counter(item["language"] for item in items)
    gpus = assign_gpus(per_lang, args.gpus.split(","))
    files = {
        "pool.jsonl": write_new(
            out / "pool.jsonl",
            jsonl_bytes(
                dict(r, judged=r["cid"] in judged_ids)
                for r in sorted(pool, key=lambda r: r["cid"])
            ),
        ),
        "twin-status.jsonl": write_new(
            out / "twin-status.jsonl", jsonl_bytes(twin_status)
        ),
    }
    for gpu in sorted(set(gpus.values())):
        name = f"items.{gpu}.jsonl"
        files[name] = write_new(
            out / name, jsonl_bytes(i for i in items if gpus[i["language"]] == gpu)
        )
    status_counts = collections.Counter(
        f"{t['language']}|{t['status']}" for t in twin_status
    )
    reasons = collections.Counter(
        f"{t['language']}|{kind}|{reason or 'kept'}"
        for t in twin_status
        if t["status"] == "parsed"
        for kind, reason in t["reasons"].items()
    )
    receipt = {
        "schema": SCHEMA + "/judgeset",
        "created_utc": utc(),
        "code": code_identity(),
        "inputs_sha256": {
            "candidates.jsonl": file_sha256(work / "candidates.jsonl"),
            "seeds.jsonl": file_sha256(work / "seeds.jsonl"),
        },
        "generation": gen_receipts,
        "editor": editor,
        "twin_status": dict(sorted(status_counts.items())),
        "twin_edit_reasons": dict(sorted(reasons.items())),
        "pool": dict(
            sorted(
                collections.Counter(
                    f"{r['language']}|{r['family']}|{r['label']}" for r in pool
                ).items()
            )
        ),
        "judge_factor": JUDGE_FACTOR,
        "plan": plan,
        "judged_rows": dict(
            sorted(
                collections.Counter(
                    f"{r['language']}|{r['family']}|{r['label']}" for r in judged
                ).items()
            )
        ),
        "items": dict(
            sorted(
                collections.Counter(
                    f"{i['language']}|{i['kind']}" for i in items
                ).items()
            )
        ),
        "gpu_of_language": gpus,
        "files_sha256": files,
    }
    write_json(out / "judgeset.receipt.json", receipt)
    print(
        json.dumps(
            {
                "pool": len(pool),
                "judged_rows": len(judged),
                "items": len(items),
                "gpus": gpus,
            }
        )
    )
    return 0


# --- selection --------------------------------------------------------------------------------


def row_sentence_ids(record: dict[str, Any]) -> set[int]:
    ids = set(record["ids"])
    if record.get("pivot") is not None:
        ids.add(record["pivot"])
    return ids


class Blocklist:
    """Dev neighbourhoods (ids within two link hops, any language) and dev sentence norms."""

    def __init__(self, dev: list[dict[str, Any]], graph: source.LinkGraph) -> None:
        ids = set()
        for record in dev:
            ids |= row_sentence_ids(record)
        self.neighbourhood = graph.within(ids, 2)
        self.norms = sorted({norm(text) for record in dev for text in record["texts"]})
        self.exact = set(self.norms)
        self.index: dict[str, list[int]] = collections.defaultdict(list)
        for position, value in enumerate(self.norms):
            for g in grams(value, 4) if len(value) >= 4 else ():
                self.index[g].append(position)

    def reason(self, record: dict[str, Any]) -> str | None:
        if row_sentence_ids(record) & self.neighbourhood:
            return "dev_neighbourhood"
        for text in record["texts"]:
            value = norm(text)
            if value in self.exact:
                return "dev_norm_equal"
            if len(value) < 4:
                continue
            candidates = {p for g in grams(value, 4) for p in self.index.get(g, ())}
            if any(near_duplicate(value, self.norms[p]) for p in sorted(candidates)):
                return "dev_near_duplicate"
        return None


def family_mix(records: list[dict[str, Any]]) -> dict[str, float]:
    counts = collections.Counter(r["family"] for r in records)
    total = max(len(records), 1)
    return {family: round(counts[family] / total, 6) for family in FAMILIES}


def match_stratum(record: dict[str, Any]) -> str:
    return f"{record['group']}|ms{int(bool(record['metrics']['same_multiset']))}"


def match_pairs(
    records: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any], float]]]:
    """Greedy 1:1 nearest-neighbour matching of yes to no rows (amendment 3).

    Per construction group x same-multiset flag (the caller passes one language), no
    rows are taken in seed-hash order; each takes the unmatched yes row with every
    coordinate of MATCH_COORDINATES within CALIPER that is nearest in Euclidean
    distance (ties: seed-hash order). Returns (no, yes, distance) per stratum in that order.
    """
    by_stratum: dict[str, dict[int, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: {0: [], 1: []}
    )
    for record in records:
        by_stratum[match_stratum(record)][record["label"]].append(record)
    out: dict[str, list[tuple[dict[str, Any], dict[str, Any], float]]] = {}
    for s in sorted(by_stratum):
        yes = sorted(by_stratum[s][1], key=seed_key)
        no = sorted(by_stratum[s][0], key=seed_key)
        buckets: dict[int, list[int]] = collections.defaultdict(list)
        coords = [[float(r["metrics"][c]) for c in MATCH_COORDINATES] for r in yes]
        for index, point in enumerate(coords):
            buckets[math.floor(point[0] / CALIPER)].append(index)
        used = [False] * len(yes)
        pairs = []
        for record in no:
            mine = [float(record["metrics"][c]) for c in MATCH_COORDINATES]
            home = math.floor(mine[0] / CALIPER)
            best = None
            for bucket in (home - 1, home, home + 1):
                for index in buckets.get(bucket, ()):
                    if used[index]:
                        continue
                    point = coords[index]
                    if any(abs(p - q) > CALIPER + 1e-12 for p, q in zip(point, mine)):
                        continue
                    distance = math.sqrt(sum((p - q) ** 2 for p, q in zip(point, mine)))
                    if best is None or (distance, index) < best:
                        best = (distance, index)
            if best is not None:
                used[best[1]] = True
                pairs.append((record, yes[best[1]], round(best[0], 6)))
        out[s] = pairs
    return out


def balance_matched(
    records: list[dict[str, Any]], targets: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[str, int], set[str]]:
    """Matched pairs per group apportioned over the ms strata (largest remainder); the
    first pairs in matching order. Returns rows, units per stratum and all matched cids.
    """
    pairs = match_pairs(records)
    matched = {r["cid"] for ps in pairs.values() for p in ps for r in p[:2]}
    chosen: list[dict[str, Any]] = []
    units: dict[str, int] = {}
    for group in ("natural", "swap"):
        avail = {s: len(ps) for s, ps in pairs.items() if s.startswith(group + "|")}
        for s, n in allocate(targets[group] // 2, avail, avail).items():
            units[s] = n
            for no, yes, distance in pairs[s][:n]:
                no["match"] = {"partner": yes["cid"], "distance": distance}
                yes["match"] = {"partner": no["cid"], "distance": distance}
                chosen += [yes, no]
    return chosen, units, matched


def draw_dev(
    records: list[dict[str, Any]],
    lang: str,
    plan: list[dict[str, Any]],
    plan_units_by_stratum: dict[str, int],
) -> list[dict[str, Any]]:
    """Matched pairs in sha256("pn1-dev:" + no-row key) order, with stratum units and
    (yes family, no family) quotas proportional to the planned TRAIN selection."""
    quota = DEV_QUOTAS[lang]
    share = sum(r["group"] == "swap" for r in plan) / len(plan) if plan else None
    targets = group_targets(lang, quota, share)
    pairs = {
        s: sorted(ps, key=lambda p: dev_key(p[0]))
        for s, ps in match_pairs(records).items()
    }
    by_cid = {r["cid"]: r for r in plan}
    combo_weights = collections.Counter(
        (match_stratum(r), r["family"], by_cid[r["match"]["partner"]]["family"])
        for r in plan
        if r["label"] == 0 and r.get("match", {}).get("partner") in by_cid
    )
    dev: list[dict[str, Any]] = []
    for group in ("natural", "swap"):
        avail = {s: len(ps) for s, ps in pairs.items() if s.startswith(group + "|")}
        weights = {s: plan_units_by_stratum.get(s, 0) for s in avail}
        for s, n in allocate(targets[group] // 2, weights, avail).items():
            cell = pairs[s]
            available = collections.Counter(
                (p[0]["family"], p[1]["family"]) for p in cell
            )
            quotas = allocate(
                n,
                {c: combo_weights[(s, *c)] for c in available},
                dict(available),
            )
            taken, counts = [], collections.Counter()
            for pair in cell:
                combo = (pair[0]["family"], pair[1]["family"])
                if counts[combo] < quotas.get(combo, 0):
                    taken.append(pair)
                    counts[combo] += 1
            chosen = {p[0]["cid"] for p in taken}
            for pair in cell:
                if len(taken) >= n:
                    break
                if pair[0]["cid"] not in chosen:
                    taken.append(pair)
            for no, yes, distance in taken[:n]:
                dev += [
                    dict(yes, match={"partner": no["cid"], "distance": distance}),
                    dict(no, match={"partner": yes["cid"], "distance": distance}),
                ]
    return dev


def select_language(
    records: list[dict[str, Any]], lang: str, graph: source.LinkGraph
) -> dict[str, Any]:
    targets = group_targets(lang, TRAIN_TARGETS[lang])
    plan, units, _ = balance_matched(records, targets)
    passes = []
    for attempt in range(DEV_PASSES):
        dev = draw_dev(records, lang, plan, units)
        dev_ids = {r["cid"] for r in dev}
        blocklist = Blocklist(dev, graph)
        blocked: dict[str, str] = {}
        train_pool = []
        for record in records:
            if record["cid"] in dev_ids:
                continue
            reason = blocklist.reason(record)
            if reason is None:
                train_pool.append(record)
            else:
                blocked[record["cid"]] = reason
        train, train_units, matched = balance_matched(train_pool, targets)
        dev_mix, train_mix = family_mix(dev), family_mix(train)
        worst = (
            max(abs(dev_mix[f] - train_mix[f]) for f in FAMILIES)
            if dev and train
            else 0.0
        )
        passes.append(
            {
                "pass": attempt + 1,
                "dev": len(dev),
                "train": len(train),
                "max_family_share_gap": round(worst, 6),
            }
        )
        if worst <= MIX_TOLERANCE:
            break
        plan, units = train, train_units
    return {
        "dev": dev,
        "train": train,
        "blocked": blocked,
        "dev_neighbourhood": len(blocklist.neighbourhood),
        "passes": passes,
        "mix": {
            "dev": dev_mix,
            "train": train_mix,
            "max_gap": round(worst, 6),
            "ok": worst <= MIX_TOLERANCE,
        },
        "train_units": train_units,
        "train_pool_matched": matched,
        "train_pool_size": len(train_pool),
    }


def trim(
    records: list[dict[str, Any]], key=None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-match the rows (amendment 3 matching) and keep every matched pair."""
    kept = []
    for pairs in match_pairs(records).values():
        for no, yes, distance in pairs:
            kept += [
                dict(yes, match={"partner": no["cid"], "distance": distance}),
                dict(no, match={"partner": yes["cid"], "distance": distance}),
            ]
    ids = {r["cid"] for r in kept}
    return kept, [r for r in records if r["cid"] not in ids]


def apply_drops(selection: dict[str, dict[str, Any]], drop: set[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for lang, chosen in selection.items():
        entry = {}
        for split, key in (("train", seed_key), ("dev", dev_key)):
            rows = chosen[split]
            left = [r for r in rows if group_id(r["group_key"]) not in drop]
            kept, removed = trim(left, key)
            chosen[split] = kept
            entry[split] = {"dropped": len(rows) - len(left), "trimmed": len(removed)}
            for record in removed:
                chosen.setdefault("trimmed", []).append(record["cid"])
            for record in rows:
                if group_id(record["group_key"]) in drop:
                    chosen.setdefault("dropped", []).append(record["cid"])
        report[lang] = entry
    return report


# --- rows and outputs -------------------------------------------------------------------------


def make_row(
    record: dict[str, Any],
    split: str,
    export_date: str,
    judge_model: str,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    rid = row_id(export_date, record)
    k, j, swap = (
        pick(rid, "instructions", len(INSTRUCTIONS)),
        pick(rid, "layout", len(LAYOUTS)),
        pick(rid, "order", 2),
    )
    first, second = record["texts"][::-1] if swap else record["texts"]
    edited = [i for i, flag in enumerate(record["edited"]) if flag]
    audit = {
        "source_local_id": record["cid"],
        "tatoeba_ids": record["ids"],
        "group_key": record["group_key"],
        "export": f"tatoeba-{export_date}",
        "pivot_id": record.get("pivot"),
        "pivot_language": record.get("pivot_lang"),
        "edit": record.get("kind"),
        "editor": record.get("editor"),
        "edited_position": (1 - edited[0] if swap else edited[0]) if edited else None,
        "sentence_order": "swapped" if swap else "as_built",
        "overlap": {
            k2: record["metrics"][k2]
            for k2 in (
                "bigram_jaccard",
                "multiset_jaccard",
                "same_multiset",
                "length_ratio",
                "bin",
            )
        }
        | extra_metrics(first, second),
        "stratum": record["stratum"],
        "match": record.get("match"),
        "judge": {
            "model": judge_model,
            "prompt_version": prompt_version,
            "label_p_yes": record["judge"]["label"],
            "label_order": record["judge"]["label_order"],
            "fluency_p_yes": record["judge"].get("fluency"),
        },
    }
    row = {
        "id": rid,
        "state": LAYOUTS[j].format(a=first, b=second),
        "instructions": INSTRUCTIONS[k],
        "options": noul_options("en"),
        "label": record["label"],
        "task_type": "noul",
        "family": record["family"],
        "group_id": group_id(record["group_key"]),
        "language": record["language"],
        "split": "train" if split == "train" else "select",
        "source": source_key(export_date, record),
        "evaluation_role": "train" if split == "train" else "select",
        "render_template": f"pn1/{record['family']}/i{k + 1}/l{j + 1}",
        "audit_metadata": audit,
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return validate_row(row, "train" if split == "train" else "select")


def input_chars(state: Any, question: dict[str, Any]) -> int:
    return len(
        json.dumps(
            {"state": state, "questions": {"decision": question}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def dev_prompt(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gold-free prompt (``training.model.infer.load_prompts`` format) and its gold line."""
    pid = "pn1dev-" + sha(f"pn1-dev-prompt:{row['id']}")[:16]
    question = {
        "type": "noul",
        "instructions": row["instructions"],
        "criteria": {option["key"]: option["description"] for option in row["options"]},
    }
    prompt = {"id": pid, "state": row["state"], "questions": {"decision": question}}
    restored = question_to_row(prompt, "decision", question)
    if (
        digest({field: restored[field] for field in INPUT_FIELDS})
        != row["input_sha256"]
    ):
        raise ValueError(
            f"{row['id']}: dev prompt does not round-trip to the row input"
        )
    chars = input_chars(row["state"], question)
    value = row["label"] == 1
    gold = {
        "id": pid,
        "task": f"pn1/{row['family']}",
        "source": row["source"],
        "split": "select",
        "source_item_id": row["id"],
        "group_id": row["group_id"],
        "cluster_id": row["group_id"],
        "language": row["language"],
        "input_chars": chars,
        "long": chars >= LONG_INPUT_CHARS,
        "provenance": {"file": "pn1.dev.jsonl"},
        "questions": {"decision": question},
        "gold": {"decision": {"type": "noul", "value": value, "semantic_value": value}},
    }
    return prompt, gold


def attribution_lines(records: list[dict[str, Any]]) -> list[str]:
    seen: dict[int, dict[str, Any]] = {}
    for record in records:
        for sid in record["ids"]:
            entry = seen.setdefault(
                sid,
                {
                    "language": record["language"],
                    "edited": False,
                    **record["attribution"][str(sid)],
                },
            )
            entry["edited"] |= any(record["edited"])
    lines = ["sentence_id\tlanguage\tusername\tlicence\tedited"]
    for sid in sorted(seen):
        entry = seen[sid]
        user = entry["user"] if entry["user"] not in ("", "\\N") else "(none)"
        licence = "CC0" if entry["cc0"] else "CC-BY-2.0-FR"
        lines.append(
            f"{sid}\t{entry['language']}\t{user}\t{licence}\t{int(entry['edited'])}"
        )
    return lines


def counts_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for lang in LANGS:
        mine = [r for r in records if r["language"] == lang]
        table[lang] = {
            "rows": len(mine),
            "yes": sum(r["label"] == 1 for r in mine),
            "by_family_label": dict(
                sorted(
                    collections.Counter(
                        f"{r['family']}|{r['label']}" for r in mine
                    ).items()
                )
            ),
            "by_stratum_label": dict(
                sorted(
                    collections.Counter(
                        f"{r['stratum']}|y{r['label']}" for r in mine
                    ).items()
                )
            ),
            "swap_share": round(
                sum(r["group"] == "swap" for r in mine) / max(len(mine), 1), 6
            ),
            "family_mix": family_mix(mine),
        }
    table["total"] = {
        "rows": len(records),
        "yes": sum(r["label"] == 1 for r in records),
        "by_family_label": dict(
            sorted(
                collections.Counter(
                    f"{r['family']}|{r['label']}" for r in records
                ).items()
            )
        ),
    }
    return table


# --- self-check: overlap-only logistic regression ---------------------------------------------


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] if abs(a[i][i]) >= 1e-12 else 0.0 for i in range(n)]


def fit_logistic(
    x: list[list[float]], y: list[int], l2: float = 1e-3, iterations: int = 25
) -> list[float]:
    """Newton-IRLS logistic regression with a bias term and a small L2 penalty."""
    width = len(x[0]) + 1
    w = [0.0] * width
    rows = [[1.0] + list(values) for values in x]
    for _ in range(iterations):
        grad = [l2 * wi for wi in w]
        hess = [[l2 if i == j else 0.0 for j in range(width)] for i in range(width)]
        for features, label in zip(rows, y):
            z = sum(wi * fi for wi, fi in zip(w, features))
            p = 1 / (1 + math.exp(-max(min(z, 35), -35)))
            for i in range(width):
                grad[i] += (p - label) * features[i]
                weight = p * (1 - p) * features[i]
                for j in range(i, width):
                    hess[i][j] += weight * features[j]
        for i in range(width):
            for j in range(i):
                hess[i][j] = hess[j][i]
        step = _solve(hess, grad)
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return w


def overlap_features(row: dict[str, Any], features=FEATURES) -> list[float]:
    overlap = row["audit_metadata"]["overlap"]
    return [float(overlap[name]) for name in features]


def cv_accuracy(
    rows: list[dict[str, Any]], folds: int = 5, features=FEATURES
) -> dict[str, Any]:
    if len(rows) < 2 * folds:
        return {"n": len(rows), "skipped": "too few rows"}
    fold_of = [int(sha(f"pn1-fold:{r['group_id']}"), 16) % folds for r in rows]
    x = [overlap_features(r, features) for r in rows]
    width = len(features)
    y = [r["label"] for r in rows]
    correct = majority = 0
    for fold in range(folds):
        train = [i for i in range(len(rows)) if fold_of[i] != fold]
        test = [i for i in range(len(rows)) if fold_of[i] == fold]
        if not train or not test:
            continue
        means = [sum(x[i][c] for i in train) / len(train) for c in range(width)]
        scales = [
            math.sqrt(sum((x[i][c] - means[c]) ** 2 for i in train) / len(train)) or 1.0
            for c in range(width)
        ]
        z = [
            [(x[i][c] - means[c]) / scales[c] for c in range(width)]
            for i in range(len(rows))
        ]
        w = fit_logistic([z[i] for i in train], [y[i] for i in train])
        base = int(sum(y[i] for i in train) * 2 >= len(train))
        for i in test:
            score = w[0] + sum(wi * zi for wi, zi in zip(w[1:], z[i]))
            correct += int((score > 0) == bool(y[i]))
            majority += int(base == y[i])
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 6),
        "majority": round(majority / len(rows), 6),
        "margin": round((correct - majority) / len(rows), 6),
    }


def self_check(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "features": list(FEATURES),
        "features_extended": list(FEATURES_EXTENDED),
        "folds": "5, group-disjoint by sha256('pn1-fold:'+group_id)",
    }
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        entry = {}
        for lang in (*LANGS, "pooled"):
            mine = (
                rows if lang == "pooled" else [r for r in rows if r["language"] == lang]
            )
            if not mine:
                continue
            entry[lang] = {
                "label_rate": round(sum(r["label"] for r in mine) / len(mine), 6),
                "family_mix": family_mix(mine),
                "overlap_lr": cv_accuracy(mine),
                "overlap_lr_extended": cv_accuracy(mine, features=FEATURES_EXTENDED),
            }
        report[split] = entry
    return report


# --- finalize ---------------------------------------------------------------------------------


def load_judgments(
    judge_dirs: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    out: dict[str, dict[str, Any]] = {}
    receipts = []
    for directory in judge_dirs:
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        path = directory / "judgments.jsonl"
        if file_sha256(path) != receipt["outputs_sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from its receipt")
        for record in read_jsonl(path):
            if record["item"] in out:
                raise ValueError(f"item {record['item']} judged twice")
            out[record["item"]] = dict(record, job=directory.name)
        receipts.append(
            {
                "dir": directory.name,
                **{
                    k: receipt[k]
                    for k in ("model", "outputs_sha256", "counts", "timing")
                },
            }
        )
    return out, receipts


def apply_judgments(
    pool: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Attach P(yes) to judged rows; keep yes rows at P >= .5, no rows below, fluency >= .5."""
    kept: list[dict[str, Any]] = []
    outcome: dict[str, str] = {}
    for record in pool:
        if not record["judged"]:
            outcome[record["cid"]] = "not_judged"
            continue
        label = judgments.get(f"{record['cid']}#label")
        fluency = judgments.get(f"{record['cid']}#fluency")
        needs_fluency = any(record["edited"])
        complete = label is not None and (fluency is not None or not needs_fluency)
        record["judge"] = {
            "complete": complete,
            "label": label["p_yes"] if label else None,
            "label_order": label["order"] if label else None,
            "fluency": fluency["p_yes"] if fluency else None,
        }
        if not complete:
            outcome[record["cid"]] = "judge_incomplete"
            continue
        label_ok = (record["judge"]["label"] >= 0.5) == (record["label"] == 1)
        fluency_ok = not needs_fluency or record["judge"]["fluency"] >= 0.5
        if label_ok and fluency_ok:
            kept.append(record)
            outcome[record["cid"]] = "kept"
        else:
            outcome[record["cid"]] = (
                "rejected_label" if not label_ok else "rejected_fluency"
            )
    return kept, outcome


def keep_rates(
    pool: list[dict[str, Any]], outcome: dict[str, str], minimum: int = 20
) -> dict[tuple[str, str, int], float]:
    """Kept share of the judged rows per (language, family, label); pooled over languages below ``minimum``."""
    judged: collections.Counter = collections.Counter()
    kept: collections.Counter = collections.Counter()
    for record in pool:
        if outcome[record["cid"]] in ("kept", "rejected_label", "rejected_fluency"):
            for key in (
                (record["language"], record["family"], record["label"]),
                ("*", record["family"], record["label"]),
            ):
                judged[key] += 1
                kept[key] += outcome[record["cid"]] == "kept"
    rates = {}
    for lang in LANGS:
        for family in FAMILIES:
            for label in (0, 1):
                key = (lang, family, label)
                source_key_ = key if judged[key] >= minimum else ("*", family, label)
                rates[key] = (
                    kept[source_key_] / judged[source_key_]
                    if judged[source_key_]
                    else 0.0
                )
    return rates


def extension(
    pool: list[dict[str, Any]], outcome: dict[str, str], margin: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Unjudged rows to judge next: per language x stratum x label, in seed-hash order, until
    the kept rows plus the expected keeps (measured keep rates) reach ``margin`` x planned units.
    """
    rates = keep_rates(pool, outcome)
    added: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    by_lang = collections.defaultdict(list)
    for record in pool:
        by_lang[record["language"]].append(record)
    for lang in LANGS:
        records = by_lang.get(lang, [])
        units = plan_units(records, lang)
        for (s, label), cell in sorted(cells_of(records, seed_key).items()):
            need = margin * units.get(s, 0)
            expected = sum(outcome[r["cid"]] == "kept" for r in cell)
            before = expected
            count = 0
            for record in cell:
                if expected >= need:
                    break
                if record["judged"]:
                    continue
                record["rank"] = round(expected / max(need, 1), 6)
                added.append(record)
                expected += rates[(lang, record["family"], label)]
                count += 1
            info[f"{lang}|{s}|y{label}"] = {
                "planned_units": units.get(s, 0),
                "kept_pass1": before,
                "added": count,
                "expected_kept": round(expected, 1),
            }
    return added, {
        "keep_rates": {"|".join(map(str, k)): round(v, 6) for k, v in rates.items()},
        "cells": info,
    }


def cmd_judgeset_extend(args: argparse.Namespace) -> int:
    work = args.work_dir
    base = json.loads(
        (work / "judgeset" / "judgeset.receipt.json").read_text(encoding="utf-8")
    )
    pool_path = work / "judgeset" / "pool.jsonl"
    if file_sha256(pool_path) != base["files_sha256"]["pool.jsonl"]:
        raise ValueError("pool.jsonl differs from the judge-set receipt")
    pool = read_jsonl(pool_path)
    judgments, receipts = load_judgments(args.judge_dir)
    _, outcome = apply_judgments(pool, judgments)
    added, plan = extension(pool, outcome, args.margin)
    items = [
        item
        for record in sorted(added, key=lambda r: (r["rank"], r["cid"]))
        for item in judge_items(record)
    ]
    counts = {
        "rows": dict(
            sorted(
                collections.Counter(
                    f"{r['language']}|{r['family']}|{r['label']}" for r in added
                ).items()
            )
        ),
        "items": dict(
            sorted(
                collections.Counter(
                    f"{i['language']}|{i['kind']}" for i in items
                ).items()
            )
        ),
        "total_items": len(items),
    }
    if args.dry_run:
        print(json.dumps(counts, indent=1))
        return 0
    out = work / args.out_name
    out.mkdir(mode=0o700)
    added_ids = {r["cid"] for r in added}
    gpus = assign_gpus(
        collections.Counter(i["language"] for i in items), args.gpus.split(",")
    )
    files = {
        "pool.jsonl": write_new(
            out / "pool.jsonl",
            jsonl_bytes(
                {
                    k: v
                    for k, v in dict(
                        r, judged=r["judged"] or r["cid"] in added_ids
                    ).items()
                    if k != "judge"
                }
                for r in sorted(pool, key=lambda r: r["cid"])
            ),
        )
    }
    for gpu in sorted(set(gpus.values())):
        name = f"items.{gpu}.jsonl"
        files[name] = write_new(
            out / name, jsonl_bytes(i for i in items if gpus[i["language"]] == gpu)
        )
    judged = [r for r in pool if r["judged"] or r["cid"] in added_ids]
    receipt = {
        **{
            k: base[k]
            for k in (
                "generation",
                "editor",
                "twin_status",
                "twin_edit_reasons",
                "pool",
            )
        },
        "schema": SCHEMA + "/judgeset-extend",
        "created_utc": utc(),
        "code": code_identity(),
        "extends": {
            "judgeset.receipt.json": file_sha256(
                work / "judgeset" / "judgeset.receipt.json"
            ),
            "judge_passes": receipts,
        },
        "twin_status_file": "judgeset/twin-status.jsonl",
        "margin": args.margin,
        "judge_factor": base["judge_factor"],
        "plan": plan,
        "added": counts,
        "judged_rows": dict(
            sorted(
                collections.Counter(
                    f"{r['language']}|{r['family']}|{r['label']}" for r in judged
                ).items()
            )
        ),
        "items": dict(
            sorted(
                collections.Counter(
                    f"{i['language']}|{i['kind']}" for i in items
                ).items()
            )
        ),
        "items_pass1": base["items"],
        "gpu_of_language": gpus,
        "files_sha256": files,
    }
    write_json(out / "judgeset.receipt.json", receipt)
    print(json.dumps({"added_rows": len(added), "items": len(items), "gpus": gpus}))
    return 0


PROBE_PER_LANGUAGE = 25
PROBE_PASS = {
    "identical_median": 0.90,
    "coord_median": 0.80,
    "seed_fluency_share": 0.80,
}


def original_sentences(pool: list[dict[str, Any]]) -> dict[str, dict[int, str]]:
    """Unedited Tatoeba sentences of the pool, per language."""
    out: dict[str, dict[int, str]] = collections.defaultdict(dict)
    for record in pool:
        for index, (value, edited) in enumerate(zip(record["texts"], record["edited"])):
            if not edited:
                sid = (
                    record["ids"][index]
                    if len(record["ids"]) > index
                    else record["ids"][0]
                )
                out[record["language"]][sid] = value
    return out


def cmd_probe_items(args: argparse.Namespace) -> int:
    """Amendment 3 control probe: identical pairs, coordination swaps and seed fluency,
    25 per language each (hash order), under both prompt versions."""
    work = args.work_dir
    pool = read_jsonl(work / "judgeset" / "pool.jsonl")
    seeds = read_jsonl(work / "seeds.jsonl")
    originals = original_sentences(pool)
    items = []
    for lang in LANGS:
        same = sorted(
            originals.get(lang, {}), key=lambda s: sha(f"pn1-probe-identical:{s}")
        )
        coords = sorted(
            (
                r
                for r in pool
                if r["language"] == lang
                and r["family"] == "pn-name"
                and r.get("kind") == "coord"
            ),
            key=lambda r: sha(f"pn1-probe-coord:{r['cid']}"),
        )
        mine = sorted(
            (s for s in seeds if s["language"] == lang),
            key=lambda s: sha(f"pn1-probe-seed:{s['seed']}"),
        )
        for version in ("v1", "v2"):
            label_, fluency_ = PROMPTS[version]
            for sid in same[:PROBE_PER_LANGUAGE]:
                value = originals[lang][sid]
                items.append(
                    {
                        "item": f"probe:{version}:identical:{sid}",
                        "cid": f"probe:identical:{sid}",
                        "kind": "label",
                        "language": lang,
                        "order": "ab",
                        "rank": 0,
                        "prompt": label_(lang, value, value),
                    }
                )
            for record in coords[:PROBE_PER_LANGUAGE]:
                order = "ab" if pick(record["cid"], "judge-order", 2) == 0 else "ba"
                a, b = record["texts"] if order == "ab" else record["texts"][::-1]
                items.append(
                    {
                        "item": f"probe:{version}:coord:{record['cid']}",
                        "cid": f"probe:coord:{record['cid']}",
                        "kind": "label",
                        "language": lang,
                        "order": order,
                        "rank": 0,
                        "prompt": label_(lang, a, b),
                    }
                )
            for seed in mine[:PROBE_PER_LANGUAGE]:
                items.append(
                    {
                        "item": f"probe:{version}:seed:{seed['seed']}",
                        "cid": f"probe:seed:{seed['seed']}",
                        "kind": "fluency",
                        "language": lang,
                        "rank": 0,
                        "prompt": fluency_(lang, seed["text"]),
                    }
                )
    items.sort(key=lambda i: (i["cid"], i["item"]))
    out = work / "probe"
    out.mkdir(mode=0o700)
    digest_ = write_new(out / "items.probe.jsonl", jsonl_bytes(items))
    counts = dict(
        sorted(
            collections.Counter(
                i["item"].split(":")[1] + "|" + i["item"].split(":")[2] for i in items
            ).items()
        )
    )
    write_json(
        out / "probe-items.receipt.json",
        {
            "schema": SCHEMA + "/probe-items",
            "code": code_identity(),
            "counts": counts,
            "items_sha256": digest_,
            "pass_rule": PROBE_PASS,
        },
    )
    print(json.dumps({"items": len(items), "counts": counts}))
    return 0


def cmd_probe_eval(args: argparse.Namespace) -> int:
    judgments, receipts = load_judgments(args.judge_dir)
    values: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    per_lang: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for item, record in judgments.items():
        _, version, kind = item.split(":")[:3]
        values[(version, kind)].append(record["p_yes"])
        per_lang[(version, kind, record["language"])].append(record["p_yes"])

    def median(v: list[float]) -> float | None:
        v = sorted(v)
        return round((v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2, 6) if v else None

    def summary(get) -> dict[str, Any]:
        return {
            "identical_median": median(get("identical")),
            "coord_median": median(get("coord")),
            "seed_fluency_share": round(
                sum(p >= 0.5 for p in get("seed")) / max(len(get("seed")), 1), 6
            ),
            "n": {k: len(get(k)) for k in ("identical", "coord", "seed")},
        }

    report = {
        version: {
            **summary(lambda k, v=version: values[(v, k)]),
            "per_language": {
                lang: summary(lambda k, v=version, l=lang: per_lang[(v, k, l)])
                for lang in LANGS
            },
        }
        for version in ("v1", "v2")
    }
    v2 = report["v2"]
    passed = all(
        v2["n"][k] == PROBE_PER_LANGUAGE * len(LANGS) for k in v2["n"]
    ) and all(v2[k] is not None and v2[k] >= t for k, t in PROBE_PASS.items())
    receipt = {
        "schema": SCHEMA + "/probe",
        "code": code_identity(),
        "pass_rule": PROBE_PASS,
        "passed": passed,
        "results": report,
        "judge": receipts,
    }
    write_json(args.output, receipt)
    print(
        json.dumps(
            {
                "passed": passed,
                "v1": {k: report["v1"][k] for k in PROBE_PASS},
                "v2": {k: v2[k] for k in PROBE_PASS},
                "n": v2["n"],
            }
        )
    )
    return 0 if passed else 4


def cmd_judgeset_rejudge(args: argparse.Namespace) -> int:
    """Amendment 3: every pool row judged again with the given prompt version (label for
    all, fluency for every edited sentence), rows ordered by need as in the first pass.
    """
    work = args.work_dir
    base = json.loads(
        (work / "judgeset" / "judgeset.receipt.json").read_text(encoding="utf-8")
    )
    pool_path = work / "judgeset" / "pool.jsonl"
    if file_sha256(pool_path) != base["files_sha256"]["pool.jsonl"]:
        raise ValueError("pool.jsonl differs from the judge-set receipt")
    pool = read_jsonl(pool_path)
    by_lang = collections.defaultdict(list)
    for record in pool:
        record.pop("judge", None)
        record["judged"] = True
        by_lang[record["language"]].append(record)
    for lang, records in by_lang.items():
        units = plan_units(records, lang)
        for (s, _), cell in cells_of(records, seed_key).items():
            for index, record in enumerate(cell):
                record["rank"] = round(index / max(units.get(s, 0), 1), 6)
    items = [
        item
        for record in sorted(pool, key=lambda r: (r["rank"], r["cid"]))
        for item in judge_items(record, args.prompt_version)
    ]
    out = work / args.out_name
    out.mkdir(mode=0o700)
    gpus = assign_gpus(
        collections.Counter(i["language"] for i in items), args.gpus.split(",")
    )
    files = {
        "pool.jsonl": write_new(
            out / "pool.jsonl", jsonl_bytes(sorted(pool, key=lambda r: r["cid"]))
        )
    }
    for gpu in sorted(set(gpus.values())):
        name = f"items.{gpu}.jsonl"
        files[name] = write_new(
            out / name, jsonl_bytes(i for i in items if gpus[i["language"]] == gpu)
        )
    counts = dict(
        sorted(
            collections.Counter(f"{i['language']}|{i['kind']}" for i in items).items()
        )
    )
    receipt = {
        **{
            k: base[k]
            for k in (
                "generation",
                "editor",
                "twin_status",
                "twin_edit_reasons",
                "pool",
            )
        },
        "schema": SCHEMA + "/judgeset-rejudge",
        "created_utc": utc(),
        "code": code_identity(),
        "rejudge_of": {
            "judgeset.receipt.json": file_sha256(
                work / "judgeset" / "judgeset.receipt.json"
            )
        },
        "prompt_version": args.prompt_version,
        "twin_status_file": "judgeset/twin-status.jsonl",
        "judge_factor": base["judge_factor"],
        "judged_rows": dict(
            sorted(
                collections.Counter(
                    f"{r['language']}|{r['family']}|{r['label']}" for r in pool
                ).items()
            )
        ),
        "items": counts,
        "total_items": len(items),
        "gpu_of_language": gpus,
        "files_sha256": files,
    }
    write_json(out / "judgeset.receipt.json", receipt)
    print(
        json.dumps(
            {
                "rows": len(pool),
                "items": len(items),
                "gpus": gpus,
                "per_gpu": dict(
                    collections.Counter(gpus[i["language"]] for i in items)
                ),
            }
        )
    )
    return 0


def disagreement(pool: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for family in FAMILIES:
        for lang in LANGS:
            mine = [
                r
                for r in pool
                if r["family"] == family
                and r["language"] == lang
                and r.get("judge", {}).get("complete")
            ]
            if not mine:
                continue
            label_bad = sum(
                (r["judge"]["label"] >= 0.5) != (r["label"] == 1) for r in mine
            )
            fluent = [r for r in mine if any(r["edited"])]
            fluency_bad = sum(r["judge"]["fluency"] < 0.5 for r in fluent)
            table.setdefault(family, {})[lang] = {
                "judged": len(mine),
                "label_disagreement": round(label_bad / len(mine), 6),
                "fluency_fail": round(fluency_bad / len(fluent), 6) if fluent else None,
                "by_label": {
                    str(y): round(
                        sum(
                            (r["judge"]["label"] >= 0.5) != (y == 1)
                            for r in mine
                            if r["label"] == y
                        )
                        / max(sum(r["label"] == y for r in mine), 1),
                        6,
                    )
                    for y in (0, 1)
                    if any(r["label"] == y for r in mine)
                },
            }
    return table


def read_drop_groups(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        values = json.loads(text)
    else:
        values = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    groups = {value for value in values if value}
    bad = sorted(g for g in groups if not g.startswith("m4pn1:"))
    if bad:
        raise ValueError(f"not PN1 group ids: {bad[:3]}")
    return groups


def gpu_jobs(directory: Path | None) -> dict[str, Any]:
    if directory is None or not directory.is_dir():
        return {"jobs": [], "total_gpu_hours": None}
    jobs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("*.json"))
    ]
    return {
        "jobs": jobs,
        "total_gpu_hours": round(sum(j["gpu_hours"] for j in jobs), 6),
    }


def cmd_finalize(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    work = args.work_dir
    judgeset = json.loads(
        (work / args.judgeset / "judgeset.receipt.json").read_text(encoding="utf-8")
    )
    pool_path = work / args.judgeset / "pool.jsonl"
    if file_sha256(pool_path) != judgeset["files_sha256"]["pool.jsonl"]:
        raise ValueError("pool.jsonl differs from the judge-set receipt")
    pool = read_jsonl(pool_path)
    judgments, judge_receipts = load_judgments(args.judge_dir)
    judge_models = {
        r["model"]["repo_id"] + "@" + r["model"]["revision"] for r in judge_receipts
    }
    if len(judge_models) != 1:
        raise ValueError(f"judgments come from {sorted(judge_models)}")
    judge_model = next(iter(judge_models))
    candidates_receipt = json.loads(
        (work / "candidates.receipt.json").read_text(encoding="utf-8")
    )
    export_date = candidates_receipt["export_date"]
    registry = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "data"
            / "records"
            / "license-registry-m4.json"
        ).read_text(encoding="utf-8")
    )
    for key in (f"tatoeba-{export_date}", f"tatoeba-{export_date}-edited"):
        if key not in registry:
            raise ValueError(f"licence registry lacks {key}")
    manifest = verify_source(args.source_dir, ["links"])
    kept, outcome = apply_judgments(pool, judgments)
    noise = disagreement(pool)

    links_started = time.perf_counter()
    graph = source.load_link_graph(args.source_dir / source.export_files()["links"])
    links_seconds = time.perf_counter() - links_started
    by_lang = collections.defaultdict(list)
    for record in kept:
        by_lang[record["language"]].append(record)
    selection = {
        lang: select_language(by_lang.get(lang, []), lang, graph) for lang in LANGS
    }
    selected_ids = sorted(
        f"{split}:{r['cid']}"
        for chosen in selection.values()
        for split in ("train", "dev")
        for r in chosen[split]
    )
    selection_sha = sha("\n".join(selected_ids))
    if args.expect_selection and args.expect_selection != selection_sha:
        raise ValueError(
            f"pre-drop selection {selection_sha} differs from --expect-selection"
        )
    drop_report = None
    drop_sha = None
    if args.drop_groups:
        drop = read_drop_groups(args.drop_groups)
        drop_sha = file_sha256(args.drop_groups)
        drop_report = {
            "groups": len(drop),
            "file_sha256": drop_sha,
            "per_language": apply_drops(selection, drop),
        }

    train_records = [r for lang in LANGS for r in selection[lang]["train"]]
    dev_records = [r for lang in LANGS for r in selection[lang]["dev"]]
    for chosen in selection.values():
        for cid, reason in chosen["blocked"].items():
            outcome[cid] = f"blocked_{reason}"
        for cid in chosen.get("dropped", []):
            outcome[cid] = "dropped_group"
        for cid in chosen.get("trimmed", []):
            outcome[cid] = "trimmed_after_drop"
    for record in train_records:
        outcome[record["cid"]] = "train"
    for record in dev_records:
        outcome[record["cid"]] = "dev"
    matched_any = set().union(*(c["train_pool_matched"] for c in selection.values()))
    for record in kept:
        if outcome[record["cid"]] == "kept":
            outcome[record["cid"]] = (
                "unselected_matched" if record["cid"] in matched_any else "unmatched"
            )
    version = judgeset.get("prompt_version", "v1")
    train_rows = sorted(
        (
            make_row(r, "train", export_date, judge_model, version)
            for r in train_records
        ),
        key=lambda r: r["id"],
    )
    dev_rows = sorted(
        (make_row(r, "dev", export_date, judge_model, version) for r in dev_records),
        key=lambda r: r["id"],
    )
    ids = [r["id"] for r in train_rows + dev_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row id")
    if {r["group_id"] for r in train_rows} & {r["group_id"] for r in dev_rows}:
        raise ValueError("a group is in both TRAIN and dev")
    if {r["input_sha256"] for r in train_rows} & {r["input_sha256"] for r in dev_rows}:
        raise ValueError("an input is in both TRAIN and dev")
    for rows, name in ((train_rows, "train"), (dev_rows, "dev")):
        for lang in LANGS:
            mine = [r for r in rows if r["language"] == lang]
            if 2 * sum(r["label"] for r in mine) != len(mine):
                raise ValueError(f"{name} {lang}: labels are not 50/50")
    pairs = [dev_prompt(row) for row in dev_rows]
    pairs.sort(key=lambda pair: pair[0]["id"])
    row_of = {
        r["audit_metadata"]["source_local_id"]: r["id"] for r in train_rows + dev_rows
    }

    code = code_identity()
    inputs = {
        "candidates.jsonl": file_sha256(work / "candidates.jsonl"),
        "seeds.jsonl": file_sha256(work / "seeds.jsonl"),
        f"{args.judgeset}/pool.jsonl": judgeset["files_sha256"]["pool.jsonl"],
        **{
            f"judge/{r['dir']}/judgments.jsonl": r["outputs_sha256"]
            for r in judge_receipts
        },
        **{
            f"gen/{g['dir']}/generations.jsonl": g["outputs_sha256"]
            for g in judgeset["generation"]
        },
    }
    build_sha12 = sha(
        canonical({"code": code, "inputs": inputs, "drop_groups_sha256": drop_sha})
    )[:12]
    out = args.output_root / build_sha12
    out.mkdir(parents=True, mode=0o700)
    files = {
        "pn1.train.jsonl": write_new(out / "pn1.train.jsonl", jsonl_bytes(train_rows)),
        "pn1.dev.jsonl": write_new(out / "pn1.dev.jsonl", jsonl_bytes(dev_rows)),
        "pn1.dev.prompts.jsonl": write_new(
            out / "pn1.dev.prompts.jsonl",
            b"".join(
                (
                    json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                for p, _ in pairs
            ),
        ),
        "pn1.dev.gold.jsonl": write_new(
            out / "pn1.dev.gold.jsonl", jsonl_bytes(g for _, g in pairs)
        ),
        "attribution.tsv": write_new(
            out / "attribution.tsv",
            ("\n".join(attribution_lines(train_records + dev_records)) + "\n").encode(
                "utf-8"
            ),
        ),
        "pn1.candidates.receipt.jsonl": write_new(
            out / "pn1.candidates.receipt.jsonl",
            jsonl_bytes(
                {
                    "cid": r["cid"],
                    "language": r["language"],
                    "family": r["family"],
                    "label": r["label"],
                    "group_id": group_id(r["group_key"]),
                    "stratum": r["stratum"],
                    "metrics": r["metrics"],
                    "tatoeba_ids": r["ids"],
                    "pivot_id": r.get("pivot"),
                    "judged": r["judged"],
                    "judge": r.get("judge"),
                    "outcome": outcome[r["cid"]],
                    "row_id": row_of.get(r["cid"]),
                }
                for r in pool
            ),
        ),
        "pn1.judge.receipt.jsonl": write_new(
            out / "pn1.judge.receipt.jsonl",
            jsonl_bytes(
                {k: v for k, v in judgments[item].items() if k != "prompt"}
                for item in sorted(judgments)
            ),
        ),
        "pn1.generation.receipt.jsonl": write_new(
            out / "pn1.generation.receipt.jsonl",
            (
                work / judgeset.get("twin_status_file", "judgeset/twin-status.jsonl")
            ).read_bytes(),
        ),
    }
    check = self_check(train_rows, dev_rows)
    files["self-check.json"] = write_json(out / "self-check.json", check)
    shortfalls = {}
    for lang in LANGS:
        train_n = len(selection[lang]["train"])
        dev_n = len(selection[lang]["dev"])
        targets = group_targets(lang, TRAIN_TARGETS[lang])
        got = collections.Counter(r["group"] for r in selection[lang]["train"])
        shortfalls[lang] = {
            "train": {
                "target": TRAIN_TARGETS[lang],
                "rows": train_n,
                "short": TRAIN_TARGETS[lang] - train_n,
            },
            "train_groups": {
                g: {"target": targets[g], "rows": got[g]} for g in targets
            },
            "dev": {
                "target": DEV_QUOTAS[lang],
                "rows": dev_n,
                "short": DEV_QUOTAS[lang] - dev_n,
            },
        }
    build = {
        "schema": SCHEMA,
        "prereg": PREREG,
        "amendments": args.amendment or [],
        "build_sha12": build_sha12,
        "build_sha12_rule": "sha256(canonical({code, inputs, drop_groups_sha256}))[:12]",
        "created_utc": utc(),
        "code": code,
        "source": {
            "export_date": export_date,
            "manifest_sha256": file_sha256(args.source_dir / "source-manifest.json"),
            "files": manifest["files"],
        },
        "licence_keys": {
            key: registry[key]
            for key in (f"tatoeba-{export_date}", f"tatoeba-{export_date}-edited")
        },
        "models": {
            "generator": (
                judgeset["generation"][0]["model"] if judgeset["generation"] else None
            ),
            "editor_tag": judgeset["editor"],
            "judge": judge_receipts[0]["model"] if judge_receipts else None,
        },
        "inputs_sha256": inputs,
        "parameters": {
            "train_targets": TRAIN_TARGETS,
            "dev_quotas": DEV_QUOTAS,
            "swap_min_share": SWAP_MIN_SHARE,
            "judge_factor": JUDGE_FACTOR,
            "mix_tolerance": MIX_TOLERANCE,
            "prompt_version": judgeset.get("prompt_version", "v1"),
            "matching": {
                "strata": "language x group x same-multiset flag",
                "coordinates": list(MATCH_COORDINATES),
                "caliper": CALIPER,
                "order": "no rows in sha256('pn1-seed:'+group key) order; nearest Euclidean yes row within the caliper",
            },
            "near_mining": {
                "rare_bigrams": source.NEAR_RARE_BIGRAMS,
                "postings": source.NEAR_POSTINGS,
                "max_queries": source.NEAR_MAX_QUERIES,
            },
            "planned_seeds": source.PLANNED_SEEDS,
            "two_hop_rule": "neighbourhood = ids plus every sentence within two links in any language; near pairs need disjoint neighbourhoods",
        },
        "steps": {
            "candidates": candidates_receipt["counts"],
            "candidates_per_language": candidates_receipt["per_language"],
            "twin_status": judgeset["twin_status"],
            "twin_edit_reasons": judgeset["twin_edit_reasons"],
            "pool": judgeset["pool"],
            "judged_rows": judgeset["judged_rows"],
            "judge_items": judgeset["items"],
            "judge_items_pass1": judgeset.get("items_pass1"),
            "judge_extension": (
                {k: judgeset[k] for k in ("margin", "added", "plan")}
                if "margin" in judgeset
                else None
            ),
            "outcomes": dict(
                sorted(
                    collections.Counter(
                        f"{r['language']}|{r['family']}|{outcome[r['cid']]}"
                        for r in pool
                    ).items()
                )
            ),
        },
        "judge_disagreement_before_filtering": noise,
        "selection": {
            lang: {
                "passes": selection[lang]["passes"],
                "mix": selection[lang]["mix"],
                "blocked": dict(
                    collections.Counter(selection[lang]["blocked"].values())
                ),
                "dev_neighbourhood_ids": selection[lang]["dev_neighbourhood"],
                "train_units_by_stratum": selection[lang]["train_units"],
                "train_pool_rows": selection[lang]["train_pool_size"],
                "train_pool_matched_rows": len(selection[lang]["train_pool_matched"]),
            }
            for lang in LANGS
        },
        "selection_sha256_pre_drop": selection_sha,
        "drop_groups": drop_report,
        "train": counts_table(train_records),
        "dev": counts_table(dev_records),
        "shortfalls": shortfalls,
        "gpu": gpu_jobs(args.gpu_time_dir),
        "generation_receipts": judgeset["generation"],
        "judge_receipts": judge_receipts,
        "timing": {
            "links_seconds": round(links_seconds, 1),
            "total_seconds": round(time.perf_counter() - started, 1),
        },
        "files_sha256": files,
    }
    write_json(out / "build-manifest.json", build)
    print(
        json.dumps(
            {
                "out": str(out),
                "train": len(train_rows),
                "dev": len(dev_rows),
                "selection_sha256": selection_sha,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cand = commands.add_parser("candidates")
    cand.add_argument("--source-dir", type=Path, required=True)
    cand.add_argument("--work-dir", type=Path, required=True)
    cand.add_argument("--workers", type=int, default=8)
    judge = commands.add_parser("judgeset")
    judge.add_argument("--work-dir", type=Path, required=True)
    judge.add_argument("--gen-dir", type=Path, action="append", required=True)
    judge.add_argument("--gpus", default="gpu3,gpu4")
    extend = commands.add_parser("judgeset-extend")
    extend.add_argument("--work-dir", type=Path, required=True)
    extend.add_argument("--judge-dir", type=Path, action="append", required=True)
    extend.add_argument("--out-name", default="judgeset2")
    extend.add_argument("--margin", type=float, default=1.25)
    extend.add_argument("--gpus", default="gpu3,gpu4")
    extend.add_argument("--dry-run", action="store_true")
    probe = commands.add_parser("probe-items")
    probe.add_argument("--work-dir", type=Path, required=True)
    probe_eval = commands.add_parser("probe-eval")
    probe_eval.add_argument("--judge-dir", type=Path, action="append", required=True)
    probe_eval.add_argument("--output", type=Path, required=True)
    rejudge = commands.add_parser("judgeset-rejudge")
    rejudge.add_argument("--work-dir", type=Path, required=True)
    rejudge.add_argument("--out-name", default="judgeset3")
    rejudge.add_argument("--prompt-version", default="v2", choices=sorted(PROMPTS))
    rejudge.add_argument("--gpus", default="gpu3,gpu4")
    fin = commands.add_parser("finalize")
    fin.add_argument("--work-dir", type=Path, required=True)
    fin.add_argument("--source-dir", type=Path, required=True)
    fin.add_argument("--judge-dir", type=Path, action="append", required=True)
    fin.add_argument("--gpu-time-dir", type=Path)
    fin.add_argument("--output-root", type=Path, required=True)
    fin.add_argument("--drop-groups", type=Path)
    fin.add_argument("--expect-selection")
    fin.add_argument("--amendment", action="append")
    fin.add_argument(
        "--judgeset", default="judgeset", help="judge-set directory in WORK"
    )
    args = parser.parse_args(argv)
    return {
        "candidates": cmd_candidates,
        "judgeset": cmd_judgeset,
        "judgeset-extend": cmd_judgeset_extend,
        "probe-items": cmd_probe_items,
        "probe-eval": cmd_probe_eval,
        "judgeset-rejudge": cmd_judgeset_rejudge,
        "finalize": cmd_finalize,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
