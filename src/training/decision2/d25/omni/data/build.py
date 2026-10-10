"""Build the Omni multimodal decision corpus (CPU only, resumable).

    python -m d25.omni.data.build generate --out $D25_OMNI_DATA/mm-v1   # staging shards per family
    python -m d25.omni.data.build bank                                   # local bank with crop variants
    python -m d25.omni.data.build decontam --out $D25_OMNI_DATA/mm-v1    # calibrate, filter, manifest

``generate`` writes ``staging/<family>/shard-<k>.jsonl.gz`` (skipping finished shards) and the images
under ``<out>/images``. ``decontam`` builds the protected text index and the image index (official bank
plus the local extended bank), calibrates both rules on planted positives, filters every staging row
(holdouts, text, image sha256 and perceptual hashes) into ``rows-<family>-<k>.jsonl.gz`` and writes
``manifest.json`` with licences, counts per source, family and skill, and drop counts per filter.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from d25.omni.data import decontam, sources
from d25.omni.data.rows import make_row

DATA = Path(os.environ.get("D25_OMNI_DATA", "/data/d25/omni/data"))
SHARD = 500
VERSION = "mm-v1"
TEXT_SUITE = [
    "/data/d25/shared/index-suite-0.3/selected-rows.jsonl.gz",
    "/data/d25/shared/index-suite-0.3/added-rows.jsonl.gz",
    "/data/d25/shared/index-suite-0.3/gsm8k-rows.jsonl.gz",
]
VISION_ROWS = [
    "/data/d25/omni/suite/vision-0.3.1/rows.jsonl.gz",
    str(DATA / "protected" / "vision-rows.jsonl.gz"),
]
PROXY_ROWS = "/data/d25/omni/proxy/*/rows*.jsonl.gz"
EXTRA_PROTECTED = [
    str(DATA / "raw" / "bench" / "text.jsonl.gz"),
    str(DATA / "protected" / "proxy-protected-items.jsonl.gz"),
]
BANKS = [
    "/data/d25/omni/suite/image-bank/bank.parquet",
    str(DATA / "protected" / "bank.parquet"),
    str(DATA / "bank-ext" / "bank.parquet"),
]
HOLDOUTS = [
    str(DATA / "protected" / "holdouts.json"),
    str(DATA / "protected" / "proxy-holdouts.json"),
]

# family: (module, function, quota, registry source of the row, source split)
FAMILIES: dict[str, tuple[str, str, int, str, str]] = {
    "coco-count": ("photos", "coco-count", 8000, "coco2017-train-ccby", "train2017"),
    "coco-relation": (
        "photos",
        "coco-relation",
        6000,
        "coco2017-train-ccby",
        "train2017",
    ),
    "oi-relation": ("photos", "oi-relation", 6000, "openimages-v7-train", "train"),
    "oi-nearest": ("photos", "oi-nearest", 3000, "openimages-v7-train", "train"),
    "coco-localize": (
        "photos",
        "coco-localize",
        3000,
        "coco2017-train-ccby",
        "train2017",
    ),
    "sat-static": ("photos", "sat-static", 8000, "sat-train", "train"),
    "warp-corr": ("photos", "warp-corr", 8000, "gen-photo", "train"),
    "kp-corr": ("photos", "kp-corr", 4000, "coco2017-train-ccby", "train2017"),
    "forensic": ("photos", "forensic", 5000, "gen-photo", "train"),
    "camera-motion": ("photos", "camera-motion", 4000, "gen-photo", "train"),
    "similarity": ("photos", "similarity", 3000, "gen-photo", "train"),
    "jigsaw": ("photos", "jigsaw", 2000, "gen-photo", "train"),
    "sat-dynamic": ("photos", "sat-dynamic", 4000, "sat-train", "train"),
    "chart": ("charts", "generate", 14000, "gen-chart", "generated"),
    "infographic": ("documents", "infographic", 5000, "gen-document", "generated"),
    "document": ("documents", "document", 3000, "gen-document", "generated"),
    "gui-som": ("gui", "generate", 16000, "gen-gui", "generated"),
    "caption-swap": (
        "photos",
        "caption-swap",
        10000,
        "coco2017-train-ccby",
        "train2017",
    ),
    "receipt": ("documents", "receipt", 6000, "gen-document", "generated"),
    "form": ("documents", "form", 4000, "gen-document", "generated"),
    "meme": ("photos", "meme", 14000, "gen-photo", "train"),
    "stem": ("stem", "generate", 14000, "gen-stem", "generated"),
}
REGISTRY_OF = {
    "coco": "coco2017-train-ccby",
    "openimages": "openimages-v7-train",
    "diffusiondb": "diffusiondb",
    "measuring-hate-speech": "measuring-hate-speech",
    "hatexplain": "hatexplain",
    "aqua-rat": "aqua-rat",
    "medmcqa": "medmcqa",
    "qasc": "qasc",
}


def log(*parts: Any) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def generator(family: str):
    module, name, *_ = FAMILIES[family]
    if module == "photos":
        from d25.omni.data.gen import photos

        return photos.FAMILIES[name]
    mod = __import__(f"d25.omni.data.gen.{module}", fromlist=[name])
    return getattr(mod, name)


def row_licence(family: str, item) -> str:
    licences = item.meta.get("licences")
    if licences:
        extra = [item.meta["text_licence"]] if item.meta.get("text_licence") else []
        return "+".join(
            sorted({part for value in licences + extra for part in value.split("+")})
        )
    return sources.BY_ID[FAMILIES[family][3]].licence


def generate_shard(task: tuple[str, int, str, int]) -> dict[str, Any]:
    family, shard, out, quota = task
    target = Path(out) / "staging" / family / f"shard-{shard:04d}.jsonl.gz"
    if target.exists():
        return {"family": family, "shard": shard, "skipped": True}
    want = min(SHARD, quota - shard * SHARD)
    fn = generator(family)
    rows, failures, attempts = [], collections.Counter(), 0
    base = shard * 100_000
    while len(rows) < want and attempts < 6 * want:
        index = base + attempts
        attempts += 1
        try:
            item = fn(index)
        except Exception as error:
            failures[type(error).__name__] += 1
            if failures[type(error).__name__] <= 2:
                traceback.print_exc()
            continue
        if item is None:
            failures["none"] += 1
            continue
        try:
            row = make_row(
                item, index, out, row_licence(family, item), FAMILIES[family][4]
            )
        except Exception as error:
            failures[f"row:{type(error).__name__}"] += 1
            continue
        row["family"] = family
        row["id"] = f"{VERSION}/{family}/{index:08d}"
        rows.append(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(target)
    stats = {
        "family": family,
        "shard": shard,
        "rows": len(rows),
        "attempts": attempts,
        "failures": dict(failures),
    }
    target.with_suffix(".stats.json").write_text(json.dumps(stats))
    return stats


def cmd_generate(args) -> None:
    families = args.families.split(",") if args.families else list(FAMILIES)
    tasks = []
    for family in families:
        quota = max(1, int(FAMILIES[family][2] * args.scale))
        for shard in range((quota + SHARD - 1) // SHARD):
            tasks.append((family, shard, args.out, quota))
    random.Random(0).shuffle(tasks)
    log(
        f"generate: {len(tasks)} shards over {len(families)} families with {args.workers} workers"
    )
    done = collections.Counter()
    with multiprocessing.get_context("fork").Pool(
        args.workers, maxtasksperchild=8
    ) as pool:
        for stats in pool.imap_unordered(generate_shard, tasks):
            done[stats["family"]] += stats.get("rows", 0)
            if not stats.get("skipped"):
                log(
                    f"{stats['family']} shard {stats['shard']}: {stats['rows']} rows / {stats['attempts']} attempts "
                    f"{stats['failures']}"
                )
    log("generate: done", dict(done))


# ------------------------------------------------------------------ local extended bank


def _bank_rows(record: dict) -> list[dict]:
    from PIL import Image

    try:
        with Image.open(DATA / "raw" / "bench" / record["path"]) as image:
            image.load()
            width, height = image.size
            variants = decontam.bank_variants(image)
    except Exception:
        return []
    return [
        {
            "sha256": record["sha256"],
            "phash64": ph,
            "dhash64": dh,
            "width": width,
            "height": height,
            "benchmark": record["benchmark"],
            "split": record["split"],
            "item_id": record["item_id"],
            "variant": variant,
        }
        for variant, ph, dh in variants
    ]


def cmd_bank(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    index = list(decontam.read_jsonl(DATA / "raw" / "bench" / "index.jsonl.gz"))
    unique = list({r["sha256"]: r for r in index}.values())
    log(f"bank: {len(index)} bench image refs, {len(unique)} unique")
    rows: list[dict] = []
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_bank_rows, unique, chunksize=64)):
            rows += part
            if i % 10000 == 0:
                log(f"bank: {i}/{len(unique)}")
    for r in rows:
        r["phash64"] = r["phash64"] & decontam.MASK64
        r["dhash64"] = r["dhash64"] & decontam.MASK64
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("sha256", pa.string()),
                ("phash64", pa.uint64()),
                ("dhash64", pa.uint64()),
                ("width", pa.int32()),
                ("height", pa.int32()),
                ("benchmark", pa.string()),
                ("split", pa.string()),
                ("item_id", pa.string()),
                ("variant", pa.string()),
            ]
        ),
    )
    out = DATA / "bank-ext" / "bank.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    log(f"bank: wrote {len(rows)} hash rows for {len(unique)} images to {out}")


# ------------------------------------------------------------------ decontamination and manifest


STATE: dict[str, Any] = {}


def identities(row: dict) -> list[str]:
    names = (
        set(sources.identities(row["source"]))
        if row["source"] in sources.BY_ID
        else {row["source"]}
    )
    meta = row.get("meta") or {}
    for ref in meta.get("source_ids") or []:
        registry = REGISTRY_OF.get(ref.split(":", 1)[0])
        if registry:
            names.update(sources.identities(registry))
    for key in ("text_source",):
        registry = REGISTRY_OF.get(meta.get(key) or "")
        if registry:
            names.update(sources.identities(registry))
    return sorted(names)


def filter_shard(path: str) -> dict[str, Any]:
    checker: decontam.Decontaminator = STATE["checker"]
    out_dir = Path(STATE["out"])
    family = Path(path).parent.name
    shard = Path(path).name.split(".")[0].split("-")[1]
    kept_path = out_dir / f"rows-{family}-{shard}.jsonl.gz"
    drop_path = out_dir / "dropped" / f"{family}-{shard}.jsonl.gz"
    kept, dropped = [], []
    for row in decontam.read_jsonl(path):
        try:
            reason, evidence = checker.check(row, identities(row))
        except Exception as error:
            reason, evidence = "check_error", {
                "error": f"{type(error).__name__}: {error}"[:200]
            }
        if reason:
            dropped.append({"id": row["id"], "reason": reason, **evidence})
        else:
            kept.append(row)
    for target, rows in ((kept_path, kept), (drop_path, dropped)):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(target)
    return {
        "family": family,
        "kept": len(kept),
        "dropped": collections.Counter(d["reason"] for d in dropped),
    }


def _sample_staging(out: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    paths = sorted(glob.glob(str(out / "staging" / "*" / "shard-*.jsonl.gz")))
    picked: list[dict] = []
    per = max(1, n // max(1, len(paths)))
    for path in paths:
        rows = list(decontam.read_jsonl(path))
        picked += rng.sample(rows, min(per, len(rows)))
    return picked


def cmd_decontam(args) -> None:
    from PIL import Image

    out = Path(args.out)
    protected = [
        p
        for p in TEXT_SUITE
        + VISION_ROWS
        + sorted(glob.glob(PROXY_ROWS))
        + EXTRA_PROTECTED
        if Path(p).exists()
    ]
    log("text: protected files", protected)
    text, per_file = decontam.build_text_index(protected)
    log(
        f"text: {text.items} items, {len(text.grams)} grams, {text.boilerplate} boilerplate"
    )
    banks = [p for p in BANKS if Path(p).exists()]
    images = decontam.ImageIndex.from_parquet(banks)
    bank_summary = collections.Counter(
        (r.get("benchmark"), r.get("variant", "full")) for r in images.records
    )
    log(f"images: {len(images)} hash rows from {banks}")
    holdouts = decontam.Holdouts.load(HOLDOUTS)
    sample = _sample_staging(out, 3000, 1)
    rng = random.Random(2)
    pool_rows: list[dict] = []
    for p in protected:
        rows = list(decontam.read_jsonl(p))
        pool_rows += rng.sample(rows, min(len(rows), 400))
    text_report = decontam.calibrate_text(
        text, pool_rows, [decontam.training_units(r) for r in sample], n_planted=2000
    )
    log("text calibration", text_report)
    bench_index = (
        list(decontam.read_jsonl(DATA / "raw" / "bench" / "index.jsonl.gz"))
        if (DATA / "raw" / "bench" / "index.jsonl.gz").exists()
        else []
    )
    by_bench: dict[str, list[dict]] = collections.defaultdict(list)
    for r in {r["sha256"]: r for r in bench_index}.values():
        by_bench[r["benchmark"]].append(r)
    planted_refs = [
        r
        for b, rs in sorted(by_bench.items())
        for r in rng.sample(rs, min(len(rs), args.planted // max(1, len(by_bench))))
    ]

    def open_image(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    planted = [open_image(DATA / "raw" / "bench" / r["path"]) for r in planted_refs]
    clean_refs = [(r, ref) for r in sample for ref in r.get("images") or []]
    clean = [
        open_image(out / ref)
        for _, ref in rng.sample(clean_refs, min(len(clean_refs), args.clean))
    ]
    image_report = decontam.calibrate_images(planted, clean, bank=images)
    image_report["planted_benchmarks"] = dict(
        collections.Counter(r["benchmark"] for r in planted_refs)
    )
    log(
        "image calibration",
        {k: v for k, v in image_report.items() if k != "per_variant"},
    )
    rule = decontam.Rule(**image_report["rule"])
    STATE["checker"] = decontam.Decontaminator(
        text, text_report["threshold"], images, rule, holdouts, out
    )
    STATE["out"] = str(out)
    staging = sorted(glob.glob(str(out / "staging" / "*" / "shard-*.jsonl.gz")))
    totals: dict[str, Any] = collections.defaultdict(
        lambda: {"kept": 0, "dropped": collections.Counter()}
    )
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for stats in pool.imap_unordered(filter_shard, staging):
            t = totals[stats["family"]]
            t["kept"] += stats["kept"]
            t["dropped"].update(stats["dropped"])
    write_manifest(
        out,
        totals,
        {"text": text_report, "image": image_report},
        per_file,
        banks,
        bank_summary,
        holdouts,
    )


def write_manifest(
    out: Path, totals, calibration, per_file, banks, bank_summary, holdouts
) -> None:
    per_skill: collections.Counter = collections.Counter()
    per_source: collections.Counter = collections.Counter()
    licences: collections.Counter = collections.Counter()
    gold_positions: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    n_images: collections.Counter = collections.Counter()
    unverified = 0
    for path in sorted(glob.glob(str(out / "rows-*.jsonl.gz"))):
        for row in decontam.read_jsonl(path):
            meta = row["meta"]
            per_skill[meta["skill"]] += 1
            per_source[row["source"]] += 1
            licences[meta["licence"]] += 1
            n_images[len(row["images"])] += 1
            unverified += not meta.get("verified", True)
            if row["question"]["type"] == "choice":
                gold_positions[f"{meta['skill']}:{meta['n_options']}"][
                    meta["gold_position"]
                ] += 1
            else:
                gold_positions[f"{meta['skill']}:noul"][int(meta["gold"])] += 1
    manifest = {
        "corpus": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row_contract": "d25.omni.common.vision_format training row (one question per row)",
        "rows": sum(t["kept"] for t in totals.values()),
        "per_family": {
            f: {
                "kept": t["kept"],
                "dropped": dict(t["dropped"]),
                "quota": FAMILIES[f][2],
            }
            for f, t in sorted(totals.items())
        },
        "drops_by_filter": dict(
            sum((t["dropped"] for t in totals.values()), collections.Counter())
        ),
        "per_skill": dict(per_skill),
        "per_source": dict(per_source),
        "licences": dict(licences),
        "images_per_row": {str(k): v for k, v in sorted(n_images.items())},
        "unverified_rows": unverified,
        "gold_positions": {
            k: dict(sorted(v.items())) for k, v in sorted(gold_positions.items())
        },
        "calibration": calibration,
        "protected_text_files": per_file,
        "image_banks": banks,
        "image_bank_entries": {
            f"{b}/{v}": n for (b, v), n in sorted(bank_summary.items(), key=str)
        },
        "holdout_files": holdouts.files,
        "sources": [s.__dict__ for s in sources.SOURCES if s.decision == "train"],
        "max_pixels": 1_638_400,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=list))
    log(
        "manifest:",
        json.dumps({k: manifest[k] for k in ("rows", "per_skill", "drops_by_filter")}),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["generate", "bank", "decontam"])
    parser.add_argument("--out", default=str(DATA / VERSION))
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("D25_WORKERS", "8"))
    )
    parser.add_argument("--families", default="")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--planted", type=int, default=1500)
    parser.add_argument("--clean", type=int, default=3000)
    args = parser.parse_args(argv)
    {"generate": cmd_generate, "bank": cmd_bank, "decontam": cmd_decontam}[args.cmd](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
