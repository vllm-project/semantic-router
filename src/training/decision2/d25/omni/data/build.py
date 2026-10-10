"""Build the Omni multimodal decision corpus (CPU only, resumable, idempotent).

    python -m d25.omni.data.build generate --profile v1a      # staging shards under mm-v1/staging, images in mm-v1/images
    python -m d25.omni.data.build textindex                   # Vega decontam index over every protected text set
    python -m d25.omni.data.build bank                        # extended image bank (crop variants)
    python -m d25.omni.data.build decontam --profile v1a      # calibrate, filter, manifest, validate, READY
    python -m d25.omni.data.build proxybank --out FILE        # hash proxy images of this node (bank rows + COCO ids)

Profiles: ``v1a`` (no-GPU-verification families, about 50k rows) is a shard prefix of ``v1`` (all families,
about 150k rows), so ``v1`` reuses every ``v1a`` shard. Staging rows and images live in ``mm-v1``; the
``v1a`` corpus directory gets its own rows and hard links to the images it uses. A corpus is marked
ready with ``READY`` (the manifest sha256) only after decontamination and validation pass.

Text decontamination is Vega's ``d25.vega.data.decontam`` (index v4 rule) run on each row and on every
text rendered into its images (``meta.image_text``) as extra pseudo-rows. Image decontamination is
``d25.omni.data.decontam`` (sha256, pHash/dHash against bank crop variants, calibrated on planted copies).
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import importlib
import json
import multiprocessing
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from d25.omni.data import decontam, sources
from d25.omni.data.rows import make_row

DATA = Path(os.environ.get("D25_OMNI_DATA", "/data/d25/omni/data"))
STAGE_ROOT = DATA / "mm-v1"
SHARD = 500
TEXT_SUITE = [
    "/data/d25/shared/index-suite-0.3/selected-rows.jsonl.gz",
    "/data/d25/shared/index-suite-0.3/added-rows.jsonl.gz",
    "/data/d25/shared/index-suite-0.3/gsm8k-rows.jsonl.gz",
]
VISION_SUITE = Path("/data/d25/omni/suite/vision-0.3.1")
PROTECTED = DATA / "protected"
TEXT_INDEX = DATA / "decontam" / "text-index"
BANK_EXT = DATA / "bank-ext" / "bank.parquet"
OFFICIAL_BANK = Path("/data/d25/omni/suite/image-bank/bank.parquet")
HOLDOUTS = [PROTECTED / "holdouts.json", PROTECTED / "proxy-holdouts.json"]

# family: (module, function, registry source, source split, raw prerequisites)
FAMILIES: dict[str, tuple[str, str, str, str, tuple[str, ...]]] = {
    "chart": ("charts", "generate", "gen-chart", "generated", ()),
    "infographic": ("documents", "infographic", "gen-document", "generated", ()),
    "document": ("documents", "document", "gen-document", "generated", ()),
    "receipt": ("documents", "receipt", "gen-document", "generated", ()),
    "form": ("documents", "form", "gen-document", "generated", ()),
    "gui-som": ("gui", "generate", "gen-gui", "generated", ()),
    "stem-vega": ("stem", "vega_render", "gen-stem", "train", ("vega",)),
    "stem-synthetic": ("stem", "generate", "gen-stem", "generated", ()),
    "coco-count": (
        "photos",
        "coco-count",
        "coco2017-train-ccby",
        "train2017",
        ("coco",),
    ),
    "coco-relation": (
        "photos",
        "coco-relation",
        "coco2017-train-ccby",
        "train2017",
        ("coco",),
    ),
    "coco-localize": (
        "photos",
        "coco-localize",
        "coco2017-train-ccby",
        "train2017",
        ("coco",),
    ),
    "kp-corr": ("photos", "kp-corr", "coco2017-train-ccby", "train2017", ("coco",)),
    "sat-static": ("photos", "sat-static", "sat-train", "train", ("sat",)),
    "sat-dynamic": ("photos", "sat-dynamic", "sat-train", "train", ("sat",)),
    "warp-corr": ("photos", "warp-corr", "gen-photo", "train", ("coco", "openimages")),
    "camera-motion": (
        "photos",
        "camera-motion",
        "gen-photo",
        "train",
        ("coco", "openimages"),
    ),
    "similarity": (
        "photos",
        "similarity",
        "gen-photo",
        "train",
        ("coco", "openimages"),
    ),
    "jigsaw": ("photos", "jigsaw", "gen-photo", "train", ("coco", "openimages")),
    "forensic": (
        "photos",
        "forensic",
        "gen-photo",
        "train",
        ("coco", "openimages", "diffusiondb"),
    ),
    "oi-relation": (
        "photos",
        "oi-relation",
        "openimages-v7-train",
        "train",
        ("openimages",),
    ),
    "oi-nearest": (
        "photos",
        "oi-nearest",
        "openimages-v7-train",
        "train",
        ("openimages",),
    ),
    "caption-swap": (
        "photos",
        "caption-swap",
        "coco2017-train-ccby",
        "train2017",
        ("coco",),
    ),
    "meme": ("photos", "meme", "gen-photo", "train", ("coco", "openimages", "text")),
}
PROFILES: dict[str, dict[str, int]] = {
    "v1a": {
        "chart": 7000,
        "infographic": 2500,
        "document": 1500,
        "receipt": 3000,
        "form": 2000,
        "gui-som": 8000,
        "stem-vega": 6000,
        "stem-synthetic": 1000,
        "coco-count": 4000,
        "coco-relation": 3000,
        "coco-localize": 1500,
        "kp-corr": 1500,
        "sat-static": 3000,
        "warp-corr": 4000,
        "camera-motion": 1500,
    },
    "v1": {
        "chart": 14000,
        "infographic": 5000,
        "document": 3000,
        "receipt": 6000,
        "form": 4000,
        "gui-som": 16000,
        "stem-vega": 10000,
        "stem-synthetic": 4000,
        "coco-count": 8000,
        "coco-relation": 6000,
        "coco-localize": 3000,
        "kp-corr": 4000,
        "sat-static": 8000,
        "sat-dynamic": 4000,
        "warp-corr": 8000,
        "camera-motion": 4000,
        "similarity": 3000,
        "jigsaw": 2000,
        "forensic": 5000,
        "oi-relation": 6000,
        "oi-nearest": 3000,
        "caption-swap": 10000,
        "meme": 14000,
    },
}
CORPUS_DIR = {"v1a": DATA / "mm-v1a", "v1": DATA / "mm-v1"}
REGISTRY_OF = {
    "coco": "coco2017-train-ccby",
    "openimages": "openimages-v7-train",
    "diffusiondb": "diffusiondb",
    "measuring-hate-speech": "measuring-hate-speech",
    "hatexplain": "hatexplain",
    "vega": "vega-m2t-v5",
}


def log(*parts: Any) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def generator(family: str):
    module, name, *_ = FAMILIES[family]
    if module == "photos":
        from d25.omni.data.gen import photos

        return photos.FAMILIES[name]
    return getattr(importlib.import_module(f"d25.omni.data.gen.{module}"), name)


def row_licence(family: str, item) -> str:
    licences = item.meta.get("licences")
    if licences:
        extra = [item.meta["text_licence"]] if item.meta.get("text_licence") else []
        return "+".join(
            sorted(
                {
                    part.strip()
                    for value in licences + extra
                    for part in value.split("+")
                }
            )
        )
    return sources.BY_ID[FAMILIES[family][2]].licence


OPTIONAL_RAW = {"openimages"}


def raw_ready(family: str) -> bool:
    """True when the family's raw inputs exist; Open Images is optional next to another photo root."""
    steps = FAMILIES[family][4]
    required = [step for step in steps if step not in OPTIONAL_RAW] or list(steps)
    return all((DATA / "raw" / step / ".done").exists() for step in required)


def generate_shard(task: tuple[str, int, int]) -> dict[str, Any]:
    family, shard, quota = task
    target = STAGE_ROOT / "staging" / family / f"shard-{shard:04d}.jsonl.gz"
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
                item, index, STAGE_ROOT, row_licence(family, item), FAMILIES[family][3]
            )
        except Exception as error:
            failures[f"row:{type(error).__name__}"] += 1
            continue
        row["family"] = family
        row["id"] = f"mm-v1/{family}/{index:08d}"
        rows.append(row)
    write_jsonl(target, rows)
    stats = {
        "family": family,
        "shard": shard,
        "rows": len(rows),
        "attempts": attempts,
        "failures": dict(failures),
    }
    target.with_suffix(".stats.json").write_text(json.dumps(stats))
    return stats


def shard_tasks(profile: str) -> list[tuple[str, int, int]]:
    scale = float(os.environ.get("D25_SCALE", "1"))
    tasks = []
    for family, full in PROFILES[profile].items():
        quota = max(1, int(full * scale))
        for shard in range((quota + SHARD - 1) // SHARD):
            tasks.append((family, shard, quota))
    return tasks


def cmd_generate(args) -> None:
    pending = shard_tasks(args.profile)
    random.Random(0).shuffle(pending)
    pending.sort(key=lambda t: FAMILIES[t[0]][0] == "gui")
    log(f"generate {args.profile}: {len(pending)} shards with {args.workers} workers")
    deadline = time.time() + args.wait_h * 3600
    done = collections.Counter()
    with multiprocessing.get_context("fork").Pool(
        args.workers, maxtasksperchild=8
    ) as pool:
        while pending:
            ready = [t for t in pending if raw_ready(t[0])]
            if not ready:
                if time.time() > deadline:
                    log(
                        "generate: raw inputs missing for",
                        sorted({t[0] for t in pending}),
                    )
                    raise SystemExit(2)
                time.sleep(60)
                continue
            pending = [t for t in pending if t not in ready]
            for stats in pool.imap_unordered(generate_shard, ready):
                done[stats["family"]] += stats.get("rows", 0)
                if not stats.get("skipped"):
                    log(
                        f"{stats['family']} shard {stats['shard']}: {stats['rows']} rows / {stats['attempts']} "
                        f"attempts {stats['failures']}"
                    )
    log("generate: done", dict(done))


# ------------------------------------------------------------------ protected text and image bank


def protected_text_files() -> list[str]:
    paths = [p for p in TEXT_SUITE if Path(p).exists()]
    if (VISION_SUITE / "rows.jsonl.gz").exists():
        paths.append(str(VISION_SUITE / "rows.jsonl.gz"))
    paths += sorted(glob.glob("/data/d25/omni/proxy/*/rows.jsonl.gz"))
    paths += sorted(glob.glob(str(PROTECTED / "proxy-rows-*.jsonl.gz")))
    if (PROTECTED / "proxy-protected-items.jsonl.gz").exists():
        paths.append(str(PROTECTED / "proxy-protected-items.jsonl.gz"))
    bench_text = DATA / "raw" / "bench" / "text.jsonl.gz"
    if bench_text.exists():
        kit = DATA / "decontam" / "bench-text-kit.jsonl.gz"
        if not kit.exists() or kit.stat().st_mtime < bench_text.stat().st_mtime:
            write_jsonl(
                kit,
                (
                    {
                        "id": r["id"],
                        "benchmark": r["id"].split(":")[0],
                        "state": "\n".join(r["texts"]),
                        "questions": {},
                    }
                    for r in decontam.read_jsonl(bench_text)
                ),
            )
        paths.append(str(kit))
    return paths


def cmd_textindex(args) -> None:
    from d25.vega.data import decontam as vdecon

    paths = protected_text_files()
    stamp = hashlib.sha256(
        json.dumps([[p, Path(p).stat().st_size] for p in paths]).encode()
    ).hexdigest()
    marker = TEXT_INDEX / "INPUTS"
    if marker.exists() and marker.read_text().strip() == stamp and not args.force:
        log("textindex: up to date")
        return
    log("textindex: inputs", paths)
    stats = vdecon.build_index(paths, TEXT_INDEX, args.workers)["stats"]
    (TEXT_INDEX / "inputs.json").write_text(
        json.dumps({"files": paths, "stats": stats}, indent=1)
    )
    marker.write_text(stamp)
    log("textindex:", json.dumps(stats)[:600])


def _bank_rows(record: dict) -> list[dict]:
    from PIL import Image

    try:
        with Image.open(record["path"]) as image:
            image.load()
            width, height = image.size
            small = image.convert("RGB")
            if max(width, height) > 768:
                small.thumbnail((768, 768), Image.Resampling.LANCZOS)
            variants = decontam.bank_variants(small)
            variants[0] = ("full", decontam.phash64(image), decontam.dhash64(image))
    except Exception:
        return []
    return [
        {
            "sha256": record["sha256"],
            "phash64": ph & decontam.MASK64,
            "dhash64": dh & decontam.MASK64,
            "width": width,
            "height": height,
            "benchmark": record["benchmark"],
            "split": record["split"],
            "item_id": record["item_id"],
            "variant": variant,
        }
        for variant, ph, dh in variants
    ]


def _write_bank(records: list[dict], out: Path, workers: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, part in enumerate(
            pool.imap_unordered(_bank_rows, records, chunksize=32)
        ):
            rows += part
            if i % 10000 == 0:
                log(f"bank: {i}/{len(records)}")
    schema = pa.schema(
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
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp)
    tmp.replace(out)
    return len(rows)


def _suite_images(root: Path, label: str) -> list[dict]:
    """Image records of a suite-layout directory (rows give benchmark and item id where available)."""
    owner: dict[str, tuple[str, str]] = {}
    rows_path = root / "rows.jsonl.gz"
    if rows_path.exists():
        for row in decontam.read_jsonl(rows_path):
            bench = (
                (row.get("metadata") or {}).get("benchmark")
                or row.get("family")
                or label
            )
            for ref in row.get("images") or []:
                owner[Path(ref).stem] = (f"{label}:{bench}", str(row.get("id")))
    out = []
    for path in glob.glob(str(root / "images" / "*" / "*")):
        sha = Path(path).stem
        bench, item = owner.get(sha, (label, ""))
        out.append(
            {
                "sha256": sha,
                "path": path,
                "benchmark": bench,
                "split": label,
                "item_id": item,
            }
        )
    return out


def cmd_bank(args) -> None:
    records: dict[str, dict] = {}
    bench_index = DATA / "raw" / "bench" / "index.jsonl.gz"
    if bench_index.exists():
        for r in decontam.read_jsonl(bench_index):
            records.setdefault(
                r["sha256"], {**r, "path": str(DATA / "raw" / "bench" / r["path"])}
            )
    for r in _suite_images(VISION_SUITE, "vision-0.3.1"):
        records.setdefault(r["sha256"], r)
    for root in sorted(glob.glob("/data/d25/omni/proxy/*/")):
        if Path(root).name.startswith("_"):
            continue
        for r in _suite_images(Path(root), f"proxy:{Path(root).name}"):
            records.setdefault(r["sha256"], r)
    stamp = hashlib.sha256("".join(sorted(records)).encode()).hexdigest()
    marker = BANK_EXT.with_suffix(".inputs")
    if (
        BANK_EXT.exists()
        and marker.exists()
        and marker.read_text().strip() == stamp
        and not args.force
    ):
        log(f"bank: up to date ({len(records)} images)")
        return
    log(
        f"bank: {len(records)} unique images "
        f"{dict(collections.Counter(r['benchmark'].split(':')[0] for r in records.values()))}"
    )
    n = _write_bank(list(records.values()), BANK_EXT, args.workers)
    marker.write_text(stamp)
    log(f"bank: wrote {n} hash rows to {BANK_EXT}")


COCO_URL = re.compile(r"(?:train|val)2017/0*(\d+)\.jpg")


def cmd_proxybank(args) -> None:
    """Hash every proxy image under /data/d25/omni/proxy of this node; collect COCO ids from provenance."""
    records, coco = [], set()
    for root in sorted(glob.glob("/data/d25/omni/proxy/*/")):
        if Path(root).name.startswith("_"):
            continue
        records += _suite_images(Path(root), f"proxy:{Path(root).name}")
        rows_path = Path(root) / "rows.jsonl.gz"
        if rows_path.exists():
            for row in decontam.read_jsonl(rows_path):
                coco.update(
                    int(m)
                    for m in COCO_URL.findall(json.dumps(row.get("metadata") or {}))
                )
        for part in glob.glob(str(Path(root) / "parts" / "*.json")):
            coco.update(
                int(m) for m in COCO_URL.findall(Path(part).read_text(errors="ignore"))
            )
    out = Path(args.out)
    n = _write_bank(records, out, args.workers)
    out.with_name("coco-blacklist-proxy.json").write_text(json.dumps(sorted(coco)))
    log(
        f"proxybank: {len(records)} images, {n} hash rows, {len(coco)} COCO ids -> {out}"
    )


# ------------------------------------------------------------------ decontamination, manifest, validation


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
    registry = REGISTRY_OF.get(meta.get("text_source") or "")
    if registry:
        names.update(sources.identities(registry))
    return sorted(names)


def text_flags(row: dict) -> tuple[str | None, dict]:
    from d25.vega.data import decontam as vdecon

    index = STATE["text"]
    pseudo = [row] + [
        {
            "id": f"{row['id']}#text{k}",
            "state": text,
            "question": {"type": "choice", "instructions": "", "criteria": {}},
        }
        for k, text in enumerate((row.get("meta") or {}).get("image_text") or [])
    ]
    for candidate in pseudo:
        flag = vdecon.decide(index, candidate)
        if flag["drop"]:
            where = "image_text" if candidate is not row else "row"
            return f"text_{flag['reasons'][0]}", {
                "bench": flag.get("bench"),
                "where": where,
                "hits": flag["hits"],
            }
    return None, {}


def filter_shard(path: str) -> dict[str, Any]:
    checker: decontam.Decontaminator = STATE["checker"]
    out_dir = Path(STATE["out"])
    family = Path(path).parent.name
    shard = Path(path).name.split(".")[0].split("-")[1]
    kept, dropped = [], []
    for row in decontam.read_jsonl(path):
        try:
            reason, evidence = text_flags(row)
            if not reason:
                reason, evidence = checker.check(row, identities(row))
        except Exception as error:
            reason, evidence = "check_error", {
                "error": f"{type(error).__name__}: {error}"[:200]
            }
        if reason:
            dropped.append({"id": row["id"], "reason": reason, **evidence})
        else:
            kept.append(row)
    write_jsonl(out_dir / f"rows-{family}-{shard}.jsonl.gz", kept)
    write_jsonl(out_dir / "dropped" / f"{family}-{shard}.jsonl.gz", dropped)
    return {
        "family": family,
        "kept": len(kept),
        "dropped": collections.Counter(d["reason"] for d in dropped),
    }


def _sample_rows(paths: list[str], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    per = max(1, n // max(1, len(paths)))
    picked: list[dict] = []
    for path in paths:
        rows = list(decontam.read_jsonl(path))
        picked += rng.sample(rows, min(per, len(rows)))
    return picked


def calibrate_text(staging: list[str], workers: int) -> dict:
    from d25.vega.data import decontam as vdecon

    rule_path = TEXT_INDEX / "rule.json"
    clean = DATA / "decontam" / "clean-sample.jsonl.gz"
    sample = _sample_rows(staging, 6000, 11)
    pseudo = []
    for row in sample:
        pseudo.append(row)
        for k, text in enumerate((row.get("meta") or {}).get("image_text") or []):
            pseudo.append(
                {
                    "id": f"{row['id']}#text{k}",
                    "source": row["source"],
                    "family": row["family"],
                    "state": text,
                    "question": {"type": "choice", "instructions": "", "criteria": {}},
                    "target": [1.0],
                    "label": 0,
                }
            )
    write_jsonl(clean, pseudo)
    suite = json.loads((TEXT_INDEX / "inputs.json").read_text())["files"]
    return vdecon.calibrate(
        TEXT_INDEX, suite, [str(clean)], rule_path, workers, 60, 20000, 20261010
    )


def cmd_decontam(args) -> None:
    from PIL import Image

    from d25.vega.data import decontam as vdecon

    out = CORPUS_DIR[args.profile]
    out.mkdir(parents=True, exist_ok=True)
    tasks = shard_tasks(args.profile)
    staging = [
        str(STAGE_ROOT / "staging" / f / f"shard-{s:04d}.jsonl.gz") for f, s, _ in tasks
    ]
    missing = [p for p in staging if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"decontam: {len(missing)} staging shards missing, e.g. {missing[:3]}"
        )
    text_rule = calibrate_text(staging, args.workers)
    log("text rule", {k: v for k, v in text_rule.items() if k != "grid"})
    STATE["text"] = vdecon.Index(TEXT_INDEX)
    banks = [
        str(p)
        for p in (
            OFFICIAL_BANK,
            BANK_EXT,
            *sorted(PROTECTED.glob("proxy-bank-*.parquet")),
        )
        if p.exists()
    ]
    images = decontam.ImageIndex.from_parquet(banks)
    log(f"images: {len(images)} hash rows from {banks}")
    holdouts = decontam.Holdouts.load([str(p) for p in HOLDOUTS])
    for path in PROTECTED.glob("coco-blacklist-*.json"):
        holdouts.item_ids.update(f"coco:{i}" for i in json.loads(path.read_text()))
    rng = random.Random(3)
    by_bench: dict[str, list[dict]] = collections.defaultdict(list)
    for r in images.records:
        if r.get("variant", "full") == "full":
            by_bench[str(r.get("benchmark")).split(":")[0]].append(r)
    planted_refs = []
    bench_paths = {
        r["sha256"]: str(DATA / "raw" / "bench" / r["path"])
        for r in decontam.read_jsonl(DATA / "raw" / "bench" / "index.jsonl.gz")
    }
    for path in glob.glob(str(VISION_SUITE / "images" / "*" / "*")):
        bench_paths.setdefault(Path(path).stem, path)
    for bench, rows in sorted(by_bench.items()):
        usable = [r for r in rows if r["sha256"] in bench_paths]
        planted_refs += rng.sample(
            usable, min(len(usable), args.planted // max(1, len(by_bench)))
        )

    def load_rgb(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    planted = [load_rgb(bench_paths[r["sha256"]]) for r in planted_refs]
    sample = _sample_rows(staging, 3000, 1)
    refs = [ref for r in sample for ref in r.get("images") or []]
    clean = [
        load_rgb(STAGE_ROOT / ref)
        for ref in rng.sample(refs, min(len(refs), args.clean))
    ]
    image_report = decontam.calibrate_images(planted, clean, bank=images)
    image_report["planted_benchmarks"] = dict(
        collections.Counter(str(r.get("benchmark")) for r in planted_refs)
    )
    log("image rule", {k: v for k, v in image_report.items() if k != "per_variant"})
    STATE["checker"] = decontam.Decontaminator(
        None, 1.0, images, decontam.Rule(**image_report["rule"]), holdouts, STAGE_ROOT
    )
    STATE["out"] = str(out)
    totals: dict[str, Any] = collections.defaultdict(
        lambda: {"kept": 0, "dropped": collections.Counter()}
    )
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for stats in pool.imap_unordered(filter_shard, staging):
            t = totals[stats["family"]]
            t["kept"] += stats["kept"]
            t["dropped"].update(stats["dropped"])
    if out != STAGE_ROOT:
        link_images(out)
    manifest = write_manifest(
        args.profile,
        out,
        totals,
        {"text": text_rule, "image": image_report},
        banks,
        holdouts,
    )
    problems = validate(out)
    (out / "validation.json").write_text(json.dumps(problems, indent=1))
    if problems["errors"]:
        log("validation FAILED", json.dumps(problems)[:800])
        raise SystemExit(3)
    digest = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    (out / "READY").write_text(f"{digest}  manifest.json\n")
    log(f"READY {out} rows={manifest['rows']} manifest sha256={digest}")


def link_images(out: Path) -> None:
    for path in sorted(out.glob("rows-*.jsonl.gz")):
        for row in decontam.read_jsonl(path):
            for ref in row["images"]:
                target = out / ref
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(STAGE_ROOT / ref, target)


def validate(out: Path) -> dict[str, Any]:
    from PIL import Image

    from d25.omni.common import vision_format

    errors: list[str] = []
    ids: set[str] = set()
    checked = 0
    for path in sorted(out.glob("rows-*.jsonl.gz")):
        for row in decontam.read_jsonl(path):
            try:
                vision_format.validate_row(row)
            except Exception as error:
                errors.append(f"{row.get('id')}: {error}")
            if row["id"] in ids:
                errors.append(f"{row['id']}: duplicate id")
            ids.add(row["id"])
            for ref, digest in zip(row["images"], row["meta"]["image_sha256"]):
                target = out / ref
                if not target.exists():
                    errors.append(f"{row['id']}: missing {ref}")
                    continue
                if checked < 20000 or random.random() < 0.02:
                    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                        errors.append(f"{row['id']}: sha256 mismatch {ref}")
                    with Image.open(target) as image:
                        if image.width * image.height > vision_format.MAX_PIXELS:
                            errors.append(f"{row['id']}: {ref} over the pixel cap")
                    checked += 1
            if len(errors) > 200:
                break
    return {"rows": len(ids), "images_checked": checked, "errors": errors[:200]}


def write_manifest(
    profile: str, out: Path, totals, calibration, banks, holdouts
) -> dict:
    per_skill: collections.Counter = collections.Counter()
    per_source: collections.Counter = collections.Counter()
    licences: collections.Counter = collections.Counter()
    gold_positions: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    n_images: collections.Counter = collections.Counter()
    unique_images: set[str] = set()
    files = {}
    unverified = 0
    for path in sorted(out.glob("rows-*.jsonl.gz")):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in decontam.read_jsonl(path):
            meta = row["meta"]
            per_skill[meta["skill"]] += 1
            per_source[row["source"]] += 1
            licences[meta["licence"]] += 1
            n_images[len(row["images"])] += 1
            unique_images.update(meta["image_sha256"])
            unverified += not meta.get("verified", True)
            key = f"{meta['skill']}:{meta['n_options'] if row['question']['type'] == 'choice' else 'noul'}"
            gold_positions[key][
                meta.get("gold_position", int(bool(meta.get("gold"))))
            ] += 1
    manifest = {
        "corpus": f"mm-{profile}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row_contract": "d25.omni.common.vision_format training row (one question per row); target = gold one-hot; "
        "teacher soft labels go to meta.teachers.<name>",
        "rows": sum(t["kept"] for t in totals.values()),
        "unique_images": len(unique_images),
        "per_family": {
            f: {
                "kept": t["kept"],
                "dropped": dict(t["dropped"]),
                "quota": PROFILES[profile][f],
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
        "text_index": json.loads((TEXT_INDEX / "inputs.json").read_text()),
        "image_banks": banks,
        "holdout_files": holdouts.files,
        "sources": [s.__dict__ for s in sources.SOURCES if s.decision == "train"],
        "files": files,
        "max_pixels": 1_638_400,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=list))
    log(
        "manifest:",
        json.dumps({k: manifest[k] for k in ("rows", "per_skill", "drops_by_filter")}),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cmd", choices=["generate", "textindex", "bank", "decontam", "proxybank"]
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v1a")
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("D25_WORKERS", "8"))
    )
    parser.add_argument("--wait-h", type=float, default=8.0)
    parser.add_argument("--planted", type=int, default=1500)
    parser.add_argument("--clean", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", default=str(PROTECTED / "proxy-bank.parquet"))
    args = parser.parse_args(argv)
    {
        "generate": cmd_generate,
        "textindex": cmd_textindex,
        "bank": cmd_bank,
        "decontam": cmd_decontam,
        "proxybank": cmd_proxybank,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
