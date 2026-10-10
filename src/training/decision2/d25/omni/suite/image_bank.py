"""Eval image bank for decontamination: one row per image of every split of the 11 public Vision-board
benchmarks and of their source pools.

Columns (``bank.parquet``): ``sha256`` (of the stored bytes), ``phash64`` and ``dhash64`` (uint64; the
``imagehash.phash`` / ``imagehash.dhash`` bits at hash size 8, row-major, most significant bit first),
``width``, ``height``, ``benchmark``, ``split``, ``item_id``, ``source`` (fetched source key) and
``region``: empty for the whole image, ``"x0,y0,x1,y1"`` for the square tiles added to tall images
(aspect ratio above ``TILE_ASPECT``), so a viewport crop of a long screenshot or infographic still
meets a bank hash. Use ``phash64`` / ``dhash64`` from this module for training images so both sides
hash identically.

    python -m d25.omni.suite.image_bank --sources /data/d25/omni/suite/sources \
        --out /data/d25/omni/suite/image-bank [--suite-dir <proxy suite dir> ...] [--workers 8]

Each (source, shard) task writes ``shards/<task>.parquet``; finished shards are skipped, then all
shards are merged into ``bank.parquet`` with ``BANK.json`` (counts per benchmark and split).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import io
import json
import os
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TILE_ASPECT = 2.0
COLUMNS = (
    "sha256",
    "phash64",
    "dhash64",
    "width",
    "height",
    "benchmark",
    "split",
    "item_id",
    "source",
    "region",
)


def _dct_matrix(n: int, k: int) -> np.ndarray:
    i = np.arange(n)
    return 2.0 * np.cos(np.pi * np.arange(k)[:, None] * (2 * i[None, :] + 1) / (2 * n))


_DCT32 = _dct_matrix(32, 8)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.flatten():
        value = (value << 1) | int(b)
    return value


def phash64(image) -> int:
    """``imagehash.phash(image)`` (hash size 8, high-frequency factor 4) as an int."""
    from PIL import Image

    small = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    low = _DCT32 @ pixels @ _DCT32.T
    return _bits_to_int(low > np.median(low))


def dhash64(image) -> int:
    """``imagehash.dhash(image)`` (hash size 8) as an int."""
    from PIL import Image

    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    return _bits_to_int(pixels[:, 1:] > pixels[:, :-1])


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Square tiles (side = the short edge) at half-side steps along the long edge of a tall image."""
    short, long_ = min(width, height), max(width, height)
    if short <= 0 or long_ / short <= TILE_ASPECT:
        return []
    starts = list(range(0, long_ - short + 1, max(1, short // 2)))
    if starts[-1] != long_ - short:
        starts.append(long_ - short)
    if width <= height:
        return [(0, s, width, s + short) for s in starts]
    return [(s, 0, s + short, height) for s in starts]


def records(
    payload: bytes, benchmark: str, split: str, item_id: str, source: str
) -> list[dict]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    digest = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as im:
        im.load()
        out = [
            {
                "sha256": digest,
                "phash64": phash64(im),
                "dhash64": dhash64(im),
                "width": im.width,
                "height": im.height,
                "benchmark": benchmark,
                "split": split,
                "item_id": item_id,
                "source": source,
                "region": "",
            }
        ]
        for box in tiles(im.width, im.height):
            crop = im.crop(box)
            out.append(
                {
                    **out[0],
                    "phash64": phash64(crop),
                    "dhash64": dhash64(crop),
                    "region": ",".join(str(v) for v in box),
                }
            )
    return out


# ---- tasks: each yields (benchmark, split, item_id, payload) -------------------------------------


def _parquet_images(
    path: str, columns: list[str], id_col: str | None, benchmark: str, split: str
) -> Iterator:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    image_cols = [c for c in columns if c in names]
    read = image_cols + ([id_col] if id_col and id_col in names else [])
    index = 0
    for batch in pf.iter_batches(batch_size=32, columns=read):
        for r in batch.to_pylist():
            item = str(r[id_col]) if id_col and id_col in r else str(index)
            for c in image_cols:
                cell = r.get(c)
                if cell and cell.get("bytes"):
                    yield benchmark, split, f"{item}#{c}", cell["bytes"]
            index += 1


def task_list(sources: Path) -> list[tuple]:
    """(task name, source key, kind, args) for every shard of every bank source."""
    out = []

    def add(key, kind, path, **kw):
        name = f"{key}__{Path(path).as_posix().replace('/', '_').replace(' ', '')}"
        out.append((name, key, kind, str(path), kw))

    s = sources
    for f in ("test_2d.parquet", "test_3d.parquet"):
        add("cvbench", "cvbench", s / "cvbench" / f)
    for p in sorted(glob.glob(str(s / "blink" / "*" / "*.parquet"))):
        add(
            "blink",
            "parquet",
            p,
            benchmark="BLINK",
            split=Path(p).name.split("-")[0],
            columns=[f"image_{i}" for i in range(1, 5)],
            id="idx",
        )
    for p in sorted(glob.glob(str(s / "realworldqa" / "data" / "*.parquet"))):
        add(
            "realworldqa",
            "parquet",
            p,
            benchmark="RealWorldQA",
            split="test",
            columns=["image"],
            id=None,
        )
    for split in ("val", "test"):
        add(
            "charxiv",
            "parquet",
            s / "charxiv" / f"{split}.parquet",
            benchmark="CharXiv",
            split=split,
            columns=["image"],
            id="figure_path",
        )
    for p in sorted(glob.glob(str(s / "infovqa" / "InfographicVQA" / "*.parquet"))):
        add(
            "infovqa",
            "parquet",
            p,
            benchmark="InfographicVQA",
            split=Path(p).name.split("-")[0],
            columns=["image"],
            id="questionId",
        )
    for key in ("mind2web", "mind2web_train"):
        for p in sorted(glob.glob(str(s / key / "data" / "*.parquet"))):
            add(
                key,
                "parquet",
                p,
                benchmark="Mind2Web",
                split=Path(p).name.split("-")[0],
                columns=["screenshot"],
                id="action_uid",
            )
    add(
        "winoground",
        "zip",
        s / "winoground" / "data" / "images.zip",
        benchmark="Winoground",
        split="test",
        prefix="",
    )
    for p in sorted(glob.glob(str(s / "cord" / "data" / "*.parquet"))):
        add(
            "cord",
            "parquet",
            p,
            benchmark="KIE (CORD+FUNSD)",
            split="cord-" + Path(p).name.split("-")[0],
            columns=["image"],
            id=None,
        )
    for p in sorted(glob.glob(str(s / "funsd" / "data" / "*.parquet"))):
        add(
            "funsd",
            "parquet",
            p,
            benchmark="KIE (CORD+FUNSD)",
            split="funsd-" + Path(p).name.split("-")[0],
            columns=["image"],
            id="id",
        )
    add("hateful_memes", "hateful_memes", s / "hateful_memes")
    for lang in ("en", "zh"):
        for p in sorted(
            glob.glob(str(s / "rbench" / f"rbench-m_{lang}" / "*.parquet"))
        ):
            add(
                "rbench",
                "parquet",
                p,
                benchmark="R-Bench-M",
                split=f"rbench-m_{lang}",
                columns=["image"],
                id="index",
            )
    for p in sorted(glob.glob(str(s / "mmmu_pro" / "*" / "*.parquet"))):
        config = Path(p).parent.name
        add(
            "mmmu_pro",
            "parquet",
            p,
            benchmark="MMMU-Pro vision",
            split=config,
            columns=["image"] + [f"image_{i}" for i in range(1, 8)],
            id="id",
        )
    for p in sorted(glob.glob(str(s / "mmmu" / "*" / "*.parquet"))):
        add(
            "mmmu",
            "parquet",
            p,
            benchmark="MMMU",
            split=Path(p).name.split("-")[0],
            columns=[f"image_{i}" for i in range(1, 8)],
            id="id",
        )
    add(
        "coco_val2017",
        "zip",
        s / "coco_val2017" / "val2017.zip",
        benchmark="COCO",
        split="val2017",
        prefix="val2017/",
    )
    add(
        "ade20k",
        "zip",
        s / "ade20k" / "ADEChallengeData2016.zip",
        benchmark="ADE20K",
        split="validation",
        prefix="ADEChallengeData2016/images/validation/",
    )
    return out


def iter_task(key: str, kind: str, path: str, kw: dict) -> Iterator:
    if kind == "parquet":
        yield from _parquet_images(
            path, kw["columns"], kw.get("id"), kw["benchmark"], kw["split"]
        )
    elif kind == "cvbench":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        index = 0
        for batch in pf.iter_batches(
            batch_size=32,
            columns=["type", "task", "source_dataset", "source_filename", "image"],
        ):
            for r in batch.to_pylist():
                payload = r["image"]["bytes"]
                yield "CV-Bench", "test", f"{r['type']}-{index}", payload
                if r["type"] == "3D":
                    yield "Omni3D", r["source_dataset"], r["source_filename"], payload
                index += 1
    elif kind == "zip":
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if name.endswith("/") or not name.startswith(kw["prefix"]):
                    continue
                if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                yield kw["benchmark"], kw["split"], name[len(kw["prefix"]) :], z.read(
                    name
                )
    elif kind == "hateful_memes":
        root = Path(path)
        splits: dict[str, list[str]] = {}
        for jsonl in sorted(root.glob("*.jsonl")):
            for line in jsonl.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    splits.setdefault(Path(r["img"]).name, []).append(jsonl.stem)
        for image in sorted((root / "img").iterdir()):
            payload = image.read_bytes()
            for split in splits.get(image.name, ["unlisted"]):
                yield "Moderation (Hateful Memes)", split, image.stem, payload
    elif kind == "suite_dir":
        root = Path(path)
        for rows in sorted(root.glob("rows*.jsonl.gz")):
            with gzip.open(rows, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    for i, ref in enumerate(r.get("images") or []):
                        yield r.get(
                            "family", kw["name"]
                        ), f"proxy:{kw['name']}", f"{r['id']}#{i}", (
                            root / ref
                        ).read_bytes()
    else:
        raise ValueError(kind)


def run_task(task: tuple, out_dir: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    name, key, kind, path, kw = task
    target = Path(out_dir) / "shards" / f"{name}.parquet"
    if target.exists():
        return {
            "task": name,
            "rows": pq.ParquetFile(target).metadata.num_rows,
            "skipped": True,
        }
    started = time.time()
    rows, cache, seen = [], {}, set()
    for benchmark, split, item_id, payload in iter_task(key, kind, path, kw):
        digest = hashlib.sha256(payload).hexdigest()
        if (benchmark, split, digest) in seen:
            continue
        seen.add((benchmark, split, digest))
        if digest not in cache:
            cache[digest] = records(payload, "", "", "", key)
        for rec in cache[digest]:
            rows.append(
                {**rec, "benchmark": benchmark, "split": split, "item_id": item_id}
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "sha256": pa.array([r["sha256"] for r in rows], pa.string()),
            "phash64": pa.array([r["phash64"] for r in rows], pa.uint64()),
            "dhash64": pa.array([r["dhash64"] for r in rows], pa.uint64()),
            "width": pa.array([r["width"] for r in rows], pa.int32()),
            "height": pa.array([r["height"] for r in rows], pa.int32()),
            "benchmark": pa.array([r["benchmark"] for r in rows], pa.string()),
            "split": pa.array([r["split"] for r in rows], pa.string()),
            "item_id": pa.array([r["item_id"] for r in rows], pa.string()),
            "source": pa.array([key] * len(rows), pa.string()),
            "region": pa.array([r["region"] for r in rows], pa.string()),
        }
    )
    tmp = target.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(target)
    return {
        "task": name,
        "rows": len(rows),
        "images": len(cache),
        "seconds": round(time.time() - started, 1),
    }


def merge(out_dir: Path) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    shards = sorted((out_dir / "shards").glob("*.parquet"))
    table = pa.concat_tables([pq.read_table(p) for p in shards])
    order = pc.sort_indices(
        table,
        sort_keys=[
            (c, "ascending")
            for c in ("benchmark", "split", "item_id", "region", "sha256")
        ],
    )
    table = table.take(order)
    tmp = out_dir / "bank.parquet.tmp"
    pq.write_table(table, tmp, compression="zstd", row_group_size=65536)
    tmp.replace(out_dir / "bank.parquet")
    full = table.filter(pc.equal(table["region"], ""))
    counts = Counter(zip(full["benchmark"].to_pylist(), full["split"].to_pylist()))
    summary = {
        "rows": table.num_rows,
        "whole_image_rows": full.num_rows,
        "tile_rows": table.num_rows - full.num_rows,
        "distinct_sha256": len(set(table["sha256"].to_pylist())),
        "per_benchmark_split": {
            f"{b} / {s}": n for (b, s), n in sorted(counts.items())
        },
        "columns": list(COLUMNS),
        "hash_note": "phash64/dhash64 = imagehash phash/dhash bits (hash size 8, row-major, MSB first) of the "
        "stored bytes decoded without EXIF transpose; use d25.omni.suite.image_bank.phash64/dhash64",
        "tile_rule": f"square tiles (side = short edge, half-side stride) for aspect ratio > {TILE_ASPECT}",
        "shards": len(shards),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "BANK.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--suite-dir",
        action="append",
        default=[],
        help="NAME=DIR of a suite-layout proxy set",
    )
    parser.add_argument("--only", default="", help="comma-separated source keys")
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("D25_WORKERS", "8"))
    )
    args = parser.parse_args()
    out = Path(args.out)
    tasks = task_list(Path(args.sources))
    for spec in args.suite_dir:
        name, path = spec.split("=", 1)
        tasks.append(
            (f"proxy__{name}", f"proxy:{name}", "suite_dir", path, {"name": name})
        )
    only = {k for k in args.only.split(",") if k}
    tasks = [t for t in tasks if not only or t[1] in only]
    tasks.sort(key=lambda t: -os.path.getsize(t[3]) if os.path.isfile(t[3]) else 0)
    failures = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_task, t, str(out)): t[0] for t in tasks}
        for fut in as_completed(futures):
            try:
                print(json.dumps(fut.result()), flush=True)
            except Exception as exc:
                failures.append(futures[fut])
                print(
                    json.dumps(
                        {"task": futures[fut], "error": f"{type(exc).__name__}: {exc}"}
                    ),
                    flush=True,
                )
    if failures:
        raise SystemExit(f"{len(failures)} bank tasks failed: {failures[:5]}")
    print(json.dumps(merge(out)), flush=True)


if __name__ == "__main__":
    main()
