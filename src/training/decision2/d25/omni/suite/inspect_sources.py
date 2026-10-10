"""Text-only summaries of fetched sources, used to pin the board's constructions.

Writes ``<out>/<key>.jsonl.gz`` (one record per source row, no image bytes) for the sources
whose board construction is not fully stated: Mind2Web steps (candidate boxes against the
screenshot), InfographicVQA questions, CORD and FUNSD annotations, Winoground examples, and the
option lists of MMMU-Pro, CV-Bench, BLINK and CharXiv.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq


def _rows(paths, columns=None, batch=64):
    for path in paths:
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=batch, columns=columns):
            yield path, b.to_pylist()


def _size(payload):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    if not payload:
        return None
    with Image.open(io.BytesIO(payload)) as im:
        return im.size


def _box(candidate):
    attributes = json.loads(candidate["attributes"])
    rect = attributes.get("bounding_box_rect")
    if not rect:
        return None
    return [round(float(v)) for v in rect.split(",")]


def mind2web(src: Path):
    for split in ("test_task", "test_website", "test_domain"):
        paths = sorted(glob.glob(str(src / "mind2web" / "data" / f"{split}-*.parquet")))
        for path, rows in _rows(paths):
            for r in rows:
                pos = [json.loads(c) for c in r["pos_candidates"]]
                neg = [json.loads(c) for c in r["neg_candidates"]]
                shot = (r.get("screenshot") or {}).get("bytes")
                yield {
                    "split": split,
                    "file": Path(path).name,
                    "action_uid": r["action_uid"],
                    "annotation_id": r["annotation_id"],
                    "website": r["website"],
                    "domain": r["domain"],
                    "subdomain": r["subdomain"],
                    "task": r["confirmed_task"],
                    "operation": json.loads(r["operation"]),
                    "target_action_index": r["target_action_index"],
                    "n_actions": len(r["action_reprs"]),
                    "target_repr": r["target_action_reprs"],
                    "screenshot": _size(shot),
                    "pos": [
                        {"tag": c["tag"], "box": _box(c), "id": c["backend_node_id"]}
                        for c in pos
                    ],
                    "neg": [[c["tag"], _box(c)] for c in neg],
                }


def infovqa(src: Path):
    for split in ("validation", "test"):
        paths = sorted(
            glob.glob(str(src / "infovqa" / "InfographicVQA" / f"{split}-*.parquet"))
        )
        for path, rows in _rows(paths):
            for r in rows:
                image = r.pop("image", None)
                ocr = r.pop("ocr", None)
                r["split"] = split
                r["ocr_chars"] = len(ocr) if isinstance(ocr, str) else None
                if image and image.get("bytes"):
                    r["image_size"] = _size(image["bytes"])
                yield r


def cord(src: Path):
    for path, rows in _rows(
        sorted(glob.glob(str(src / "cord" / "data" / "*.parquet")))
    ):
        split = Path(path).name.split("-")[0]
        for i, r in enumerate(rows):
            image = r.pop("image", None)
            yield {
                "split": split,
                "file": Path(path).name,
                "index": i,
                "ground_truth": r["ground_truth"],
                "image_size": _size(image["bytes"]) if image else None,
            }


def winoground(src: Path):
    path = src / "winoground" / "data" / "examples.jsonl"
    names = set()
    with zipfile.ZipFile(src / "winoground" / "data" / "images.zip") as z:
        names = {Path(n).name for n in z.namelist()}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        r["images_in_zip"] = [
            f"{r['image_0']}.png" in names,
            f"{r['image_1']}.png" in names,
        ]
        yield r


def options_only(src: Path):
    for path, rows in _rows(
        sorted(glob.glob(str(src / "mmmu_pro" / "vision" / "*.parquet"))),
        ["id", "options", "answer", "subject"],
    ):
        for r in rows:
            yield {"source": "mmmu_pro/vision", **r}
    for name in ("test_2d.parquet", "test_3d.parquet"):
        pf = pq.ParquetFile(src / "cvbench" / name)
        cols = [c for c in pf.schema_arrow.names if c != "image"]
        for _, rows in _rows([src / "cvbench" / name], cols):
            for r in rows:
                yield {
                    "source": f"cvbench/{name}",
                    **{k: v for k, v in r.items() if k != "bbox"},
                }
    for path, rows in _rows(
        sorted(glob.glob(str(src / "blink" / "*" / "test-*.parquet"))),
        ["idx", "sub_task", "choices", "answer", "question"],
    ):
        for r in rows:
            yield {"source": "blink/test", **r}
    pf = pq.ParquetFile(src / "charxiv" / "test.parquet")
    cols = [c for c in pf.schema_arrow.names if c != "image"]
    for _, rows in _rows([src / "charxiv" / "test.parquet"], cols):
        for r in rows:
            yield {"source": "charxiv/test", **r}


INSPECTORS = {
    "mind2web": mind2web,
    "infovqa": infovqa,
    "cord": cord,
    "winoground": winoground,
    "options": options_only,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in [k for k in args.only.split(",") if k] or INSPECTORS:
        target = out / f"{key}.jsonl.gz"
        if target.exists():
            continue
        tmp = target.with_suffix(".tmp")
        n = 0
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for rec in INSPECTORS[key](Path(args.sources)):
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
        tmp.replace(target)
        print(json.dumps({"inspected": key, "records": n}), flush=True)


if __name__ == "__main__":
    main()
