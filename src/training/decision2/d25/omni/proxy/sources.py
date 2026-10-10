"""Pinned third-party sources for photo-based proxies, downloaded into a per-node work cache.

Every source records its licence and download location for the manifest and the holdout registry.
Only evaluation use is made of these files; nothing is redistributed or uploaded.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

OPEN_IMAGES = {
    "name": "Open Images V7 test split",
    "licence": "images CC BY 2.0 (per-image author in metadata); annotations CC BY 4.0 (Google LLC)",
    "boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "labels": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv",
    "classes": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
    "metadata": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    "image": "https://open-images-dataset.s3.amazonaws.com/test/{id}.jpg",
    "cc_by_2": "https://creativecommons.org/licenses/by/2.0/",
}
COCO_KEYPOINTS = {
    "name": "COCO 2017 train person keypoints",
    "licence": "annotations CC BY 4.0; images Flickr (COCO terms of use)",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "member": "annotations/person_keypoints_train2017.json",
    "image": "http://images.cocodataset.org/train2017/{id:012d}.jpg",
}
CUB = {
    "name": "Caltech-UCSD Birds-200-2011",
    "licence": "research and non-commercial use (Caltech); images from Flickr",
    "archive": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
}


def fetch(url: str, dest: Path, retries: int = 4, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` once (atomic rename); returns the path."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "d25-omni-proxy/1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response, open(
                tmp, "wb"
            ) as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # network errors are retried
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {error}")


def fetch_many(
    jobs: list[tuple[str, Path]], workers: int = 16
) -> dict[Path, str | None]:
    """Parallel downloads; returns ``{dest: error or None}``."""
    out: dict[Path, str | None] = {}
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, url, dest): dest for url, dest in jobs}
        for future in cf.as_completed(futures):
            dest = futures[future]
            try:
                future.result()
                out[dest] = None
            except Exception as exc:
                out[dest] = str(exc)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pick(key: str, salt: str, mod: int, keep: int) -> bool:
    """Deterministic hash slice used to select source items."""
    value = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16)
    return value % mod < keep


def open_images_tables(work: Path) -> dict[str, Path]:
    root = work / "openimages"
    return {
        k: fetch(OPEN_IMAGES[k], root / Path(OPEN_IMAGES[k]).name)
        for k in ("boxes", "labels", "classes", "metadata")
    }


def open_images_metadata(work: Path) -> dict[str, dict[str, str]]:
    path = open_images_tables(work)["metadata"]
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rotation = row.get("Rotation") or "0"
            if row["License"] != OPEN_IMAGES["cc_by_2"] or float(rotation or 0) != 0.0:
                continue
            out[row["ImageID"]] = {
                "author": row["Author"],
                "landing": row["OriginalLandingURL"],
                "title": row["Title"],
            }
    return out


def open_images_classes(work: Path) -> dict[str, str]:
    with open(
        open_images_tables(work)["classes"], newline="", encoding="utf-8"
    ) as handle:
        return {mid: name for mid, name in csv.reader(handle)}


def open_images_boxes(work: Path) -> dict[str, list[dict[str, Any]]]:
    """Boxes per image: ``{image_id: [{label, x0, x1, y0, y1, occluded, truncated, group, depiction, inside}]}``."""
    cache = work / "openimages" / "boxes-by-image.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out: dict[str, list[dict[str, Any]]] = {}
    with open(
        open_images_tables(work)["boxes"], newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["ImageID"], []).append(
                {
                    "label": row["LabelName"],
                    "x0": float(row["XMin"]),
                    "x1": float(row["XMax"]),
                    "y0": float(row["YMin"]),
                    "y1": float(row["YMax"]),
                    "occluded": int(row["IsOccluded"]),
                    "truncated": int(row["IsTruncated"]),
                    "group": int(row["IsGroupOf"]),
                    "depiction": int(row["IsDepiction"]),
                    "inside": int(row["IsInside"]),
                }
            )
    cache.write_text(json.dumps(out))
    return out


def open_images_positive_labels(work: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with open(
        open_images_tables(work)["labels"], newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if row["Confidence"] == "1":
                out.setdefault(row["ImageID"], set()).add(row["LabelName"])
    return out


def open_images_files(work: Path, ids: list[str]) -> dict[str, Path]:
    root = work / "openimages" / "images"
    jobs = [(OPEN_IMAGES["image"].format(id=i), root / f"{i}.jpg") for i in ids]
    status = fetch_many(jobs)
    return {i: root / f"{i}.jpg" for i in ids if status.get(root / f"{i}.jpg") is None}


def open_images_provenance(
    image_id: str, meta: dict[str, str], path: Path
) -> dict[str, Any]:
    return {
        "source": "open-images-v7-test",
        "source_id": image_id,
        "file": OPEN_IMAGES["image"].format(id=image_id),
        "sha256": sha256_file(path),
        "licence": "CC BY 2.0",
        "author": meta.get("author", ""),
        "landing": meta.get("landing", ""),
    }


def coco_keypoints(work: Path) -> dict[str, Any]:
    root = work / "coco"
    cache = root / "person_keypoints_train2017.json"
    if not cache.exists():
        archive = fetch(
            COCO_KEYPOINTS["annotations"],
            root / "annotations_trainval2017.zip",
            timeout=600,
        )
        with zipfile.ZipFile(archive) as zf:
            cache.write_bytes(zf.read(COCO_KEYPOINTS["member"]))
    return json.loads(cache.read_text())


def coco_files(work: Path, ids: list[int]) -> dict[int, Path]:
    root = work / "coco" / "train2017"
    jobs = [(COCO_KEYPOINTS["image"].format(id=i), root / f"{i:012d}.jpg") for i in ids]
    status = fetch_many(jobs)
    return {
        i: root / f"{i:012d}.jpg"
        for i in ids
        if status.get(root / f"{i:012d}.jpg") is None
    }


def cub(work: Path) -> Path:
    """Extracted CUB_200_2011 directory (images, parts/part_locs.txt, bounding_boxes.txt)."""
    import tarfile

    root = work / "cub"
    target = root / "CUB_200_2011"
    if (target / "parts" / "part_locs.txt").exists():
        return target
    archive = fetch(CUB["archive"], root / "CUB_200_2011.tgz", timeout=1800)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return target


def image_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def decode(path: Path):
    from PIL import Image

    with Image.open(io.BytesIO(Path(path).read_bytes())) as image:
        return image.convert("RGB")
