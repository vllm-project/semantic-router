"""Download and index the raw inputs of the Omni corpus under ``$D25_OMNI_DATA/raw`` (resumable).

    python -m d25.omni.data.fetch coco openimages sat diffusiondb text bench sscd

Steps write ``raw/<step>/index.jsonl.gz`` (one record per usable item, with the sha256 of the original
bytes) and ``raw/<step>/.done``. ``bench`` downloads every image and question text of the board
benchmarks that are openly downloadable (all splits) for the local extended bank, planted-positive
calibration and the protected text pool; it never feeds training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

DATA = Path(os.environ.get("D25_OMNI_DATA", "/data/d25/omni/data"))
RAW = DATA / "raw"
COCO_LICENCES = {4: "CC-BY-2.0", 7: "no-known-copyright", 8: "US-Government-Work"}
OI_LICENCE = "https://creativecommons.org/licenses/by/2.0/"


def log(*parts: Any) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    import requests

    for attempt in range(retries):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": "d25-omni-data"}
            )
            if response.status_code == 200:
                return response.content
            if response.status_code in (403, 404, 410):
                raise FileNotFoundError(f"{response.status_code} {url}")
        except FileNotFoundError:
            raise
        except Exception:
            pass
        time.sleep(2**attempt)
    raise IOError(f"failed {url}")


def download(url: str, path: Path, size: int | None = None) -> Path:
    import requests

    if path.exists() and (size is None or path.stat().st_size == size):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp, "wb") as stream:
            for chunk in response.iter_content(1 << 22):
                stream.write(chunk)
    tmp.replace(path)
    return path


def hf_file(repo: str, filename: str, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def fetch_images(jobs: list[tuple[str, Path]], workers: int = 32) -> dict[str, str]:
    """Download ``(url, path)`` pairs in parallel; returns path -> sha256 for the ones that worked."""
    from PIL import Image

    def one(job):
        url, path = job
        if path.exists() and path.stat().st_size > 0:
            return str(path), sha256(path.read_bytes())
        try:
            payload = http_get(url)
            Image.open(io.BytesIO(payload)).verify()
        except Exception:
            return str(path), None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        return str(path), sha256(payload)

    out: dict[str, str] = {}
    with ThreadPoolExecutor(workers) as pool:
        for i, (path, digest) in enumerate(pool.map(one, jobs)):
            if digest:
                out[path] = digest
            if i % 2000 == 0:
                log(f"images {i}/{len(jobs)} ok {len(out)}")
    return out


# ------------------------------------------------------------------ COCO


def coco_blacklist() -> set[int]:
    """COCO image ids of VSR (all splits; BLINK Spatial_Relation) and the TallyQA test split (BLINK Counting).

    TallyQA's train split covers most of COCO, so it is left to the image-bank check, which holds the BLINK
    images themselves.
    """
    from huggingface_hub import list_repo_files

    ids: set[int] = set()
    for repo in ("cambridgeltl/vsr_random", "cambridgeltl/vsr_zeroshot"):
        try:
            files = list_repo_files(repo, repo_type="dataset")
        except Exception as error:
            log("vsr listing failed", repo, error)
            continue
        for name in files:
            if name.endswith((".jsonl", ".json")):
                text = hf_file(repo, name).read_text(errors="ignore")
                ids.update(int(m) for m in re.findall(r"(\d{6,12})\.jpg", text))
    payload = http_get(
        "https://github.com/manoja328/TallyQA_dataset/raw/master/tallyqa.zip",
        timeout=300,
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.endswith("test.json"):
                text = archive.read(name).decode("utf-8", errors="ignore")
                ids.update(
                    int(m)
                    for m in re.findall(r"COCO_(?:train|val)2014_0*(\d+)\.jpg", text)
                )
    return ids


def step_coco(limit: int) -> None:
    root = RAW / "coco"
    zpath = download(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        root / "annotations_trainval2017.zip",
        252907541,
    )
    with zipfile.ZipFile(zpath) as archive:
        load = lambda n: json.loads(archive.read(f"annotations/{n}_train2017.json"))
        instances, keypoints, captions = (
            load("instances"),
            load("person_keypoints"),
            load("captions"),
        )
    blacklist = coco_blacklist()
    (root / "blacklist.json").write_text(json.dumps(sorted(blacklist)))
    cats = {c["id"]: c["name"] for c in instances["categories"]}
    kp_names = keypoints["categories"][0]["keypoints"]
    images = {
        im["id"]: im
        for im in instances["images"]
        if im["license"] in COCO_LICENCES and im["id"] not in blacklist
    }
    log(
        f"coco: {len(instances['images'])} train2017, {len(images)} CC BY/PD after blacklist ({len(blacklist)} ids)"
    )
    objects: dict[int, list] = {}
    for ann in instances["annotations"]:
        if ann["image_id"] in images:
            objects.setdefault(ann["image_id"], []).append(
                {
                    "cat": cats[ann["category_id"]],
                    "bbox": [round(v, 1) for v in ann["bbox"]],
                    "area": round(ann["area"], 1),
                    "crowd": ann["iscrowd"],
                }
            )
    people: dict[int, list] = {}
    for ann in keypoints["annotations"]:
        if (
            ann["image_id"] in images
            and ann["num_keypoints"] >= 8
            and not ann["iscrowd"]
        ):
            people.setdefault(ann["image_id"], []).append(
                {"bbox": [round(v, 1) for v in ann["bbox"]], "kps": ann["keypoints"]}
            )
    caps: dict[int, list] = {}
    for ann in captions["annotations"]:
        if ann["image_id"] in images:
            caps.setdefault(ann["image_id"], []).append(ann["caption"].strip())
    chosen = sorted(images, key=lambda i: sha256(f"coco:{i}".encode()))[:limit]
    jobs = [
        (images[i]["coco_url"], root / "images" / images[i]["file_name"])
        for i in chosen
    ]
    digests = fetch_images(jobs)
    records = []
    for i in chosen:
        path = root / "images" / images[i]["file_name"]
        if str(path) not in digests:
            continue
        records.append(
            {
                "id": i,
                "file": f"images/{images[i]['file_name']}",
                "w": images[i]["width"],
                "h": images[i]["height"],
                "licence": COCO_LICENCES[images[i]["license"]],
                "flickr": images[i].get("flickr_url"),
                "sha256": digests[str(path)],
                "objects": objects.get(i, []),
                "people": people.get(i, []),
                "captions": caps.get(i, []),
            }
        )
    n = write_jsonl(root / "index.jsonl.gz", records)
    (root / "keypoints.json").write_text(json.dumps(kp_names))
    log(f"coco: indexed {n}")


# ------------------------------------------------------------------ Open Images


def step_openimages(limit: int) -> None:
    root = RAW / "openimages"
    base = "https://storage.googleapis.com/openimages"
    classes_path = download(
        f"{base}/v7/oidv7-class-descriptions-boxable.csv", root / "classes.csv"
    )
    classes = dict(row[:2] for row in csv.reader(open(classes_path)) if len(row) >= 2)
    meta_path = download(
        f"{base}/2018_04/train/train-images-boxable-with-rotation.csv",
        root / "train-images.csv",
        638407721,
    )
    candidates: dict[str, dict] = {}
    with open(meta_path, newline="") as stream:
        for row in csv.DictReader(stream):
            rotation = (row.get("Rotation") or "").strip()
            if row.get("License") == OI_LICENCE and rotation in ("", "0", "0.0"):
                candidates[row["ImageID"]] = {
                    "author": row.get("Author"),
                    "url": row.get("OriginalURL"),
                    "md5": row.get("OriginalMD5"),
                }
    keep = set(
        sorted(candidates, key=lambda i: sha256(f"oi:{i}".encode()))[: limit * 3]
    )
    log(f"openimages: {len(candidates)} CC BY 2.0 upright, sampling {len(keep)}")
    boxes_path = download(
        f"{base}/v6/oidv6-train-annotations-bbox.csv",
        root / "train-bbox.csv",
        2258447590,
    )
    boxes: dict[str, list] = {}
    with open(boxes_path, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["ImageID"] in keep:
                boxes.setdefault(row["ImageID"], []).append(
                    {
                        "label": classes.get(row["LabelName"], row["LabelName"]),
                        "box": [
                            float(row["XMin"]),
                            float(row["YMin"]),
                            float(row["XMax"]),
                            float(row["YMax"]),
                        ],
                        "occluded": row["IsOccluded"] == "1",
                        "truncated": row["IsTruncated"] == "1",
                        "group": row["IsGroupOf"] == "1",
                        "depiction": row["IsDepiction"] == "1",
                    }
                )
    usable = [
        i
        for i in sorted(keep, key=lambda i: sha256(f"oi:{i}".encode()))
        if len(boxes.get(i, [])) >= 2 and not all(b["depiction"] for b in boxes[i])
    ][:limit]
    jobs = [
        (
            f"https://s3.amazonaws.com/open-images-dataset/train/{i}.jpg",
            root / "images" / f"{i}.jpg",
        )
        for i in usable
    ]
    digests = fetch_images(jobs)
    records = []
    for i in usable:
        path = root / "images" / f"{i}.jpg"
        if str(path) in digests:
            records.append(
                {
                    "id": i,
                    "file": f"images/{i}.jpg",
                    "licence": "CC-BY-2.0",
                    "sha256": digests[str(path)],
                    "author": candidates[i]["author"],
                    "url": candidates[i]["url"],
                    "boxes": boxes[i],
                }
            )
    log(f"openimages: indexed {write_jsonl(root / 'index.jsonl.gz', records)}")


# ------------------------------------------------------------------ SAT


def step_sat(per_type: int) -> None:
    import pyarrow.parquet as pq

    root = RAW / "sat"
    path = hf_file(
        "array/SAT", "SAT_train.parquet", "bda5dde942d7f7b41bff7935f086ed9a9e348ae3"
    )
    table = pq.ParquetFile(path)
    types = table.read(columns=["question_type"]).column(0).to_pylist()
    by_type: dict[str, list[int]] = {}
    for i, t in enumerate(types):
        by_type.setdefault(t or "unknown", []).append(i)
    chosen: set[int] = set()
    for t, idx in by_type.items():
        rng = random.Random(f"sat:{t}")
        chosen.update(
            rng.sample(idx, min(per_type * (3 if t == "other" else 1), len(idx)))
        )
    log(
        f"sat: {len(types)} rows, types { {t: len(v) for t, v in by_type.items()} }, chosen {len(chosen)}"
    )
    records, offset = [], 0
    for batch in table.iter_batches(batch_size=32):
        n = batch.num_rows
        wanted = [i for i in range(offset, offset + n) if i in chosen]
        if wanted:
            rows = batch.to_pylist()
            for i in wanted:
                row = rows[i - offset]
                files, digests = [], []
                for k, image in enumerate(row["image_bytes"] or []):
                    payload = image["bytes"] if isinstance(image, dict) else image
                    ext = "png" if payload[:4] == b"\x89PNG" else "jpg"
                    rel = f"images/{i:06d}_{k}.{ext}"
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    (root / rel).write_bytes(payload)
                    files.append(rel)
                    digests.append(sha256(payload))
                records.append(
                    {
                        "id": i,
                        "files": files,
                        "sha256": digests,
                        "question": row["question"],
                        "answers": row["answers"],
                        "correct": row["correct_answer"],
                        "type": row["question_type"],
                    }
                )
        offset += n
    log(f"sat: indexed {write_jsonl(root / 'index.jsonl.gz', records)}")


# ------------------------------------------------------------------ DiffusionDB


def step_diffusiondb(parts: int) -> None:
    import pyarrow.parquet as pq

    root = RAW / "diffusiondb"
    rev = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
    meta = pq.read_table(
        hf_file("poloclub/diffusiondb", "metadata.parquet", rev),
        columns=[
            "image_name",
            "prompt",
            "part_id",
            "width",
            "height",
            "image_nsfw",
            "prompt_nsfw",
        ],
    )
    rows = [
        r
        for r in meta.to_pylist()
        if r["part_id"] <= parts
        and (r["image_nsfw"] or 1) < 0.2
        and (r["prompt_nsfw"] or 1) < 0.2
    ]
    wanted = {r["image_name"]: r for r in rows}
    records = []
    for part in range(1, parts + 1):
        zpath = hf_file("poloclub/diffusiondb", f"images/part-{part:06d}.zip", rev)
        with zipfile.ZipFile(zpath) as archive:
            for name in archive.namelist():
                if name in wanted:
                    payload = archive.read(name)
                    out = root / "images" / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(payload)
                    r = wanted[name]
                    records.append(
                        {
                            "id": name,
                            "file": f"images/{name}",
                            "sha256": sha256(payload),
                            "prompt": r["prompt"],
                            "w": r["width"],
                            "h": r["height"],
                        }
                    )
        log(f"diffusiondb: part {part} -> {len(records)}")
    log(f"diffusiondb: indexed {write_jsonl(root / 'index.jsonl.gz', records)}")


# ------------------------------------------------------------------ text sources


CLEAN = re.compile(r"(@\w+|<user>|<url>|https?://\S+|#\w+|&amp;|\bRT\b)", re.I)


def _clean(text: str) -> str:
    text = CLEAN.sub(" ", text or "")
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -:\"'")


def step_text() -> None:
    import pyarrow.parquet as pq

    root = RAW / "text"
    hate: list[dict] = []
    mhs = pq.read_table(
        hf_file(
            "ucberkeley-dlab/measuring-hate-speech",
            "measuring-hate-speech.parquet",
            "5468f6e118396646b02a2f691e771f6b6d9502ea",
        ),
        columns=["comment_id", "text", "hate_speech_score"],
    ).to_pylist()
    seen: set[int] = set()
    for row in mhs:
        if row["comment_id"] in seen:
            continue
        seen.add(row["comment_id"])
        text, score = _clean(row["text"]), row["hate_speech_score"]
        if not 4 <= len(text.split()) <= 35:
            continue
        if score > 0.5:
            hate.append(
                {
                    "id": f"mhs:{row['comment_id']}",
                    "source": "measuring-hate-speech",
                    "text": text,
                    "hateful": True,
                }
            )
        elif score < -1.0:
            hate.append(
                {
                    "id": f"mhs:{row['comment_id']}",
                    "source": "measuring-hate-speech",
                    "text": text,
                    "hateful": False,
                }
            )
    payload = http_get(
        "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
        timeout=300,
    )
    (root / "hatexplain.sha256").parent.mkdir(parents=True, exist_ok=True)
    (root / "hatexplain.sha256").write_text(sha256(payload))
    for post_id, post in json.loads(payload).items():
        labels = [a["label"] for a in post["annotators"]]
        top = max(set(labels), key=labels.count)
        if labels.count(top) < 2 or top == "offensive":
            continue
        text = _clean(" ".join(post["post_tokens"]))
        if 4 <= len(text.split()) <= 35:
            hate.append(
                {
                    "id": f"hatexplain:{post_id}",
                    "source": "hatexplain",
                    "text": text,
                    "hateful": top == "hatespeech",
                }
            )
    log(f"text: hate pool {len(hate)} ({sum(h['hateful'] for h in hate)} hateful)")
    write_jsonl(root / "hate.jsonl.gz", hate)
    mcq: list[dict] = []
    aqua = pq.read_table(
        hf_file(
            "deepmind/aqua_rat",
            "raw/train-00000-of-00001.parquet",
            "33301c6a050c96af81f63cad5562cb5363e88971",
        )
    ).to_pylist()
    for i, row in enumerate(aqua):
        options = [re.sub(r"^[A-E]\)\s*", "", o).strip() for o in row["options"]]
        if len(options) == 5 and row["correct"] in "ABCDE" and len(set(options)) == 5:
            mcq.append(
                {
                    "id": f"aqua:{i}",
                    "source": "aqua-rat",
                    "question": row["question"].strip(),
                    "options": options,
                    "gold": "ABCDE".index(row["correct"]),
                    "subject": "math",
                }
            )
    med = pq.read_table(
        hf_file(
            "openlifescienceai/medmcqa",
            "data/train-00000-of-00001.parquet",
            "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        )
    ).to_pylist()
    for row in med:
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        if (
            row.get("choice_type") == "single"
            and row["cop"] in (0, 1, 2, 3)
            and len(set(options)) == 4
            and all(o and len(o) < 120 for o in options)
            and len(row["question"]) < 400
        ):
            mcq.append(
                {
                    "id": f"medmcqa:{row['id']}",
                    "source": "medmcqa",
                    "question": row["question"].strip(),
                    "options": [o.strip() for o in options],
                    "gold": row["cop"],
                    "subject": row.get("subject_name") or "medicine",
                }
            )
    qasc = pq.read_table(
        hf_file(
            "allenai/qasc",
            "data/train-00000-of-00001.parquet",
            "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
        )
    ).to_pylist()
    for row in qasc:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        if row["answerKey"] in labels and len(texts) == 8:
            mcq.append(
                {
                    "id": f"qasc:{row['id']}",
                    "source": "qasc",
                    "question": row["question"].strip(),
                    "options": texts,
                    "gold": labels.index(row["answerKey"]),
                    "subject": "science",
                }
            )
    log(f"text: mcq pool {write_jsonl(root / 'mcq.jsonl.gz', mcq)}")


# ------------------------------------------------------------------ benchmark images and texts


BENCH_REPOS = {
    "cv-bench": (
        "nyu-visionx/CV-Bench",
        "bc284db50d036958861cb60cdd7b77612052ce0d",
        r"\.parquet$",
    ),
    "blink": (
        "BLINK-Benchmark/BLINK",
        "a3666eb249237ba3d5eca8db21176cc47967e040",
        r"\.parquet$",
    ),
    "realworldqa": (
        "xai-org/RealworldQA",
        "17e7f75e092e47169732462ea3cdfebe911105dd",
        r"\.parquet$",
    ),
    "charxiv": (
        "princeton-nlp/CharXiv",
        "f441eb632fc62f6f777830a0f47619e6e86459b0",
        r"^(val|test)\.parquet$",
    ),
    "mmmu-pro": (
        "MMMU/MMMU_Pro",
        "0d7426df4ccb3d8704a992fe6850abee7c262127",
        r"^vision/.*\.parquet$",
    ),
    "r-bench": (
        "R-Bench/R-Bench",
        "c0e92a6d90dad980ea51a47af64c12ec67b670ce",
        r"\.parquet$",
    ),
    "cord-v2": (
        "naver-clova-ix/cord-v2",
        "7f0115a4b758a71d6473b8d085751692da2fef98",
        r"\.parquet$",
    ),
    "funsd": (
        "nielsr/funsd",
        "7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5",
        r"\.parquet$",
    ),
    "infographicvqa": (
        "lmms-lab/DocVQA",
        "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        r"^InfographicVQA/.*\.parquet$",
    ),
    "mmmu": ("MMMU/MMMU", "876ce5cb130f7f7e290ce4d9984357737d4db5cf", r"\.parquet$"),
}
TEXT_FIELDS = (
    "question",
    "prompt",
    "options",
    "choices",
    "text",
    "reasoning_q",
    "descriptive_q1",
    "descriptive_q2",
    "descriptive_q3",
    "descriptive_q4",
    "caption",
    "caption_0",
    "caption_1",
    "answers",
    "answer",
)


def _image_payloads(value: Any) -> list[bytes]:
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return [bytes(value["bytes"])]
    if isinstance(value, (bytes, bytearray)) and value[:4] in (
        b"\x89PNG",
        b"\xff\xd8\xff\xe0",
        b"\xff\xd8\xff\xe1",
        b"\xff\xd8\xff\xdb",
        b"GIF8",
        b"RIFF",
    ):
        return [bytes(value)]
    if isinstance(value, list):
        return [p for v in value for p in _image_payloads(v)]
    return []


def _store_bench(payload: bytes, root: Path) -> tuple[str, str]:
    digest = sha256(payload)
    ext = (
        "png"
        if payload[:4] == b"\x89PNG"
        else ("gif" if payload[:4] == b"GIF8" else "jpg")
    )
    rel = f"images/{digest[:2]}/{digest}.{ext}"
    out = root / rel
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
    return rel, digest


def step_bench(names: list[str]) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files, snapshot_download

    root = RAW / "bench"
    index: list[dict] = []
    texts: list[dict] = []
    for name in names:
        marker = root / f".{name}.done"
        part_index, part_text = (
            root / f"index-{name}.jsonl.gz",
            root / f"text-{name}.jsonl.gz",
        )
        if marker.exists():
            continue
        items, strings = [], []
        if name == "hateful-memes":
            local = Path(
                snapshot_download(
                    "neuralcatcher/hateful_memes",
                    repo_type="dataset",
                    revision="d201c488dc7024623d1ecbcc987b3f132c4c2e12",
                    max_workers=16,
                )
            )
            for split in (
                "train",
                "dev_seen",
                "dev_unseen",
                "test_seen",
                "test_unseen",
            ):
                for line in open(local / f"{split}.jsonl"):
                    row = json.loads(line)
                    path = local / row["img"]
                    if path.exists():
                        rel, digest = _store_bench(path.read_bytes(), root)
                        items.append(
                            {
                                "benchmark": name,
                                "split": split,
                                "item_id": str(row["id"]),
                                "sha256": digest,
                                "path": rel,
                            }
                        )
                    strings.append(
                        {
                            "id": f"{name}:{split}:{row['id']}",
                            "texts": [row.get("text", "")],
                        }
                    )
        elif name == "coco2017-val":
            zpath = download(
                "http://images.cocodataset.org/zips/val2017.zip",
                root / "val2017.zip",
                815585330,
            )
            with zipfile.ZipFile(zpath) as archive:
                for member in archive.namelist():
                    if member.endswith(".jpg"):
                        rel, digest = _store_bench(archive.read(member), root)
                        items.append(
                            {
                                "benchmark": name,
                                "split": "val2017",
                                "item_id": Path(member).stem,
                                "sha256": digest,
                                "path": rel,
                            }
                        )
        else:
            repo, revision, pattern = BENCH_REPOS[name]
            files = [
                f
                for f in list_repo_files(repo, repo_type="dataset", revision=revision)
                if re.search(pattern, f)
            ]
            if name == "charxiv":
                zpath = hf_file(repo, "images.zip", revision)
                with zipfile.ZipFile(zpath) as archive:
                    for member in archive.namelist():
                        if member.lower().endswith((".jpg", ".png", ".jpeg")):
                            rel, digest = _store_bench(archive.read(member), root)
                            items.append(
                                {
                                    "benchmark": name,
                                    "split": "all",
                                    "item_id": Path(member).stem,
                                    "sha256": digest,
                                    "path": rel,
                                }
                            )
            for filename in files:
                split = Path(filename).stem.split("-")[0]
                parquet = pq.ParquetFile(hf_file(repo, filename, revision))
                for batch in parquet.iter_batches(batch_size=64):
                    for k, row in enumerate(batch.to_pylist()):
                        item_id = str(
                            row.get("id")
                            or row.get("idx")
                            or row.get("questionId")
                            or row.get("original_id")
                            or f"{filename}:{k}"
                        )
                        for key, value in row.items():
                            for payload in _image_payloads(value):
                                rel, digest = _store_bench(payload, root)
                                items.append(
                                    {
                                        "benchmark": name,
                                        "split": split,
                                        "item_id": item_id,
                                        "sha256": digest,
                                        "path": rel,
                                    }
                                )
                        found = [
                            (
                                row[f]
                                if isinstance(row[f], str)
                                else json.dumps(row[f], ensure_ascii=False)
                            )
                            for f in TEXT_FIELDS
                            if row.get(f)
                        ]
                        if found:
                            strings.append(
                                {"id": f"{name}:{split}:{item_id}", "texts": found}
                            )
                log(f"bench {name}: {filename} -> {len(items)} images")
        write_jsonl(part_index, items)
        write_jsonl(part_text, strings)
        marker.write_text(str(len(items)))
        log(f"bench {name}: {len(items)} images, {len(strings)} text items")
    for name in names:
        index += list(read_jsonl(root / f"index-{name}.jsonl.gz"))
        texts += list(read_jsonl(root / f"text-{name}.jsonl.gz"))
    write_jsonl(root / "index.jsonl.gz", index)
    write_jsonl(root / "text.jsonl.gz", texts)


VEGA_ROWS = Path("/data/d25/shared/data/v1/M2T-v5")
PERMISSIVE_PART = re.compile(
    r"^(apache[- ]?2\.0|apache license 2\.0|mit( license)?|bsd(-[23]-clause)?|cc0([- ]1\.0)?|cc[- ]by[- ][234]\.0|"
    r"odc[- ]by|cdla[- ]permissive(-[12]\.0)?|public[- ]domain|d20-permissive)( \(.*\))?$",
    re.I,
)


def permissive(licence: str | None) -> bool:
    parts = [p.strip() for p in re.split(r",(?![^()]*\))", licence or "") if p.strip()]
    return bool(parts) and all(PERMISSIVE_PART.match(p) for p in parts)


def step_vega(limit: int) -> None:
    """Vega M2T-v5 choice rows (permissive licence, 3-10 options, page-sized text) to render with options in the image."""
    from d25.vega.common import decision_format as df

    root = RAW / "vega"
    caps = {"knowledge": 0.45, "tasksource": 0.40, "d20": 0.10, "indist": 0.05}
    pools: dict[str, list[dict]] = {k: [] for k in caps}
    for path in sorted(VEGA_ROWS.glob("train-*.jsonl.gz")):
        for row in read_jsonl(path):
            q, meta = row["question"], row.get("meta") or {}
            part = row["source"].split(":")[0]
            if (
                part not in pools
                or q["type"] != "choice"
                or not permissive(meta.get("licence"))
            ):
                continue
            keys, texts = df.options(q)
            gold = meta.get("gold_target") or row["target"]
            if not 3 <= len(keys) <= 10 or max(gold) < 0.999:
                continue
            state = (
                df.describe(row.get("state"))
                if row.get("state") not in (None, "")
                else ""
            )
            if (
                len(state) > 900
                or any(len(t) > 160 for t in texts)
                or len(q.get("instructions") or "") > 300
            ):
                continue
            if state.startswith(("{", "[")) and len(state) > 300:
                continue
            pools[part].append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "family": row["family"],
                    "state": state,
                    "instructions": q.get("instructions") or "",
                    "options": texts,
                    "gold": int(max(range(len(gold)), key=gold.__getitem__)),
                    "teacher": (meta.get("teachers") or {}).get("pplx11"),
                    "licence": meta.get("licence"),
                    "dataset": meta.get("dataset"),
                }
            )
    chosen: list[dict] = []
    for part, share in caps.items():
        rows = sorted(pools[part], key=lambda r: sha256(f"vega:{r['id']}".encode()))
        per_family: dict[str, int] = {}
        cap_family = max(50, int(limit * share) // 40)
        taken = 0
        for r in rows:
            if per_family.get(r["family"], 0) >= cap_family:
                continue
            per_family[r["family"]] = per_family.get(r["family"], 0) + 1
            chosen.append(r)
            taken += 1
            if taken >= int(limit * share):
                break
    log(
        f"vega: candidates { {k: len(v) for k, v in pools.items()} }, chosen {len(chosen)}"
    )
    write_jsonl(
        root / "pool.jsonl.gz",
        sorted(chosen, key=lambda r: sha256(f"order:{r['id']}".encode())),
    )


def step_sscd() -> None:
    download(
        "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt",
        RAW / "sscd" / "sscd_disc_mixup.torchscript.pt",
        98791638,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="+")
    parser.add_argument("--coco", type=int, default=16000)
    parser.add_argument("--openimages", type=int, default=24000)
    parser.add_argument("--sat-per-type", type=int, default=3000)
    parser.add_argument("--diffusiondb-parts", type=int, default=12)
    parser.add_argument("--vega", type=int, default=30000)
    parser.add_argument(
        "--bench",
        default="cv-bench,blink,realworldqa,charxiv,mmmu-pro,r-bench,cord-v2,funsd,"
        "infographicvqa,hateful-memes,coco2017-val,mmmu",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for step in args.steps:
        done = RAW / step / ".done"
        if done.exists() and not args.force:
            log(f"{step}: done")
            continue
        log(f"{step}: start")
        if step == "coco":
            step_coco(args.coco)
        elif step == "openimages":
            step_openimages(args.openimages)
        elif step == "sat":
            step_sat(args.sat_per_type)
        elif step == "diffusiondb":
            step_diffusiondb(args.diffusiondb_parts)
        elif step == "text":
            step_text()
        elif step == "bench":
            step_bench(args.bench.split(","))
        elif step == "sscd":
            step_sscd()
        elif step == "vega":
            step_vega(args.vega)
        else:
            raise SystemExit(f"unknown step {step}")
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        log(f"{step}: done")


if __name__ == "__main__":
    sys.exit(main())
