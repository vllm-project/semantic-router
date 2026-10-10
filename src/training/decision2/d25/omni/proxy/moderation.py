"""Moderation proxy (Hateful Memes style): binary hateful / not hateful on memes outside Facebook Hateful Memes.

Memes come from three research meme sets with hate-type labels, read from the MemeLens collection
(CC BY-NC-4.0, pinned revision), and never from its ``Hateful_en_FHM`` config (the public benchmark):

- MAMI (SemEval-2022 Task 5) test memes, label misogynous / not misogynous (hate against women);
- Multi3Hate English memes (all splits), label hateful / not hateful from US annotators;
- MIMIC Islamophobia memes (test and validation), label hateful / not hateful.

The set is balanced per source. The meme text stays inside the image only (the state is empty), as
in Hateful Memes, where the meaning needs the picture and the caption together.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

BENCHMARK = "Moderation (Hateful Memes)"
NAME = "moderation-proxy"
VERSION = "1"

REPO = "QCRI/MemeLens"
CONFIGS = {
    "mami": (
        "misogynous_en__MAMI",
        ["test"],
        {"misogynous": 1, "not-misogynous": 0, "not misogynous": 0},
        260,
    ),
    "multi3hate": (
        "Hateful_en__Multi3Hate",
        ["test", "validation", "train"],
        {"hateful": 1, "not-hateful": 0},
        120,
    ),
    "mimic": (
        "Hateful_en__MIMIC_Islamophpbia",
        ["test", "validation"],
        {"hateful": 1, "not-hateful": 0},
        120,
    ),
}
QUESTION = (
    "Is this meme hateful? A meme is hateful when its picture and its text, taken together, attack or "
    "demean people because of a protected characteristic such as race, ethnicity, religion, nationality, "
    "gender, sexual orientation or disability."
)
CRITERIA = {"hateful": "Hateful", "not hateful": "Not hateful"}


def _revision() -> str:
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{REPO}", timeout=60
    ) as response:
        return json.loads(response.read())["sha"]


def prepare(work: Path, seed: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from d25.omni.proxy import sources as src

    root = work / "memelens"
    pin = root / "revision.txt"
    if not pin.exists():
        root.mkdir(parents=True, exist_ok=True)
        pin.write_text(_revision())
    revision = pin.read_text().strip()
    items = []
    for source, (config, splits, labels, quota) in CONFIGS.items():
        pool = {0: [], 1: []}
        for split in splits:
            api = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{config}"
            with urllib.request.urlopen(api, timeout=60) as response:
                files = [
                    f["path"]
                    for f in json.loads(response.read())
                    if f["path"].split("/")[-1].startswith(split + "-")
                ]
            for path in sorted(files):
                local = src.fetch(
                    f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}",
                    root / path,
                    timeout=900,
                )
                table = pq.read_table(local, columns=["id", "image", "label", "text"])
                for row in table.to_pylist():
                    label = labels.get(str(row["label"]).strip().lower())
                    payload = (row["image"] or {}).get("bytes")
                    if label is None or not payload:
                        continue
                    pool[label].append(
                        {
                            "source": source,
                            "config": config,
                            "split": split,
                            "id": row["id"],
                            "file": path,
                            "label": label,
                            "text": row["text"],
                        }
                    )
        rnd = random.Random(f"{seed}:{source}")
        for label in (0, 1):
            rnd.shuffle(pool[label])
        half = min(quota // 2, len(pool[0]), len(pool[1]))
        items += pool[0][:half] + pool[1][:half]
        print(
            f"moderation {source}: {len(pool[0])} not / {len(pool[1])} hateful available, {half} each used",
            flush=True,
        )
    random.Random(seed).shuffle(items)
    return {"items": items, "revision": revision, "root": str(root)}


def sources(context: Any) -> dict[str, Any]:
    return {
        "memelens": {
            "repo": REPO,
            "revision": context["revision"],
            "licence": "CC BY-NC-4.0 (collection); underlying sets research-only",
            "configs": {k: v[0] for k, v in CONFIGS.items()},
            "excluded_config": "Hateful_en_FHM",
        }
    }


_tables: dict[str, Any] = {}


def _image(context: dict[str, Any], item: dict[str, Any]) -> bytes:
    import pyarrow.parquet as pq

    path = Path(context["root"]) / item["file"]
    key = str(path)
    if key not in _tables:
        table = pq.read_table(path, columns=["id", "image"])
        _tables[key] = {r["id"]: r["image"]["bytes"] for r in table.to_pylist()}
    return _tables[key][item["id"]]


def size(context: Any) -> int:
    return len(context["items"])


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from PIL import Image

    item = context["items"][index]
    payload = _image(context, item)
    with Image.open(io.BytesIO(payload)) as im:
        fmt = (im.format or "JPEG").lower()
    ext = {"jpeg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}.get(fmt, "jpg")
    return {
        "item_id": f"{index:05d}",
        "subtask": item["source"],
        "payloads": [(payload, ext)],
        "instructions": QUESTION,
        "criteria": dict(CRITERIA),
        "answer": "hateful" if item["label"] else "not hateful",
        "provenance": [
            {
                "source": f"memelens/{item['config']}",
                "repo": REPO,
                "revision": context["revision"],
                "file": item["file"],
                "source_id": item["id"],
                "split": item["split"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "licence": "CC BY-NC-4.0 (MemeLens); research use",
            }
        ],
        "extra": {"meme_text": item["text"]},
    }
