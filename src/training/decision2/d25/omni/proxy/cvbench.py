"""CV-Bench proxy: counting, 2D relation, depth and 3D distance on images outside CV-Bench's pools.

CV-Bench draws 2D items from COCO val2017 and ADE20K val and 3D items from Omni3D test frames. Here:

- ``count`` and ``relation`` use Open Images V7 test photos (CC BY 2.0) with exhaustive box
  annotations: counts come from classes with a verified positive image label, no group-of or
  depiction box and no tiny boxes; relations use two classes with exactly one instance each that are
  fully separated along the asked axis. Counts span 1-12 (the private set is harder than the public
  one, whose counts are mostly 1-2).
- ``depth`` and ``distance`` use Hypersim validation scenes (CC BY-SA 3.0; none of the 23 Hypersim
  scenes behind CV-Bench): per-pixel depth, instance masks and world positions give exact answers;
  objects are boxed in red, blue and green as in CV-Bench.

Question and option wording follows CV-Bench; counts get 4-6 numeric options, the rest 2.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BENCHMARK = "CV-Bench"
NAME = "cvbench-proxy"
VERSION = "1"

WEIGHTS = {"count": 0.35, "relation": 0.25, "depth": 0.2, "distance": 0.2}
COUNTABLE = {
    "Car",
    "Bus",
    "Truck",
    "Bicycle",
    "Motorcycle",
    "Dog",
    "Cat",
    "Horse",
    "Bird",
    "Cattle",
    "Sheep",
    "Boat",
    "Airplane",
    "Chair",
    "Bottle",
    "Coffee cup",
    "Bowl",
    "Book",
    "Clock",
    "Vase",
    "Apple",
    "Orange",
    "Banana",
    "Pizza",
    "Doughnut",
    "Cake",
    "Laptop",
    "Television",
    "Mobile phone",
    "Traffic light",
    "Stop sign",
    "Bench",
    "Umbrella",
    "Handbag",
    "Backpack",
    "Suitcase",
    "Ball",
    "Kite",
    "Skateboard",
    "Surfboard",
    "Teddy bear",
    "Balloon",
    "Candle",
    "Lamp",
    "Pillow",
    "Mug",
    "Wine glass",
    "Duck",
    "Goose",
    "Penguin",
    "Elephant",
    "Giraffe",
    "Zebra",
    "Deer",
    "Hat",
    "Helmet",
    "Taxi",
    "Van",
    "Tomato",
    "Strawberry",
    "Lemon",
    "Egg",
    "Cookie",
    "Muffin",
    "Pumpkin",
    "Street light",
    "Tire",
    "Flag",
    "Poster",
    "Picture frame",
    "Bicycle wheel",
    "Drum",
    "Guitar",
    "Swan",
    "Chicken",
    "Monkey",
    "Bear",
    "Lion",
    "Tiger",
    "Fish",
    "Butterfly",
    "Tin can",
    "Traffic sign",
    "Wheel",
    "Sunglasses",
    "Watch",
    "Shelf",
    "Table",
    "Desk",
    "Sofa bed",
    "Couch",
}
CVBENCH_HYPERSIM_SCENES = {
    "ai_001_010",
    "ai_010_004",
    "ai_011_007",
    "ai_013_010",
    "ai_014_006",
    "ai_018_001",
    "ai_021_001",
    "ai_022_003",
    "ai_026_002",
    "ai_028_005",
    "ai_028_006",
    "ai_033_002",
    "ai_034_002",
    "ai_037_009",
    "ai_039_002",
    "ai_041_007",
    "ai_048_008",
    "ai_048_010",
    "ai_051_001",
    "ai_052_009",
    "ai_053_007",
    "ai_054_005",
    "ai_054_007",
}
HYPERSIM = {
    "name": "Hypersim (Apple ml-hypersim) validation scenes",
    "licence": "CC BY-SA 3.0",
    "split": "https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv",
    "zip": "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip",
}
NYU40 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]
OBJECT_IDS = {
    i + 1
    for i, n in enumerate(NYU40)
    if n
    not in (
        "wall",
        "floor",
        "ceiling",
        "otherstructure",
        "otherfurniture",
        "otherprop",
        "floor mat",
        "blinds",
    )
}


def _hypersim_frames(
    work: Path, seed: int, n_scenes: int = 30, per_scene: int = 16
) -> list[dict[str, Any]]:
    from remotezip import RemoteZip

    from d25.omni.proxy import sources as src

    root = work / "hypersim"
    index_path = root / f"frames-{seed}-{n_scenes}x{per_scene}.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    split = src.fetch(HYPERSIM["split"], root / "split.csv")
    frames: dict[str, list[tuple[str, int]]] = {}
    with open(split, newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["included_in_public_release"] == "True"
                and row["split_partition_name"] == "val"
                and row["scene_name"] not in CVBENCH_HYPERSIM_SCENES
            ):
                frames.setdefault(row["scene_name"], []).append(
                    (row["camera_name"], int(row["frame_id"]))
                )
    scenes = sorted(
        frames, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
    )[:n_scenes]
    out = []
    for scene in scenes:
        rnd = random.Random(f"{seed}:{scene}")
        chosen = rnd.sample(frames[scene], min(per_scene, len(frames[scene])))
        url = HYPERSIM["zip"].format(scene=scene)
        try:
            with RemoteZip(url) as zf:
                names = set(zf.namelist())
                for cam, fid in chosen:
                    stem = f"{scene}/images/scene_{cam}_geometry_hdf5/frame.{fid:04d}"
                    jpg = f"{scene}/images/scene_{cam}_final_preview/frame.{fid:04d}.tonemap.jpg"
                    need = [jpg] + [
                        f"{stem}.{k}.hdf5"
                        for k in (
                            "depth_meters",
                            "semantic",
                            "semantic_instance",
                            "position",
                        )
                    ]
                    if not all(n in names for n in need):
                        continue
                    local = root / scene / f"{cam}_{fid:04d}"
                    local.mkdir(parents=True, exist_ok=True)
                    for member in need:
                        target = local / member.rsplit("/", 1)[1]
                        if not target.exists():
                            target.write_bytes(zf.read(member))
                    out.append(
                        {
                            "scene": scene,
                            "camera": cam,
                            "frame": fid,
                            "dir": str(local),
                            "zip": url,
                            "member": jpg,
                        }
                    )
        except Exception as exc:
            print(f"hypersim {scene}: {exc}", flush=True)
    index_path.write_text(json.dumps(out))
    return out


def prepare(work: Path, seed: int, n: int = 600) -> dict[str, Any]:
    from d25.omni.proxy import sources as src
    from d25.omni.proxy.rows import allocate

    meta = src.open_images_metadata(work)
    boxes = src.open_images_boxes(work)
    positive = src.open_images_positive_labels(work)
    classes = src.open_images_classes(work)
    names = {mid: n for mid, n in classes.items() if n in COUNTABLE}
    count_items, relation_items = [], []
    for image_id in sorted(boxes):
        if image_id not in meta or not src.pick(image_id, f"{NAME}:oi", 12, 1):
            continue
        by_label: dict[str, list[dict[str, Any]]] = {}
        for b in boxes[image_id]:
            by_label.setdefault(b["label"], []).append(b)
        for label, items in by_label.items():
            if label not in names or label not in positive.get(image_id, set()):
                continue
            if any(b["group"] or b["depiction"] for b in items):
                continue
            areas = [(b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in items]
            if min(areas) < 0.002 or len(items) > 12:
                continue
            count_items.append({"id": image_id, "label": names[label], "n": len(items)})
        singles = [
            (names[l], v[0])
            for l, v in by_label.items()
            if l in names
            and len(v) == 1
            and not v[0]["group"]
            and not v[0]["depiction"]
            and l in positive.get(image_id, set())
        ]
        for i in range(len(singles)):
            for j in range(len(singles)):
                (na, a), (nb, b) = singles[i], singles[j]
                if na == nb or i == j:
                    continue
                if a["x1"] + 0.02 < b["x0"]:
                    relation_items.append(
                        {
                            "id": image_id,
                            "a": na,
                            "b": nb,
                            "axis": "x",
                            "answer": "left",
                        }
                    )
                elif a["y1"] + 0.02 < b["y0"]:
                    relation_items.append(
                        {
                            "id": image_id,
                            "a": na,
                            "b": nb,
                            "axis": "y",
                            "answer": "above",
                        }
                    )
    rnd = random.Random(seed)
    rnd.shuffle(count_items)
    rnd.shuffle(relation_items)
    by_n: dict[int, list[dict[str, Any]]] = {}
    for it in count_items:
        by_n.setdefault(it["n"], []).append(it)
    balanced = []
    for n, items in sorted(by_n.items()):
        balanced += items[: 60 if n <= 2 else 45]
    relation_items = relation_items[:500]
    ids = sorted({it["id"] for it in balanced + relation_items})
    files = src.open_images_files(work, ids)
    seen: set[str] = set()
    relation_unique = []
    for it in relation_items:
        if it["id"] not in seen:
            seen.add(it["id"])
            relation_unique.append(it)
    ctx = {
        "count": [
            dict(it, path=str(files[it["id"]]), meta=meta[it["id"]])
            for it in balanced
            if it["id"] in files
        ],
        "relation": [
            dict(it, path=str(files[it["id"]]), meta=meta[it["id"]])
            for it in relation_unique
            if it["id"] in files
        ],
        "frames": _hypersim_frames(work, seed),
    }
    sizes = {
        "count": len(ctx["count"]),
        "relation": len(ctx["relation"]),
        "depth": len(ctx["frames"]),
        "distance": len(ctx["frames"]),
    }
    ctx["plan"] = allocate(sizes, WEIGHTS, n, seed)
    return ctx


def sources(context: Any) -> dict[str, Any]:
    from d25.omni.proxy import sources as src

    return {
        "open-images-v7-test": {
            k: src.OPEN_IMAGES[k]
            for k in ("name", "licence", "boxes", "labels", "metadata", "image")
        },
        "hypersim-val": dict(HYPERSIM, excluded_scenes=sorted(CVBENCH_HYPERSIM_SCENES)),
    }


def count_item(
    r: random.Random, ctx: dict[str, Any], it: dict[str, Any]
) -> dict[str, Any] | None:
    from d25.omni.proxy import sources as src

    n = it["n"]
    k = r.choice([4, 5, 6])
    pool = sorted(
        {max(0, n + d) for d in (-3, -2, -1, 1, 2, 3)} | {0} - {n},
        key=lambda v: (abs(v - n), r.random()),
    )
    options = [str(n)] + [str(v) for v in pool[: k - 1]]
    if len(set(options)) < k:
        return None
    noun = it["label"].lower()
    plural = (
        noun
        if noun.endswith("s")
        else (
            noun[:-1] + "ies"
            if noun.endswith("y") and noun[-2] not in "aeiou"
            else noun + "s"
        )
    )
    question = f"How many {plural} are in the image?"
    return {
        "images": [Path(it["path"]).read_bytes()],
        "ext": "jpg",
        "question": question,
        "correct": str(n),
        "distractors": options[1:],
        "provenance": [
            src.open_images_provenance(it["id"], it["meta"], Path(it["path"]))
        ],
        "extra": {"count": n, "label": it["label"]},
    }


def relation_item(
    r: random.Random, ctx: dict[str, Any], it: dict[str, Any]
) -> dict[str, Any] | None:
    from d25.omni.proxy import sources as src

    a, b = it["a"].lower(), it["b"].lower()
    if r.random() < 0.5:
        a, b = b, a
        answer = {"left": "right", "above": "below"}[it["answer"]]
    else:
        answer = it["answer"]
    other = {"left": "right", "right": "left", "above": "below", "below": "above"}[
        answer
    ]
    question = (
        f"Considering the relative positions of the {a} and the {b} in the image provided, where is the "
        f"{a} located with respect to the {b}?"
    )
    order = ["left", "right"] if answer in ("left", "right") else ["above", "below"]
    return {
        "images": [Path(it["path"]).read_bytes()],
        "ext": "jpg",
        "question": question,
        "correct": answer,
        "distractors": [other],
        "keep_order": order,
        "provenance": [
            src.open_images_provenance(it["id"], it["meta"], Path(it["path"]))
        ],
        "extra": {"axis": it["axis"]},
    }


def _frame_objects(
    frame: dict[str, Any],
) -> tuple[Image.Image, list[dict[str, Any]]] | None:
    import h5py

    d = Path(frame["dir"])

    def load(kind: str) -> np.ndarray:
        with h5py.File(d / f"frame.{frame['frame']:04d}.{kind}.hdf5", "r") as handle:
            return np.array(handle["dataset"])

    depth, sem, inst, pos = (
        load("depth_meters"),
        load("semantic"),
        load("semantic_instance"),
        load("position"),
    )
    image = Image.open(d / f"frame.{frame['frame']:04d}.tonemap.jpg").convert("RGB")
    h, w = inst.shape
    objects = []
    for iid in np.unique(inst):
        if iid < 0:
            continue
        mask = inst == iid
        area = mask.mean()
        if area < 0.006 or area > 0.35:
            continue
        labels, counts = np.unique(sem[mask], return_counts=True)
        label = int(labels[np.argmax(counts)])
        if label not in OBJECT_IDS or counts.max() < 0.8 * mask.sum():
            continue
        dvals = depth[mask]
        dvals = dvals[np.isfinite(dvals)]
        pvals = pos[mask]
        pvals = pvals[np.all(np.isfinite(pvals), axis=1)]
        if len(dvals) < 50 or len(pvals) < 50:
            continue
        ys, xs = np.nonzero(mask)
        objects.append(
            {
                "name": NYU40[label - 1],
                "depth": float(np.median(dvals)),
                "center": np.median(pvals, axis=0),
                "box": (
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ),
                "area": float(area),
            }
        )
    return image, objects


def _hypersim_prov(frame: dict[str, Any]) -> dict[str, Any]:
    from d25.omni.proxy import sources as src

    path = Path(frame["dir"]) / f"frame.{frame['frame']:04d}.tonemap.jpg"
    return {
        "source": "hypersim-val",
        "source_id": f"{frame['scene']}/{frame['camera']}/{frame['frame']}",
        "file": frame["member"],
        "repo": frame["zip"],
        "sha256": src.sha256_file(path),
        "licence": HYPERSIM["licence"],
    }


def three_d_item(
    r: random.Random, ctx: dict[str, Any], kind: str, frame: dict[str, Any]
) -> dict[str, Any] | None:
    from d25.omni.proxy import augment

    loaded = _frame_objects(frame)
    if loaded is None:
        return None
    image, objects = loaded
    names = {}
    for o in objects:
        names.setdefault(o["name"], []).append(o)
    unique = [v[0] for v in names.values() if len(v) == 1]
    if kind == "depth":
        if len(unique) < 2:
            return None
        a, b = r.sample(unique, 2)
        ratio = max(a["depth"], b["depth"]) / min(a["depth"], b["depth"])
        if ratio < 1.15:
            return None
        question = (
            f"Which object is closer to the camera taking this photo, the {a['name']} (highlighted by a red box) "
            f"or the {b['name']} (highlighted by a blue box)?"
        )
        correct = a["name"] if a["depth"] < b["depth"] else b["name"]
        options = [a["name"], b["name"]]
        boxes = [(a, (255, 0, 0)), (b, (0, 0, 255))]
        extra = {"depth_ratio": round(ratio, 2)}
    else:
        if len(unique) < 3:
            return None
        a, b, c = r.sample(unique, 3)
        db = float(np.linalg.norm(a["center"] - b["center"]))
        dc = float(np.linalg.norm(a["center"] - c["center"]))
        ratio = max(db, dc) / max(1e-6, min(db, dc))
        if ratio < 1.3:
            return None
        question = (
            f"Estimate the real-world distances between objects in this image. Which object is closer to the "
            f"{a['name']} (highlighted by a red box), the {b['name']} (highlighted by a blue box) or the "
            f"{c['name']} (highlighted by a green box)?"
        )
        correct = b["name"] if db < dc else c["name"]
        options = [b["name"], c["name"]]
        boxes = [(a, (255, 0, 0)), (b, (0, 0, 255)), (c, (0, 255, 0))]
        extra = {"distance_ratio": round(ratio, 2)}
    for obj, color in boxes:
        augment.box_marker(image, obj["box"], color, width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    distractor = [o for o in options if o != correct]
    return {
        "images": [buffer.getvalue()],
        "ext": "jpg",
        "question": question,
        "correct": correct,
        "distractors": distractor,
        "keep_order": options,
        "provenance": [_hypersim_prov(frame)],
        "extra": extra,
    }


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy.rows import LETTERS, lettered, rng

    candidates = list(context["plan"][index])
    fallback = rng(NAME, seed, index, "fallback")
    for attempt in range(len(candidates) + 400):
        r = rng(NAME, seed, index, attempt)
        if attempt < len(candidates):
            kind, j = candidates[attempt]
        else:
            kind = candidates[0][0]
            j = fallback.randrange(
                len(
                    context["frames"]
                    if kind in ("depth", "distance")
                    else context[kind]
                )
            )
        if kind == "count":
            item = count_item(r, context, context["count"][j])
        elif kind == "relation":
            item = relation_item(r, context, context["relation"][j])
        else:
            item = three_d_item(r, context, kind, context["frames"][j])
        if item is None:
            continue
        item["extra"]["reused_source"] = attempt >= len(candidates)
        if "keep_order" in item:
            criteria = {LETTERS[i]: t for i, t in enumerate(item["keep_order"])}
            answer = LETTERS[item["keep_order"].index(item["correct"])]
        else:
            criteria, answer = lettered(item["correct"], item["distractors"], r)
        return {
            "item_id": f"{index:05d}",
            "subtask": kind,
            "payloads": [(p, item["ext"]) for p in item["images"]],
            "instructions": item["question"],
            "criteria": criteria,
            "answer": answer,
            "provenance": item["provenance"],
            "extra": {"attempt": attempt, **item["extra"]},
        }
    raise RuntimeError(f"no valid CV-Bench item for {index}")
