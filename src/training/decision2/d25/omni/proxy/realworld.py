"""RealWorldQA proxy: 4-option questions about fresh real-world photos taken from a car or a phone.

The private RealWorldQA set is 4-option (from the board's score lattice), so every row has 4 options.

- ``driving``: Argoverse 2 sensor validation logs (CC BY-NC-SA 4.0), front-centre camera. 3D cuboids
  (ego frame) projected with the published calibration give exact answers to RealWorldQA-style
  questions: cars within a distance, visible pedestrians, the closest road user in our corridor and
  its distance band. Objects near a threshold, outside the image or mostly hidden behind nearer
  objects are excluded from the counts, and items whose answer is close to a boundary are dropped.
- ``everyday``: Open Images V7 test photos (CC BY 2.0, outside the CV-Bench proxy slice): relative
  position with four directions, the object taking the most space, and counts phrased as sentences.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK = "RealWorldQA"
NAME = "realworldqa-proxy"
VERSION = "1"

AV2 = {
    "name": "Argoverse 2 sensor dataset, validation split",
    "licence": "CC BY-NC-SA 4.0 (Argoverse terms of use)",
    "bucket": "https://s3.amazonaws.com/argoverse",
    "prefix": "datasets/av2/sensor/val/",
}
CAMERA = "ring_front_center"
FRIENDLY = {
    "REGULAR_VEHICLE": "a car",
    "PEDESTRIAN": "a pedestrian",
    "BUS": "a bus",
    "BOX_TRUCK": "a truck",
    "TRUCK": "a truck",
    "LARGE_VEHICLE": "a large vehicle",
    "BICYCLIST": "a cyclist",
    "MOTORCYCLIST": "a motorcyclist",
    "BICYCLE": "a bicycle",
    "MOTORCYCLE": "a motorcycle",
    "SCHOOL_BUS": "a school bus",
    "ARTICULATED_BUS": "a bus",
}
WEIGHTS = {"driving": 0.5, "everyday": 0.5}
EVERYDAY_CLASSES = {
    "Car",
    "Bus",
    "Truck",
    "Bicycle",
    "Motorcycle",
    "Dog",
    "Cat",
    "Horse",
    "Bird",
    "Boat",
    "Chair",
    "Bottle",
    "Bench",
    "Umbrella",
    "Traffic light",
    "Stop sign",
    "Lamp",
    "Television",
    "Laptop",
    "Book",
    "Clock",
    "Vase",
    "Table",
    "Tree",
    "Building",
    "Person",
    "Street light",
    "Traffic sign",
    "Window",
    "Door",
    "Flower",
    "Backpack",
    "Handbag",
    "Suitcase",
    "Ball",
    "Bowl",
    "Coffee cup",
    "Mug",
    "Wine glass",
    "Pillow",
    "Couch",
    "Bed",
    "Desk",
    "Sink",
    "Toilet",
    "Mirror",
    "Plate",
    "Fountain",
    "Tower",
    "Skyscraper",
    "House",
    "Taxi",
    "Van",
    "Train",
    "Airplane",
}


def _list(prefix: str, delimiter: str = "/") -> tuple[list[str], list[str]]:
    keys, prefixes, token = [], [], None
    while True:
        url = f"{AV2['bucket']}?list-type=2&prefix={prefix}&delimiter={delimiter}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", text)
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", text)[1:]
        match = re.search(
            r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text
        )
        if not match:
            return keys, prefixes
        token = match.group(1)


def _quat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )


def _av2_frames(
    work: Path, seed: int, n_logs: int = 100, per_log: int = 4
) -> list[dict[str, Any]]:
    import pyarrow.feather as feather

    from d25.omni.proxy import sources as src

    root = work / "av2"
    index = root / "frames.json"
    if index.exists():
        return json.loads(index.read_text())
    _, logs = _list(AV2["prefix"])
    logs = sorted(logs, key=lambda p: src.item_seed(seed, p))[:n_logs]
    out = []
    for log_prefix in logs:
        log = log_prefix.rstrip("/").rsplit("/", 1)[1]
        local = root / log
        try:
            for name in (
                "annotations.feather",
                "calibration/egovehicle_SE3_sensor.feather",
                "calibration/intrinsics.feather",
            ):
                src.fetch(f"{AV2['bucket']}/{log_prefix}{name}", local / name)
            keys, _ = _list(f"{log_prefix}sensors/cameras/{CAMERA}/")
            cams = sorted(int(Path(k).stem) for k in keys if k.endswith(".jpg"))
            ann = feather.read_table(local / "annotations.feather").to_pandas()
            sweeps = sorted(ann["timestamp_ns"].unique())
            rnd = random.Random(f"{seed}:{log}")
            for ts in rnd.sample(sweeps[5:-5], min(per_log, max(0, len(sweeps) - 10))):
                cam = min(cams, key=lambda c: abs(c - ts)) if cams else None
                if cam is None or abs(cam - ts) > 30_000_000:
                    continue
                key = f"{log_prefix}sensors/cameras/{CAMERA}/{cam}.jpg"
                image = src.fetch(f"{AV2['bucket']}/{key}", local / f"{cam}.jpg")
                out.append(
                    {
                        "log": log,
                        "sweep": int(ts),
                        "camera_ts": int(cam),
                        "image": str(image),
                        "dir": str(local),
                        "key": key,
                    }
                )
        except Exception as exc:
            print(f"av2 {log}: {exc}", flush=True)
    index.write_text(json.dumps(out))
    return out


def _av2_objects(frame: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    import pyarrow.feather as feather

    local = Path(frame["dir"])
    ann = feather.read_table(local / "annotations.feather").to_pandas()
    ann = ann[ann["timestamp_ns"] == frame["sweep"]]
    ext = feather.read_table(
        local / "calibration/egovehicle_SE3_sensor.feather"
    ).to_pandas()
    intr = feather.read_table(local / "calibration/intrinsics.feather").to_pandas()
    e = ext[ext["sensor_name"] == CAMERA].iloc[0]
    k = intr[intr["sensor_name"] == CAMERA].iloc[0]
    R, t = _quat(e.qw, e.qx, e.qy, e.qz), np.array([e.tx_m, e.ty_m, e.tz_m])
    W, H = int(k.width_px), int(k.height_px)
    objects = []
    for row in ann.itertuples():
        center = np.array([row.tx_m, row.ty_m, row.tz_m])
        rot = _quat(row.qw, row.qx, row.qy, row.qz)
        corners = (
            np.array(
                [
                    [sx * row.length_m / 2, sy * row.width_m / 2, sz * row.height_m / 2]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ]
            )
            @ rot.T
            + center
        )
        cam = (corners - t) @ R
        if np.any(cam[:, 2] <= 0.5):
            continue
        u = k.fx_px * cam[:, 0] / cam[:, 2] + k.cx_px
        v = k.fy_px * cam[:, 1] / cam[:, 2] + k.cy_px
        box = (max(0.0, u.min()), max(0.0, v.min()), min(W, u.max()), min(H, v.max()))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        full = (u.max() - u.min()) * (v.max() - v.min())
        visible = (box[2] - box[0]) * (box[3] - box[1]) / max(full, 1e-6)
        objects.append(
            {
                "category": row.category,
                "distance": float(np.hypot(row.tx_m, row.ty_m)),
                "x": float(row.tx_m),
                "y": float(row.ty_m),
                "box": box,
                "in_frame": visible,
                "points": int(row.num_interior_pts),
            }
        )
    objects.sort(key=lambda o: o["distance"])
    for i, o in enumerate(objects):
        area = (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])
        covered = 0.0
        for nearer in objects[:i]:
            b = nearer["box"]
            ix = max(0.0, min(o["box"][2], b[2]) - max(o["box"][0], b[0]))
            iy = max(0.0, min(o["box"][3], b[3]) - max(o["box"][1], b[1]))
            covered = max(covered, ix * iy / max(area, 1e-6))
        o["occluded"] = covered
    return objects, (W, H)


def driving_item(
    r: random.Random, ctx: dict[str, Any], frame: dict[str, Any]
) -> dict[str, Any] | None:
    from d25.omni.proxy import sources as src

    objects, _ = _av2_objects(frame)
    good = [
        o
        for o in objects
        if o["in_frame"] >= 0.6
        and o["occluded"] < 0.5
        and o["points"] >= 15
        and o["x"] > 0
        and o["box"][3] - o["box"][1] >= 30
    ]
    kind = r.choice(["cars_within", "pedestrians", "closest", "distance_band"])
    if kind == "cars_within":
        d = r.choice([10, 15, 20, 30])
        cars = [
            o
            for o in objects
            if o["category"] == "REGULAR_VEHICLE" and o["x"] > 0 and o["in_frame"] > 0
        ]
        if any(abs(o["distance"] - d) < 2.5 for o in cars):
            return None
        n = sum(
            1
            for o in cars
            if o["distance"] < d and o["in_frame"] >= 0.6 and o["occluded"] < 0.5
        )
        if n > 6 or (n == 0 and r.random() < 0.7):
            return None
        question = f"How many cars are within {d} meters from us?"
        correct = str(n)
        pool = sorted(
            {v for v in range(0, 9) if v != n}, key=lambda v: (abs(v - n), r.random())
        )
        distractors = [str(v) for v in pool[:3]]
    elif kind == "pedestrians":
        peds = [
            o
            for o in objects
            if o["category"] == "PEDESTRIAN" and o["x"] > 0 and o["in_frame"] > 0
        ]
        if any(30 < o["distance"] < 45 for o in peds):
            return None
        n = sum(
            1
            for o in peds
            if o["distance"] <= 30 and o["in_frame"] >= 0.6 and o["occluded"] < 0.5
        )
        if not 1 <= n <= 8:
            return None
        question = "How many pedestrians are visible within about 30 meters of us?"
        correct = str(n)
        pool = sorted(
            {v for v in range(0, 11) if v != n}, key=lambda v: (abs(v - n), r.random())
        )
        distractors = [str(v) for v in pool[:3]]
    else:
        corridor = [o for o in good if abs(o["y"]) < 2.0 and o["category"] in FRIENDLY]
        if len(corridor) < 1:
            return None
        first = corridor[0]
        if len(corridor) > 1 and corridor[1]["distance"] - first["distance"] < 4:
            return None
        if kind == "closest":
            question = "What is the closest road user directly ahead of us in our lane?"
            correct = FRIENDLY[first["category"]]
            pool = sorted({v for v in FRIENDLY.values() if v != correct})
            distractors = r.sample(pool, 3)
        else:
            bands = [
                (0, 10, "less than 10 meters"),
                (10, 20, "between 10 and 20 meters"),
                (20, 40, "between 20 and 40 meters"),
                (40, 1e9, "more than 40 meters"),
            ]
            dist = first["distance"]
            if any(abs(dist - edge) < 2.0 for edge in (10, 20, 40)):
                return None
            noun = FRIENDLY.get(first["category"], "the vehicle").split(" ", 1)[1]
            question = (
                f"Approximately how far ahead of us is the closest {noun} in our lane?"
            )
            correct = next(label for lo, hi, label in bands if lo <= dist < hi)
            distractors = [label for _, _, label in bands if label != correct]
    prov = {
        "source": "argoverse2-sensor-val",
        "source_id": f"{frame['log']}/{frame['camera_ts']}",
        "file": frame["key"],
        "repo": AV2["bucket"],
        "sha256": src.sha256_file(Path(frame["image"])),
        "licence": AV2["licence"],
    }
    return {
        "images": [Path(frame["image"]).read_bytes()],
        "question": question,
        "correct": correct,
        "distractors": distractors,
        "provenance": [prov],
        "subtask": f"driving:{kind}",
    }


def everyday_item(
    r: random.Random, ctx: dict[str, Any], it: dict[str, Any]
) -> dict[str, Any] | None:
    from d25.omni.proxy import sources as src

    objs = it["objects"]
    kind = r.choice(["relpos", "largest", "count"])
    if kind == "relpos":
        singles = [o for o in objs if o["n"] == 1]
        if len(singles) < 2:
            return None
        a, b = r.sample(singles, 2)
        dx = (a["x0"] + a["x1"]) / 2 - (b["x0"] + b["x1"]) / 2
        dy = (a["y0"] + a["y1"]) / 2 - (b["y0"] + b["y1"]) / 2
        if abs(dx) > 2 * abs(dy) and (a["x1"] < b["x0"] or b["x1"] < a["x0"]):
            rel = "to the left of" if dx < 0 else "to the right of"
        elif abs(dy) > 2 * abs(dx) and (a["y1"] < b["y0"] or b["y1"] < a["y0"]):
            rel = "above" if dy < 0 else "below"
        else:
            return None
        an, bn = a["name"].lower(), b["name"].lower()
        texts = {
            k: f"The {an} is {k} the {bn}."
            for k in ("to the left of", "to the right of", "above", "below")
        }
        question = f"Where is the {an} relative to the {bn}?"
        correct, distractors = texts[rel], [v for k, v in texts.items() if k != rel]
    elif kind == "largest":
        singles = [o for o in objs if o["n"] == 1]
        if len(singles) < 4:
            return None
        pick = r.sample(singles, 4)
        areas = [(o["x1"] - o["x0"]) * (o["y1"] - o["y0"]) for o in pick]
        order = sorted(range(4), key=lambda i: -areas[i])
        if areas[order[0]] < 1.4 * areas[order[1]]:
            return None
        question = "Which of these objects takes up the most space in the image?"
        names = [f"The {o['name'].lower()}" for o in pick]
        correct, distractors = names[order[0]], [names[i] for i in order[1:]]
    else:
        counts = [o for o in objs if 2 <= o["n"] <= 9 and o["countable"]]
        if not counts:
            return None
        o = r.choice(counts)
        n = o["n"]
        noun = o["name"].lower()
        plural = noun if noun.endswith("s") else noun + "s"
        pool = sorted(
            {v for v in range(1, 13) if v != n}, key=lambda v: (abs(v - n), r.random())
        )
        question = f"How many {plural} are in this image?"
        correct = f"There are {n} {plural}."
        distractors = [f"There are {v} {plural}." for v in pool[:3]]
    prov = src.open_images_provenance(it["id"], it["meta"], Path(it["path"]))
    return {
        "images": [Path(it["path"]).read_bytes()],
        "question": question,
        "correct": correct,
        "distractors": distractors,
        "provenance": [prov],
        "subtask": f"everyday:{kind}",
    }


def prepare(work: Path, seed: int, n: int = 500) -> dict[str, Any]:
    from d25.omni.proxy import sources as src
    from d25.omni.proxy.rows import allocate

    meta = src.open_images_metadata(work)
    boxes = src.open_images_boxes(work)
    positive = src.open_images_positive_labels(work)
    classes = src.open_images_classes(work)
    names = {mid: n for mid, n in classes.items() if n in EVERYDAY_CLASSES}
    candidates = []
    for image_id in sorted(boxes):
        if image_id not in meta or not src.pick(image_id, f"{NAME}:oi", 25, 1):
            continue
        if src.pick(image_id, "cvbench-proxy:oi", 12, 1) or src.pick(
            image_id, "blink-proxy:visual", 50, 1
        ):
            continue
        by_label: dict[str, list[dict[str, Any]]] = {}
        for b in boxes[image_id]:
            by_label.setdefault(b["label"], []).append(b)
        objs = []
        for label, items in by_label.items():
            if (
                label not in names
                or label not in positive.get(image_id, set())
                or any(b["depiction"] for b in items)
            ):
                continue
            group = any(b["group"] for b in items)
            areas = [(b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in items]
            if group and len(items) == 1:
                continue
            b = items[0]
            objs.append(
                {
                    "name": names[label],
                    "n": len(items),
                    "countable": not group and min(areas) >= 0.002,
                    "x0": b["x0"],
                    "x1": b["x1"],
                    "y0": b["y0"],
                    "y1": b["y1"],
                }
            )
        if len(objs) >= 3:
            candidates.append({"id": image_id, "objects": objs})
    rnd = random.Random(seed)
    rnd.shuffle(candidates)
    candidates = candidates[:700]
    files = src.open_images_files(work, [c["id"] for c in candidates])
    everyday = [
        dict(c, path=str(files[c["id"]]), meta=meta[c["id"]])
        for c in candidates
        if c["id"] in files
    ]
    ctx = {"everyday": everyday, "frames": _av2_frames(work, seed)}
    ctx["plan"] = allocate(
        {"driving": len(ctx["frames"]), "everyday": len(everyday)}, WEIGHTS, n, seed
    )
    return ctx


def sources(context: Any) -> dict[str, Any]:
    from d25.omni.proxy import sources as src

    return {
        "argoverse2-sensor-val": dict(AV2),
        "open-images-v7-test": {
            k: src.OPEN_IMAGES[k]
            for k in ("name", "licence", "boxes", "metadata", "image")
        },
    }


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy.rows import lettered, rng

    pools = {"driving": context["frames"], "everyday": context["everyday"]}
    candidates = list(context["plan"][index])
    fallback = rng(NAME, seed, index, "fallback")
    for attempt in range(len(candidates) + 600):
        r = rng(NAME, seed, index, attempt)
        if attempt < len(candidates):
            kind, j = candidates[attempt]
        else:
            kind = candidates[0][0]
            j = fallback.randrange(len(pools[kind]))
        entry = pools[kind][j]
        item = (
            driving_item(r, context, entry)
            if kind == "driving"
            else everyday_item(r, context, entry)
        )
        if item is None or len(set([item["correct"], *item["distractors"]])) != 4:
            continue
        criteria, answer = lettered(item["correct"], item["distractors"], r)
        return {
            "item_id": f"{index:05d}",
            "subtask": item["subtask"],
            "payloads": [(p, "jpg") for p in item["images"]],
            "instructions": item["question"],
            "criteria": criteria,
            "answer": answer,
            "provenance": item["provenance"],
            "extra": {"attempt": attempt, "reused_source": attempt >= len(candidates)},
        }
    raise RuntimeError(f"no valid RealWorldQA item for {index}")
