"""BLINK proxy (private-set style): 4-option, two-image point correspondence on fresh photos.

The private BLINK set is all multi-image and all 4-option (single-image engines score exactly
-33.3 there), so every row has two images and four lettered points, drawn as in BLINK.

- ``visual``: an Open Images test photo and a second view made by a random homography plus a
  lighting, blur, noise and JPEG change (HPatches-style). The reference is a corner in view 1; the
  answer is its mapped position in view 2; distractors are other corners of view 2, half of them
  picked for local appearance similar to the reference.
- ``semantic-bird``: the same CUB-200-2011 part (beak, left wing, tail, ...) on two different birds;
  distractors are other visible parts of the second bird.
- ``semantic-person``: the same COCO train2017 body keypoint on two different people.

Prompts and option texts are BLINK's own. A forensic (real vs generated, 4 images) part needs a GPU
image generator and is specified in ws-proxy/DESIGN.md.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BENCHMARK = "BLINK"
NAME = "blink-proxy"
VERSION = "1"

VISUAL_PROMPT = (
    "A point is circled on the first image, labeled with REF. We change the camera position or lighting and "
    'shoot the second image. You are given multiple red-circled points on the second image, choices of "A, B, '
    'C, D" are drawn beside each circle. Which point on the second image corresponds to the point in the first image?'
)
SEMANTIC_PROMPT = (
    "Humans can find corresponding points for different objects in the same category. For instance, if there "
    "are images of two different cats, then the left ear tip of one cat corresponds to the left ear tip of the "
    "other cat, and the right front paw of one cat corresponds to the right front paw of the other cat.\n"
    "Given the following two images, a reference point is annotated on the first image, labeled with REF. You "
    'are given multiple red-circled points on the second image, choices of "A, B, C, D" are drawn beside each '
    "circle. Select between the choices on the second image and find the corresponding point for the reference "
    "point. Which point is corresponding to the reference point?"
)
OPTIONS = {"A": "Point A", "B": "Point B", "C": "Point C", "D": "Point D"}
WEIGHTS = {"visual": 0.5, "semantic-bird": 0.25, "semantic-person": 0.25}
CUB_SYMMETRIC = {7: 11, 11: 7, 8: 12, 12: 8, 9: 13, 13: 9}
COCO_NAMES = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
]
COCO_SYMMETRIC = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
    9: 10,
    10: 9,
    11: 12,
    12: 11,
    13: 14,
    14: 13,
    15: 16,
    16: 15,
}


def prepare(work: Path, seed: int, n: int = 500) -> dict[str, Any]:
    from d25.omni.proxy import sources as src
    from d25.omni.proxy.rows import allocate

    meta = src.open_images_metadata(work)
    ids = sorted(i for i in meta if src.pick(i, f"{NAME}:visual", 50, 1))
    files = src.open_images_files(work, ids[:1600])
    visual = [
        {"id": i, "path": str(p), "meta": meta[i]} for i, p in sorted(files.items())
    ]

    cub_root = src.cub(work)
    names = dict(
        line.split(" ", 1)
        for line in (cub_root / "images.txt").read_text().splitlines()
    )
    parts: dict[str, dict[int, tuple[float, float]]] = {}
    for line in (cub_root / "parts" / "part_locs.txt").read_text().splitlines():
        img, part, x, y, vis = line.split()
        if vis == "1":
            parts.setdefault(img, {})[int(part)] = (float(x), float(y))
    boxes = {}
    for line in (cub_root / "bounding_boxes.txt").read_text().splitlines():
        img, x, y, w, h = line.split()
        boxes[img] = (float(x), float(y), float(w), float(h))
    part_names = {
        int(a): b
        for a, b in (
            l.split(" ", 1)
            for l in (cub_root / "parts" / "parts.txt").read_text().splitlines()
        )
    }
    birds = [
        {
            "id": img,
            "path": str(cub_root / "images" / names[img].strip()),
            "parts": parts[img],
            "box": boxes[img],
            "species": names[img].split("/")[0],
        }
        for img in sorted(parts, key=int)
        if len(parts[img]) >= 9 and src.pick(img, f"{NAME}:cub", 6, 1)
    ]

    coco = src.coco_keypoints(work)
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    people = []
    for image_id, anns in sorted(by_image.items()):
        im = images[image_id]
        big = [
            a
            for a in anns
            if not a["iscrowd"] and a["area"] >= 0.10 * im["width"] * im["height"]
        ]
        if (
            len(big) != 1
            or sum(a["area"] >= 0.02 * im["width"] * im["height"] for a in anns) > 1
        ):
            continue
        a = big[0]
        kp = a["keypoints"]
        visible = {
            k: (kp[3 * k], kp[3 * k + 1]) for k in range(17) if kp[3 * k + 2] == 2
        }
        if len(visible) >= 13 and src.pick(str(image_id), f"{NAME}:coco", 8, 1):
            people.append(
                {
                    "id": image_id,
                    "box": a["bbox"],
                    "kp": visible,
                    "license": im.get("license"),
                    "flickr": im.get("flickr_url", ""),
                }
            )
    people = people[:900]
    files = src.coco_files(work, [p["id"] for p in people])
    people = [dict(p, path=str(files[p["id"]])) for p in people if p["id"] in files]
    ctx = {"visual": visual, "birds": birds, "people": people, "cub_parts": part_names}
    sizes = {
        "visual": len(visual),
        "semantic-bird": len(birds),
        "semantic-person": len(people),
    }
    ctx["plan"] = allocate(sizes, WEIGHTS, n, seed)
    return ctx


def sources(context: Any) -> dict[str, Any]:
    from d25.omni.proxy import sources as src

    return {
        "open-images-v7-test": {
            k: src.OPEN_IMAGES[k] for k in ("name", "licence", "metadata", "image")
        },
        "cub-200-2011": dict(src.CUB),
        "coco-2017-train-keypoints": dict(src.COCO_KEYPOINTS),
    }


def _far(p: tuple[float, float], others: list[tuple[float, float]], d: float) -> bool:
    return all(math.hypot(p[0] - q[0], p[1] - q[1]) >= d for q in others)


def visual_item(
    r: random.Random, ctx: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any] | None:
    import cv2

    from d25.omni.proxy import augment

    img1 = np.array(Image.open(entry["path"]).convert("RGB"))
    h, w = img1.shape[:2]
    if min(h, w) < 500:
        return None
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    corners1 = cv2.goodFeaturesToTrack(gray1, 300, 0.01, 18)
    if corners1 is None or len(corners1) < 40:
        return None
    hard = r.random()
    angle = math.radians(r.uniform(-1, 1) * (12 + 23 * hard))
    scale = r.uniform(0.8, 1.2)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    center = np.float32([w / 2, h / 2])
    dst = []
    for x, y in src_pts:
        dx, dy = x - center[0], y - center[1]
        px, py = center[0] + c * dx - s * dy, center[1] + s * dx + c * dy
        px += r.uniform(-1, 1) * w * (0.04 + 0.08 * hard)
        py += r.uniform(-1, 1) * h * (0.04 + 0.08 * hard)
        dst.append([px, py])
    H = cv2.getPerspectiveTransform(src_pts, np.float32(dst))
    img2 = cv2.warpPerspective(
        img1,
        H,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    f = img2.astype(np.float32)
    f = f * r.uniform(0.6, 1.35) + r.uniform(-25, 25)
    f = 255 * (np.clip(f, 0, 255) / 255) ** r.uniform(0.7, 1.4)
    f = f * np.array([r.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
    if r.random() < 0.5:
        f = cv2.GaussianBlur(f, (0, 0), r.uniform(0.4, 1.4))
    f = f + np.random.default_rng(r.randrange(2**31)).normal(
        0, r.uniform(1, 6), f.shape
    )
    img2 = np.clip(f, 0, 255).astype(np.uint8)
    margin = 0.06 * max(w, h)
    diag = math.hypot(w, h)
    pts1 = corners1.reshape(-1, 2)
    mapped = cv2.perspectiveTransform(
        pts1.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)
    inside = [
        i
        for i, (x, y) in enumerate(mapped)
        if margin <= x <= w - margin
        and margin <= y <= h - margin
        and margin <= pts1[i][0] <= w - margin
        and margin <= pts1[i][1] <= h - margin
    ]
    if len(inside) < 10:
        return None
    ref_i = r.choice(inside)
    ref, gt = tuple(pts1[ref_i]), tuple(mapped[ref_i])
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    corners2 = cv2.goodFeaturesToTrack(gray2, 300, 0.01, 18)
    if corners2 is None:
        return None
    cands = [
        tuple(p)
        for p in corners2.reshape(-1, 2)
        if margin <= p[0] <= w - margin
        and margin <= p[1] <= h - margin
        and math.hypot(p[0] - gt[0], p[1] - gt[1]) >= 0.07 * diag
    ]
    if len(cands) < 6:
        return None
    if r.random() < 0.5:
        orb = cv2.ORB_create()
        k1 = [cv2.KeyPoint(float(ref[0]), float(ref[1]), 31)]
        k2 = [cv2.KeyPoint(float(x), float(y), 31) for x, y in cands]
        _, d1 = orb.compute(gray1, k1)
        k2, d2 = orb.compute(gray2, k2)
        if d1 is not None and d2 is not None and len(k2) >= 6:
            dists = [int(cv2.norm(d1[0], d, cv2.NORM_HAMMING)) for d in d2]
            order = np.argsort(dists, kind="stable")
            cands = [tuple(k2[i].pt) for i in order]
    else:
        r.shuffle(cands)
    chosen: list[tuple[float, float]] = []
    for p in cands:
        if _far(p, [gt, *chosen], 0.06 * diag):
            chosen.append(p)
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        return None
    im1, im2 = Image.fromarray(img1), Image.fromarray(img2)
    augment.point_marker(im1, ref, "REF")
    points = [gt, *chosen]
    order = list(range(4))
    r.shuffle(order)
    for letter, k in zip("ABCD", order):
        augment.point_marker(im2, points[k], letter)
    answer = "ABCD"[order.index(0)]
    from d25.omni.proxy import sources as src

    prov = src.open_images_provenance(entry["id"], entry["meta"], Path(entry["path"]))
    return {
        "images": [im1, im2],
        "prompt": VISUAL_PROMPT,
        "answer": answer,
        "provenance": [prov, dict(prov, derived="homography+photometric")],
        "extra": {
            "rotation_deg": round(math.degrees(angle), 1),
            "scale": round(scale, 2),
            "hard": round(hard, 2),
        },
    }


def _crop(
    img: Image.Image,
    box: tuple[float, float, float, float],
    r: random.Random,
    long_side: int = 1024,
):
    x, y, bw, bh = box
    m = r.uniform(0.15, 0.45)
    x0, y0 = max(0, x - m * bw), max(0, y - m * bh)
    x1, y1 = min(img.width, x + bw * (1 + m)), min(img.height, y + bh * (1 + m))
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    s = long_side / max(crop.size)
    crop = crop.resize(
        (max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS
    )
    return crop, (x0, y0, s)


def semantic_item(
    r: random.Random, ctx: dict[str, Any], kind: str, a: dict[str, Any]
) -> dict[str, Any] | None:
    from d25.omni.proxy import augment
    from d25.omni.proxy import sources as src

    if kind == "semantic-bird":
        pool, key, sym = ctx["birds"], "parts", CUB_SYMMETRIC
        others = [
            b
            for b in pool
            if b["id"] != a["id"]
            and (b["species"] != a["species"]) == (r.random() < 0.7)
        ]
        if not others:
            return None
        b = r.choice(others)
    else:
        pool, key, sym = ctx["people"], "kp", COCO_SYMMETRIC
        b = r.choice([p for p in pool if p["id"] != a["id"]])
    common = sorted(set(a[key]) & set(b[key]))
    if len(common) < 6:
        return None
    ref_part = r.choice(common)
    imgs = []
    for entry in (a, b):
        img = Image.open(entry["path"]).convert("RGB")
        crop, (x0, y0, s) = _crop(img, entry["box"], r)
        pts = {k: ((x - x0) * s, (y - y0) * s) for k, (x, y) in entry[key].items()}
        imgs.append((crop, pts))
    (im1, p1), (im2, p2) = imgs
    diag = math.hypot(*im2.size)
    gt = p2[ref_part]
    pool_parts = [
        k for k in p2 if k != ref_part and (k != sym.get(ref_part) or r.random() < 0.25)
    ]
    r.shuffle(pool_parts)
    chosen: list[int] = []
    for k in pool_parts:
        if _far(p2[k], [gt, *[p2[c] for c in chosen]], 0.07 * diag):
            chosen.append(k)
        if len(chosen) == 3:
            break
    if len(chosen) < 3 or not all(
        0 <= v < lim for v, lim in zip(p1[ref_part], im1.size)
    ):
        return None
    augment.point_marker(im1, p1[ref_part], "REF")
    points = [gt, *[p2[c] for c in chosen]]
    order = list(range(4))
    r.shuffle(order)
    for letter, k in zip("ABCD", order):
        augment.point_marker(im2, points[k], letter)
    answer = "ABCD"[order.index(0)]
    if kind == "semantic-bird":
        name = ctx["cub_parts"].get(ref_part, str(ref_part))
        prov = [
            {
                "source": "cub-200-2011",
                "source_id": e["id"],
                "file": Path(e["path"]).name,
                "sha256": src.sha256_file(Path(e["path"])),
                "licence": src.CUB["licence"],
            }
            for e in (a, b)
        ]
    else:
        name = COCO_NAMES[ref_part]
        prov = [
            {
                "source": "coco-2017-train",
                "source_id": e["id"],
                "file": src.COCO_KEYPOINTS["image"].format(id=e["id"]),
                "sha256": src.sha256_file(Path(e["path"])),
                "licence": f"COCO licence id {e.get('license')}",
                "flickr": e.get("flickr", ""),
            }
            for e in (a, b)
        ]
    return {
        "images": [im1, im2],
        "prompt": SEMANTIC_PROMPT,
        "answer": answer,
        "provenance": prov,
        "extra": {"part": name, "symmetric_distractor": sym.get(ref_part) in chosen},
    }


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy import augment
    from d25.omni.proxy.rows import rng

    pools = {
        "visual": context["visual"],
        "semantic-bird": context["birds"],
        "semantic-person": context["people"],
    }
    candidates = list(context["plan"][index])
    fallback = rng(NAME, seed, index, "fallback")
    for attempt in range(len(candidates) + 400):
        r = rng(NAME, seed, index, attempt)
        if attempt < len(candidates):
            kind, j = candidates[attempt]
        else:
            kind = candidates[0][0]
            j = fallback.randrange(len(pools[kind]))
        entry = pools[kind][j]
        item = (
            visual_item(r, context, entry)
            if kind == "visual"
            else semantic_item(r, context, kind, entry)
        )
        if item is None:
            continue
        item["extra"]["reused_source"] = attempt >= len(candidates)
        payloads = [(augment.jpeg(im, 95), "jpg") for im in item["images"]]
        return {
            "item_id": f"{index:05d}",
            "subtask": kind,
            "payloads": payloads,
            "instructions": item["prompt"],
            "criteria": dict(OPTIONS),
            "answer": item["answer"],
            "provenance": item["provenance"],
            "extra": {"attempt": attempt, **item["extra"]},
        }
    raise RuntimeError(f"no valid correspondence item for {index}")
