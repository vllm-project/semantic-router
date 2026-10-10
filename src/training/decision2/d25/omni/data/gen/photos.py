"""Questions on licensed photos: COCO train2017 (CC BY / PD subset), Open Images train (CC BY 2.0), SAT
renders (MIT) and DiffusionDB images (CC0).

Single image: counting, 2D relations, nearest object, box localisation (CV-Bench, RealWorldQA, BLINK).
Multi-image (BLINK style, 2-4 images): warped-photo correspondence, keypoint correspondence across two
people, camera motion from shifted crops, real-versus-generated forensics, same-scene similarity and
jigsaw completion. Also word-swap caption pairs (Winoground; marked unverified until a GPU check) and
memes from hate-speech text on photos (Hateful Memes). SAT rows are converted as they are.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from d25.omni.data import render
from d25.omni.data.rows import Item, rng_for

RAW = Path(os.environ.get("D25_OMNI_DATA", "/data/d25/omni/data")) / "raw"
NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)


@lru_cache(maxsize=None)
def blacklist(name: str) -> frozenset:
    """Ids reserved by proxies (``protected/<name>-blacklist-*.json``)."""
    ids: set = set()
    for path in (RAW.parent / "protected").glob(f"{name}-blacklist-*.json"):
        ids.update(json.loads(path.read_text()))
    return frozenset(ids)


@lru_cache(maxsize=None)
def pool(name: str) -> list[dict]:
    path = RAW / name / "index.jsonl.gz"
    if not path.exists():
        return []
    banned = blacklist(name)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        return [row for row in rows if row["id"] not in banned]


@lru_cache(maxsize=None)
def text_pool(name: str) -> list[dict]:
    path = RAW / "text" / f"{name}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@lru_cache(maxsize=None)
def keypoint_names() -> list[str]:
    return json.loads((RAW / "coco" / "keypoints.json").read_text())


def load(record: dict, root: str) -> Image.Image:
    with Image.open(RAW / root / record["file"]) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def licence_of(record: dict, root: str) -> str:
    return {
        "coco": record.get("licence", "CC-BY-2.0") + "+CC-BY-4.0-annotations",
        "openimages": "CC-BY-2.0",
        "diffusiondb": "CC0-1.0",
    }[root]


def attribution(record: dict, root: str) -> str:
    if root == "coco":
        return record.get("flickr") or f"coco:{record['id']}"
    if root == "openimages":
        return (
            f"{record.get('author') or 'unknown'} ({record.get('url') or record['id']})"
        )
    return record["id"]


def photo(rng: random.Random, root: str | None = None) -> tuple[dict, str]:
    roots = [r for r in ([root] if root else ["coco", "openimages"]) if pool(r)]
    root = rng.choice(roots)
    return rng.choice(pool(root)), root


def marker_font(image: Image.Image, rng: random.Random):
    return render.font("sans_bold", max(14, int(max(image.size) * 0.028)), rng)


def _item(source, family, skill, images, question, records, **kw) -> Item:
    meta = kw.pop("meta", {})
    meta.setdefault("attribution", [attribution(r, root) for r, root in records])
    meta.setdefault("source_ids", [f"{root}:{r['id']}" for r, root in records])
    licences = sorted({licence_of(r, root) for r, root in records})
    meta["licences"] = licences
    return Item(
        source=source,
        family=family,
        skill=skill,
        images=images,
        image_kinds=kw.pop("image_kinds", ["jpeg"] * len(images)),
        question=question,
        orig_sha256=kw.pop("orig_sha256", [None] * len(images)),
        meta=meta,
        **kw,
    )


# ------------------------------------------------------------------ objects from annotations


def _objects(
    record: dict, root: str
) -> list[tuple[str, tuple[float, float, float, float], bool]]:
    """(label, box in pixels x0 y0 x1 y1, countable) for COCO or Open Images records."""
    out = []
    if root == "coco":
        for o in record["objects"]:
            x, y, w, h = o["bbox"]
            out.append(
                (
                    o["cat"],
                    (x, y, x + w, y + h),
                    not o["crowd"] and o["area"] >= 0.002 * record["w"] * record["h"],
                )
            )
    else:
        w, h = record.get("w"), record.get("h")
        for b in record["boxes"]:
            x0, y0, x1, y1 = b["box"]
            out.append(
                (
                    b["label"].lower(),
                    (x0, y0, x1, y1),
                    not b["group"]
                    and not b["depiction"]
                    and (x1 - x0) * (y1 - y0) >= 0.004,
                )
            )
    return out


def _scale_boxes(objs, root, image):
    if root == "coco":
        return objs
    w, h = image.size
    return [
        (label, (b[0] * w, b[1] * h, b[2] * w, b[3] * h), ok) for label, b, ok in objs
    ]


def plural(word: str) -> str:
    if word.endswith(("s", "sh", "ch", "x")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word in ("person", "man", "woman", "child", "mouse", "knife", "sheep", "skis"):
        return {
            "person": "people",
            "man": "men",
            "woman": "women",
            "child": "children",
            "mouse": "mice",
            "knife": "knives",
            "sheep": "sheep",
            "skis": "pairs of skis",
        }[word]
    return word + "s"


def count(index: int, root: str) -> Item | None:
    rng = rng_for("gen-photo", f"{root}-count", index)
    for _ in range(20):
        record, root = photo(rng, root)
        objs = _objects(record, root)
        labels = {}
        for label, box, ok in objs:
            labels.setdefault(label, []).append(ok)
        good = [
            l
            for l, oks in labels.items()
            if all(oks) and 1 <= len(oks) <= (9 if root == "coco" else 4)
        ]
        if good:
            break
    else:
        return None
    label = rng.choice(good)
    n = len(labels[label])
    image = load(record, root)
    k = rng.choice([4, 4, 5, 6])
    pool_values = [v for v in range(0, 11) if v != n]
    near = sorted(pool_values, key=lambda v: (abs(v - n), rng.random()))[: k - 1]
    options = [str(n)] + [str(v) for v in near]
    question = rng.choice(
        [
            f"How many {plural(label)} are in the image?",
            f"Count the {plural(label)} visible in the picture.",
            f"How many {plural(label)} can you see?",
        ]
    )
    return _item(
        "gen-photo",
        f"{root}-count",
        "spatial",
        [image],
        question,
        [(record, root)],
        options=options,
        orig_sha256=[record["sha256"]],
        meta={"benchmark_target": "CV-Bench", "label": label, "count": n},
    )


def relation(index: int, root: str) -> Item | None:
    rng = rng_for("gen-photo", f"{root}-relation", index)
    for _ in range(30):
        record, root = photo(rng, root)
        image_size = (record.get("w") or 1, record.get("h") or 1)
        objs = _objects(record, root)
        labels: dict[str, list] = {}
        for label, box, ok in objs:
            labels.setdefault(label, []).append((box, ok))
        unique = [(l, v[0][0]) for l, v in labels.items() if len(v) == 1 and v[0][1]]
        if len(unique) >= 2:
            (a, box_a), (b, box_b) = rng.sample(unique, 2)
            break
    else:
        return None
    image = load(record, root)
    w, h = image.size
    if root == "openimages":
        box_a = (box_a[0] * w, box_a[1] * h, box_a[2] * w, box_a[3] * h)
        box_b = (box_b[0] * w, box_b[1] * h, box_b[2] * w, box_b[3] * h)
    ax, ay = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    bx, by = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    horizontal = box_a[2] < box_b[0] or box_b[2] < box_a[0]
    vertical = box_a[3] < box_b[1] or box_b[3] < box_a[1]
    if horizontal and abs(ax - bx) > 0.15 * w and (not vertical or rng.random() < 0.7):
        gold, other = ("left", "right") if ax < bx else ("right", "left")
    elif vertical and abs(ay - by) > 0.15 * h:
        gold, other = ("above", "below") if ay < by else ("below", "above")
    else:
        return None
    style = rng.random()
    meta = {"benchmark_target": "CV-Bench", "objects": [a, b]}
    if style < 0.6:
        question = rng.choice(
            [
                f"Where is the {a} located with respect to the {b}?",
                f"Relative to the {b}, where is the {a} in the image?",
                (
                    f"In the image, is the {a} {gold} of or {other} of the {b}?"
                    if gold in ("left", "right")
                    else f"In the image, is the {a} {gold} or {other} the {b}?"
                ),
            ]
        )
        options = [gold, other]
        keys = [gold, other] if rng.random() < 0.4 else None
        return _item(
            "gen-photo",
            f"{root}-relation",
            "spatial",
            [image],
            question,
            [(record, root)],
            options=options,
            keys=keys,
            orig_sha256=[record["sha256"]],
            meta=meta,
        )
    phrase = {
        "left": "to the left of",
        "right": "to the right of",
        "above": "above",
        "below": "below",
    }
    truth = rng.random() < 0.5
    said = gold if truth else other
    return _item(
        "gen-photo",
        f"{root}-relation",
        "spatial",
        [image],
        f"Is the {a} {phrase[said]} the {b} in the image?",
        [(record, root)],
        noul=truth,
        orig_sha256=[record["sha256"]],
        meta=meta,
    )


def nearest(index: int, root: str) -> Item | None:
    rng = rng_for("gen-photo", f"{root}-nearest", index)
    for _ in range(30):
        record, root = photo(rng, root)
        objs = _objects(record, root)
        labels: dict[str, list] = {}
        for label, box, ok in objs:
            labels.setdefault(label, []).append((box, ok))
        unique = [(l, v[0][0]) for l, v in labels.items() if len(v) == 1 and v[0][1]]
        if len(unique) >= 4:
            break
    else:
        return None
    rng.shuffle(unique)
    ref, others = unique[0], unique[1:5]
    centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    rx, ry = centre(ref[1])
    dist = sorted(
        ((math.hypot(centre(b)[0] - rx, centre(b)[1] - ry), l) for l, b in others)
    )
    if dist[1][0] < 1.6 * dist[0][0]:
        return None
    image = load(record, root)
    options = [dist[0][1]] + [l for _, l in dist[1:]]
    question = f"Which of these objects is closest to the {ref[0]} in the image?"
    return _item(
        "gen-photo",
        f"{root}-nearest",
        "spatial",
        [image],
        question,
        [(record, root)],
        options=options,
        orig_sha256=[record["sha256"]],
        meta={"benchmark_target": "RealWorldQA", "reference": ref[0]},
    )


def localize(index: int) -> Item | None:
    rng = rng_for("gen-photo", "coco-localize", index)
    for _ in range(30):
        record, root = photo(rng, "coco")
        objs = [(l, b) for l, b, ok in _objects(record, root) if ok]
        labels: dict[str, list] = {}
        for label, box in objs:
            labels.setdefault(label, []).append(box)
        unique = [
            (l, v[0])
            for l, v in labels.items()
            if len(v) == 1
            and (v[0][2] - v[0][0]) * (v[0][3] - v[0][1])
            > 0.01 * record["w"] * record["h"]
        ]
        if unique:
            break
    else:
        return None
    label, box = rng.choice(unique)
    image = load(record, root)
    w, h = image.size
    bw, bh = box[2] - box[0], box[3] - box[1]
    others = [b for l, b in objs if l != label]
    if others and rng.random() < 0.4:
        wrong = rng.choice(others)
    else:
        dx, dy = rng.choice([-1, 1]) * bw * rng.uniform(0.45, 0.9), rng.choice(
            [-1, 1]
        ) * bh * rng.uniform(0.3, 0.8)
        s = rng.uniform(0.7, 1.4)
        cx, cy = (box[0] + box[2]) / 2 + dx, (box[1] + box[3]) / 2 + dy
        wrong = (
            max(0, cx - bw * s / 2),
            max(0, cy - bh * s / 2),
            min(w - 1, cx + bw * s / 2),
            min(h - 1, cy + bh * s / 2),
        )
    if wrong[2] - wrong[0] < 8 or wrong[3] - wrong[1] < 8:
        return None
    face = marker_font(image, rng)
    draw = ImageDraw.Draw(image)
    first_gold = rng.random() < 0.5
    boxes = [box, wrong] if first_gold else [wrong, box]
    colors = rng.sample(["#ff0000", "#0050ff", "#00b050", "#ff9900"], 2)
    for label_text, b, color in zip("AB", boxes, colors):
        render.mark_box(
            draw,
            tuple(int(v) for v in b),
            label_text,
            face,
            color=color,
            width=max(3, w // 200),
        )
    return _item(
        "gen-photo",
        "coco-localize",
        "spatial",
        [image],
        f"Which marked box tightly encloses the {label}?",
        [(record, root)],
        options=["box A", "box B"],
        gold=0 if first_gold else 1,
        fixed_order=True,
        orig_sha256=[record["sha256"]],
        meta={"benchmark_target": "BLINK", "label": label},
    )


# ------------------------------------------------------------------ multi-image


def _corners(gray: np.ndarray, n: int, quality: float = 0.02) -> np.ndarray:
    import cv2

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=n,
        qualityLevel=quality,
        minDistance=max(8, gray.shape[1] // 40),
    )
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)


def warp_corr(index: int) -> Item | None:
    import cv2

    rng = rng_for("gen-photo", "warp-corr", index)
    record, root = photo(rng)
    first = render.fit_pixels(load(record, root), 900_000)
    w, h = first.size
    src = np.asarray(first)
    j = 0.12 * min(w, h)
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = quad + np.float32(
        [[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)]
    )
    scale = rng.uniform(0.75, 1.15)
    centre = np.float32([w / 2, h / 2])
    dst = (
        (dst - centre) * scale
        + centre
        + np.float32([rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h])
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(src, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    second = Image.fromarray(warped)
    second = ImageEnhance.Brightness(second).enhance(rng.uniform(0.65, 1.35))
    second = ImageEnhance.Color(second).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.4:
        second = second.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    gray1 = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.asarray(second), cv2.COLOR_RGB2GRAY)
    margin = 0.1
    candidates = [
        p
        for p in _corners(gray1, 60)
        if margin * w < p[0] < (1 - margin) * w and margin * h < p[1] < (1 - margin) * h
    ]
    rng.shuffle(candidates)
    for ref in candidates[:10]:
        mapped = cv2.perspectiveTransform(
            ref.reshape(1, 1, 2).astype(np.float32), matrix
        ).reshape(2)
        if not (0.06 * w < mapped[0] < 0.94 * w and 0.06 * h < mapped[1] < 0.94 * h):
            continue
        spread = 0.1 * min(w, h)
        others = [
            p
            for p in _corners(gray2, 80)
            if np.hypot(*(p - mapped)) > spread
            and 0.05 * w < p[0] < 0.95 * w
            and 0.05 * h < p[1] < 0.95 * h
        ]
        rng.shuffle(others)
        picked: list[np.ndarray] = []
        for p in others:
            if all(np.hypot(*(p - q)) > spread for q in picked):
                picked.append(p)
            if len(picked) == 3:
                break
        if len(picked) < 3:
            continue
        face = marker_font(first, rng)
        d1, d2 = ImageDraw.Draw(first), ImageDraw.Draw(second)
        radius = max(8, int(min(w, h) * 0.018))
        render.mark_point(d1, tuple(ref), "REF", face, radius=radius)
        points = [mapped] + picked
        order = list(range(4))
        rng.shuffle(order)
        for label, k in zip("ABCD", order):
            render.mark_point(d2, tuple(points[k]), label, face, radius=radius)
        gold = order.index(0)
        question = rng.choice(
            [
                "The first image marks a reference point labelled REF. The second image shows the same scene after a "
                "change of viewpoint and lighting. Which labelled point in the second image is the same physical point?",
                "Which of the points A to D in the second picture corresponds to the REF point in the first picture?",
            ]
        )
        return _item(
            "gen-photo",
            "warp-corr",
            "multi_image",
            [first, second],
            question,
            [(record, root)],
            options=[f"point {l}" for l in "ABCD"],
            gold=gold,
            fixed_order=True,
            orig_sha256=[record["sha256"], record["sha256"]],
            meta={"benchmark_target": "BLINK"},
        )
    return None


SEMANTIC_PARTS = {
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
    0: 0,
}


@lru_cache(maxsize=None)
def people_pool() -> list[dict]:
    return [
        r
        for r in pool("coco")
        if any(
            sum(1 for v in p["kps"][2::3] if v == 2) >= 12
            and p["bbox"][2] * p["bbox"][3] > 0.06 * r["w"] * r["h"]
            for p in r["people"]
        )
    ]


@lru_cache(maxsize=None)
def photographic_pool() -> list[dict]:
    return [g for g in pool("diffusiondb") if PHOTO_PROMPT.search(g["prompt"] or "")]


@lru_cache(maxsize=None)
def sat_pool(dynamic: bool) -> list[dict]:
    return [r for r in pool("sat") if (len(r["files"]) > 1) == dynamic]


def kp_corr(index: int) -> Item | None:
    rng = rng_for("gen-photo", "kp-corr", index)
    people = people_pool()
    if len(people) < 2:
        return None
    names = keypoint_names()
    first_rec, second_rec = rng.sample(people, 2)

    def crop(record):
        person = max(
            (
                p
                for p in record["people"]
                if sum(1 for v in p["kps"][2::3] if v == 2) >= 12
            ),
            key=lambda p: p["bbox"][2] * p["bbox"][3],
        )
        image = load(record, "coco")
        x, y, bw, bh = person["bbox"]
        m = 0.25
        box = (
            max(0, x - m * bw),
            max(0, y - m * bh),
            min(image.width, x + bw * (1 + m)),
            min(image.height, y + bh * (1 + m)),
        )
        part = image.crop(tuple(int(v) for v in box))
        scale = 520 / max(part.size) if max(part.size) < 520 else 1.0
        if scale != 1.0:
            part = part.resize(
                (int(part.width * scale), int(part.height * scale)),
                Image.Resampling.LANCZOS,
            )
        kps = person["kps"]
        pts = {
            k: ((kps[3 * k] - box[0]) * scale, (kps[3 * k + 1] - box[1]) * scale)
            for k in range(17)
            if kps[3 * k + 2] == 2
        }
        return part, pts

    first, p1 = crop(first_rec)
    second, p2 = crop(second_rec)
    shared = [k for k in p1 if k in p2 and k in SEMANTIC_PARTS and k > 4]
    if not shared:
        return None
    k = rng.choice(shared)
    distractors = (
        [SEMANTIC_PARTS[k]]
        if SEMANTIC_PARTS[k] in p2 and SEMANTIC_PARTS[k] != k
        else []
    )
    rest = [o for o in p2 if o != k and o not in distractors]
    rng.shuffle(rest)
    min_gap = 0.06 * max(second.size)
    chosen = [p2[k]]
    labels = [k]
    for o in distractors + rest:
        if all(math.hypot(p2[o][0] - c[0], p2[o][1] - c[1]) > min_gap for c in chosen):
            chosen.append(p2[o])
            labels.append(o)
        if len(chosen) == 4:
            break
    if len(chosen) < 4:
        return None
    face = marker_font(second, rng)
    radius = max(7, int(max(second.size) * 0.016))
    render.mark_point(ImageDraw.Draw(first), p1[k], "REF", face, radius=radius)
    order = list(range(4))
    rng.shuffle(order)
    d2 = ImageDraw.Draw(second)
    for label, j in zip("ABCD", order):
        render.mark_point(d2, chosen[j], label, face, radius=radius)
    return _item(
        "gen-photo",
        "kp-corr",
        "multi_image",
        [first, second],
        "A body part of the person in the first image is marked REF. Which labelled point in the second "
        "image marks the same body part of the other person?",
        [(first_rec, "coco"), (second_rec, "coco")],
        options=[f"point {l}" for l in "ABCD"],
        gold=order.index(0),
        fixed_order=True,
        orig_sha256=[first_rec["sha256"], second_rec["sha256"]],
        meta={
            "benchmark_target": "BLINK",
            "keypoint": names[k],
            "distractors": [names[o] for o in labels[1:]],
        },
    )


def _square(image: Image.Image, size: int, rng: random.Random) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    x = rng.randint(0, w - side) if w > side else 0
    y = rng.randint(0, h - side) if h > side else 0
    return image.crop((x, y, x + side, y + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


PHOTO_PROMPT = re.compile(
    r"\b(photo|photograph|realistic|35mm|dslr|portrait|landscape|street|cinematic|"
    r"hyperrealistic|8k|canon|nikon|film)\b",
    re.I,
)


def forensic(index: int) -> Item | None:
    rng = rng_for("gen-photo", "forensic", index)
    generated = pool("diffusiondb")
    if len(generated) < 10:
        return None
    photographic = photographic_pool() or generated
    size = rng.choice([384, 448, 512])
    find_real = rng.random() < 0.5
    n_real = 1 if find_real else 3
    reals = [photo(rng) for _ in range(n_real)]
    fakes = [
        rng.choice(photographic if rng.random() < 0.7 else generated)
        for _ in range(4 - n_real)
    ]
    images, shas, records, kinds = [], [], [], []
    for record, root in reals:
        images.append(_square(load(record, root), size, rng))
        shas.append(record["sha256"])
        records.append((record, root))
        kinds.append("real")
    for record in fakes:
        images.append(_square(load(record, "diffusiondb"), size, rng))
        shas.append(record["sha256"])
        records.append((record, "diffusiondb"))
        kinds.append("generated")
    order = list(range(4))
    rng.shuffle(order)
    images = [images[i] for i in order]
    shas = [shas[i] for i in order]
    kinds = [kinds[i] for i in order]
    target = "real" if find_real else "generated"
    gold = kinds.index(target)
    question = (
        "Which of the four images is most likely a real photograph rather than AI-generated?"
        if find_real
        else "Which of the four images is most likely AI-generated rather than a real photograph?"
    )
    return _item(
        "gen-photo",
        "forensic",
        "multi_image",
        images,
        question,
        records,
        options=[f"image {i + 1}" for i in range(4)],
        gold=gold,
        fixed_order=True,
        orig_sha256=shas,
        meta={"benchmark_target": "BLINK", "kinds": kinds},
    )


def camera_motion(index: int) -> Item | None:
    rng = rng_for("gen-photo", "camera-motion", index)
    record, root = photo(rng)
    image = render.fit_pixels(load(record, root), 1_200_000)
    w, h = image.size
    mode = rng.choice(["pan", "pan", "tilt", "zoom"])
    if mode == "pan":
        cw = int(w * rng.uniform(0.55, 0.72))
        shift = int(w * rng.uniform(0.15, 0.3))
        x0 = rng.randint(0, max(0, w - cw - shift))
        a, b = image.crop((x0, 0, x0 + cw, h)), image.crop(
            (x0 + shift, 0, x0 + shift + cw, h)
        )
        move = "right"
    elif mode == "tilt":
        ch = int(h * rng.uniform(0.55, 0.72))
        shift = int(h * rng.uniform(0.15, 0.3))
        y0 = rng.randint(0, max(0, h - ch - shift))
        a, b = image.crop((0, y0, w, y0 + ch)), image.crop(
            (0, y0 + shift, w, y0 + shift + ch)
        )
        move = "down"
    else:
        f = rng.uniform(0.55, 0.75)
        cw, ch = int(w * f), int(h * f)
        x0, y0 = (w - cw) // 2 + rng.randint(-w // 20, w // 20), (
            h - ch
        ) // 2 + rng.randint(-h // 20, h // 20)
        a, b = image, image.crop((x0, y0, x0 + cw, y0 + ch)).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        move = "in"
    if rng.random() < 0.5:
        a, b = b, a
        move = {"right": "left", "down": "up", "in": "out"}[move]
    if mode == "zoom":
        options, gold = ["zoomed in", "zoomed out", "moved left", "moved right"], [
            "in",
            "out",
        ].index(move)
    else:
        options = ["moved left", "moved right", "moved up", "moved down"]
        gold = ["left", "right", "up", "down"].index(move)
    if rng.random() < 0.4 and mode == "pan":
        options, gold = options[:2], gold
    question = rng.choice(
        [
            "The two images were taken one after the other with the same camera. How did the camera "
            "move from the first image to the second?",
            "From the first frame to the second frame, which way did the camera move?",
        ]
    )
    return _item(
        "gen-photo",
        "camera-motion",
        "multi_image",
        [a, b],
        question,
        [(record, root)],
        options=options,
        gold=gold,
        orig_sha256=[record["sha256"]] * 2,
        meta={"benchmark_target": "BLINK", "mode": mode},
    )


def similarity(index: int) -> Item | None:
    rng = rng_for("gen-photo", "similarity", index)
    record, root = photo(rng)
    base = load(record, root)
    w, h = base.size
    f = rng.uniform(0.6, 0.85)
    cw, ch = int(w * f), int(h * f)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    same = base.crop((x0, y0, x0 + cw, y0 + ch))
    same = ImageEnhance.Color(same).enhance(rng.uniform(0.5, 1.5))
    same = ImageEnhance.Brightness(same).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.5:
        same = ImageOps.mirror(same)
    labels = {l for l, _, _ in _objects(record, root)}
    candidates = [
        r
        for r in rng.sample(pool(root), min(400, len(pool(root))))
        if r["id"] != record["id"] and labels & {l for l, _, _ in _objects(r, root)}
    ]
    k = rng.choice([2, 3])
    if len(candidates) < k - 1:
        return None
    others = [(r, root) for r in candidates[: k - 1]]
    images = [same] + [load(r, root) for r, _ in others]
    shas = [record["sha256"]] + [r["sha256"] for r, _ in others]
    order = list(range(k))
    rng.shuffle(order)
    cands = [images[i] for i in order]
    gold = order.index(0)
    question = (
        f"The first image is a reference. Which of the next {k} images shows the same scene (possibly "
        "cropped, mirrored or recoloured)?"
    )
    return _item(
        "gen-photo",
        "similarity",
        "multi_image",
        [base] + cands,
        question,
        [(record, root)] + others,
        options=[f"image {i + 2}" for i in range(k)],
        gold=gold,
        fixed_order=True,
        orig_sha256=[record["sha256"]] + [shas[i] for i in order],
        meta={"benchmark_target": "BLINK"},
    )


def jigsaw(index: int) -> Item | None:
    rng = rng_for("gen-photo", "jigsaw", index)
    record, root = photo(rng)
    image = render.fit_pixels(load(record, root), 1_000_000)
    w, h = image.size
    pw, ph = int(w * rng.uniform(0.2, 0.3)), int(h * rng.uniform(0.2, 0.3))
    x, y = rng.randint(int(0.1 * w), int(0.9 * w) - pw), rng.randint(
        int(0.1 * h), int(0.9 * h) - ph
    )
    patch = image.crop((x, y, x + pw, y + ph))
    for _ in range(20):
        ox, oy = rng.randint(0, w - pw), rng.randint(0, h - ph)
        if abs(ox - x) > pw or abs(oy - y) > ph:
            break
    else:
        return None
    decoy = image.crop((ox, oy, ox + pw, oy + ph))
    masked = image.copy()
    ImageDraw.Draw(masked).rectangle(
        (x, y, x + pw, y + ph),
        fill=rng.choice([(0, 0, 0), (255, 255, 255), (128, 128, 128)]),
    )
    first_gold = rng.random() < 0.5
    pieces = [patch, decoy] if first_gold else [decoy, patch]
    return _item(
        "gen-photo",
        "jigsaw",
        "multi_image",
        [masked] + pieces,
        "Part of the first image has been blanked out. Which of the two patches (image 2 or image 3) "
        "fills the missing region?",
        [(record, root)],
        options=["image 2", "image 3"],
        gold=0 if first_gold else 1,
        fixed_order=True,
        orig_sha256=[record["sha256"]] * 3,
        meta={"benchmark_target": "BLINK"},
    )


# ------------------------------------------------------------------ captions (Winoground style)


COLOURS = (
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "orange",
    "pink",
    "purple",
    "gray",
    "grey",
)
NOUNS = (
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "person",
    "dog",
    "cat",
    "horse",
    "bird",
    "table",
    "chair",
    "car",
    "truck",
    "bus",
    "bike",
    "bicycle",
    "motorcycle",
    "train",
    "plate",
    "cup",
    "bowl",
    "pizza",
    "cake",
    "sandwich",
    "laptop",
    "phone",
    "book",
    "ball",
    "kite",
    "umbrella",
    "bench",
    "tree",
    "sign",
    "boat",
    "elephant",
    "giraffe",
    "zebra",
    "sheep",
    "cow",
    "bear",
    "frisbee",
    "skateboard",
    "surfboard",
    "couch",
    "bed",
    "toilet",
    "sink",
    "clock",
    "vase",
    "teddy",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "donut",
    "fence",
    "building",
    "wall",
    "window",
    "door",
    "road",
    "street",
    "player",
    "baby",
    "lady",
    "guy",
)
SYMMETRIC = re.compile(
    r"\b(and|with|next to|beside|near|by|or|together|alongside|between|both)\b"
)
OPPOSITES = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "above": "below",
    "below": "above",
    "front": "back",
    "inside": "outside",
    "outside": "inside",
    "under": "over",
    "over": "under",
    "on top of": "underneath",
    "underneath": "on top of",
    "behind": "in front of",
    "in front of": "behind",
}


def swap_caption(caption: str, rng: random.Random) -> tuple[str, str] | None:
    words = caption.strip().rstrip(".").split()
    low = [re.sub(r"\W", "", w.lower()) for w in words]
    nouns = [i for i, w in enumerate(low) if w in NOUNS]
    if len(nouns) >= 2:
        i, j = nouns[0], nouns[1]
        between = " ".join(low[i + 1 : j])
        if low[i] != low[j] and not SYMMETRIC.search(between) and between:
            out = list(words)
            out[i], out[j] = re.sub(low[i], low[j], words[i], flags=re.I), re.sub(
                low[j], low[i], words[j], flags=re.I
            )
            return " ".join(out), "noun_swap"
    colours = [i for i, w in enumerate(low) if w in COLOURS]
    if len(colours) >= 2 and low[colours[0]] != low[colours[1]]:
        out = list(words)
        i, j = colours[0], colours[1]
        out[i], out[j] = words[j], words[i]
        return " ".join(out), "colour_swap"
    text = " ".join(low)
    for phrase in sorted(OPPOSITES, key=len, reverse=True):
        if (
            re.search(rf"\b{phrase}\b", text)
            and len(re.findall(rf"\b{phrase}\b", text)) == 1
        ):
            swapped = re.sub(
                rf"\b{phrase}\b",
                OPPOSITES[phrase],
                caption.strip().rstrip("."),
                flags=re.I,
            )
            if swapped.lower() != caption.strip().rstrip(".").lower():
                return swapped, "spatial_flip"
    return None


def caption_pair(index: int) -> Item | None:
    rng = rng_for("gen-photo", "caption-swap", index)
    for _ in range(40):
        record, root = photo(rng, "coco")
        captions = list(record.get("captions") or [])
        rng.shuffle(captions)
        for caption in captions:
            swapped = swap_caption(caption, rng)
            if swapped:
                negative, kind = swapped
                image = load(record, root)
                clean = caption.strip().rstrip(".")
                question = rng.choice(
                    [
                        "Which caption describes the image?",
                        "Which caption matches this picture?",
                        "Pick the caption that is true of the image.",
                    ]
                )
                return _item(
                    "gen-photo",
                    "caption-swap",
                    "caption",
                    [image],
                    question,
                    [(record, root)],
                    options=[clean, negative],
                    orig_sha256=[record["sha256"]],
                    meta={
                        "benchmark_target": "Winoground",
                        "swap": kind,
                        "verified": False,
                    },
                    image_text=[],
                )
    return None


# ------------------------------------------------------------------ memes (Hateful Memes style)


def _outlined(draw, xy, text, face, fill="white", outline="black", width=3):
    x, y = xy
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx * dx + dy * dy <= width * width:
                draw.text((x + dx, y + dy), text, font=face, fill=outline)
    draw.text((x, y), text, font=face, fill=fill)


def meme(index: int) -> Item | None:
    rng = rng_for("gen-photo", "meme", index)
    texts = text_pool("hate")
    hateful = rng.random() < 0.45
    choices = [
        t for t in rng.sample(texts, min(300, len(texts))) if t["hateful"] == hateful
    ]
    if not choices:
        return None
    entry = rng.choice(choices)
    record, root = photo(rng)
    image = render.fit_pixels(load(record, root), 700_000)
    w, h = image.size
    text = entry["text"]
    style = rng.choice(["classic", "classic", "band"])
    if style == "classic":
        words = text.upper().split() if rng.random() < 0.8 else text.split()
        cut = (
            max(1, len(words) // 2)
            if len(words) > 6 and rng.random() < 0.7
            else len(words)
        )
        parts = [" ".join(words[:cut]), " ".join(words[cut:])]
        draw = ImageDraw.Draw(image)
        size = max(18, int(h * 0.075))
        face = render.font("display", size, rng)
        for k, part in enumerate(p for p in parts if p):
            lines = render.wrap(draw, part, face, int(w * 0.92))
            while len(lines) > 3 and size > 14:
                size -= 3
                face = render.font("display", size, rng)
                lines = render.wrap(draw, part, face, int(w * 0.92))
            lh = render.line_height(face)
            y = int(h * 0.02) if k == 0 else h - int(h * 0.03) - lh * len(lines)
            for line in lines:
                _outlined(
                    draw,
                    ((w - render.text_width(draw, line, face)) // 2, y),
                    line,
                    face,
                    width=max(2, size // 14),
                )
                y += lh
        out = image
    else:
        face = render.font(
            rng.choice(["sans", "sans_bold"]), max(18, int(w * 0.045)), rng
        )
        probe = ImageDraw.Draw(image)
        lines = render.wrap(probe, text, face, int(w * 0.92))
        band = render.line_height(face) * len(lines) + 30
        out = Image.new("RGB", (w, h + band), "white")
        out.paste(image, (0, band))
        render.draw_lines(
            ImageDraw.Draw(out),
            (int(w * 0.04), 15),
            lines,
            face,
            fill="black",
            spacing=1.0,
        )
    state = "" if rng.random() < 0.7 else f"Meme text: {text}"
    licences = [(record, root)]
    meta = {
        "benchmark_target": "Hateful Memes",
        "text_source": entry["source"],
        "text_id": entry["id"],
        "text_licence": "CC-BY-4.0",
    }
    if rng.random() < 0.5:
        return _item(
            "gen-photo",
            "meme",
            "moderation",
            [out],
            "Is this meme hateful?",
            licences,
            noul=hateful,
            state=state,
            orig_sha256=[record["sha256"]],
            meta=meta,
            image_text=[text],
        )
    options = [
        "hateful: it attacks people based on a protected characteristic",
        "not hateful",
    ]
    return _item(
        "gen-photo",
        "meme",
        "moderation",
        [out],
        "Is this meme hateful or not hateful?",
        licences,
        options=options,
        gold=0 if hateful else 1,
        keys=["hateful", "not_hateful"],
        state=state,
        orig_sha256=[record["sha256"]],
        meta=meta,
        image_text=[text],
    )


# ------------------------------------------------------------------ SAT


def sat(index: int, dynamic: bool) -> Item | None:
    rows = sat_pool(dynamic)
    if not rows:
        return None
    rng = rng_for("gen-photo", "sat", dynamic, index)
    row = rows[index % len(rows)] if index < len(rows) else rng.choice(rows)
    answers = [str(a) for a in row["answers"]]
    if (
        row["correct"] not in answers
        or len(set(answers)) != len(answers)
        or len(answers) < 2
    ):
        return None
    images = []
    for rel in row["files"]:
        with Image.open(RAW / "sat" / rel) as image:
            images.append(image.convert("RGB"))
    return Item(
        source="sat-train",
        family="sat-dynamic" if dynamic else "sat-static",
        skill="multi_image" if dynamic else "spatial",
        images=images,
        image_kinds=["png" if f.endswith("png") else "jpeg" for f in row["files"]],
        question=row["question"],
        options=answers,
        gold=answers.index(row["correct"]),
        orig_sha256=row["sha256"],
        meta={
            "benchmark_target": "CV-Bench" if not dynamic else "BLINK",
            "sat_type": row["type"],
            "source_key": row["id"],
            "licences": ["MIT"],
        },
    )


FAMILIES = {
    "coco-count": lambda i, ctx=None: count(i, "coco"),
    "oi-count": lambda i, ctx=None: count(i, "openimages"),
    "coco-relation": lambda i, ctx=None: relation(i, "coco"),
    "oi-relation": lambda i, ctx=None: relation(i, "openimages"),
    "oi-nearest": lambda i, ctx=None: nearest(i, "openimages"),
    "coco-localize": lambda i, ctx=None: localize(i),
    "warp-corr": lambda i, ctx=None: warp_corr(i),
    "kp-corr": lambda i, ctx=None: kp_corr(i),
    "forensic": lambda i, ctx=None: forensic(i),
    "camera-motion": lambda i, ctx=None: camera_motion(i),
    "similarity": lambda i, ctx=None: similarity(i),
    "jigsaw": lambda i, ctx=None: jigsaw(i),
    "caption-swap": lambda i, ctx=None: caption_pair(i),
    "meme": lambda i, ctx=None: meme(i),
    "sat-static": lambda i, ctx=None: sat(i, False),
    "sat-dynamic": lambda i, ctx=None: sat(i, True),
}
