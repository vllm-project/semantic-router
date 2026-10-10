"""Mind2Web: test_website steps with an on-screen target, 973 rows, four marked boxes A to D.

A step qualifies when it has a screenshot, at least one positive candidate whose box lies inside it
and at least three usable negative candidates; in test_website that is 973 of 1,019 steps, the
board's count exactly. The target is the first on-screen positive. Three distractors are sampled
(seeded by the step) from negatives inside a 1280 x 1280 window around the target (the board's
1,638,400-pixel cap), preferring boxes that are visible, not nested with the target or with each
other and not page-sized. The four boxes are drawn with letter tags in a shuffled order. Board rows
show the window; the ``mind2web-fullpage`` variant draws the same boxes on the full screenshot.
"""

from __future__ import annotations

import glob
import io
import json
from pathlib import Path

import pyarrow.parquet as pq

from d25.omni.common import vision_format
from d25.omni.suite import rows as R

BENCHMARK = "Mind2Web"
SOURCES = ("mind2web",)
SPLIT = "test_website"
WINDOW = 1280
COLORS = {
    "A": (230, 25, 75),
    "B": (0, 130, 200),
    "C": (60, 180, 75),
    "D": (245, 130, 48),
}
INSTRUCTIONS = (
    "The screenshot shows a web page with four marked elements labelled A, B, C and D. "
    "Which element should the next action use to continue the task?"
)


def _box(candidate: dict):
    attributes = json.loads(candidate["attributes"])
    rect = attributes.get("bounding_box_rect")
    if not rect:
        return None
    return tuple(round(float(v)) for v in rect.split(","))


def _inside(b, x0, y0, x1, y1) -> bool:
    return (
        b is not None
        and b[2] > 0
        and b[3] > 0
        and b[0] >= x0
        and b[1] >= y0
        and b[0] + b[2] <= x1
        and b[1] + b[3] <= y1
    )


def _iou(a, b) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def _contains(a, b) -> bool:
    return (
        a[0] <= b[0]
        and a[1] <= b[1]
        and a[0] + a[2] >= b[0] + b[2]
        and a[1] + a[3] >= b[1] + b[3]
    )


def window(target, width: int, height: int) -> tuple[int, int, int, int]:
    """A WINDOW-high crop (full width) centred on the target, clamped to the page."""
    if height <= WINDOW or target[3] > WINDOW:
        return 0, 0, width, height
    cy = target[1] + target[3] / 2
    top = int(min(max(0, round(cy - WINDOW / 2)), height - WINDOW))
    return 0, top, width, top + WINDOW


def pick_distractors(target, negatives, frame, rng):
    """Three negatives inside ``frame``; stricter filters first, relaxed only when too few remain."""
    x0, y0, x1, y1 = frame
    area = (x1 - x0) * (y1 - y0)
    seen, pool = set(), []
    for b in negatives:
        if b is None or b in seen or not _inside(b, x0, y0, x1, y1):
            continue
        seen.add(b)
        pool.append(b)
    tiers = [
        lambda b: b[2] >= 8
        and b[3] >= 8
        and b[2] * b[3] <= 0.25 * area
        and _iou(b, target) < 0.3
        and not _contains(b, target)
        and not _contains(target, b),
        lambda b: b[2] >= 4
        and b[3] >= 4
        and b[2] * b[3] <= 0.5 * area
        and _iou(b, target) < 0.5
        and not _contains(b, target)
        and not _contains(target, b),
        lambda b: not _contains(b, target) and not _contains(target, b),
    ]
    for tier, ok in enumerate(tiers):
        candidates = [b for b in pool if ok(b)]
        rng.shuffle(candidates)
        chosen = []
        for b in candidates:
            if tier == 0 and any(
                _iou(b, c) >= 0.3 or _contains(b, c) or _contains(c, b) for c in chosen
            ):
                continue
            chosen.append(b)
            if len(chosen) == 3:
                return chosen, tier
    return None, None


def draw(image, boxes: dict, offset=(0, 0)):
    from PIL import ImageDraw, ImageFont

    out = image.convert("RGB")
    canvas = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=22)
    for letter in sorted(boxes):
        x, y, w, h = boxes[letter]
        x, y = x - offset[0], y - offset[1]
        color = COLORS[letter]
        canvas.rectangle((x, y, x + w, y + h), outline=color, width=3)
        tw, th = 22, 26
        ty = y - th if y - th >= 0 else y
        canvas.rectangle((x, ty, x + tw, ty + th), fill=color)
        canvas.text((x + 5, ty + 1), letter, fill=(255, 255, 255), font=font)
    return out


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def steps(ctx):
    root = ctx.source("mind2web")
    for path in sorted(glob.glob(str(root / "data" / f"{SPLIT}-*.parquet"))):
        name = str(Path(path).relative_to(root))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                yield name, r


def build(ctx) -> dict:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    board, full, skipped = [], [], {}
    for name, r in steps(ctx):
        shot = (r.get("screenshot") or {}).get("bytes")
        if not shot:
            skipped["no screenshot"] = skipped.get("no screenshot", 0) + 1
            continue
        with Image.open(io.BytesIO(shot)) as page:
            page.load()
            W, H = page.size
            positives = [
                b
                for b in (_box(json.loads(c)) for c in r["pos_candidates"])
                if _inside(b, 0, 0, W, H)
            ]
            if not positives:
                skipped["no on-screen target"] = (
                    skipped.get("no on-screen target", 0) + 1
                )
                continue
            target = positives[0]
            frame = window(target, W, H)
            negatives = [_box(json.loads(c)) for c in r["neg_candidates"]]
            rng = R.stable_rng("mind2web", r["action_uid"])
            distractors, tier = pick_distractors(target, negatives, frame, rng)
            if distractors is None:
                skipped["fewer than 3 distractors"] = (
                    skipped.get("fewer than 3 distractors", 0) + 1
                )
                continue
            boxes = [target] + distractors
            letters = list("ABCD")
            rng.shuffle(letters)
            by_letter = dict(zip(letters, boxes))
            gold = letters[0]
            crop = page.crop(frame)
            window_ref = ctx.store(_png(draw(crop, by_letter, offset=frame[:2])))
            full_ref = ctx.store(_png(draw(page, by_letter)))
        index = int(r["target_action_index"])
        state = {
            "task": r["confirmed_task"],
            "previous_actions": list(r["action_reprs"][:index]),
        }
        prov = ctx.provenance(
            "mind2web",
            name,
            r["action_uid"],
            annotation_id=r["annotation_id"],
            screenshot_sha256=vision_format.sha256_bytes(shot),
        )
        extra = {
            "website": r["website"],
            "domain": r["domain"],
            "operation": json.loads(r["operation"])["op"],
            "boxes": {k: list(b) for k, b in by_letter.items()},
            "window": list(frame),
            "screenshot_size": [W, H],
            "distractor_tier": tier,
        }
        common = dict(
            benchmark=BENCHMARK,
            split=SPLIT,
            source_id=r["action_uid"],
            instructions=INSTRUCTIONS,
            criteria={k: f"Element {k}" for k in "ABCD"},
            gold=gold,
            provenance=prov,
            state=state,
            tags=[f"website:{r['website']}", f"op:{extra['operation']}"],
            extra=extra,
        )
        board.append(R.make_row(images=[window_ref], **common))
        full.append(R.make_row(images=[full_ref], **common))
    return {
        "rows": board,
        "variants": {"mind2web-fullpage": full},
        "notes": f"{SPLIT} steps with an on-screen target and 3 distractors; skipped {skipped}",
    }
