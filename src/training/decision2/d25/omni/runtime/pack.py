"""A small image request pack of our own rendered images, for checks the public suite cannot travel to.

Rows use the evaluation-row format of ``d25.omni.common.vision_format`` (``rows.jsonl.gz`` + ``images/``), so
``parity_image``, ``latency`` and ``smoke`` take the pack as ``--suite``. Content is invented and drawn with
PIL (Pillow's bundled font): bar charts, receipts and coloured shapes; 1, 2 or 4 images per row; sizes from
0.3 MP to 1.9 MP (above the 1.6 MP cap, so the processor downscales). Deterministic for a given seed.

    python -m d25.omni.runtime.pack --out PACK [--rows 48] [--seed 20261010]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
from pathlib import Path

COLORS = {
    "red": (220, 50, 47),
    "green": (40, 160, 70),
    "blue": (38, 110, 210),
    "orange": (240, 140, 30),
    "purple": (130, 70, 180),
    "gray": (120, 120, 120),
}
SHAPES = ("circle", "square", "triangle")
SIZES = ((1600, 1200), (1280, 1280), (960, 720), (640, 480), (800, 1100))
ITEMS = (
    "Kettle",
    "Toaster",
    "Mug set",
    "Cutting board",
    "Chef knife",
    "Tea towels",
    "Colander",
    "Whisk",
)


def font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def save(image, root: Path) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    relative = Path("images") / digest[:2] / f"{digest}.png"
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_bytes(data)
    return relative.as_posix()


def bar_chart(rng: random.Random, size: tuple[int, int], bars: int):
    from PIL import Image, ImageDraw

    width, height = size
    labels = list("ABCDEF")[:bars]
    values = rng.sample(range(10, 100), bars)
    image = Image.new("RGB", size, (250, 250, 248))
    draw = ImageDraw.Draw(image)
    left, bottom, top = int(width * 0.1), int(height * 0.85), int(height * 0.12)
    slot = width * 0.8 / bars
    for i, (label, value) in enumerate(zip(labels, values)):
        x0, x1 = left + i * slot + slot * 0.2, left + (i + 1) * slot - slot * 0.2
        y0 = bottom - (bottom - top) * value / 100
        draw.rectangle(
            (x0, y0, x1, bottom), fill=list(COLORS.values())[i % len(COLORS)]
        )
        draw.text(
            (x0, bottom + height * 0.02),
            label,
            font=font(max(16, height // 22)),
            fill=(30, 30, 30),
        )
        draw.text(
            (x0, y0 - height * 0.05),
            str(value),
            font=font(max(14, height // 28)),
            fill=(30, 30, 30),
        )
    draw.line((left, bottom, width * 0.92, bottom), fill=(60, 60, 60), width=3)
    return image, labels, values


def receipt(rng: random.Random, size: tuple[int, int]):
    from PIL import Image, ImageDraw

    width, height = size
    items = [
        (name, rng.randint(150, 4500) / 100)
        for name in rng.sample(ITEMS, rng.randint(2, 5))
    ]
    total = round(sum(price for _, price in items), 2)
    image = Image.new("RGB", size, (246, 244, 238))
    draw = ImageDraw.Draw(image)
    face = font(max(16, height // 30))
    y = height * 0.08
    draw.text(
        (width * 0.1, y),
        "SAMPLE STORE RECEIPT",
        font=font(max(20, height // 22)),
        fill=(25, 25, 25),
    )
    y += height * 0.1
    for name, price in items:
        draw.text((width * 0.1, y), name, font=face, fill=(25, 25, 25))
        draw.text((width * 0.7, y), f"{price:.2f}", font=face, fill=(25, 25, 25))
        y += height * 0.06
    draw.line((width * 0.1, y, width * 0.9, y), fill=(90, 90, 90), width=2)
    y += height * 0.03
    draw.text((width * 0.1, y), "TOTAL", font=face, fill=(25, 25, 25))
    draw.text((width * 0.7, y), f"{total:.2f}", font=face, fill=(25, 25, 25))
    return image, total


def shape_image(size: tuple[int, int], color: str, shape: str):
    from PIL import Image, ImageDraw

    width, height = size
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    r, cx, cy = min(width, height) * 0.3, width / 2, height / 2
    box = (cx - r, cy - r, cx + r, cy + r)
    if shape == "circle":
        draw.ellipse(box, fill=COLORS[color])
    elif shape == "square":
        draw.rectangle(box, fill=COLORS[color])
    else:
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color]
        )
    return image


def build(out: Path, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    kinds = ["chart"] * (n // 3) + ["receipt"] * (n // 4) + ["two"] * (n // 6)
    kinds += ["four"] * (n - len(kinds))
    for number, kind in enumerate(kinds):
        size = SIZES[number % len(SIZES)]
        row = {
            "id": f"pack:{kind}:{number:03d}",
            "family": f"pack-{kind}",
            "split": "pack",
            "state": {},
        }
        if kind == "chart":
            image, labels, values = bar_chart(rng, size, rng.randint(3, 6))
            row["images"] = [save(image, out)]
            row["questions"] = {
                "q1": {
                    "type": "choice",
                    "instructions": "Which bar is the tallest?",
                    "criteria": {label: None for label in labels},
                }
            }
            row["expected"] = {"q1": labels[values.index(max(values))]}
        elif kind == "receipt":
            image, total = receipt(rng, size)
            row["images"] = [save(image, out)]
            row["state"] = {"note": "Receipt photo attached by a customer."}
            row["questions"] = {
                "q1": {
                    "type": "noul",
                    "instructions": "Is the receipt total above 50.00?",
                }
            }
            row["expected"] = {"q1": total > 50}
        elif kind == "two":
            counts = rng.sample(range(3, 7), 2)
            images = [
                bar_chart(rng, SIZES[(number + k) % len(SIZES)], c)[0]
                for k, c in enumerate(counts)
            ]
            row["images"] = [save(image, out) for image in images]
            row["questions"] = {
                "q1": {
                    "type": "choice",
                    "instructions": "Which chart has more bars?",
                    "criteria": {
                        "first": "The first image",
                        "second": "The second image",
                    },
                }
            }
            row["expected"] = {"q1": "first" if counts[0] > counts[1] else "second"}
        else:
            target = rng.randrange(4)
            images = []
            for k in range(4):
                if k == target:
                    color, shape = "red", "circle"
                else:
                    color, shape = rng.choice(
                        [
                            ("red", "square"),
                            ("blue", "circle"),
                            ("green", "triangle"),
                            ("orange", "circle"),
                            ("red", "triangle"),
                        ]
                    )
                images.append(
                    shape_image(SIZES[(number + k) % len(SIZES)], color, shape)
                )
            row["images"] = [save(image, out) for image in images]
            row["questions"] = {
                "q1": {
                    "type": "choice",
                    "instructions": "Which image shows a red circle?",
                    "criteria": {str(k + 1): f"Image {k + 1}" for k in range(4)},
                }
            }
            row["expected"] = {"q1": str(target + 1)}
        row["metadata"] = {
            "n_images": len(row["images"]),
            "sizes": size,
            "content": "rendered, invented",
        }
        rows.append(row)
    with gzip.open(out / "rows.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    digest = hashlib.sha256((out / "rows.jsonl.gz").read_bytes()).hexdigest()
    return {
        "pack": str(out),
        "rows": len(rows),
        "seed": seed,
        "rows_sha256": digest,
        "images": sum(len(r["images"]) for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    print(json.dumps(build(args.out, args.rows, args.seed), indent=1))


if __name__ == "__main__":
    main()
