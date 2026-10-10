"""Render the model card's Quickstart image: a store receipt drawn with PIL (no third-party content).

The store, items, numbers and barcode are invented. Text uses DejaVu Sans when the system or matplotlib
ships it (Bitstream Vera licence), else Pillow's bundled default font (Aileron Regular, CC0).

    python -m d25.omni.runtime.example_image --out assets/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NAME = "example-receipt.png"
WIDTH, HEIGHT = 900, 1000
STORE = ("SAMPLE HOME & KITCHEN", "Store 112 - 48 Example Street")
ORDER = ("Order A-20481", "2026-09-28 14:12")
ITEMS = (
    ("1", "Countertop blender BL-200", "89.00"),
    ("1", "Glass carafe 1.5 L", "19.50"),
    ("1", "Silicone spatula set", "7.25"),
)
TOTALS = (("Subtotal", "115.75"), ("Tax 8%", "9.26"), ("TOTAL", "125.01"))
PAYMENT = ("Paid by card", "VISA **** 4421")
FOOTER = ("Returns and replacements accepted", "within 30 days with this receipt.")
THANKS = "Thank you for shopping with us!"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
INK = (34, 34, 38)


def font_path() -> str | None:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    try:
        import matplotlib

        bundled = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if bundled.is_file():
            return str(bundled)
    except Exception:  # noqa: BLE001
        pass
    return None


def render(out: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    path = font_path()

    def font(size: int):
        return (
            ImageFont.truetype(path, size)
            if path
            else ImageFont.load_default(size=size)
        )

    title, body, small = font(40), font(28), font(23)
    image = Image.new("RGB", (WIDTH, HEIGHT), (236, 232, 224))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 50, WIDTH - 90, HEIGHT - 50
    draw.rectangle(
        (left, top, right, bottom),
        fill=(253, 252, 248),
        outline=(205, 200, 190),
        width=2,
    )
    inner_left, inner_right = left + 40, right - 40

    def height(text, face):
        box = draw.textbbox((0, 0), text, font=face)
        return box[3] - box[1]

    def centered(y, text, face):
        box = draw.textbbox((0, 0), text, font=face)
        draw.text(((WIDTH - (box[2] - box[0])) // 2, y), text, font=face, fill=INK)
        return y + height("Hg", face) + 16

    def row(y, first, last, face=body, indent=0, bold=False):
        draw.text((inner_left + indent, y), first, font=face, fill=INK)
        box = draw.textbbox((0, 0), last, font=face)
        x = inner_right - (box[2] - box[0])
        draw.text((x, y), last, font=face, fill=INK)
        if bold:
            draw.text((inner_left + indent + 1, y), first, font=face, fill=INK)
            draw.text((x + 1, y), last, font=face, fill=INK)
        return y + height("Hg", face) + 18

    def rule(y):
        draw.line((inner_left, y, inner_right, y), fill=(150, 150, 150), width=2)
        return y + 20

    y = top + 44
    y = centered(y, STORE[0], title)
    y = centered(y, STORE[1], small) + 14
    y = row(y, *ORDER)
    y = rule(y)
    for quantity, name, price in ITEMS:
        draw.text((inner_left, y), quantity, font=body, fill=INK)
        y = row(y, name, price, indent=36)
    y = rule(y)
    for label, amount in TOTALS:
        y = row(y, label, amount, indent=36, bold=label == "TOTAL")
    y = rule(y)
    y = row(y, *PAYMENT) + 12
    for line in FOOTER:
        y = centered(y, line, small)
    y += 10
    bits = "".join(f"{ord(c):08b}" for c in ORDER[0])
    bar_x = (WIDTH - len(bits) * 4) // 2
    for i, bit in enumerate(bits):
        if bit == "1" or i % 7 == 0:
            draw.rectangle((bar_x + 4 * i, y, bar_x + 4 * i + 2, y + 64), fill=INK)
    y += 84
    centered(y, THANKS, small)

    out.mkdir(parents=True, exist_ok=True)
    target = out / NAME
    image.save(target, format="PNG", optimize=True)
    return {
        "file": str(target),
        "size": [WIDTH, HEIGHT],
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "font": Path(path).name if path else "Pillow default (Aileron Regular, CC0)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(render(args.out), indent=1))


if __name__ == "__main__":
    main()
