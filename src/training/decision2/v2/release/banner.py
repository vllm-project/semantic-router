"""DEV2.0 owl model-name banners in the Decision 1.0 composition (needs Pillow).

Each banner is a transparent 2,172 x 724 sticker layout like Decision 1.0: the
tier's mosaic owl on the left (pixels copied unchanged from the pinned 1.0
banner of that tier; the new 27B owl comes from ``brand/sources``), a cream
``DEV2.0`` wordmark with a navy outline and an accent-coloured extruded shadow,
a ``DECISION 2.0`` pill and the size in the tier accent. Inputs are verified by
SHA-256 and the output carries no PNG text chunks.

    python banner.py --sources DIR_WITH_1.0_HEADERS --fonts DIR --output-dir brand/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WIDTH, HEIGHT = 2172, 724
NAVY, CREAM, WHITE = "#071943", "#fffdf5", "#ffffff"
OWL_BOX = (0, 0, 760, 724)
FONTS = {
    "Poppins-Black.ttf": "d82aaaf98a9283f9a8edd24e51173337d8eaf09e25cd3d98831f8ec8461748a1",
    "DejaVuSans-Bold.ttf": None,
}
# Pinned Decision 1.0 banners (current main of each public 1.0 repository).
SIZES = {
    "0.6B": {
        "accent": "#63d2f0",
        "owl": (
            "llm-semantic-router/Decision-1.0-Kai-0.6B",
            "9d6872cde6950c2c2b5786d182ec9a06ca1bdd66",
            "assets/decision-kai-header.png",
            "6e1ed4119bd1554a997aa89c60529e8b1fc1d147a393799b9ca0f3f32ae7605d",
        ),
    },
    "0.8B": {
        "accent": "#ffb574",
        "owl": (
            "llm-semantic-router/Decision-1.0-Eos-0.8B",
            "363c4a5e56afc115b1c78c837633956d0bbb63ab",
            "assets/decision-eos-header.png",
            "02673d96602b3328b59a12831cebfb32e4553d4ac8a417528e17b2c55409a6ca",
        ),
    },
    "2B": {
        "accent": "#ffd84e",
        "owl": (
            "llm-semantic-router/Decision-1.0-Sol-2B",
            "ce0c018a28de16d6639b1cd203b761bf643b89e6",
            "assets/decision-sol-2b-header.png",
            "fe6b65ab04dc2044c38d383091a03cbc0f176fbb04a13948c12d5ae50d1de710",
        ),
    },
    "4B": {
        "accent": "#b4a0f6",
        "owl": (
            "llm-semantic-router/Decision-1.0-Nox-4B",
            "cde2a68dbaa557ea65dc458104d410a0802ee259",
            "assets/decision-nox-4b-header.png",
            "58dbd7cf5ff49b760a162c32366bd5312a2acfded1273b79b937afcd31ca6ee4",
        ),
    },
    "9B": {
        "accent": "#f8ca69",
        "owl": (
            "llm-semantic-router/Decision-1.0-Lux-9B",
            "cdf4d3ef2dda21518e599fe99ebbe468486b197c",
            "assets/decision-lux-9b-header.png",
            "64e55deb635334185a01e1e9787de95503b8f49d11c44a2be5dd1b925d5e8ce1",
        ),
    },
    "27B": {
        "accent": "#6fdcb4",
        "owl": ("new", None, "DEV2.0-27B-owl-sticker.png", None),
    },
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owl_sticker(size: str, sources: Path, own_sources: Path):
    from PIL import Image

    repo, _, name, digest = SIZES[size]["owl"]
    if repo == "new":
        path = own_sources / name
        sticker = Image.open(path).convert("RGBA")
        scale = 650 / sticker.height
        sticker = sticker.resize((round(sticker.width * scale), 650), Image.LANCZOS)
        canvas = Image.new("RGBA", (OWL_BOX[2], OWL_BOX[3]), (0, 0, 0, 0))
        canvas.alpha_composite(
            sticker, ((OWL_BOX[2] - sticker.width) // 2 + 10, (OWL_BOX[3] - 650) // 2)
        )
        return canvas, {"source": f"brand/sources/{name}", "sha256": sha_file(path)}
    path = sources / Path(repo).name / name
    if sha_file(path) != digest:
        raise ValueError(f"1.0 banner differs from its pinned bytes: {path}")
    return Image.open(path).convert("RGBA").crop(OWL_BOX), {
        "source": f"{repo}@{SIZES[size]['owl'][1]}:{name}",
        "sha256": digest,
    }


def _fit(font_path: Path, text: str, width: int, height: int, start: int):
    from PIL import ImageFont

    size = start
    while size > 20:
        font = ImageFont.truetype(str(font_path), size)
        left, top, right, bottom = font.getbbox(text, stroke_width=16)
        if right - left <= width and bottom - top <= height:
            return font
        size -= 4
    raise ValueError("Text does not fit")


def render(size: str, sources: Path, own_sources: Path, fonts: Path):
    from PIL import Image, ImageDraw, ImageFont

    accent = SIZES[size]["accent"]
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    owl, provenance = owl_sticker(size, sources, own_sources)
    image.alpha_composite(owl, (0, 0))
    draw = ImageDraw.Draw(image)
    black = fonts / "Poppins-Black.ttf"

    word = _fit(black, "DEV2.0", 1150, 270, 340)
    left, top, right, bottom = word.getbbox("DEV2.0", stroke_width=16)
    x, y = 800 - left, 178 - top
    draw.text(
        (x + 14, y + 14),
        "DEV2.0",
        font=word,
        fill=accent,
        stroke_width=16,
        stroke_fill=NAVY,
    )
    draw.text(
        (x, y), "DEV2.0", font=word, fill=CREAM, stroke_width=16, stroke_fill=NAVY
    )

    pill_font = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 52)
    label, tracking = "DECISION 2.0", 7
    widths = [pill_font.getlength(ch) for ch in label]
    text_width = sum(widths) + tracking * (len(label) - 1)
    pad_x, pill_h = 44, 96
    right_edge, pill_top = 2086, 40
    box = (right_edge - text_width - 2 * pad_x, pill_top, right_edge, pill_top + pill_h)
    draw.rounded_rectangle(box, radius=pill_h // 2, fill=WHITE, outline=NAVY, width=9)
    cursor = box[0] + pad_x
    ascent, descent = pill_font.getmetrics()
    baseline_y = pill_top + (pill_h - (ascent + descent)) // 2 + 2
    for ch, w in zip(label, widths):
        draw.text((cursor, baseline_y), ch, font=pill_font, fill=NAVY)
        cursor += w + tracking

    size_font = _fit(black, size, 560, 215, 260)
    left, top, right, bottom = size_font.getbbox(size, stroke_width=14)
    sx, sy = right_edge - right - 10, 690 - bottom - 10
    draw.text(
        (sx + 10, sy + 10),
        size,
        font=size_font,
        fill=NAVY,
        stroke_width=14,
        stroke_fill=NAVY,
    )
    draw.text(
        (sx, sy), size, font=size_font, fill=accent, stroke_width=14, stroke_fill=NAVY
    )
    return image, provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sources", type=Path, required=True, help="<dir>/<1.0 repo name>/assets/*.png"
    )
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="*", default=list(SIZES))
    args = parser.parse_args()
    own_sources = Path(__file__).resolve().parent / "brand" / "sources"
    fonts = {name: sha_file(args.fonts / name) for name in FONTS}
    if fonts["Poppins-Black.ttf"] != FONTS["Poppins-Black.ttf"]:
        raise ValueError("Wordmark font differs from the pinned Poppins Black")
    receipt = {"schema": "dev2-banners/1", "fonts": fonts, "banners": {}}
    for size in args.sizes:
        image, provenance = render(size, args.sources, own_sources, args.fonts)
        path = args.output_dir / f"DEV2.0-{size}-owl-banner.png"
        image.save(path, format="PNG", optimize=True)
        receipt["banners"][size] = {
            "file": path.name,
            "sha256": sha_file(path),
            "accent": SIZES[size]["accent"],
            "owl": provenance,
        }
    (args.output_dir / "BANNERS.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v["sha256"][:12] for k, v in receipt["banners"].items()}))


if __name__ == "__main__":
    main()
