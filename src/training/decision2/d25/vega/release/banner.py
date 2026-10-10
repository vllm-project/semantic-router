"""Decision family banner in the Vela 2.0 layout: a 16:9 PNG rendered with Pillow (no other dependency).

    python banner.py --background bg-interstellar.png --logo vllm-sr-logo.white.png --fonts <dir> \
        --version 3.0 --tagline "Towards Open Multimodal Foundation Decision Models" \
        --spec "d3 · 27B · TEXT · IMAGES" --url huggingface.co/vllm-sr/d3 --out decision-3.0.png

The background is cropped to 16:9 (``--focus-y`` picks the vertical window) and resized with Lanczos; a global
darkening plus an elliptical vignette keep the centered stack readable: "Decision" in warm white with the version
in an orange gradient, the tagline (light sans), the spec line (monospace, letter-spaced, orange), the white vLLM
Semantic Router logo and the URL (monospace, muted). ``--fonts`` holds ``InterDisplay-SemiBold.ttf``,
``InterVariable.ttf`` (tagline at weight 300; ``Inter-Regular.ttf`` otherwise) and a monospace TTF
(``JetBrainsMono-Regular.ttf``, else any ``*Mono*.ttf``). Prints the input and output SHA-256 as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WARM_WHITE = (246, 240, 231)
GRADIENT = ((255, 211, 168), (234, 118, 48))
SPEC = (238, 150, 90)
TAGLINE = (242, 240, 236)
MUTED = (170, 162, 152)
# Vertical centres as fractions of the height.
LAYOUT = {"title": 0.335, "tagline": 0.462, "spec": 0.524, "logo": 0.700, "url": 0.862}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cover(image: Image.Image, size: tuple[int, int], focus_y: float) -> Image.Image:
    w, h = image.size
    width, height = size
    if w * height > h * width:
        crop = round(h * width / height)
        left = (w - crop) // 2
        box = (left, 0, left + crop, h)
    else:
        crop = round(w * height / width)
        top = round((h - crop) * focus_y)
        box = (0, top, w, top + crop)
    return image.crop(box).resize(size, Image.LANCZOS)


def darken(
    image: Image.Image,
    overall: float,
    center: tuple[float, float],
    radii: tuple[float, float],
    strength: float,
) -> Image.Image:
    """Multiply by (1 - overall) and by an elliptical vignette (``strength`` at the centre, 0 outside)."""
    width, height = image.size
    scale = 8
    small = Image.new("L", (width // scale, height // scale))
    pixels = small.load()
    for y in range(small.height):
        for x in range(small.width):
            d = math.hypot(
                (x * scale - center[0]) / radii[0], (y * scale - center[1]) / radii[1]
            )
            t = min(1.0, max(0.0, (d - 0.25) / 0.85))
            v = strength * (1 - t * t * (3 - 2 * t))
            pixels[x, y] = round(255 * (1 - (1 - overall) * (1 - v)))
    mask = small.resize((width, height), Image.BICUBIC).filter(
        ImageFilter.GaussianBlur(6)
    )
    black = Image.new("RGB", image.size, (4, 3, 6))
    return Image.composite(black, image, mask)


def light(path: Path, size: int, weight: int = 300) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), size)
    try:
        values = []
        for axis in font.get_variation_axes():
            name = axis.get("name")
            name = (name.decode() if isinstance(name, bytes) else str(name)).lower()
            if name.startswith("weight"):
                values.append(weight)
            elif name.startswith("optical"):
                values.append(axis["maximum"])
            else:
                values.append(axis.get("default", axis["minimum"]))
        font.set_variation_by_axes(values)
    except (OSError, AttributeError):
        pass
    return font


def mono_font(fonts: Path) -> Path:
    preferred = fonts / "JetBrainsMono-Regular.ttf"
    if preferred.is_file():
        return preferred
    found = sorted(fonts.glob("*Mono*.ttf"))
    if not found:
        raise FileNotFoundError(f"no monospace TTF in {fonts}")
    return found[0]


def shadow(
    base: Image.Image, mask: Image.Image, radius: float, alpha: float
) -> Image.Image:
    soft = mask.filter(ImageFilter.GaussianBlur(radius)).point(
        lambda v: round(v * alpha)
    )
    return Image.composite(Image.new("RGB", base.size, (0, 0, 0)), base, soft)


def tracked(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y: float,
    width: int,
    tracking: float,
    fill,
) -> None:
    """Centered single line with extra letter spacing (``tracking`` in em)."""
    gap = tracking * font.size
    total = sum(font.getlength(c) for c in text) + gap * (len(text) - 1)
    x = (width - total) / 2
    for c in text:
        draw.text((x, y), c, font=font, fill=fill, anchor="lm")
        x += font.getlength(c) + gap


def render(args: argparse.Namespace) -> Image.Image:
    width, height = args.width, args.height
    s = height / 1080
    background = Image.open(args.background).convert("RGB")
    image = cover(background, (width, height), args.focus_y)
    image = darken(
        image, 0.10, (width / 2, height * 0.52), (width * 0.47, height * 0.40), 0.66
    )

    title = ImageFont.truetype(
        str(args.fonts / "InterDisplay-SemiBold.ttf"), round(160 * s)
    )
    variable = args.fonts / "InterVariable.ttf"
    tagline = (
        light(variable, round(40 * s))
        if variable.is_file()
        else ImageFont.truetype(str(args.fonts / "Inter-Regular.ttf"), round(40 * s))
    )
    mono = mono_font(args.fonts)
    spec = ImageFont.truetype(str(mono), round(22 * s))
    url = ImageFont.truetype(str(mono), round(24 * s))

    word, version = "Decision ", args.version
    total = title.getlength(word + version)
    x0 = (width - total) / 2
    _, top, _, bottom = title.getbbox("D", anchor="ls")
    baseline = height * LAYOUT["title"] + (bottom - top) / 2

    text_mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(text_mask)
    mask_draw.text((x0, baseline), word + version, font=title, fill=255, anchor="ls")
    lines = [
        (args.tagline, tagline, LAYOUT["tagline"], 0.0),
        (args.spec, spec, LAYOUT["spec"], 0.42),
        (args.url, url, LAYOUT["url"], 0.04),
    ]
    for text, font, fy, track in lines:
        tracked(mask_draw, text, font, height * fy, width, track, 255)
    image = shadow(image, text_mask, 22 * s, 0.55)

    draw = ImageDraw.Draw(image)
    draw.text((x0, baseline), word, font=title, fill=WARM_WHITE, anchor="ls")
    version_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(version_mask).text(
        (x0 + title.getlength(word), baseline),
        version,
        font=title,
        fill=255,
        anchor="ls",
    )
    left, _, right, _ = version_mask.getbbox()
    ramp = Image.linear_gradient("L").rotate(90).resize((max(1, right - left), height))
    ramp = ramp.transpose(Image.FLIP_LEFT_RIGHT)
    gradient = Image.new("RGB", image.size, GRADIENT[1])
    start = Image.new("RGB", (right - left, height), GRADIENT[0])
    end = Image.new("RGB", (right - left, height), GRADIENT[1])
    gradient.paste(Image.composite(start, end, ramp), (left, 0))
    image = Image.composite(gradient, image, version_mask)

    draw = ImageDraw.Draw(image)
    tracked(
        draw, args.tagline, tagline, height * LAYOUT["tagline"], width, 0.0, TAGLINE
    )
    tracked(draw, args.spec, spec, height * LAYOUT["spec"], width, 0.42, SPEC)
    tracked(draw, args.url, url, height * LAYOUT["url"], width, 0.04, MUTED)

    logo = Image.open(args.logo).convert("RGBA")
    logo = logo.crop(logo.getchannel("A").getbbox())
    logo_height = round(80 * s)
    logo = logo.resize(
        (round(logo.width * logo_height / logo.height), logo_height), Image.LANCZOS
    )
    alpha = logo.getchannel("A")
    image = shadow(image, _placed(alpha, image.size, width, height), 14 * s, 0.45)
    image.paste(
        logo,
        ((width - logo.width) // 2, round(height * LAYOUT["logo"] - logo_height / 2)),
        logo,
    )
    return image


def _placed(
    alpha: Image.Image, size: tuple[int, int], width: int, height: int
) -> Image.Image:
    mask = Image.new("L", size, 0)
    mask.paste(
        alpha,
        ((width - alpha.width) // 2, round(height * LAYOUT["logo"] - alpha.height / 2)),
    )
    return mask


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--background", required=True, type=Path)
    ap.add_argument("--logo", required=True, type=Path)
    ap.add_argument("--fonts", required=True, type=Path)
    ap.add_argument(
        "--version", required=True, help='the number after "Decision", e.g. 3.0'
    )
    ap.add_argument("--tagline", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument(
        "--focus-y",
        type=float,
        default=0.0,
        help="vertical crop window, 0 = top, 1 = bottom",
    )
    args = ap.parse_args(argv)
    if args.width * 9 != args.height * 16:
        raise SystemExit("the banner is 16:9")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(args).save(args.out, format="PNG", optimize=True)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "size": [args.width, args.height],
                "sha256": {
                    "background": sha256(args.background),
                    "logo": sha256(args.logo),
                    "png": sha256(args.out),
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
