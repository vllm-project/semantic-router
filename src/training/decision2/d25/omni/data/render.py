"""Rendering helpers shared by the generators: fonts, text layout, marks and photo degradation."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT_DIR = Path(os.environ.get("D25_FONTS", "/data/d25/omni/pylib/fonts"))
FONT_ROLES = {
    "sans": (
        "Lato-Regular",
        "FiraSans-Regular",
        "OpenSans[wdth,wght]",
        "NotoSans[wdth,wght]",
        "Inter[opsz,wght]",
        "Roboto[wdth,wght]",
        "PT_Sans-Web-Regular",
        "LiberationSans-Regular",
        "DejaVuSans",
        "Montserrat[wght]",
        "Nunito[wght]",
        "Raleway[wght]",
        "Poppins-Regular",
    ),
    "sans_bold": (
        "Lato-Bold",
        "FiraSans-Bold",
        "PT_Sans-Web-Bold",
        "LiberationSans-Bold",
        "DejaVuSans-Bold",
        "Poppins-Bold",
        "ArchivoBlack-Regular",
    ),
    "serif": (
        "PT_Serif-Web-Regular",
        "LiberationSerif-Regular",
        "DejaVuSerif",
        "Merriweather[opsz,wdth,wght]",
        "SourceSerif4[opsz,wght]",
        "PlayfairDisplay[wght]",
    ),
    "serif_bold": ("PT_Serif-Web-Bold", "LiberationSerif-Bold", "DejaVuSerif-Bold"),
    "mono": (
        "CourierPrime-Regular",
        "LiberationMono-Regular",
        "DejaVuSansMono",
        "IBMPlexMono-Regular",
        "SourceCodePro[wght]",
        "RobotoMono[wght]",
        "ShareTechMono-Regular",
    ),
    "receipt": (
        "VT323-Regular",
        "ShareTechMono-Regular",
        "CourierPrime-Regular",
        "LiberationMono-Regular",
        "DejaVuSansMono",
        "IBMPlexMono-Regular",
        "SpecialElite-Regular",
    ),
    "display": (
        "Anton-Regular",
        "BebasNeue-Regular",
        "Oswald[wght]",
        "ArchivoBlack-Regular",
    ),
    "hand": (
        "Caveat[wght]",
        "HomemadeApple-Regular",
        "ShadowsIntoLight",
        "Kalam-Regular",
    ),
}
PALETTES = (
    (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ),
    (
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#8ab17d",
        "#b56576",
        "#6d597a",
    ),
    (
        "#003f5c",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
    ),
    (
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
    ),
    (
        "#0b3954",
        "#087e8b",
        "#bfd7ea",
        "#ff5a5f",
        "#c81d25",
        "#5c946e",
        "#f6ae2d",
        "#33658a",
    ),
)


@lru_cache(maxsize=None)
def _available() -> dict[str, str]:
    return {p.stem: str(p) for p in FONT_DIR.glob("*.ttf")}


def font_names(role: str) -> list[str]:
    names = [name for name in FONT_ROLES[role] if name in _available()]
    if not names:
        raise FileNotFoundError(f"no fonts for role {role} under {FONT_DIR}")
    return names


@lru_cache(maxsize=4096)
def _load(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_available()[name], size)


def font(
    role: str, size: int, rng: random.Random | None = None, name: str | None = None
):
    names = font_names(role)
    chosen = name if name in names else (rng.choice(names) if rng else names[0])
    return _load(chosen, max(6, int(size)))


def font_path(role: str, rng: random.Random) -> str:
    return _available()[rng.choice(font_names(role))]


def text_width(draw: ImageDraw.ImageDraw, text: str, face) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=face)
    return right - left


def line_height(face) -> int:
    ascent, descent = face.getmetrics()
    return ascent + descent


def wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if current and text_width(draw, trial, face) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    return lines


def draw_lines(draw, xy, lines, face, fill="black", spacing=1.25) -> int:
    x, y = xy
    step = int(line_height(face) * spacing)
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += step
    return y


def label_tag(draw, xy, label: str, face, fill="#e00000", text_fill="white", pad=3):
    """A filled tag with ``label`` whose top-left corner is ``xy``; returns its box."""
    x, y = xy
    w = text_width(draw, label, face)
    h = line_height(face)
    box = (x, y, x + w + 2 * pad, y + h + pad)
    draw.rectangle(box, fill=fill)
    draw.text((x + pad, y + pad // 2), label, font=face, fill=text_fill)
    return box


def mark_point(draw, xy, label: str, face, color="#ff0000", radius=9, width=3):
    x, y = xy
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline=color, width=width
    )
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    tx, ty = x + radius + 2, y - radius - line_height(face) - 2
    if ty < 0:
        ty = y + radius + 2
    label_tag(draw, (tx, ty), label, face, fill=color)


def mark_box(draw, box, label: str, face, color="#ff0000", width=3):
    draw.rectangle(box, outline=color, width=width)
    x0, y0 = box[0], box[1] - line_height(face) - 4
    if y0 < 0:
        y0 = box[1] + 2
    label_tag(draw, (x0, y0), label, face, fill=color)


def noise(shape, sigma: float, rng: random.Random) -> np.ndarray:
    return (
        np.random.default_rng(rng.randrange(1 << 30)).standard_normal(
            shape, dtype=np.float32
        )
        * sigma
    )


def paper(size, rng: random.Random, tint=None) -> Image.Image:
    base = tint or rng.choice(
        [(255, 255, 255), (252, 250, 245), (248, 248, 244), (255, 253, 240)]
    )
    arr = np.full((size[1], size[0], 3), base, dtype=np.float32)
    arr += noise(arr.shape[:2], rng.uniform(0, 3), rng)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def degrade_scan(image: Image.Image, rng: random.Random) -> Image.Image:
    """Scanner look: slight rotation, blur, noise, contrast change, optional grayscale."""
    if rng.random() < 0.5:
        image = image.rotate(
            rng.uniform(-1.5, 1.5),
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=(250, 250, 250),
        )
    if rng.random() < 0.4:
        image = image.convert("L").convert("RGB")
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.BoxBlur(rng.uniform(0.3, 0.8)))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    arr = np.asarray(image, dtype=np.float32)
    arr += noise(arr.shape[:2], rng.uniform(0, 6), rng)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def photograph(
    image: Image.Image, rng: random.Random, background: Image.Image | None = None
) -> Image.Image:
    """A phone photo of a printed page: perspective, background, lighting gradient, blur, noise."""
    import cv2

    src = np.asarray(image.convert("RGB"))
    h, w = src.shape[:2]
    margin = int(max(w, h) * rng.uniform(0.06, 0.18))
    out_w, out_h = w + 2 * margin, h + 2 * margin
    jitter = lambda: rng.uniform(-0.04, 0.04) * max(w, h)
    dst = np.float32(
        [
            [margin + jitter(), margin + jitter()],
            [margin + w + jitter(), margin + jitter()],
            [margin + w + jitter(), margin + h + jitter()],
            [margin + jitter(), margin + h + jitter()],
        ]
    )
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst
    )
    if background is None:
        tone = np.array([rng.randint(60, 200) for _ in range(3)], dtype=np.float32)
        bg = np.tile(tone, (out_h, out_w, 1))
        bg += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 12, bg.shape)
        bg = np.clip(bg, 0, 255).astype(np.uint8)
    else:
        bg = np.asarray(background.convert("RGB").resize((out_w, out_h)))
    warped = cv2.warpPerspective(
        src, matrix, (out_w, out_h), borderMode=cv2.BORDER_TRANSPARENT, dst=bg.copy()
    )
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    grad = np.cos(angle) * xx / out_w + np.sin(angle) * yy / out_h
    light = (
        1 + rng.uniform(-0.18, 0.12) * (grad - grad.mean()) / (np.ptp(grad) + 1e-6) * 2
    )
    arr = warped.astype(np.float32) * light[..., None]
    arr += noise(arr.shape, rng.uniform(1, 5), rng)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    return out


def fit_pixels(image: Image.Image, max_pixels: int = 1_638_400) -> Image.Image:
    w, h = image.size
    if w * h <= max_pixels:
        return image
    s = (max_pixels / (w * h)) ** 0.5
    return image.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
