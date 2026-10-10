"""Photo- and scan-like degradations for rendered documents, and marker drawing on images."""

from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def font(size: int) -> ImageFont.ImageFont:
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        import matplotlib

        path = f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size)


def perspective_coeffs(
    src: list[tuple[float, float]], dst: list[tuple[float, float]]
) -> list[float]:
    """Coefficients for ``Image.transform(..., PERSPECTIVE)`` mapping output ``dst`` to input ``src``."""
    rows = []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(rows, dtype=float)
    b = np.array([c for p in src for c in p], dtype=float)
    return list(np.linalg.solve(A, b))


def texture(size: tuple[int, int], r: random.Random) -> Image.Image:
    w, h = size
    base = np.array([r.randint(60, 230) for _ in range(3)], dtype=float)
    rng = np.random.default_rng(r.randrange(2**31))
    noise = rng.normal(0, r.uniform(4, 14), (h // 8 + 1, w // 8 + 1, 1))
    noise = (
        np.array(
            Image.fromarray(
                np.clip(noise + 128, 0, 255).astype(np.uint8)[:, :, 0]
            ).resize((w, h), Image.BICUBIC),
            dtype=float,
        )[:, :, None]
        - 128
    )
    grain = rng.normal(0, 3, (h, w, 1))
    img = np.clip(base[None, None, :] + noise + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def photo(
    doc: Image.Image,
    r: random.Random,
    target_long: int = 1400,
    tilt: float = 0.02,
    max_angle: float = 2.5,
) -> Image.Image:
    """Place a rendered document on a textured surface with perspective, light falloff and noise."""
    doc = doc.convert("RGB")
    dw, dh = doc.size
    pad = int(max(dw, dh) * r.uniform(0.08, 0.2))
    W, H = dw + 2 * pad, dh + 2 * pad
    canvas = texture((W, H), r)
    jitter = lambda: r.uniform(-tilt, tilt) * max(dw, dh)  # noqa: E731
    corners = [
        (pad + jitter(), pad + jitter()),
        (pad + dw + jitter(), pad + jitter()),
        (pad + dw + jitter(), pad + dh + jitter()),
        (pad + jitter(), pad + dh + jitter()),
    ]
    coeffs = perspective_coeffs([(0, 0), (dw, 0), (dw, dh), (0, dh)], corners)
    warped = doc.transform(
        (W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=None
    )
    mask = Image.new("L", (dw, dh), 255).transform(
        (W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC
    )
    shadow = mask.filter(ImageFilter.GaussianBlur(r.uniform(6, 16)))
    canvas = Image.composite(
        Image.new("RGB", (W, H), (30, 30, 30)),
        canvas,
        shadow.point(lambda v: int(v * 0.35)),
    )
    canvas.paste(warped, (0, 0), mask)
    angle = r.uniform(-max_angle, max_angle)
    canvas = canvas.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=tuple(int(v) for v in np.array(canvas).reshape(-1, 3).mean(0)),
    )
    arr = np.array(canvas, dtype=float)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = r.uniform(0, W), r.uniform(0, H)
    falloff = 1 - r.uniform(0.1, 0.35) * np.sqrt(
        ((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2
    )
    warm = np.array([1.0, r.uniform(0.95, 1.0), r.uniform(0.85, 1.0)])
    arr = arr * falloff[:, :, None] * warm[None, None, :]
    arr += np.random.default_rng(r.randrange(2**31)).normal(
        0, r.uniform(2, 6), arr.shape
    )
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if r.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.9)))
    scale = target_long / max(img.size)
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def scan(doc: Image.Image, r: random.Random, target_long: int = 1100) -> Image.Image:
    """Grey, slightly skewed, noisy photocopy look."""
    img = doc.convert("L")
    img = img.rotate(
        r.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=255
    )
    arr = np.array(img, dtype=float)
    arr = 255 - (255 - arr) * r.uniform(0.7, 1.0)
    arr = arr * r.uniform(0.85, 0.97) + r.uniform(0, 20)
    rng = np.random.default_rng(r.randrange(2**31))
    arr += rng.normal(0, r.uniform(3, 9), arr.shape)
    speckle = rng.random(arr.shape) < r.uniform(0.0005, 0.003)
    arr[speckle] = rng.uniform(0, 120, speckle.sum())
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.2, 0.7)))
    scale = target_long / max(img.size)
    return img.resize(
        (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
    ).convert("RGB")


def jpeg(img: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    fill=(0, 0, 0),
    color=(255, 255, 255),
) -> None:
    f = font(size)
    l, t, rgt, b = draw.textbbox((0, 0), text, font=f)
    w, h = rgt - l, b - t
    x, y = xy
    draw.rectangle([x - 2, y - 2, x + w + 3, y + h + 4], fill=fill)
    draw.text((x - l, y - t), text, font=f, fill=color)


def point_marker(
    img: Image.Image, xy: tuple[float, float], label: str, color=(255, 0, 0)
) -> None:
    """BLINK-style marker: a red ring with a white rim and a black label tag above it."""
    draw = ImageDraw.Draw(img)
    x, y = xy
    radius = max(5, int(round(0.009 * max(img.size))))
    width = max(2, radius // 3)
    draw.ellipse(
        [x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1],
        outline=(255, 255, 255),
        width=width + 2,
    )
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], outline=color, width=width
    )
    size = max(12, int(round(radius * 2.0)))
    f = font(size)
    l, t, rgt, b = draw.textbbox((0, 0), label, font=f)
    label_box(draw, (x - (rgt - l) / 2, y - radius - (b - t) - 8), label, size)


def box_marker(
    img: Image.Image,
    box: tuple[float, float, float, float],
    color,
    width: int | None = None,
    label: str | None = None,
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    width = width or max(2, int(round(0.004 * max(img.size))))
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    if label:
        size = max(14, int(round(0.018 * max(img.size))))
        label_box(
            draw,
            (x0 + width, max(0, y0 - size - 8) if y0 > size + 10 else y0 + width + 2),
            label,
            size,
            fill=color,
            color=(255, 255, 255),
        )


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
