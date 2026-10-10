"""Questions whose text and options live only inside the image (MMMU-Pro vision and R-Bench-M style).

``vega_render``: Vega M2T-v5 text decision rows (permissive licences, selected by ``fetch vega``) rendered
with state, question and options inside the image; five-option rows sometimes gain R-Bench-M's sixth
option "All other answers are incorrect". ``generate``: synthetic geometry and graph problems with an
exact answer and a diagram, 4-10 numeric options. Layouts: exam sheet, quiz screenshot, slide, and a
phone photo of a printed sheet.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import random
from functools import lru_cache

from PIL import Image, ImageDraw

from d25.omni.data import render
from d25.omni.data.gen.photos import RAW
from d25.omni.data.rows import Item, numeric_distractors, rng_for

LETTERS = "ABCDEFGHIJ"
NONE_CORRECT = "All other answers are incorrect"
INSTRUCTIONS = (
    "Answer the multiple-choice question shown in the image.",
    "Read the question and the options in the picture and choose the correct option.",
    "Which option is correct for the question in the image?",
)


def _triangle(rng):
    a, b = rng.randint(25, 95), rng.randint(25, 95)
    if a + b >= 165:
        return None
    size = 520
    image = Image.new("RGB", (size, int(size * 0.75)), "white")
    draw = ImageDraw.Draw(image)
    base = (60, int(size * 0.75) - 60), (size - 60, int(size * 0.75) - 60)
    ta, tb = math.radians(a), math.radians(b)
    width = base[1][0] - base[0][0]
    x = width * math.tan(tb) / (math.tan(ta) + math.tan(tb))
    top = (base[0][0] + x, base[0][1] - min(x * math.tan(ta), size * 0.6))
    draw.polygon([base[0], base[1], top], outline="black", width=3)
    face = render.font("serif", 26, rng)
    draw.text((base[0][0] + 28, base[0][1] - 40), f"{a}°", font=face, fill="black")
    draw.text((base[1][0] - 70, base[1][1] - 40), f"{b}°", font=face, fill="black")
    draw.text((top[0] - 10, top[1] + 30), "x", font=face, fill="black")
    return (
        image,
        "In the triangle shown, what is the value of x (in degrees)?",
        float(180 - a - b),
        "°",
    )


def _rectangle(rng):
    w, h = rng.randint(3, 25), rng.randint(3, 25)
    ask = rng.choice(["area", "perimeter"])
    image = Image.new("RGB", (560, 420), "white")
    draw = ImageDraw.Draw(image)
    scale = 300 / max(w, h)
    x0, y0 = 100, 60
    draw.rectangle((x0, y0, x0 + w * scale, y0 + h * scale), outline="black", width=3)
    face = render.font("sans", 24, rng)
    unit = rng.choice(["cm", "m", "in"])
    draw.text(
        (x0 + w * scale / 2 - 20, y0 + h * scale + 10),
        f"{w} {unit}",
        font=face,
        fill="black",
    )
    draw.text(
        (x0 + w * scale + 12, y0 + h * scale / 2 - 12),
        f"{h} {unit}",
        font=face,
        fill="black",
    )
    value = float(w * h if ask == "area" else 2 * (w + h))
    return (
        image,
        f"What is the {ask} of the rectangle shown?",
        value,
        f"{unit}²" if ask == "area" else unit,
    )


def _line_graph(rng):
    from d25.omni.data.gen.charts import _plt

    plt = _plt()
    slope = rng.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3])
    icpt = rng.randint(-4, 4)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([-5, 5], [slope * -5 + icpt, slope * 5 + icpt], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(-5, 6))
    ax.set_yticks(range(-10, 11, 2))
    ax.grid(True, alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-10, 10)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    if rng.random() < 0.5:
        return (
            image,
            "What is the slope of the line shown in the graph?",
            float(slope),
            "",
        )
    return (
        image,
        "What is the y-intercept of the line shown in the graph?",
        float(icpt),
        "",
    )


def synthetic(rng: random.Random):
    made = rng.choice([_triangle, _rectangle, _line_graph])(rng)
    if made is None:
        return None
    diagram, question, value, unit = made
    n = rng.choice([10, 10, 6, 4])
    decimals = 1 if not float(value).is_integer() else 0
    values = [value] + numeric_distractors(
        value, rng, n - 1, integer=decimals == 0, spread=0.5
    )
    if len(values) < n:
        return None
    suffix = unit if unit == "°" else (f" {unit}" if unit else "")
    options = [f"{v:.{decimals}f}{suffix}" for v in values]
    if len(set(options)) != n:
        return None
    order = list(range(n))
    rng.shuffle(order)
    return question, [options[i] for i in order], order.index(0), diagram


def _layout(
    question: str, options: list[str], diagram: Image.Image | None, rng: random.Random
) -> Image.Image:
    style = rng.choice(["sheet", "sheet", "quiz", "slide"])
    width = rng.choice([900, 1000, 1100, 1240])
    canvas = Image.new("RGB", (width, 2600), "white")
    draw = ImageDraw.Draw(canvas)
    serif = rng.random() < 0.5 and style == "sheet"
    face = render.font("serif" if serif else "sans", rng.randint(22, 28), rng)
    bold = render.font("serif_bold" if serif else "sans_bold", rng.randint(24, 30), rng)
    ink = "black"
    if style == "slide":
        bg = rng.choice(["#1f3b57", "#f3efe6", "#e8f1f8", "#2b2d42"])
        ink = "white" if bg in ("#1f3b57", "#2b2d42") else "black"
        draw.rectangle((0, 0, width, 2600), fill=bg)
    y = 50
    if style == "quiz":
        draw.rectangle((0, 0, width, 70), fill=rng.choice(render.PALETTES)[0])
        draw.text(
            (30, 18),
            f"Practice quiz · Question {rng.randint(1, 40)}",
            font=bold,
            fill="white",
        )
        y = 110
    else:
        draw.text((50, y), f"{rng.randint(1, 60)}.", font=bold, fill=ink)
    lines = render.wrap(draw, question, face, width - 160)
    y = (
        render.draw_lines(
            draw, (100 if style != "quiz" else 40, y), lines, face, fill=ink
        )
        + 20
    )
    if diagram is not None:
        d = diagram
        if d.width > width - 200:
            d = d.resize((width - 200, int(d.height * (width - 200) / d.width)))
        canvas.paste(d, ((width - d.width) // 2, y))
        y += d.height + 20
    marker = rng.choice(["({})", "{}.", "{})", "{}:"])
    two_col = (
        len(options) >= 8
        and max(render.text_width(draw, o, face) for o in options) < width / 2 - 140
    )
    col_x = [100, width // 2 + 20] if two_col else [100]
    rows_per_col = math.ceil(len(options) / len(col_x))
    start = y
    for i, option in enumerate(options):
        c = i // rows_per_col if two_col else 0
        if two_col and i == rows_per_col:
            y = start
        label = marker.format(LETTERS[i])
        if style == "quiz":
            wrapped = render.wrap(draw, option, face, width - 220)
            box_h = render.line_height(face) * len(wrapped) + 16
            right = (width // 2 if two_col else width) - 40 + (width // 2 - 40) * c
            draw.rounded_rectangle(
                (col_x[c] - 20, y, right, y + box_h),
                radius=10,
                outline="#999999",
                width=2,
            )
            draw.text((col_x[c], y + 8), label, font=bold, fill=ink)
            render.draw_lines(
                draw, (col_x[c] + 60, y + 8), wrapped, face, fill=ink, spacing=1.0
            )
            y += box_h + 12
        else:
            draw.text((col_x[c], y), label, font=bold, fill=ink)
            wrapped = render.wrap(
                draw, option, face, (width // 2 - 200) if two_col else width - 260
            )
            y = render.draw_lines(draw, (col_x[c] + 60, y), wrapped, face, fill=ink) + 8
    image = canvas.crop((0, 0, width, min(2600, max(y, start) + 40)))
    if style == "sheet" and rng.random() < 0.35:
        image = render.photograph(image, rng)
    return image


@lru_cache(maxsize=None)
def vega_pool() -> list[dict]:
    with gzip.open(RAW / "vega" / "pool.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def vega_render(index: int, ctx=None) -> Item | None:
    """A Vega M2T-v5 text decision row rendered with its state, question and options inside the image.

    The text-side teacher distribution (pplx-decider-v1.1 on the text row) is kept, permuted with the
    options, as ``meta.teachers.pplx11_text``; the target is the gold answer.
    """
    pool = vega_pool()
    rng = rng_for("gen-stem-vega", index)
    row = (
        pool[index % len(pool)] if index < len(pool) else pool[rng.randrange(len(pool))]
    )
    n = len(row["options"])
    order = list(range(n))
    rng.shuffle(order)
    options = [row["options"][i] for i in order]
    gold = order.index(row["gold"])
    teacher = (
        [row["teacher"][i] for i in order]
        if row.get("teacher") and len(row["teacher"]) == n
        else None
    )
    if n == 5 and rng.random() < 0.4:
        options.append(NONE_CORRECT)
        teacher = teacher + [0.0] if teacher else None
    text = "\n\n".join(
        (
            part
            if isinstance(part, str)
            else json.dumps(part, ensure_ascii=False, indent=1)
        )
        for part in (row["state"], row["instructions"])
        if part
    )
    image = _layout(text, options, None, rng)
    meta = {
        "benchmark_target": "MMMU-Pro" if len(options) >= 8 else "R-Bench-M",
        "text_source": "vega-m2t-v5",
        "vega_id": row["id"],
        "vega_source": row["source"],
        "licences": [row["licence"] or "unknown"],
        "source_ids": [f"vega:{row['id']}"],
    }
    if teacher:
        meta["teachers"] = {"pplx11_text": teacher}
    return Item(
        source="gen-stem",
        family="stem-vega",
        skill="stem_in_image",
        images=[image],
        image_kinds=["png"],
        question=rng.choice(INSTRUCTIONS),
        options=options,
        gold=gold,
        keys=list(LETTERS[: len(options)]),
        fixed_order=True,
        describe_keys=False,
        meta=meta,
        image_text=[text + " " + " ".join(options)],
    )


def generate(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-stem", index)
    made = synthetic(rng)
    if made is None:
        return None
    question, options, gold, diagram = made
    image = _layout(question, options, diagram, rng)
    return Item(
        source="gen-stem",
        family="stem-synthetic",
        skill="stem_in_image",
        images=[image],
        image_kinds=["png"],
        question=rng.choice(INSTRUCTIONS),
        options=options,
        gold=gold,
        keys=list(LETTERS[: len(options)]),
        fixed_order=True,
        describe_keys=False,
        meta={
            "benchmark_target": "MMMU-Pro" if len(options) == 10 else "R-Bench-M",
            "text_source": "synthetic",
            "licences": ["Apache-2.0"],
        },
        image_text=[question + " " + " ".join(options)],
    )
