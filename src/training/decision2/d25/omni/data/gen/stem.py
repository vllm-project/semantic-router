"""Questions whose text and options live only inside the image (MMMU-Pro vision and R-Bench-M style).

Sources: AQuA-RAT train (expanded to 10 numeric options), MedMCQA train (6 options with 'All other answers
are incorrect', gold on it one time in six), QASC train (8 options) and synthetic geometry, graph and
arithmetic problems with exact answers and a diagram. Layouts: exam sheet, quiz screenshot, slide, and a
phone photo of a printed sheet.
"""

from __future__ import annotations

import io
import math
import random
import re

from PIL import Image, ImageDraw

from d25.omni.data import render
from d25.omni.data.gen.photos import text_pool
from d25.omni.data.rows import Item, fmt_number, numeric_distractors, rng_for

LETTERS = "ABCDEFGHIJ"
NONE_CORRECT = "All other answers are incorrect"


def _number(text: str) -> float | None:
    m = re.fullmatch(
        r"\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|[a-zA-Z/. ]{0,12})?\s*", text
    )
    return float(m.group(1).replace(",", "")) if m else None


def _expand_numeric(
    options: list[str], gold: int, n: int, rng
) -> tuple[list[str], int] | None:
    value = _number(options[gold])
    if value is None:
        return None
    suffix = re.sub(r"^[\s$\-\d,.]+", "", options[gold]).strip()
    decimals = (
        len(options[gold].split(".")[1].split()[0]) if "." in options[gold] else 0
    )
    seen = {o.strip() for o in options}
    out = list(options)
    for v in numeric_distractors(value, rng, 3 * n, integer=decimals == 0, spread=0.6):
        text = f"{v:.{decimals}f}" + (f" {suffix}" if suffix else "")
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) == n:
            break
    return (out, gold) if len(out) == n else None


def from_mcq(
    index: int, rng: random.Random
) -> tuple[str, list[str], int, str, Image.Image | None] | None:
    pool = text_pool("mcq")
    entry = pool[rng.randrange(len(pool))]
    options, gold = list(entry["options"]), entry["gold"]
    if entry["source"] == "aqua-rat":
        if rng.random() < 0.65:
            expanded = _expand_numeric(options, gold, 10, rng)
            if expanded is None:
                return None
            options, gold = expanded
        elif rng.random() < 1 / 6:
            expanded = _expand_numeric(options, gold, 6, rng)
            if expanded is None:
                return None
            options = [o for i, o in enumerate(expanded[0]) if i != gold] + [
                NONE_CORRECT
            ]
            gold = 5
        else:
            options = options + [NONE_CORRECT]
    elif entry["source"] == "medmcqa":
        same = [
            e
            for e in rng.sample(pool, 60)
            if e["source"] == "medmcqa" and e["id"] != entry["id"]
        ]
        extras = list(
            dict.fromkeys(o for e in same for o in e["options"] if o not in options)
        )
        if len(extras) < 2:
            return None
        options = options + [extras[0]]
        if rng.random() < 1 / 6:
            options = [o for i, o in enumerate(options) if i != gold] + [extras[1]]
            gold = 5
        options.append(NONE_CORRECT)
    order = list(range(len(options)))
    if options[-1] == NONE_CORRECT:
        head = order[:-1]
        rng.shuffle(head)
        order = head + [order[-1]]
    else:
        rng.shuffle(order)
    options = [options[i] for i in order]
    gold = order.index(gold)
    return entry["question"], options, gold, entry["source"], None


def _triangle(rng):
    a, b = rng.randint(25, 95), rng.randint(25, 95)
    if a + b >= 165:
        return None
    c = 180 - a - b
    size = 520
    image = Image.new("RGB", (size, int(size * 0.75)), "white")
    draw = ImageDraw.Draw(image)
    base = (60, int(size * 0.75) - 60), (size - 60, int(size * 0.75) - 60)
    ta, tb = math.radians(a), math.radians(b)
    width = base[1][0] - base[0][0]
    x = width * math.tan(tb) / (math.tan(ta) + math.tan(tb))
    y = x * math.tan(ta)
    top = (base[0][0] + x, base[0][1] - min(y, size * 0.6))
    draw.polygon([base[0], base[1], top], outline="black", width=3)
    face = render.font("serif", 26, rng)
    draw.text((base[0][0] + 28, base[0][1] - 40), f"{a}°", font=face, fill="black")
    draw.text((base[1][0] - 70, base[1][1] - 40), f"{b}°", font=face, fill="black")
    draw.text((top[0] - 10, top[1] + 30), "x", font=face, fill="black")
    return (
        image,
        "In the triangle shown, what is the value of x (in degrees)?",
        float(c),
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
    xs = [-5, 5]
    ax.plot(xs, [slope * x + icpt for x in xs], linewidth=2)
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
    builder = rng.choice([_triangle, _rectangle, _line_graph])
    made = builder(rng)
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
    options = [
        f"{v:.{decimals}f}{(' ' + unit) if unit and unit != '°' else unit}"
        for v in values
    ]
    if len(set(options)) != n:
        return None
    order = list(range(n))
    rng.shuffle(order)
    return question, [options[i] for i in order], order.index(0), "synthetic", diagram


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
    bg, ink = "white", "black"
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
    y = (
        render.draw_lines(
            draw,
            (100 if style != "quiz" else 40, y),
            render.wrap(draw, question, face, width - 160),
            face,
            fill=ink,
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
            lines = render.wrap(draw, option, face, width - 220)
            box_h = render.line_height(face) * len(lines) + 16
            draw.rounded_rectangle(
                (
                    col_x[c] - 20,
                    y,
                    (width // 2 if two_col else width) - 40 + (width // 2 - 40) * c,
                    y + box_h,
                ),
                radius=10,
                outline="#999999",
                width=2,
            )
            draw.text((col_x[c], y + 8), label, font=bold, fill=ink)
            render.draw_lines(
                draw, (col_x[c] + 60, y + 8), lines, face, fill=ink, spacing=1.0
            )
            y += box_h + 12
        else:
            draw.text((col_x[c], y), label, font=bold, fill=ink)
            lines = render.wrap(
                draw, option, face, (width // 2 - 200) if two_col else width - 260
            )
            y = render.draw_lines(draw, (col_x[c] + 60, y), lines, face, fill=ink) + 8
    bottom = max(y, start) + 40
    image = canvas.crop((0, 0, width, min(2600, bottom)))
    if style == "sheet" and rng.random() < 0.35:
        image = render.photograph(image, rng)
    return image


def generate(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-stem", index)
    made = synthetic(rng) if rng.random() < 0.3 else from_mcq(index, rng)
    if made is None:
        return None
    question, options, gold, source, diagram = made
    image = _layout(question, options, diagram, rng)
    instruction = rng.choice(
        [
            "Answer the multiple-choice question shown in the image.",
            "Read the question and the options in the picture and choose the correct option.",
            "Which option is correct for the question in the image?",
        ]
    )
    licences = {
        "aqua-rat": "Apache-2.0",
        "medmcqa": "Apache-2.0",
        "qasc": "CC-BY-4.0",
        "synthetic": "Apache-2.0",
    }
    return Item(
        source="gen-stem",
        family=f"stem-{source}",
        skill="stem_in_image",
        images=[image],
        image_kinds=["png"],
        question=instruction,
        options=options,
        gold=gold,
        keys=list(LETTERS[: len(options)]),
        fixed_order=True,
        describe_keys=False,
        meta={
            "benchmark_target": "MMMU-Pro" if len(options) == 10 else "R-Bench-M",
            "text_source": source,
            "licences": [licences[source]],
        },
        image_text=[question + " " + " ".join(options)],
    )


__all__ = ["generate", "fmt_number"]
