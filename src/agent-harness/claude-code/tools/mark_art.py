"""Draws assets/mark.png in character cells, for terminals that cannot show the picture.

Each cell gets the character and the two colors (the hue drawn, the hue behind it or the terminal's own
background) whose shape is closest to the picture in that cell. Two sets of characters:

- smooth: block elements, sextants and the diagonal wedges of Symbols for Legacy Computing, whose straight
  slanted edges follow the V's sides. Only terminals that draw these themselves show them (VS Code 1.108 and
  later, VTE 0.60 and later, Konsole 22.04 and later, Windows Terminal 1.25 and later, foot, kitty, Ghostty).
- blocks: block elements alone (halves, quadrants, eighths), which every terminal draws.

Every character's shape comes from its definition, not from a font: a wedge's from its Unicode name.

Writes hooks/lib/mark-art.ts (both sets at 2, 3 and 4 rows) and assets/mark.ansi (blocks, 4 rows, for the
demo's title card); with --preview PATH, a PNG of every size drawn as a terminal draws the characters.
Needs Pillow and numpy: python3 tools/mark_art.py [--preview PATH] from the plugin root.
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/mark.png"
ROWS = (2, 3, 4)
# A cell is about half as wide as it is high; the mark's box is two columns per row.
CELL_ASPECT = 0.5
# Samples per cell, across and down: the grid the shapes are compared on.
SX, SY = 12, 24
INKS = ("Y", "B")

X, Y = np.meshgrid((np.arange(SX) + 0.5) / SX, (np.arange(SY) + 0.5) / SY)


def rect(x0, y0, x1, y1):
    return (x0 <= X) & (x1 > X) & (y0 <= Y) & (y1 > Y)


def block_elements():
    shapes = {
        "█": rect(0, 0, 1, 1),
        "▀": rect(0, 0, 1, 0.5),
        "▄": rect(0, 0.5, 1, 1),
        "▌": rect(0, 0, 0.5, 1),
        "▐": rect(0.5, 0, 1, 1),
        "▔": rect(0, 0, 1, 1 / 8),
        "▕": rect(7 / 8, 0, 1, 1),
    }
    for k in range(1, 8):
        shapes[chr(0x2580 + k)] = rect(0, 1 - k / 8, 1, 1)  # lower k eighths
        shapes[chr(0x2590 - k)] = rect(0, 0, k / 8, 1)  # left k eighths
    parts = {
        "tl": rect(0, 0, 0.5, 0.5),
        "tr": rect(0.5, 0, 1, 0.5),
        "bl": rect(0, 0.5, 0.5, 1),
        "br": rect(0.5, 0.5, 1, 1),
    }
    quads = {
        "▘": ("tl",),
        "▝": ("tr",),
        "▖": ("bl",),
        "▗": ("br",),
        "▚": ("tl", "br"),
        "▞": ("tr", "bl"),
        "▙": ("tl", "bl", "br"),
        "▛": ("tl", "tr", "bl"),
        "▜": ("tl", "tr", "br"),
        "▟": ("tr", "bl", "br"),
    }
    for char, names in quads.items():
        shapes[char] = np.logical_or.reduce([parts[n] for n in names])
    return shapes


def sextants():
    parts = [
        rect(c / 2, r / 3, (c + 1) / 2, (r + 1) / 3) for r in range(3) for c in range(2)
    ]
    shapes = {}
    for p in range(1, 63):
        if p in (21, 42):
            continue  # the left and right halves are block elements
        shapes[chr(0x1FB00 + p - 1 - (p > 21) - (p > 42))] = np.logical_or.reduce(
            [parts[i] for i in range(6) if p >> i & 1]
        )
    return shapes


POINTS = {
    "UPPER LEFT": (0, 0),
    "UPPER CENTRE": (0.5, 0),
    "UPPER RIGHT": (1, 0),
    "UPPER MIDDLE LEFT": (0, 1 / 3),
    "UPPER MIDDLE RIGHT": (1, 1 / 3),
    "LOWER MIDDLE LEFT": (0, 2 / 3),
    "LOWER MIDDLE RIGHT": (1, 2 / 3),
    "LOWER LEFT": (0, 1),
    "LOWER CENTRE": (0.5, 1),
    "LOWER RIGHT": (1, 1),
}


def side(p, q, x, y):
    return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])


def wedges():
    """U+1FB3C to U+1FB6F: the part of the cell on the named corner's side of the named diagonal, and the triangles."""
    shapes = {}
    for cp in range(0x1FB3C, 0x1FB68):
        corner, line = unicodedata.name(chr(cp)).split(" BLOCK DIAGONAL ")
        start, end = line.split(" TO ")
        p, q, c = POINTS[start], POINTS[end], POINTS[corner]
        shapes[chr(cp)] = np.sign(side(p, q, X, Y)) == np.sign(side(p, q, *c))
    quarter = {
        "LEFT": (X <= Y) & (X <= 1 - Y),
        "UPPER": (Y <= X) & (Y <= 1 - X),
        "RIGHT": (1 - X <= Y) & (1 - X <= 1 - Y),
        "LOWER": (1 - Y <= X) & (1 - Y <= 1 - X),
    }
    for cp in range(0x1FB68, 0x1FB70):
        name = unicodedata.name(chr(cp))
        if "ONE QUARTER" in name:
            shapes[chr(cp)] = quarter[name.split(" ")[0]]
        else:
            named = name.split(" TRIANGULAR")[0].split(" AND ")
            shapes[chr(cp)] = ~quarter[next(k for k in quarter if k not in named)]
    return shapes


SETS = {
    "blocks": block_elements(),
    "smooth": {**block_elements(), **sextants(), **wedges()},
}


def coverage(img, rows):
    """Per sample of every cell: how much of it is yellow and how much blue, the mark fitted to its box."""
    cols = 2 * rows
    box_w, box_h = cols * CELL_ASPECT, rows  # in cell heights
    scale = min(box_w / img.width, box_h / img.height)
    w, h = img.width * scale, img.height * scale
    sub = 3
    px = cols * SX * sub, rows * SY * sub
    canvas = Image.new("RGBA", px, (0, 0, 0, 0))
    fitted = img.resize(
        (max(1, round(w / box_w * px[0])), max(1, round(h / box_h * px[1]))),
        Image.LANCZOS,
    )
    canvas.paste(
        fitted,
        (
            round((box_w - w) / 2 / box_w * px[0]),
            round((box_h - h) / 2 / box_h * px[1]),
        ),
        fitted,
    )
    a = np.asarray(canvas).astype(float) / 255
    alpha = a[..., 3]
    yellow = alpha * (a[..., 0] > a[..., 2])
    blue = alpha * (a[..., 0] <= a[..., 2])

    def pool(m):
        return m.reshape(rows * SY, sub, cols * SX, sub).mean(axis=(1, 3))

    return pool(yellow), pool(blue)


ONEHOT = {
    "Y": np.array([1.0, 0, 0]),
    "B": np.array([0, 1.0, 0]),
    None: np.array([0, 0, 1.0]),
}


def best_cell(target, shapes):
    """The character and colors closest to `target`, a (SY, SX, 3) array of yellow, blue and background."""
    best = (((target - ONEHOT[None]) ** 2).sum(), " ", None, None)
    for char, shape in shapes.items():
        m = shape[..., None]
        for fg in INKS:
            for bg in (None, *[i for i in INKS if i != fg]):
                e = ((target - (m * ONEHOT[fg] + (~m) * ONEHOT[bg])) ** 2).sum()
                if e < best[0]:
                    best = (e, char, fg, bg)
    return best[1:]


def polygon_mask(vertices, width, height):
    """Which samples of a grid of `width` by `height` steps lie inside a polygon given in steps (even-odd rule)."""
    xs, ys = np.meshgrid(
        (np.arange(width * SX // 2) + 0.5) / (SX // 2),
        (np.arange(height * SY // 3) + 0.5) / (SY // 3),
    )
    inside = np.zeros(xs.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1], strict=False):
        if y0 == y1:
            continue
        crosses = (ys >= min(y0, y1)) & (ys < max(y0, y1))
        x_at = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (xs < x_at)
    return inside


# The slopes, as (across, down) in steps of half a column and a third of a row, along which the wedges draw a
# straight edge from cell to cell, measured: 1:3 and 1:1 from any grid point, 2:1 from a cell's corner.
STEEP = ((1, 3), (1, 1))
FLAT = ((2, 1), (1, 1))


def run(x, y, slope, to_y):
    """Where an edge from (x, y) along `slope` reaches row `to_y`, or None off the grid points."""
    dx, dy = slope
    return None if (to_y - y) * dx % dy else x + (to_y - y) * dx // dy


def lattice_blue(blue, xt, rows):
    """The blue bar on the wedges' grid, from the yellow's tip at column step `xt`: sides at 1:3, top at 2:1 or
    1:1, its corners on grid points; the one closest to the picture's blue, with its error.
    """
    w, h = 4 * rows, 3 * rows
    best = (np.inf, None)
    for x_bl in (xt, xt + 1):
        for yb in range(h % 3, h, 3):
            x_tl = x_bl + (h - yb) // 3
            for dx, dy in FLAT:
                if yb * dx % dy:
                    continue
                xr = x_tl + yb * dx // dy  # up and to the right
                x_br = xr - h // 3
                if xr > w or x_br <= x_bl or ((dx, dy) == (2, 1) and xr % 2):
                    continue
                mask = polygon_mask([(x_bl, h), (x_tl, yb), (xr, 0), (x_br, h)], w, h)
                err = np.abs(mask - blue).sum()
                if err < best[0]:
                    best = (err, mask)
    return best


def lattice_design(yellow, blue, rows):
    """The V redrawn on the wedges' grid, every corner on a grid point and every slanted edge along a slope the
    wedges draw straight. The yellow's slant (about 0.7 steps across per step down) has no such slope, so it is
    drawn as 1:1 then 1:3, the bend chosen with the rest to cover the picture best. At 2 rows that leaves too
    little yellow, so there the yellow follows the picture cell by cell and only the blue is redrawn.
    """
    w, h = 4 * rows, 3 * rows
    if rows == 2:
        xt = round(float((yellow.sum(axis=0) > 0).nonzero()[0].max() + 1) / (SX // 2))
        _, b_mask = lattice_blue(blue, xt, rows)
        return yellow * ~b_mask, b_mask.astype(float)
    best = (np.inf, None)
    for yt in range(h):
        for xl in range(w):
            for ym in range(yt + 1, h):
                xm = run(xl, yt, (1, 1), ym)
                xt = run(xm, ym, (1, 3), h)
                if xt is None or xt >= w or xt <= xl:
                    continue
                y_mask = polygon_mask([(xl, yt), (xt, yt), (xt, h), (xm, ym)], w, h)
                y_err = np.abs(y_mask - yellow).sum()
                if y_err >= best[0]:
                    continue
                b_err, b_mask = lattice_blue(blue, xt, rows)
                if b_mask is not None and y_err + b_err < best[0]:
                    best = (
                        y_err + b_err,
                        (y_mask.astype(float), (b_mask & ~y_mask).astype(float)),
                    )
    return best[1]


def art(img, rows, shapes, snap=False):
    yellow, blue = coverage(img, rows)
    if snap:
        yellow, blue = lattice_design(yellow, blue, rows)
    target = np.stack([yellow, blue, np.clip(1 - yellow - blue, 0, 1)], axis=-1)
    return [
        [
            list(
                best_cell(target[r * SY : (r + 1) * SY, c * SX : (c + 1) * SX], shapes)
            )
            for c in range(2 * rows)
        ]
        for r in range(rows)
    ]


def hues(img):
    a = np.asarray(img).astype(float)
    opaque = a[..., 3] > 250
    masks = (
        ("Y", opaque & (a[..., 0] > a[..., 2])),
        ("B", opaque & (a[..., 0] <= a[..., 2])),
    )
    return {
        k: "#{:02x}{:02x}{:02x}".format(
            *tuple(round(v) for v in a[m][:, :3].mean(axis=0))
        )
        for k, m in masks
    }


def rgb(hex_):
    return tuple(int(hex_[i : i + 2], 16) for i in (1, 3, 5))


def esc(layer, color):
    """A truecolor escape: layer 38 sets the foreground, 48 the background."""
    r, g, b = color
    return f"\033[{layer};2;{r};{g};{b}m"


def ansi(rows, palette):
    lines = []
    for line in rows:
        text = ""
        for char, fg, bg in line:
            if fg is None:
                text += char
                continue
            text += esc(38, rgb(palette[fg])) + (
                esc(48, rgb(palette[bg])) if bg else ""
            )
            text += char + "\033[0m"
        lines.append(text)
    return "\n".join(lines) + "\n"


def draw(rows_art, shapes, palette, cell=(16, 34), background=(39, 40, 34)):
    """The art as a terminal that draws these characters itself shows it."""
    cw, ch = cell
    img = np.zeros((len(rows_art) * ch, len(rows_art[0]) * cw, 3), dtype=np.uint8)
    img[:] = background
    xs = ((np.arange(cw) + 0.5) / cw * SX).astype(int)
    ys = ((np.arange(ch) + 0.5) / ch * SY).astype(int)
    for r, line in enumerate(rows_art):
        for c, (char, fg, bg) in enumerate(line):
            tile = img[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
            if bg:
                tile[:] = rgb(palette[bg])
            if fg:
                tile[shapes[char][np.ix_(ys, xs)]] = rgb(palette[fg])
    return Image.fromarray(img)


def preview(path, arts, palette):
    pad = 24
    rows_of_tiles = [
        [draw(arts[name][rows], SETS[name], palette) for rows in ROWS]
        for name in ("blocks", "smooth")
    ]
    width = max(sum(t.width for t in line) for line in rows_of_tiles) + pad * 4
    height = sum(max(t.height for t in line) for line in rows_of_tiles) + pad * 3
    sheet = Image.new("RGB", (width, height), (39, 40, 34))
    y = pad
    for line in rows_of_tiles:
        x = pad
        for t in line:
            sheet.paste(t, (x, y))
            x += t.width + pad
        y += max(t.height for t in line) + pad
    sheet.save(path)


def main():
    img = Image.open(SOURCE).convert("RGBA")
    img = img.crop(img.getbbox())
    palette = hues(img)
    arts = {
        name: {rows: art(img, rows, shapes, snap=name == "smooth") for rows in ROWS}
        for name, shapes in SETS.items()
    }
    ts = [
        "// Generated by tools/mark_art.py from assets/mark.png; edit the picture and rerun it, not this file.",
        "",
        f"export const HUES = {{ Y: '{palette['Y']}', B: '{palette['B']}' }} as const",
        "",
        "export type Hue = keyof typeof HUES",
        "",
        "// A character, the hue it is drawn in, and the hue behind it (null: the terminal's own).",
        "export type Cell = readonly [char: string, fg: Hue | null, bg: Hue | null]",
        "",
        "export type ArtSet = 'smooth' | 'blocks'",
        "",
        "// `smooth` uses the sextants and diagonal wedges of Symbols for Legacy Computing; `blocks` only block elements.",
        "export const MARK_ART: Readonly<Record<ArtSet, Readonly<Record<2 | 3 | 4, readonly (readonly Cell[])[]>>>> = {",
    ]
    for name in ("smooth", "blocks"):
        ts.append(f"  {name}: {{")
        for rows in ROWS:
            ts.append(f"    {rows}: [")
            for line in arts[name][rows]:
                ts.append(
                    "      ["
                    + ", ".join(
                        json.dumps(cell, ensure_ascii=False).replace('"', "'")
                        for cell in line
                    )
                    + "],"
                )
            ts.append("    ],")
        ts.append("  },")
    ts.append("}")
    (ROOT / "hooks/lib/mark-art.ts").write_text("\n".join(ts) + "\n")
    (ROOT / "assets/mark.ansi").write_text(ansi(arts["blocks"][4], palette))
    if "--preview" in sys.argv:
        preview(sys.argv[sys.argv.index("--preview") + 1], arts, palette)
    for name in ("blocks", "smooth"):
        for rows in ROWS:
            print(f"{name}, {rows} rows")
            sys.stdout.write(ansi(arts[name][rows], palette))


if __name__ == "__main__":
    main()
