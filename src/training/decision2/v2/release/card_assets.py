"""Banner and evaluation charts of a Decision 2.0 product card (PNG, matplotlib).

Run outside the package build, where matplotlib and the Inter fonts are present:

    python -m v2.release.card_assets --spec SPEC --index INDEX --logo LOGO \
        --fonts DIR --output DIR

It writes ``assets/banner.png`` and the four charts of ``layout.CHART_FILES`` plus
``card-assets.json``, a receipt of input and output SHA-256 digests (no values).
The build copies the PNGs only after re-checking that receipt against its own
reports and Index input, so a chart can never show other numbers than the card.

Style: white background, the vLLM-SR palette (blue #30A0FC, yellow #FCB414),
Inter, the logo bottom-right of every chart. PNGs carry no text chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from v2.release import card, card_index, layout

SCHEMA = "dev2-card-assets/1"
RECEIPT = "card-assets.json"
BANNER = "assets/banner.png"
FONTS = (
    "Inter-Regular",
    "Inter-Medium",
    "Inter-SemiBold",
    "Inter-Bold",
    "InterDisplay-SemiBold",
    "InterDisplay-Bold",
)
TAGLINE = "Structured decisions in one forward pass"
BANNER_INCHES = (12.0, 4.0)
BANNER_UNITS = (120.0, 40.0)
GRADIENT = ("#0A5BD8", "#5CC8FF")
VMARK_ALPHA = 0.2

BLUE = "#30A0FC"
BLUE_DEEP = "#0B6BCB"
YELLOW = "#FCB414"
INK = "#0B1220"
MUTED = "#5B6577"
GREY = "#A3ACB9"
GREY_LIGHT = "#E6E9EF"
GRID = "#F1F3F6"
V1 = "#B9C7D8"
PEERS = ("#D5DAE2", "#C3C9D3", "#AEB6C2", "#9AA3B0", "#868F9D")
DPI = 200
TYPE_PANELS = (
    ("choice", "Choice"),
    ("noul", "Yes / No"),
    ("score_accuracy", "Score"),
    ("transfer", "Human-labelled\ntransfer"),
)


def _rgb(colour: str):
    import numpy as np

    return np.array([int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)])


def short_name(name: str) -> str:
    """``Decision-2.0-Nox-4B`` -> ``Nox 4B`` for the banner."""
    match = layout.MODEL_NAME.match(name)
    if not match:
        raise ValueError(f"not a Decision 2.0 model name: {name}")
    return f"{match['codename']} {match['size']}B"


class Renderer:
    def __init__(self, logo: Path, fonts: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from PIL import Image

        for name in FONTS:
            font_manager.fontManager.addfont(str(fonts / f"{name}.ttf"))
        plt.rcParams.update(
            {
                "font.family": "Inter",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "text.color": INK,
            }
        )
        self.plt = plt
        self.logo = Image.open(logo).convert("RGBA")

    def add_logo(self, fig, width_frac: float = 0.11, pad: float = 0.018) -> None:
        w, h = self.logo.size
        fw, fh = fig.get_size_inches() * fig.dpi
        lh = width_frac * fw * h / w
        ax = fig.add_axes([1 - width_frac - pad, pad, width_frac, lh / fh])
        ax.imshow(self.logo)
        ax.axis("off")

    @staticmethod
    def footnote(fig, x: float, text: str) -> None:
        """One line per sentence, kept left of the logo."""
        lines = [
            s if s.endswith(".") else s + "." for s in text.rstrip(".").split(". ")
        ]
        fig.text(
            x,
            0.018,
            "\n".join(lines),
            fontsize=7.8,
            color=MUTED,
            va="bottom",
            linespacing=1.5,
        )

    @staticmethod
    def header(fig, title: str, subtitle: str, x: float = 0.06) -> None:
        fig.text(
            x,
            0.915,
            title,
            fontsize=19,
            fontfamily="Inter Display",
            fontweight="semibold",
            color=INK,
        )
        fig.text(x, 0.862, subtitle, fontsize=11, color=MUTED)

    def save(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, metadata={"Software": None})
        self.plt.close(fig)

    def banner(
        self, name: str, path: Path
    ) -> dict[str, tuple[float, float, float, float]]:
        """The codename as the focal point: a deep-blue-to-cyan gradient in Inter Display Bold, the size in ink
        on its baseline, the DECISION 2.0 eyebrow above, the tagline with a yellow full stop below, the small
        logo top-left and the translucent V-mark of the logo bleeding off the right edge, on white.

        Returns the pixel box ``(x0, y0, x1, y1)`` of every element (``vmark`` clipped to the canvas).
        """
        import numpy as np
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Circle, PathPatch
        from matplotlib.textpath import TextPath
        from matplotlib.transforms import Affine2D

        plt = self.plt
        match = layout.MODEL_NAME.match(name)
        if not match:
            raise ValueError(f"not a Decision 2.0 model name: {name}")
        codename, size = match["codename"], f"{match['size']}B"
        fig = plt.figure(figsize=BANNER_INCHES, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, BANNER_UNITS[0])
        ax.set_ylim(0, BANNER_UNITS[1])
        ax.axis("off")
        boxes: dict[str, tuple[float, float, float, float]] = {}

        vmark = self.vmark()
        vh, vw = vmark.shape[:2]
        aspect = BANNER_INCHES[0] / BANNER_INCHES[1]
        height = 1.25
        width = height * (vw / vh) / aspect
        left, bottom = 0.985 - 0.92 * width, -0.14
        vax = fig.add_axes([left, bottom, width, height])
        vax.imshow(vmark, interpolation="lanczos")
        vax.axis("off")

        w, h = self.logo.size
        logo_w = 0.12
        lax = fig.add_axes([0.052, 0.82, logo_w, logo_w * (h / w) * aspect])
        lax.imshow(self.logo)
        lax.axis("off")

        x0, baseline = 5.6, 10.6
        eyebrow = ax.text(
            x0 + 0.5,
            27.6,
            "DECISION 2.0",
            fontsize=12.5,
            fontweight="bold",
            color=GRADIENT[0],
        )
        display = FontProperties(family="Inter Display", weight="bold")
        code = TextPath((0, 0), codename, size=15.5, prop=display)
        cb = code.get_extents()
        place = Affine2D().translate(x0 - cb.x0, baseline)
        patch = PathPatch(
            code, transform=place + ax.transData, facecolor="none", edgecolor="none"
        )
        ax.add_patch(patch)
        t = np.linspace(0, 1, 512)[None, :, None]
        ramp = _rgb(GRADIENT[0]) * (1 - t) + _rgb(GRADIENT[1]) * t
        image = ax.imshow(
            np.repeat(ramp, 8, axis=0),
            extent=(x0, x0 + cb.width, baseline + cb.y0, baseline + cb.y1),
            aspect="auto",
            interpolation="bicubic",
            zorder=3,
        )
        image.set_clip_path(patch)
        code_right = x0 + cb.width
        tier = TextPath((0, 0), size, size=8.4, prop=display)
        sb = tier.get_extents()
        size_left = code_right + 3.2
        ax.add_patch(
            PathPatch(
                Affine2D().translate(size_left - sb.x0, baseline).transform_path(tier),
                facecolor=INK,
                edgecolor="none",
                zorder=3,
            )
        )
        tagline = ax.text(x0 + 0.5, 3.4, TAGLINE, fontsize=13.5, color=MUTED)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def pixels(x_0, y_0, x_1, y_1):
            (a, b), (c, d) = ax.transData.transform([(x_0, y_0), (x_1, y_1)])
            return (float(a), float(b), float(c), float(d))

        def window(artist):
            e = artist.get_window_extent(renderer)
            return (float(e.x0), float(e.y0), float(e.x1), float(e.y1))

        tag_box = window(tagline)
        tag_right = ax.transData.inverted().transform((tag_box[2], tag_box[1]))[0]
        dot = (tag_right + 0.75, 3.4 + 0.45)
        ax.add_patch(Circle(dot, 0.45, facecolor=YELLOW, edgecolor="none", zorder=3))
        fw, fh = fig.get_size_inches() * fig.dpi
        boxes["vmark"] = (
            max(0.0, left * fw),
            max(0.0, bottom * fh),
            min(fw, (left + width) * fw),
            min(fh, (bottom + height) * fh),
        )
        boxes["logo"] = window(lax)
        boxes["eyebrow"] = window(eyebrow)
        boxes["codename"] = pixels(x0, baseline + cb.y0, code_right, baseline + cb.y1)
        boxes["size"] = pixels(
            size_left, baseline + sb.y0, size_left + sb.width, baseline + sb.y1
        )
        boxes["tagline"] = tag_box
        boxes["dot"] = pixels(
            dot[0] - 0.45, dot[1] - 0.45, dot[0] + 0.45, dot[1] + 0.45
        )
        boxes["canvas"] = (0.0, 0.0, float(fw), float(fh))
        self.save(fig, path)
        return boxes

    def vmark(self):
        """The yellow and blue shapes of the logo only, cropped to them, at ``VMARK_ALPHA``."""
        import numpy as np

        rgba = np.asarray(self.logo).astype(float) / 255
        chroma = rgba[..., :3].max(axis=2) - rgba[..., :3].min(axis=2)
        coloured = (rgba[..., 3] > 0.15) & (chroma > 0.25)
        coloured[:, int(rgba.shape[1] * 0.4) :] = False
        ys, xs = np.where(coloured)
        crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        return np.dstack([rgba[crop][..., :3], coloured[crop] * VMARK_ALPHA])

    def jevarena(self, rows: list[dict[str, Any]], path: Path) -> None:
        """Overall JevArena of this model, its counterpart and the same-size peers."""
        plt = self.plt
        rows = sorted(rows, key=lambda r: -r["score"])
        fig, ax = plt.subplots(figsize=(11, 1.9 + 0.62 * len(rows)), dpi=DPI)
        top = 1 - 1.15 / fig.get_size_inches()[1]
        fig.subplots_adjust(
            left=0.24,
            right=0.93,
            top=top,
            bottom=0.42 / fig.get_size_inches()[1] + 0.06,
        )
        self.header(fig, "JevArena", "Overall decision quality versus same-size models")
        fig.texts[0].set_y(1 - 0.38 / fig.get_size_inches()[1])
        fig.texts[1].set_y(1 - 0.68 / fig.get_size_inches()[1])
        ys = list(range(len(rows)))[::-1]
        limit = max(r["score"] for r in rows) * 1.14
        for y, row in zip(ys, rows):
            ours = row["role"] == "candidate"
            color = BLUE if ours else V1 if row["role"] == "own-1.0" else GREY_LIGHT
            ax.barh(y, row["score"], height=0.56, color=color, zorder=2)
            ax.text(
                row["score"] + limit * 0.01,
                y,
                f"{row['score']:.1f}",
                va="center",
                fontsize=12,
                fontweight="semibold" if ours else "regular",
                color=BLUE_DEEP if ours else MUTED,
            )
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=12)
        for tick, row in zip(ax.get_yticklabels(), rows):
            if row["role"] == "candidate":
                tick.set_fontweight("semibold")
        ax.set_xlim(0, limit)
        ax.xaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        self.add_logo(fig)
        self.save(fig, path)

    def types(self, rows: list[dict[str, Any]], path: Path) -> None:
        """Accuracy per decision type and human-labelled transfer, same models."""
        plt = self.plt
        order = sorted(
            rows,
            key=lambda r: (
                r["role"] != "candidate",
                r["role"] not in ("own-1.0", "reference"),
                -r["score"],
            ),
        )
        colors, peer = [], 0
        for row in order:
            if row["role"] == "candidate":
                colors.append(BLUE)
            elif row["role"] == "own-1.0":
                colors.append(V1)
            else:
                colors.append(PEERS[peer % len(PEERS)])
                peer += 1
        fig, axes = plt.subplots(1, len(TYPE_PANELS), figsize=(12, 4.6), dpi=DPI)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.72, bottom=0.2, wspace=0.3)
        self.header(
            fig,
            "JevArena by decision type",
            "Accuracy (%) on Choice, Yes / No and Score decisions · macro-F1 (×100) on human-labelled transfer tasks",
        )
        for ax, (kind, title) in zip(axes, TYPE_PANELS):
            values = [r[kind] for r in order]
            ax.bar(range(len(order)), values, color=colors, width=0.74, zorder=2)
            for i, value in enumerate(values):
                ax.text(
                    i,
                    value + 1.4,
                    f"{value:.0f}",
                    ha="center",
                    fontsize=9.5 if len(order) < 6 else 8.5,
                    color=BLUE_DEEP if i == 0 else MUTED,
                    fontweight="semibold" if i == 0 else "regular",
                )
            ax.set_title(title, fontsize=11.5, fontweight="medium", color=INK, pad=10)
            ax.set_ylim(0, 105)
            ax.set_xticks([])
            ax.yaxis.set_visible(False)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GREY_LIGHT)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
        fig.legend(
            handles,
            [r["label"] for r in order],
            loc="lower left",
            ncol=len(order),
            frameon=False,
            fontsize=9.5 if len(order) < 5 else 8.8,
            bbox_to_anchor=(0.055, 0.0),
            columnspacing=1.4,
            handlelength=1.6,
        )
        self.add_logo(fig, width_frac=0.1)
        self.save(fig, path)

    def pareto(self, view: dict[str, Any], path: Path) -> None:
        """Balanced skill against parameters: the family, Decision 1.0 and public entrants."""
        from matplotlib.ticker import FixedLocator, NullLocator

        plt = self.plt
        own, compare = view["own"], view["compare"]
        family = view["family"]
        line_1_0 = sorted(
            (p for p in view["decision1"] if p.get("tier")),
            key=lambda p: p["parameters"],
        )
        other_1_0 = [p for p in view["decision1"] if not p.get("tier")]
        entrants = view["entrants"]

        def xy(points):
            return [p["parameters"] / 1e9 for p in points], [
                p["balanced_skill"] for p in points
            ]

        fig, ax = plt.subplots(figsize=(11, 6.6), dpi=DPI)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.18)
        self.header(
            fig,
            "Jev Decision Index",
            f"Balanced skill against model size · Decision 2.0 vs. Decision 1.0 and {len(entrants)} public entrants",
            x=0.08,
        )
        ax.scatter(
            *xy(entrants),
            s=24,
            color=GREY_LIGHT,
            edgecolor=GREY,
            linewidth=0.6,
            zorder=2,
        )
        points = sorted(
            (p["parameters"] / 1e9, p["balanced_skill"])
            for p in [*entrants, *family, *view["decision1"]]
        )
        best, frontier = -math.inf, []
        for x, y in points:
            if y > best:
                frontier.append((x, y))
                best = y
        frontier.append((60, best))
        ax.step(
            *zip(*frontier),
            where="post",
            color=GREY,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
        ax.plot(*xy(line_1_0), color=V1, linewidth=1.8, zorder=3)
        ax.scatter(
            *xy(line_1_0 + other_1_0),
            s=46,
            facecolor="white",
            edgecolor=V1,
            linewidth=1.8,
            zorder=4,
        )
        ax.plot(*xy(family), color=BLUE, linewidth=2.4, zorder=5)
        ax.scatter(
            *xy(family), s=64, color=BLUE, edgecolor="white", linewidth=1.4, zorder=6
        )
        x, y = own["parameters"] / 1e9, own["balanced_skill"]
        ax.scatter(
            [x], [y], s=280, facecolor="none", edgecolor=YELLOW, linewidth=2.4, zorder=7
        )
        backing = {"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9}
        ax.annotate(
            f"{own['name']}  {y:.1f}",
            (x, y),
            xytext=(-16, 10),
            textcoords="offset points",
            ha="right",
            fontsize=10.5,
            fontweight="semibold",
            color=BLUE_DEEP,
            zorder=8,
            bbox=backing,
        )
        offset, align = (
            ((-12, 12), "right")
            if view["compare_kind"] == "family"
            else ((10, -14), "left")
        )
        ax.annotate(
            f"{compare['name']}  {compare['balanced_skill']:.1f}",
            (compare["parameters"] / 1e9, compare["balanced_skill"]),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            fontsize=9,
            color=MUTED,
            zorder=8,
            bbox=backing,
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.1B", "0.3B", "1B", "3B", "10B", "30B"])
        ax.set_xlim(0.06, 45)
        top = max(y for _, y in points)
        ax.set_ylim(0, 10 * math.ceil((top + 6) / 10))
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY_LIGHT)
        ax.tick_params(colors=MUTED)
        ax.set_xlabel("Parameters (log scale)", fontsize=10.5, color=MUTED, labelpad=8)
        ax.set_ylabel("Balanced skill", fontsize=10.5, color=MUTED, labelpad=8)
        handles = [
            plt.Line2D(
                [],
                [],
                color=BLUE,
                marker="o",
                markersize=7,
                linewidth=2.4,
                label="Decision 2.0",
            ),
            plt.Line2D(
                [],
                [],
                color=V1,
                marker="o",
                markerfacecolor="white",
                markersize=7,
                linewidth=1.8,
                label="Decision 1.0",
            ),
            plt.Line2D(
                [],
                [],
                color=GREY,
                marker="o",
                markerfacecolor=GREY_LIGHT,
                linestyle="none",
                markersize=6,
                label="Public entrants",
            ),
            plt.Line2D(
                [],
                [],
                color=GREY,
                linestyle=(0, (3, 3)),
                linewidth=1.0,
                label="Pareto frontier",
            ),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5)
        self.footnote(fig, 0.08, view["footnote"])
        self.add_logo(fig)
        self.save(fig, path)

    def areas(self, view: dict[str, Any], path: Path) -> None:
        """Balanced skill per area against the comparison model, with the overall delta."""
        plt = self.plt
        own, compare = view["own"], view["compare"]
        fig, ax = plt.subplots(figsize=(11, 5.0), dpi=DPI)
        fig.subplots_adjust(left=0.13, right=0.93, top=0.72, bottom=0.16)
        delta = view["delta"]
        self.header(
            fig,
            "Jev Decision Index by area",
            f"Balanced skill · {own['name']} vs. {compare['name']}  ·  overall "
            f"{own['balanced_skill']:.1f} vs. {compare['balanced_skill']:.1f} ({delta:+.1f})".replace(
                "(-", "(−"
            ),
        )
        names = [label for _, label in card_index.AREAS]
        ys = list(range(len(names)))[::-1]
        height = 0.34
        limit = max(max(own["areas"].values()), max(compare["areas"].values())) * 1.2
        for y, (area, _) in zip(ys, card_index.AREAS):
            mine, theirs = own["areas"][area], compare["areas"][area]
            ax.barh(y + height / 2, mine, height=height, color=BLUE, zorder=2)
            ax.barh(y - height / 2, theirs, height=height, color=V1, zorder=2)
            ax.text(
                mine + limit * 0.01,
                y + height / 2,
                f"{mine:.1f}",
                va="center",
                fontsize=10,
                color=BLUE_DEEP,
                fontweight="semibold",
            )
            ax.text(
                theirs + limit * 0.01,
                y - height / 2,
                f"{theirs:.1f}",
                va="center",
                fontsize=10,
                color=MUTED,
            )
            diff = mine - theirs
            ax.text(
                limit,
                y,
                f"{diff:+.1f}".replace("-", "−"),
                va="center",
                ha="right",
                fontsize=11.5,
                fontweight="semibold",
                color=BLUE_DEEP if diff > 0 else MUTED,
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=12)
        ax.set_xlim(0, limit * 1.01)
        ax.xaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=BLUE),
            plt.Rectangle((0, 0), 1, 1, color=V1),
        ]
        fig.legend(
            handles,
            [own["name"], compare["name"]],
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=9.5,
            bbox_to_anchor=(0.93, 0.80),
        )
        self.footnote(fig, 0.06, view["footnote"])
        self.add_logo(fig)
        self.save(fig, path)


def chart_rows(shown: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in shown:
        typed = {}
        for kind, key in (
            ("choice", "choice"),
            ("noul", "noul"),
            ("score", "score_accuracy"),
        ):
            correct, n = card._typed(entry["data"], kind)
            typed[key] = 100 * correct / n
        rows.append(
            {
                "key": entry["key"],
                "role": entry["role"],
                "label": card._label(entry),
                "score": card._score(entry),
                "transfer": 100 * card._transfer(entry),
                **typed,
            }
        )
    return rows


def render(
    spec: dict[str, Any],
    index_path: Path,
    logo: Path,
    fonts: Path,
    output: Path,
    model_sha256: str,
    source_root: Path,
) -> dict[str, Any]:
    match = layout.MODEL_NAME.match(spec["model_name"])
    if not match:
        raise ValueError(f"not a Decision 2.0 model name: {spec['model_name']}")
    tier = next(t for t, c in layout.CODENAMES.items() if c == match["codename"])
    roster = Path(spec["card"]["roster"])
    comparison = card.comparison_of(spec)
    selection = card.select_reports(
        spec["card"]["reports"],
        roster if roster.is_absolute() else source_root / roster,
        comparison,
    )
    shown = selection["shown"]
    candidate = next(e for e in shown if e["role"] == "candidate")
    if card._label(candidate) != spec["model_name"]:
        raise ValueError("the candidate report label must be the model name")
    view = card_index.view(card_index.load(index_path), tier, model_sha256)
    renderer = Renderer(logo, fonts)
    rows = chart_rows(shown)
    renderer.banner(spec["model_name"], output / BANNER)
    jevarena, types, pareto, areas = layout.CHART_FILES
    renderer.jevarena(rows, output / jevarena)
    renderer.types(rows, output / types)
    renderer.pareto(view, output / pareto)
    renderer.areas(view, output / areas)
    import matplotlib
    import PIL

    receipt = {
        "schema": SCHEMA,
        "model_name": spec["model_name"],
        "tier": tier,
        "model_sha256": model_sha256,
        "inputs": {
            "reports": {e["key"]: e["sha256"] for e in shown},
            "index_sha256": view["sha256"],
            "logo_sha256": layout.sha_file(logo),
            "fonts_sha256": {
                name: layout.sha_file(fonts / f"{name}.ttf") for name in FONTS
            },
        },
        "software": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "files": {
            name: layout.sha_file(output / name)
            for name in (BANNER, *layout.CHART_FILES)
        },
    }
    (output / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--logo", type=Path, required=True)
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument(
        "--model-sha256", required=True, help="identity.model_sha256 of the package"
    )
    ap.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    receipt = render(
        spec,
        args.index,
        args.logo,
        args.fonts,
        args.output,
        args.model_sha256,
        args.source_root,
    )
    print(json.dumps({"files": receipt["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
