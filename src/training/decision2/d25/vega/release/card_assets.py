"""Banner and Jev Decision Index charts of a Decision 2.5 card (PNG, matplotlib), in the Decision 2.0 style.

The renderer is the Decision 2.0 one (``v2.release.card_assets.Renderer``: Inter on white, blue #30A0FC and
yellow #FCB414, the vLLM-SR logo); the banner takes the generation eyebrow ("DECISION 2.5"), and the two
charts plot the 0.3 Full score against model size and the public-suite area skills against the previous
generation. Rendering needs matplotlib, Pillow, the Inter fonts and the logo (``card.py`` calls this).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from v2.release.card_assets import (
    BANNER_INCHES,
    BANNER_UNITS,
    BLUE,
    BLUE_DEEP,
    DPI,
    GRADIENT,
    GREY,
    GREY_LIGHT,
    GRID,
    INK,
    MUTED,
    TAGLINE,
    V1,
    YELLOW,
    Renderer as Renderer20,
    _rgb,
)

AREAS = (
    ("knowledge", "Knowledge"),
    ("language", "Language"),
    ("retrieval", "Retrieval"),
    ("tools", "Tools"),
    ("arts", "Arts"),
)


class Renderer(Renderer20):
    def banner(
        self,
        name: str,
        path: Path,
        *,
        codename: str,
        size: str,
        eyebrow: str = "DECISION 2.5",
    ):
        """The 2.0 banner composition with this generation's eyebrow; returns each element's pixel box."""
        import numpy as np
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Circle, PathPatch
        from matplotlib.textpath import TextPath
        from matplotlib.transforms import Affine2D

        plt = self.plt
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
        eyebrow_text = ax.text(
            x0 + 0.5, 27.6, eyebrow, fontsize=12.5, fontweight="bold", color=GRADIENT[0]
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
        boxes["eyebrow"] = window(eyebrow_text)
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

    def pareto25(self, view: dict[str, Any], path: Path) -> None:
        """Full score against served parameters: this model, the previous generation and the public entrants."""
        from matplotlib.ticker import FixedLocator, NullLocator

        plt = self.plt
        own, family, entrants = view["own"], view["family"], view["entrants"]

        def xy(points):
            return [p["parameters"] / 1e9 for p in points], [p["full"] for p in points]

        fig, ax = plt.subplots(figsize=(11, 6.6), dpi=DPI)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.18)
        self.header(
            fig,
            "Jev Decision Index",
            (
                f"Full score against model size · {view['generation']} vs. {view['previous_generation']} and "
                f"{len(entrants)} public entrants"
                if own
                else f"Full score against model size · {view['previous_generation']} and {len(entrants)} public "
                f"entrants ({view['generation']} pending)"
            ),
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
            (p["parameters"] / 1e9, p["full"])
            for p in [*entrants, *family, *([own] if own else [])]
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
        line = sorted(family, key=lambda p: p["parameters"])
        ax.plot(*xy(line), color=V1, linewidth=1.8, zorder=3)
        ax.scatter(
            *xy(line), s=46, facecolor="white", edgecolor=V1, linewidth=1.8, zorder=4
        )
        backing = {"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9}
        if own:
            x, y = own["parameters"] / 1e9, own["full"]
            ax.scatter(
                [x], [y], s=64, color=BLUE, edgecolor="white", linewidth=1.4, zorder=6
            )
            ax.scatter(
                [x],
                [y],
                s=280,
                facecolor="none",
                edgecolor=YELLOW,
                linewidth=2.4,
                zorder=7,
            )
            ax.annotate(
                f"{own['name']}  {y:.1f}{view['own_mark']}",
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
        previous = view["previous"]
        ax.annotate(
            f"{previous['name']}  {previous['full']:.1f}",
            (previous["parameters"] / 1e9, previous["full"]),
            xytext=(-12, -16),
            textcoords="offset points",
            ha="right",
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
        ax.set_ylabel("Full score", fontsize=10.5, color=MUTED, labelpad=8)
        handles = [
            *(
                [
                    plt.Line2D(
                        [],
                        [],
                        color=BLUE,
                        marker="o",
                        markersize=7,
                        linestyle="none",
                        label=view["generation"],
                    )
                ]
                if own
                else []
            ),
            plt.Line2D(
                [],
                [],
                color=V1,
                marker="o",
                markerfacecolor="white",
                markersize=7,
                linewidth=1.8,
                label=view["previous_generation"],
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

    def areas25(self, view: dict[str, Any], path: Path) -> None:
        """Public-suite skill per area against the previous generation, with the per-area delta."""
        plt = self.plt
        own, previous = view["own"], view["previous"]
        fig, ax = plt.subplots(figsize=(11, 5.0), dpi=DPI)
        fig.subplots_adjust(left=0.13, right=0.93, top=0.72, bottom=0.16)
        delta = own["public"] - previous["public"]
        self.header(
            fig,
            "Jev Decision Index by area",
            (
                f"Public-suite skill · {own['name']} vs. {previous['name']}  ·  public index "
                f"{own['public']:.1f} vs. {previous['public']:.1f} ({delta:+.1f})"
            ).replace("(-", "(−"),
        )
        names = [label for _, label in AREAS]
        ys = list(range(len(names)))[::-1]
        height = 0.34
        limit = max(max(own["areas"].values()), max(previous["areas"].values())) * 1.2
        for y, (area, _) in zip(ys, AREAS):
            mine, theirs = own["areas"][area], previous["areas"][area]
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
            [own["name"], previous["name"]],
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=9.5,
            bbox_to_anchor=(0.93, 0.80),
        )
        self.footnote(fig, 0.06, view["footnote"])
        self.add_logo(fig)
        self.save(fig, path)
