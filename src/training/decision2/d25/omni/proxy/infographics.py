"""InfographicVQA proxy: rendered infographics with short-answer questions turned into multiple choice.

Each row is one tall HTML infographic (stat tiles with icons, bar charts, donut charts, ranked lists,
timelines, two-group comparisons, icon arrays) on a random topic with random data, rendered by
headless Chromium. Questions cover reading, comparison, ranking, counting and simple arithmetic; the
short answer becomes 4-way multiple choice with distractors drawn from the same infographic.
"""

from __future__ import annotations

import html as htmlmod
import math
import random
from typing import Any

BENCHMARK = "InfographicVQA"
NAME = "infovqa-proxy"
VERSION = "1"

TOPICS = [
    ("Internet use", "hours online per day", "use the internet daily"),
    ("Coffee culture", "cups per week", "drink coffee every morning"),
    ("Remote work", "remote days per week", "work from home at least once a week"),
    (
        "Electric vehicles",
        "EV share of new cars (%)",
        "consider an EV for their next car",
    ),
    ("Recycling habits", "recycling rate (%)", "sort household waste"),
    ("Sleep health", "hours of sleep", "sleep less than 7 hours"),
    ("Online shopping", "orders per month", "shop online weekly"),
    ("Streaming", "subscriptions per household", "watch streaming video daily"),
    ("Mobile payments", "share of payments (%)", "pay by phone"),
    ("Fitness trends", "workouts per week", "exercise three times a week"),
    ("Food waste", "kg wasted per person", "throw away food weekly"),
    ("Reading habits", "books per year", "read at least one book a month"),
    ("Commuting", "minutes per trip", "commute by public transport"),
    ("Video gaming", "hours per week", "play games on a phone"),
    ("Renewable energy", "share of electricity (%)", "support new wind farms"),
    ("Pet ownership", "pets per 100 homes", "own a dog"),
    ("Travel plans", "trips per year", "plan to travel abroad"),
    ("News sources", "share of readers (%)", "get news from social media"),
]
CATEGORY_SETS = [
    ("age group", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]),
    (
        "country",
        [
            "Germany",
            "Brazil",
            "Japan",
            "Canada",
            "India",
            "Kenya",
            "Mexico",
            "Sweden",
            "Australia",
            "Spain",
            "Korea",
            "Nigeria",
        ],
    ),
    ("region", ["North", "South", "East", "West", "Central"]),
    ("device", ["Smartphone", "Laptop", "Tablet", "Desktop", "Smart TV", "Console"]),
    (
        "city",
        [
            "Lagos",
            "Lima",
            "Oslo",
            "Seoul",
            "Austin",
            "Lyon",
            "Pune",
            "Perth",
            "Cairo",
            "Quito",
        ],
    ),
    ("platform", ["Video", "Podcasts", "Social", "Forums", "Newsletters", "Blogs"]),
    ("income group", ["Low", "Lower-middle", "Upper-middle", "High"]),
]
PALETTES = [
    ["#0b3c5d", "#328cc1", "#d9b310", "#1d2731", "#e05a47", "#6aa84f"],
    ["#2e294e", "#541388", "#f1e9da", "#ffd400", "#d90368", "#00a6a6"],
    ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#2a9d8f"],
    ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ab17d"],
    ["#22223b", "#4a4e69", "#9a8c98", "#c9ada7", "#f2e9e4", "#ef476f"],
    ["#05668d", "#028090", "#00a896", "#02c39a", "#f0f3bd", "#ff8c42"],
]
FONTS = ["Liberation Sans", "DejaVu Sans", "Noto Sans", "Arial", "sans-serif"]
ICONS = {
    "person": '<circle cx="12" cy="6" r="4"/><path d="M4 22c0-5 3.6-8 8-8s8 3 8 8z"/>',
    "house": '<path d="M2 11 12 3l10 8v11H14v-7h-4v7H2z"/>',
    "phone": '<rect x="6" y="1" width="12" height="22" rx="2"/>',
    "globe": '<circle cx="12" cy="12" r="10"/>',
    "star": '<path d="M12 2l3 7h7l-5.5 4.5 2 7.5-6.5-4.5-6.5 4.5 2-7.5L2 9h7z"/>',
    "bolt": '<path d="M13 1 3 14h7l-1 9 10-13h-7z"/>',
    "heart": '<path d="M12 21 3 12a5 5 0 0 1 9-6 5 5 0 0 1 9 6z"/>',
    "cart": '<path d="M2 3h3l3 12h11l3-8H7"/><circle cx="9" cy="20" r="2"/><circle cx="18" cy="20" r="2"/>',
}


def icon(name: str, color: str, size: int = 34) -> str:
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}">{ICONS[name]}</svg>'


def pct(v: float) -> str:
    return f"{v:.0f}%"


class Builder:
    """Accumulates sections and the facts that questions can be asked about."""

    def __init__(self, r: random.Random):
        self.r = r
        self.parts: list[str] = []
        self.questions: list[tuple[str, str, str, list[str]]] = []
        self.numbers: set[str] = set()
        self.palette = r.choice(PALETTES)
        self.topic, self.measure, self.statement = r.choice(TOPICS)

    def color(self, i: int) -> str:
        return self.palette[i % len(self.palette)]

    def options_from(self, correct: str, pool: list[str]) -> list[str] | None:
        pool = sorted(set(pool) - {correct})
        if len(pool) < 3:
            return None
        return self.r.sample(pool, 3)

    def stat_tiles(self) -> None:
        r = self.r
        n = r.randint(3, 4)
        groups = r.sample(
            [
                "adults",
                "teens",
                "parents",
                "students",
                "seniors",
                "workers",
                "women",
                "men",
                "city dwellers",
                "rural residents",
            ],
            n,
        )
        values = r.sample(range(8, 93), n)
        tiles = []
        for i, (g, v) in enumerate(zip(groups, values)):
            name = r.choice(list(ICONS))
            tiles.append(
                f'<div class=tile style="border-top:6px solid {self.color(i)}">{icon(name, self.color(i))}'
                f'<div class=big style="color:{self.color(i)}">{v}%</div><div>of {g} {htmlmod.escape(self.statement)}</div></div>'
            )
            self.numbers.add(pct(v))
        self.parts.append(f'<div class=tiles>{"".join(tiles)}</div>')
        k = r.randrange(n)
        distract = self.options_from(
            pct(values[k]),
            [pct(v) for v in values] + [pct(v + d) for v in values for d in (-7, 6)],
        )
        if distract:
            self.questions.append(
                (
                    "tiles:read",
                    f"What percentage of {groups[k]} {self.statement}?",
                    pct(values[k]),
                    distract,
                )
            )
        hi = max(range(n), key=lambda i: values[i])
        if n >= 4:
            self.questions.append(
                (
                    "tiles:max",
                    f"Which group has the highest share of people who {self.statement}?",
                    groups[hi],
                    [groups[i] for i in range(n) if i != hi][:3],
                )
            )

    def bar_chart(self) -> None:
        r = self.r
        noun, cats = r.choice(CATEGORY_SETS)
        n = r.randint(4, min(7, len(cats)))
        labels = (
            r.sample(cats, n) if noun not in ("age group", "income group") else cats[:n]
        )
        unit_pct = "%" in self.measure
        values = [round(r.uniform(5, 95), 0 if unit_pct else 1) for _ in labels]
        while len(set(values)) < n:
            values = [round(r.uniform(5, 95), 0 if unit_pct else 1) for _ in labels]
        top = max(values)
        fmt = (
            (lambda v: f"{v:.0f}%") if unit_pct else (lambda v: f"{v:.1f}")
        )  # noqa: E731
        rows = "".join(
            f'<div class=brow><div class=blab>{htmlmod.escape(l)}</div><div class=bar style="width:{v / top * 70:.1f}%;background:{self.color(i if r.random() < 0.3 else 1)}"></div><div class=bval>{fmt(v)}</div></div>'
            for i, (l, v) in enumerate(zip(labels, values))
        )
        title = f"{self.measure[0].upper()}{self.measure[1:]} by {noun}"
        self.parts.append(
            f"<div class=sec><h2>{htmlmod.escape(title)}</h2>{rows}</div>"
        )
        order = sorted(range(n), key=lambda i: -values[i])
        if values[order[0]] - values[order[1]] >= 2:
            self.questions.append(
                (
                    "bar:max",
                    f"Which {noun} has the highest {self.measure.split(' (')[0]}?",
                    labels[order[0]],
                    [labels[i] for i in order[1:4]],
                )
            )
        if n >= 5:
            self.questions.append(
                (
                    "bar:second",
                    f"Which {noun} ranks second in {self.measure.split(' (')[0]}?",
                    labels[order[1]],
                    [labels[i] for i in (order[0], *order[2:4])],
                )
            )
        a, b = r.sample(range(n), 2)
        diff = abs(values[a] - values[b])
        correct = fmt(diff)
        pool = [
            fmt(abs(values[i] - values[j])) for i in range(n) for j in range(n) if i < j
        ] + [fmt(values[a] + values[b])]
        distract = self.options_from(correct, pool)
        if distract and diff > 0:
            self.questions.append(
                (
                    "bar:difference",
                    f"What is the difference in {self.measure.split(' (')[0]} between {labels[a]} and {labels[b]}?",
                    correct,
                    distract,
                )
            )
        k = r.randrange(n)
        distract = self.options_from(fmt(values[k]), [fmt(v) for v in values])
        if distract:
            self.questions.append(
                (
                    "bar:read",
                    f"What value is shown for {labels[k]} in the chart of {self.measure.split(' (')[0]} by {noun}?",
                    fmt(values[k]),
                    distract,
                )
            )
        for v in values:
            self.numbers.add(fmt(v))

    def donut(self) -> None:
        r = self.r
        options = r.choice(
            [
                ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
                ["Very satisfied", "Satisfied", "Neutral", "Unsatisfied"],
                ["Price", "Quality", "Brand", "Reviews", "Convenience"],
                ["Car", "Bus", "Bike", "Walk", "Train"],
                ["Yes", "No", "Not sure"],
                ["At home", "At work", "On the go", "Elsewhere"],
            ]
        )
        n = len(options)
        cuts = sorted(r.sample(range(3, 97), n - 1))
        shares = [b - a for a, b in zip([0, *cuts], [*cuts, 100])]
        if len(set(shares)) < n or min(shares) < 3:
            return
        radius, cx = 70, 90
        start, arcs = -math.pi / 2, []
        for i, s in enumerate(shares):
            end = start + 2 * math.pi * s / 100
            large = 1 if end - start > math.pi else 0
            x0, y0 = cx + radius * math.cos(start), cx + radius * math.sin(start)
            x1, y1 = cx + radius * math.cos(end), cx + radius * math.sin(end)
            arcs.append(
                f'<path d="M{x0:.1f} {y0:.1f} A{radius} {radius} 0 {large} 1 {x1:.1f} {y1:.1f}" stroke="{self.color(i)}" stroke-width="34" fill="none"/>'
            )
            start = end
        legend = "".join(
            f'<div><span class=sw style="background:{self.color(i)}"></span>{htmlmod.escape(o)} <b>{s}%</b></div>'
            for i, (o, s) in enumerate(zip(options, shares))
        )
        question_text = r.choice(
            [
                "How often do you",
                "What matters most when you",
                "How do you usually",
                "Do you",
            ]
        )
        title = f"{question_text} ... ? (survey)"
        self.parts.append(
            f'<div class="sec flex"><svg width="180" height="180">{"".join(arcs)}</svg><div class=legend><h3>{htmlmod.escape(title)}</h3>{legend}</div></div>'
        )
        order = sorted(range(n), key=lambda i: -shares[i])
        if n >= 4:
            self.questions.append(
                (
                    "donut:max",
                    "Which answer was chosen most often in the survey donut chart?",
                    options[order[0]],
                    [options[i] for i in order[1:4]],
                )
            )
            k = r.randrange(n)
            distract = self.options_from(
                f"{shares[k]}%",
                [f"{s}%" for s in shares] + [f"{s + 5}%" for s in shares],
            )
            if distract:
                self.questions.append(
                    (
                        "donut:read",
                        f'What share of respondents answered "{options[k]}"?',
                        f"{shares[k]}%",
                        distract,
                    )
                )
        for s in shares:
            self.numbers.add(f"{s}%")

    def ranked_list(self) -> None:
        r = self.r
        noun, cats = r.choice([c for c in CATEGORY_SETS if len(c[1]) >= 6])
        items = r.sample(cats, 5)
        values = sorted(r.sample(range(12, 990), 5), reverse=True)
        unit = r.choice(["k users", "M visits", "tonnes", "stores", "km"])
        rows = "".join(
            f'<div class=rank><span class=num style="background:{self.color(i)}">{i + 1}</span>{htmlmod.escape(it)}<span class=rv>{v} {unit}</span></div>'
            for i, (it, v) in enumerate(zip(items, values))
        )
        title = f"Top 5 {noun[:-1] + 'ies' if noun.endswith('y') else noun + 's'}"
        self.parts.append(
            f"<div class=sec><h2>{htmlmod.escape(title)}</h2>{rows}</div>"
        )
        k = r.choice([1, 2, 3])
        ordinal = {1: "second", 2: "third", 3: "fourth"}[k]
        self.questions.append(
            (
                "list:rank",
                f"Which {noun} is ranked {ordinal} in the top-5 list?",
                items[k],
                [items[i] for i in range(5) if i != k][:3],
            )
        )
        a, b = sorted(r.sample(range(5), 2))
        total = values[a] + values[b]
        pool = [
            f"{values[i] + values[j]} {unit}"
            for i in range(5)
            for j in range(5)
            if i < j
        ]
        distract = self.options_from(f"{total} {unit}", pool)
        if distract:
            self.questions.append(
                (
                    "list:sum",
                    f"What is the combined value of {items[a]} and {items[b]} in the top-5 list?",
                    f"{total} {unit}",
                    distract,
                )
            )

    def timeline(self) -> None:
        r = self.r
        years = sorted(r.sample(range(1995, 2026), 5))
        events = r.sample(
            [
                "first survey",
                "law passed",
                "record high",
                "new program launched",
                "market peak",
                "price drop",
                "national campaign",
                "industry standard set",
                "public pilot",
                "big merger",
            ],
            5,
        )
        rows = "".join(
            f'<div class=ev><div class=yr style="color:{self.color(i)}">{y}</div><div>{htmlmod.escape(e.capitalize())}</div></div>'
            for i, (y, e) in enumerate(zip(years, events))
        )
        self.parts.append(
            f"<div class=sec><h2>Milestones</h2><div class=tl>{rows}</div></div>"
        )
        k = r.randrange(5)
        self.questions.append(
            (
                "timeline:year",
                f'In which year was the milestone "{events[k]}" reached?',
                str(years[k]),
                [str(y) for i, y in enumerate(years) if i != k][:3],
            )
        )
        span = years[-1] - years[0]
        pool = [str(years[j] - years[i]) for i in range(5) for j in range(5) if i < j]
        distract = self.options_from(str(span), pool)
        if distract:
            self.questions.append(
                (
                    "timeline:span",
                    "How many years passed between the first and the last milestone?",
                    str(span),
                    distract,
                )
            )

    def comparison(self) -> None:
        r = self.r
        a, b = r.choice(
            [
                ("Men", "Women"),
                ("Urban", "Rural"),
                ("2019", "2025"),
                ("Under 35", "Over 35"),
            ]
        )
        items = r.sample(
            [
                "Video calls",
                "Online banking",
                "Streaming",
                "Shopping",
                "News",
                "Gaming",
                "Maps",
                "Fitness apps",
            ],
            4,
        )
        va = [r.randint(10, 90) for _ in items]
        vb_ = [r.randint(10, 90) for _ in items]
        gaps = [abs(x - y) for x, y in zip(va, vb_)]
        if sorted(gaps)[-1] - sorted(gaps)[-2] < 4:
            return
        rows = "".join(
            f'<tr><td>{htmlmod.escape(it)}</td><td style="color:{self.color(0)}">{x}%</td><td style="color:{self.color(1)}">{y}%</td></tr>'
            for it, x, y in zip(items, va, vb_)
        )
        self.parts.append(
            f"<div class=sec><h2>{a} vs {b}</h2><table class=cmp><tr><th></th><th>{a}</th><th>{b}</th></tr>{rows}</table></div>"
        )
        best = max(range(4), key=lambda i: gaps[i])
        self.questions.append(
            (
                "compare:gap",
                f"For which activity is the gap between {a} and {b} the largest?",
                items[best],
                [items[i] for i in range(4) if i != best],
            )
        )

    def icon_array(self) -> None:
        r = self.r
        k = r.randint(1, 9)
        name = r.choice(["person", "heart", "house"])
        icons = "".join(
            icon(name, self.color(0) if i < k else "#cfd3d6", 30) for i in range(10)
        )
        self.parts.append(
            f"<div class=sec><div class=arr>{icons}</div><div class=cap>{k} in 10 people {htmlmod.escape(self.statement)}</div></div>"
        )
        pool = [f"{v} in 10" for v in range(1, 10) if v != k]
        self.questions.append(
            (
                "icons:count",
                f"According to the infographic, how many in 10 people {self.statement}?",
                f"{k} in 10",
                r.sample(pool, 3),
            )
        )


def page_html(b: Builder) -> tuple[str, int]:
    r = b.r
    width = r.randint(760, 980)
    font = r.choice(FONTS)
    bg = r.choice(["#ffffff", "#f7f4ee", "#eef4f7", "#fdf6e3", "#f4f1fa"])
    title = f"{b.topic}: {r.choice(['the numbers', 'by the numbers', 'facts and figures', 'what the survey says', '2026 report'])}"
    css = f"""
    body {{ margin:0; background:{bg}; font-family:'{font}'; color:#222; }}
    .p {{ width:{width}px; }} .hd {{ background:{b.color(0)}; color:#fff; padding:28px 36px; }}
    .hd h1 {{ margin:0; font-size:{r.randint(34, 46)}px; }} .hd div {{ opacity:.85; margin-top:6px; }}
    .tiles {{ display:flex; gap:16px; padding:22px 28px; }} .tile {{ flex:1; background:#fff; padding:14px; font-size:15px; box-shadow:0 1px 3px #0002; }}
    .big {{ font-size:{r.randint(34, 44)}px; font-weight:800; margin:6px 0; }}
    .sec {{ margin:14px 28px; padding:16px 20px; background:#ffffffcc; border-radius:{r.choice([0, 6, 12])}px; }}
    .sec h2 {{ margin:0 0 12px; font-size:22px; color:{b.color(0)}; }} .flex {{ display:flex; gap:24px; align-items:center; }}
    .brow {{ display:flex; align-items:center; margin:7px 0; font-size:15px; }} .blab {{ width:130px; }}
    .bar {{ height:20px; border-radius:3px; }} .bval {{ margin-left:8px; font-weight:700; }}
    .legend div {{ margin:5px 0; font-size:15px; }} .legend h3 {{ margin:0 0 8px; font-size:17px; }}
    .sw {{ display:inline-block; width:14px; height:14px; margin-right:8px; vertical-align:middle; }}
    .rank {{ display:flex; align-items:center; gap:12px; margin:8px 0; font-size:16px; }} .rv {{ margin-left:auto; font-weight:700; }}
    .num {{ color:#fff; width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:700; }}
    .tl {{ display:flex; justify-content:space-between; gap:10px; }} .ev {{ flex:1; font-size:14px; border-left:3px solid #9993; padding-left:8px; }}
    .yr {{ font-size:22px; font-weight:800; }} .cmp {{ border-collapse:collapse; font-size:16px; }}
    .cmp td, .cmp th {{ padding:6px 18px; border-bottom:1px solid #ddd; text-align:left; }} .cmp td:nth-child(n+2) {{ font-weight:700; }}
    .arr svg {{ margin-right:6px; }} .cap {{ margin-top:8px; font-size:17px; font-weight:700; }}
    .ft {{ margin:18px 28px 26px; font-size:12px; color:#666; }}
    """
    body = "".join(b.parts)
    page = (
        f"<html><head><style>{css}</style></head><body><div class=p><div class=hd><h1>{htmlmod.escape(title)}</h1>"
        f"<div>Survey of {r.randint(8, 60) * 100:,} respondents</div></div>{body}"
        f"<div class=ft>Source: illustrative survey data, {r.randint(2024, 2026)}. Figures rounded.</div></div></body></html>"
    )
    return page, width


def worker_init() -> None:
    from d25.omni.proxy import render

    render.start()


def worker_close() -> None:
    from d25.omni.proxy import render

    render.stop()


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy import render
    from d25.omni.proxy.rows import lettered, rng

    for attempt in range(50):
        r = rng(NAME, seed, index, attempt)
        b = Builder(r)
        sections = [
            b.stat_tiles,
            b.bar_chart,
            b.donut,
            b.ranked_list,
            b.timeline,
            b.comparison,
            b.icon_array,
        ]
        for section in r.sample(sections, r.randint(4, 6)):
            section()
        valid = [q for q in b.questions if len(set([q[2], *q[3]])) == 4]
        if not valid:
            continue
        section = r.choice(sorted({q[0].split(":")[0] for q in valid}))
        subtask, question, correct, distractors = r.choice(
            [q for q in valid if q[0].startswith(section + ":")]
        )
        page, width = page_html(b)
        png, _, size = render.html(page, width, scale=r.choice([1.0, 1.25, 1.5]))
        criteria, answer = lettered(correct, distractors, r)
        return {
            "item_id": f"{index:05d}",
            "subtask": subtask,
            "payloads": [(png, "png")],
            "instructions": question,
            "criteria": criteria,
            "answer": answer,
            "provenance": [
                {
                    "source": "generated",
                    "generator": f"d25.omni.proxy.infographics v{VERSION}",
                    "seed": seed,
                    "index": index,
                    "licence": "generated (no third-party content)",
                }
            ],
            "extra": {"attempt": attempt, "size": list(size)},
        }
    raise RuntimeError(f"no valid infographic for {index}")
