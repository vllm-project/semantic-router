"""a6_band_rubric: map a computed (or stated) quantity onto explicitly ranged Score levels."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from datetime import timedelta
from fractions import Fraction
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import Group, Variant

FAMILY = "a6_band_rubric"
KINDS = ("late", "cost", "response", "overbudget")
INF = math.inf
TEXT = {
    "en": {
        "late": (
            "Delivery record — {co}",
            "Which lateness band in the options applies to this delivery?",
        ),
        "cost": (
            "Invoice summary — {co}",
            "Which cost band in the options does the invoice total fall into?",
        ),
        "response": (
            "Support ticket — {co}",
            "Which response-time band in the options applies to this ticket?",
        ),
        "overbudget": (
            "Project budget review — {co}",
            "Which budget band in the options describes this project's spending?",
        ),
    },
    "zh": {
        "late": ("交付记录——{co}", "这批货的延误情况属于选项中的哪一档？"),
        "cost": ("发票摘要——{co}", "这张发票的总额落在选项中的哪一档？"),
        "response": ("客服工单——{co}", "这张工单的首次响应时间属于选项中的哪一档？"),
        "overbudget": ("项目预算复核——{co}", "这个项目的支出情况属于选项中的哪一档？"),
    },
}
INTRO = {
    "en": {
        "late": "{co} shipped an order of office furniture to {cust}.",
        "cost": "{co} sent {cust} an invoice for catering at a staff event.",
        "response": "{cust} contacted the {co} help desk about a billing problem.",
        "overbudget": "{co} ran a refurbishment project for {cust}.",
    },
    "zh": {
        "late": "{co}给{cust}发了一批办公家具。",
        "cost": "{co}就一次员工活动的餐饮服务给{cust}开了发票。",
        "response": "{cust}就一个账单问题联系了{co}的客服台。",
        "overbudget": "{co}为{cust}承接了一个装修项目。",
    },
}
ITEMS = {
    "en": (
        ("sandwich platter", "sandwich platters"),
        ("fruit bowl", "fruit bowls"),
        ("coffee urn", "coffee urns"),
        ("dessert tray", "dessert trays"),
        ("salad box", "salad boxes"),
    ),
    "zh": (
        ("三明治拼盘", "份"),
        ("水果盘", "份"),
        ("咖啡壶", "壶"),
        ("甜点托盘", "盘"),
        ("沙拉盒", "盒"),
    ),
}
CO_SUFFIX = {
    "en": ("Supplies", "Services", "Contractors"),
    "zh": ("供应", "服务", "工程"),
}


def _bands(
    rng: random.Random, kind: str, levels: int, stated: bool
) -> list[tuple[float, float]]:
    """Contiguous bands; integer kinds use inclusive [lo, hi], overbudget uses (lo, hi]."""
    if kind == "overbudget":
        cuts = [0]
        for _ in range(levels - 1):
            cuts.append(cuts[-1] + rng.randint(3 if stated else 2, 8))
        bands = [(-INF, 0.0)] + [
            (float(cuts[i]), float(cuts[i + 1])) for i in range(levels - 2)
        ]
        return bands + [(float(cuts[levels - 2]), INF)]
    step = {"late": 1, "response": 5, "cost": 10}[kind]
    low_w, high_w = {
        "late": (3 if stated else 1, 7),
        "response": (2, 9),
        "cost": (5, 26),
    }[kind]
    first = {
        "late": rng.choice((1, 2, 3)) if not stated else rng.randint(3, 5),
        "response": rng.randint(2, 6) * 5,
        "cost": rng.randint(16, 30) * 10,
    }[kind]
    bands: list[tuple[float, float]] = []
    lo, hi = 0, first - 1
    for level in range(levels):
        if level == levels - 1:
            bands.append((float(lo), INF))
            break
        bands.append((float(lo), float(hi)))
        lo = hi + 1
        hi = lo + rng.randint(low_w, high_w) * step - 1
    return bands


def _describe(kind: str, band: tuple[float, float], lang: str, cur: str) -> str:
    lo, hi = band
    a, b = int(lo) if lo != -INF else 0, int(hi) if hi != INF else 0
    if kind == "overbudget":
        if lo == -INF:
            return "At or under budget" if lang == "en" else "未超出预算（含持平）"
        if hi == INF:
            return (
                f"More than {a}% over budget" if lang == "en" else f"超出预算{a}%以上"
            )
        return (
            f"More than {a}% and at most {b}% over budget"
            if lang == "en"
            else f"超出预算{a}%以上、至多{b}%"
        )
    if lang == "en":
        unit = {"late": ("day", "days"), "response": ("minute", "minutes")}.get(kind)
        money = (lambda x: f"{cur}{x:,}") if kind == "cost" else (lambda x: f"{x}")
        tail = (
            ""
            if kind == "cost"
            else f" {unit[1]}" + (" late" if kind == "late" else "")
        )
        if hi == INF:
            return {
                "cost": f"{money(a)} or more",
                "late": f"{a} or more days late",
                "response": f"{a} minutes or more",
            }[kind]
        if a == b == 0 and kind == "late":
            return "On time (0 days late)"
        if a == b:
            single = (
                ""
                if kind == "cost"
                else f" {unit[0] if a == 1 else unit[1]}"
                + (" late" if kind == "late" else "")
            )
            return f"Exactly {money(a)}{single}"
        return f"{money(a)} to {money(b)}{tail}"
    unit = {"late": "天", "response": "分钟", "cost": "元"}[kind]
    head = "延误" if kind == "late" else ""
    if hi == INF:
        return f"{head}{a}{unit}及以上"
    if a == b == 0 and kind == "late":
        return "按时（延误0天）"
    if a == b:
        return f"恰好{head}{a}{unit}"
    return f"{head}{a}{unit}至{b}{unit}" if kind == "cost" else f"{head}{a}至{b}{unit}"


def _in_band(value: Fraction | int, band: tuple[float, float], kind: str) -> bool:
    lo, hi = band
    if kind == "overbudget":
        return (lo == -INF or value > lo) and (hi == INF or value <= hi)
    return lo <= value <= hi


def _pick(
    rng: random.Random,
    band: tuple[float, float],
    kind: str,
    avoid: set[int],
    stated: bool,
) -> int:
    lo, hi = band
    if kind == "overbudget":
        low = int(lo) + 1 if lo != -INF else -15
        high = int(hi) if hi != INF else int(lo) + 25
    else:
        low, high = int(lo), (
            int(hi)
            if hi != INF
            else int(lo) + {"late": 12, "response": 90, "cost": 600}[kind]
        )
        if kind == "cost":
            low = max(low, 120)
    values = list(range(low, high + 1))
    if stated:
        values = [v for v in values if v not in avoid and v != 0] or values
    elif rng.random() < 0.3:
        edges = [v for v in (low, high) if _in_band(v, band, kind)]
        return rng.choice(edges)
    return rng.choice(values)


class _Case:
    def __init__(self, rng: random.Random, lang: str) -> None:
        self.lang = lang
        self.kind = rng.choice(KINDS)
        self.stated = rng.random() < 0.3
        self.co = core.company(rng, lang, CO_SUFFIX[lang])
        self.cust = core.people(rng, lang, 1)[0]
        self.cur = rng.choice(("$", "€", "£"))
        self.due = core.random_date(rng)
        self.opened = rng.randint(7 * 60, 11 * 60)
        self.budget = rng.randint(20, 200) * 100
        self.items = rng.sample(ITEMS[lang], 2)
        self.fixed_qty, self.fixed_price = rng.randint(1, 2), rng.randint(12, 30)
        self.unit_price = rng.randint(12, 35)
        self.style = rng.randrange(2)

    def fact(self, value: int) -> tuple[list[str], dict[str, Any]]:
        en = self.lang == "en"
        cur = self.cur
        if self.kind == "late":
            if self.stated:
                if value == 0:
                    return (
                        ["The delivery arrived exactly on its due date."]
                        if en
                        else ["这批货正好在约定日期当天送达。"]
                    ), {"late": 0}
                unit = "day" if value == 1 else "days"
                return (
                    [f"The delivery was late by {value} {unit}."]
                    if en
                    else [f"这批货晚了{value}天才送达。"]
                ), {"late": value}
            delivered = (
                self.due + timedelta(days=value)
                if value > 0
                else self.due - timedelta(days=self.style * 2)
            )
            d1, d2 = core.fmt_date(self.due, self.lang), core.fmt_date(
                delivered, self.lang
            )
            text = (
                f"The order was due on {d1}. It was delivered on {d2}."
                if en
                else f"这批货应于{d1}交付，实际于{d2}送达。"
            )
            return [text], {
                "due": self.due.isoformat(),
                "delivered": delivered.isoformat(),
            }
        if self.kind == "response":
            if self.stated:
                return (
                    [
                        f"The first reply came {value} minutes after the ticket was opened."
                    ]
                    if en
                    else [f"首次回复在工单创建{value}分钟后发出。"]
                ), {"minutes": value}
            t1, t2 = self.opened, self.opened + value
            fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"
            text = (
                f"The ticket was opened at {fmt(t1)}, and the first reply was sent at {fmt(t2)}."
                if en
                else f"工单于{fmt(t1)}创建，首次回复于{fmt(t2)}发出。"
            )
            return [text], {"opened": t1, "replied": t2}
        if self.kind == "cost":
            if self.stated:
                return (
                    [f"The invoice total came to {cur}{value:,}."]
                    if en
                    else [f"发票总额为{value}元。"]
                ), {"total": value}
            remainder = value - self.fixed_qty * self.fixed_price
            qty2, fee = divmod(remainder, self.unit_price)
            if fee < 5:
                qty2, fee = qty2 - 1, fee + self.unit_price
            if qty2 < 1:
                return [], {}
            quantities, prices = [self.fixed_qty, qty2], [
                self.fixed_price,
                self.unit_price,
            ]
            parts = []
            for (name, second), q, p in zip(self.items, quantities, prices):
                if en:
                    parts.append(f"{q} {name if q == 1 else second} at {cur}{p} each")
                else:
                    parts.append(f"{name}{q}{second}，单价{p}元")
            if en:
                body = f"The invoice lists {parts[0]}, {parts[1]} and a delivery charge of {cur}{fee}."
            else:
                body = f"发票明细：{parts[0]}；{parts[1]}；配送费{fee}元。"
            return [body], {
                "quantities": quantities,
                "prices": prices,
                "fee": fee,
                "total": value,
            }
        if self.stated:
            if value == 0:
                return (
                    ["Spending came in exactly on budget."]
                    if en
                    else ["实际支出与预算持平。"]
                ), {"pct": 0}
            if value < 0:
                return (
                    [f"Spending ended {-value}% under budget."]
                    if en
                    else [f"实际支出比预算节省了{-value}%。"]
                ), {"pct": value}
            return (
                [f"Spending ended {value}% over budget."]
                if en
                else [f"实际支出超出预算{value}%。"]
            ), {"pct": value}
        actual = self.budget + self.budget * value // 100
        text = (
            f"The approved budget was {cur}{self.budget:,}; actual spending came to {cur}{actual:,}."
            if en
            else f"批准预算为{self.budget}元，实际支出为{actual}元。"
        )
        return [text], {"budget": self.budget, "actual": actual}


def build_group(
    rng: random.Random, lang: str, task_type: str = "score", levels: int | None = None
) -> Group:
    levels = levels or 4
    for _ in range(300):
        case = _Case(rng, lang)
        bands = _bands(rng, case.kind, levels, case.stated)
        descending = rng.random() < 0.5
        ordered = list(reversed(bands)) if descending else bands
        descriptions = [_describe(case.kind, band, lang, case.cur) for band in ordered]
        if len(set(descriptions)) != levels:
            continue
        shown = {
            int(n.replace(",", ""))
            for d in descriptions
            for n in re.findall(r"\d[\d,]*", d)
        }
        specs = []
        for grade, band in enumerate(ordered):
            value = _pick(rng, band, case.kind, shown, case.stated)
            lines, facts = case.fact(value)
            if not lines:
                break
            finite = [x if math.isfinite(x) else None for x in band]
            specs.append(
                (
                    grade,
                    lines,
                    {"kind": case.kind, "value": value, "band": finite, **facts},
                )
            )
        if len(specs) != levels or len({tuple(s[1]) for s in specs}) != levels:
            continue
        intro = INTRO[lang][case.kind].format(co=case.co, cust=case.cust)
        pad = core.filler(
            rng,
            lang,
            core.words(intro + specs[0][1][0], lang) + 10,
            [case.cust, case.co],
        )
        title, question = TEXT[lang][case.kind]
        sec = core.SECTION[lang]
        variants = []
        for grade, lines, facts in specs:
            state = core.compose(
                title.format(co=case.co),
                [
                    (sec["background"], [intro, *pad[0]]),
                    (sec["record"], lines),
                    (sec["notes"], pad[1]),
                ],
                lang,
            )
            variants.append(
                Variant(
                    state, grade, facts, f"{case.kind} value set to {facts['value']}"
                )
            )
        mode = "stated" if case.stated else "computed"
        group = Group(
            "score",
            question,
            core.score_options(descriptions),
            variants,
            f"{case.kind}_{mode}_{'desc' if descending else 'asc'}",
            f"{case.kind}_{mode}",
            {"levels": levels},
        )
        core.check_group(group)
        return group
    raise RuntimeError("band rubric group construction failed")


# ---------------------------------------------------------------- oracle 2


def _parse_band(text: str, kind: str, lang: str) -> tuple[float, float]:
    nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)]
    if kind == "overbudget":
        if "At or under" in text or "未超出" in text:
            return -INF, 0.0
        if len(nums) == 1:
            return nums[0], INF
        return nums[0], nums[1]
    if "or more" in text or "及以上" in text:
        return nums[0], INF
    if "On time" in text or "按时" in text:
        return 0.0, 0.0
    if "Exactly" in text or "恰好" in text:
        return nums[0], nums[0]
    return nums[0], nums[1]


def _value(state: str, kind: str, lang: str) -> Fraction | None:
    d = core.DATE_RX[lang]
    if kind == "late":
        if m := re.search(
            rf"was due on ({d})\. It was delivered on ({d})|应于({d})交付，实际于({d})送达",
            state,
        ):
            due, got = [core.to_date(g) for g in m.groups() if g]
            return Fraction(max(0, (got - due).days))
        if m := re.search(r"late by (\d+) days?|晚了(\d+)天", state):
            return Fraction(int(m.group(1) or m.group(2)))
        return (
            Fraction(0)
            if re.search(r"exactly on its due date|正好在约定日期当天送达", state)
            else None
        )
    if kind == "response":
        if m := re.search(
            r"opened at (\d\d):(\d\d), and the first reply was sent at (\d\d):(\d\d)|于(\d\d):(\d\d)创建，首次回复于(\d\d):(\d\d)发出",
            state,
        ):
            h1, m1, h2, m2 = [int(g) for g in m.groups() if g]
            return Fraction((h2 * 60 + m2) - (h1 * 60 + m1))
        m = re.search(r"came (\d+) minutes after|创建(\d+)分钟后", state)
        return Fraction(int(m.group(1) or m.group(2))) if m else None
    if kind == "cost":
        lines = (
            re.findall(r"(\d+) [a-z ]+? at [$€£](\d+) each", state)
            if lang == "en"
            else re.findall(r"(\d+)[份壶盘盒]，单价(\d+)元", state)
        )
        fee = re.search(r"delivery charge of [$€£](\d+)|配送费(\d+)元", state)
        if lines and fee:
            return Fraction(
                sum(int(q) * int(p) for q, p in lines)
                + int(fee.group(1) or fee.group(2))
            )
        m = re.search(r"total came to [$€£]([\d,]+)|发票总额为(\d+)元", state)
        return Fraction(int((m.group(1) or m.group(2)).replace(",", ""))) if m else None
    if m := re.search(
        r"approved budget was [$€£]([\d,]+); actual spending came to [$€£]([\d,]+)|批准预算为(\d+)元，实际支出为(\d+)元",
        state,
    ):
        budget, actual = [int(g.replace(",", "")) for g in m.groups() if g]
        return Fraction(100 * (actual - budget), budget)
    if re.search(r"exactly on budget|与预算持平", state):
        return Fraction(0)
    if m := re.search(r"(\d+)% over budget|超出预算(\d+)%", state):
        return Fraction(int(m.group(1) or m.group(2)))
    if m := re.search(r"(\d+)% under budget|节省了(\d+)%", state):
        return Fraction(-int(m.group(1) or m.group(2)))
    return None


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    kind = next(
        (k for k, (title, _) in TEXT[lang].items() if title.split("{")[0] in state),
        None,
    )
    if kind is None:
        return None
    value = _value(state, kind, lang)
    if value is None:
        return None
    hits = [
        i
        for i, option in enumerate(options)
        if _in_band(value, _parse_band(option["description"], kind, lang), kind)
    ]
    return core.score_levels(options)[hits[0]] if len(hits) == 1 else None
