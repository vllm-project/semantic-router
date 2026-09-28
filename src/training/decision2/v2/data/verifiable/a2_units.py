"""a2_units: compare measurements given in mixed units (>= 3% margin after conversion)."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import A4Scenario, Group, Variant

FAMILY = "a2_units"
MARGIN = 1.03
FACTOR = {
    "g": 1.0,
    "kg": 1000.0,
    "lb": 453.59237,
    "oz": 28.349523125,
    "m": 1.0,
    "km": 1000.0,
    "mi": 1609.344,
    "ft": 0.3048,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "mL": 0.001,
    "L": 1.0,
    "gal": 3.785411784,
}
EN_UNIT = {
    "h": ("hour", "hours"),
    "min": ("minute", "minutes"),
    "s": ("second", "seconds"),
}
ZH_UNIT = {
    "g": "克",
    "kg": "千克",
    "lb": "磅",
    "oz": "盎司",
    "m": "米",
    "km": "千米",
    "mi": "英里",
    "ft": "英尺",
    "s": "秒",
    "min": "分钟",
    "h": "小时",
    "mL": "毫升",
    "L": "升",
    "gal": "加仑",
}

DOMAINS: dict[str, dict[str, Any]] = {
    "parcels": {
        "units": ("g", "kg", "lb", "oz"),
        "range": (800.0, 30000.0),
        "best": max,
        "labels": "people",
        "en": {
            "title": "Mailroom log — {co}",
            "intro": "Several colleagues dropped off parcels at the {co} mailroom on {day}.",
            "items": (
                "{x}'s parcel weighs {v}.",
                "The parcel from {x} came in at {v}.",
                "{x} sent a parcel weighing {v}.",
            ),
            "limit": "The courier accepts parcels of up to {v} each; anything heavier goes as freight.",
            "q": "Whose parcel is the heaviest?",
            "noul": "Does {x}'s parcel exceed the courier's weight limit?",
            "is_best": "Is {x}'s parcel the heaviest one?",
        },
        "zh": {
            "title": "收发室登记——{co}",
            "intro": "{day}，几位同事在{co}的收发室寄出了包裹。",
            "items": (
                "{x}的包裹重{v}。",
                "{x}寄出的包裹称重为{v}。",
                "{x}的包裹有{v}重。",
            ),
            "limit": "快递公司规定单个包裹最重{v}，超过的按货运处理。",
            "q": "谁的包裹最重？",
            "noul": "{x}的包裹超过快递公司的重量上限了吗？",
            "is_best": "最重的包裹来自{x}吗？",
        },
    },
    "trails": {
        "units": ("m", "km", "mi", "ft"),
        "range": (1500.0, 24000.0),
        "best": max,
        "labels": "trails",
        "en": {
            "title": "Park trail guide — {co}",
            "intro": "The {co} visitor centre lists these walking routes.",
            "items": (
                "The {x} is {v} long.",
                "The {x} runs for {v}.",
                "Walkers on the {x} cover {v}.",
            ),
            "limit": "Routes listed as family walks may be at most {v} long.",
            "q": "Which trail is the longest?",
            "noul": "Does the {x} exceed the length limit for family walks?",
            "is_best": "Is the {x} the longest trail?",
        },
        "zh": {
            "title": "公园步道指南——{co}",
            "intro": "{co}游客中心列出了以下几条步道。",
            "items": ("{x}全长{v}。", "{x}的长度为{v}。", "走完{x}要走{v}。"),
            "limit": "列为亲子徒步的路线，全长不得超过{v}。",
            "q": "哪条步道最长？",
            "noul": "{x}超过亲子徒步路线的长度上限了吗？",
            "is_best": "最长的步道为{x}吗？",
        },
    },
    "couriers": {
        "units": ("s", "min", "h"),
        "range": (1500.0, 12000.0),
        "best": min,
        "labels": "people",
        "en": {
            "title": "Route timing sheet — {co}",
            "intro": "Each courier at {co} ran the same delivery route on {day}.",
            "items": (
                "{x} completed the route in {v}.",
                "{x}'s run took {v}.",
                "{x} needed {v} to finish the route.",
            ),
            "limit": "The target time for the route is {v}.",
            "q": "Which courier completed the route in the shortest time?",
            "noul": "Did {x} take longer than the target time?",
            "is_best": "Was {x} the fastest courier on the route?",
        },
        "zh": {
            "title": "线路计时表——{co}",
            "intro": "{day}，{co}的几位骑手跑了同一条配送线路。",
            "items": (
                "{x}跑完这条线路用了{v}。",
                "{x}的用时为{v}。",
                "{x}花了{v}才跑完全程。",
            ),
            "limit": "这条线路的目标用时为{v}。",
            "q": "哪位骑手跑完这条线路用时最短？",
            "noul": "{x}的用时超过目标时间了吗？",
            "is_best": "用时最短的骑手为{x}吗？",
        },
    },
    "containers": {
        "units": ("mL", "L", "gal"),
        "range": (1.5, 40.0),
        "best": max,
        "labels": "containers",
        "en": {
            "title": "Event supplies list — {co}",
            "intro": "These water containers are set aside for the {co} open day. Gallon figures are US gallons.",
            "items": (
                "The {x} holds {v}.",
                "The {x} can take {v}.",
                "The {x} has a capacity of {v}.",
            ),
            "limit": "Fire rules cap any single container at {v}.",
            "q": "Which container holds the most?",
            "noul": "Does the {x} hold more than the limit for a single container?",
            "is_best": "Does the {x} hold the most?",
        },
        "zh": {
            "title": "活动物资清单——{co}",
            "intro": "以下储水容器是为{co}开放日准备的。文中的加仑均指美制加仑。",
            "items": ("{x}能装{v}。", "{x}的容量为{v}。", "{x}最多可装{v}。"),
            "limit": "消防规定单个容器的容量不得超过{v}。",
            "q": "哪个容器装得最多？",
            "noul": "{x}的容量超过单个容器的上限了吗？",
            "is_best": "装得最多的容器为{x}吗？",
        },
    },
}
LABELS = {
    "trails": {
        "en": (
            "Ridge Loop",
            "Birch Trail",
            "Old Mill Path",
            "Heron Walk",
            "Quarry Circuit",
            "Larkspur Way",
            "Fox Hollow Trail",
        ),
        "zh": (
            "山脊环线",
            "白桦步道",
            "老磨坊小径",
            "苍鹭步道",
            "采石场环线",
            "飞燕草步道",
            "狐狸谷步道",
        ),
    },
    "containers": {
        "en": (
            "red jug",
            "blue bottle",
            "grey drum",
            "green canister",
            "white tank",
            "yellow flask",
            "black keg",
        ),
        "zh": (
            "红色水壶",
            "蓝色水瓶",
            "灰色水桶",
            "绿色储水罐",
            "白色水箱",
            "黄色保温瓶",
            "黑色大桶",
        ),
    },
}
DAYS = {
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
    "zh": ("周一", "周二", "周三", "周四", "周五"),
}
CO_SUFFIX = {
    "parcels": {"en": ("Consulting", "Group"), "zh": ("咨询", "集团")},
    "trails": {
        "en": ("Regional Park", "Nature Reserve"),
        "zh": ("森林公园", "湿地公园"),
    },
    "couriers": {"en": ("Couriers", "Express"), "zh": ("快送", "速递")},
    "containers": {
        "en": ("Community Centre", "Sports Club"),
        "zh": ("社区中心", "体育俱乐部"),
    },
}


def render_number(value: float, unit: str) -> float:
    raw = value / FACTOR[unit]
    if raw >= 1000:
        return float(round(raw / 10) * 10)
    if raw >= 100:
        return float(round(raw))
    return round(raw, 1) if raw >= 10 else round(raw, 2)


def fmt_value(number: float, unit: str, lang: str) -> str:
    text = core.fmt_dec(number, 2)
    if lang == "zh":
        return f"{text}{ZH_UNIT[unit]}"
    if number >= 1000:
        text = f"{int(number):,}"
    if unit in EN_UNIT:
        return f"{text} {EN_UNIT[unit][0 if number == 1 else 1]}"
    return f"{text} {unit}"


class _Sheet:
    def __init__(self, rng: random.Random, lang: str, n: int) -> None:
        self.lang = lang
        self.domain = rng.choice(sorted(DOMAINS))
        self.d = DOMAINS[self.domain]
        self.t = self.d[lang]
        if self.d["labels"] == "people":
            self.full = core.people(rng, lang, n)
        else:
            self.full = rng.sample(LABELS[self.d["labels"]][lang], n)
        self.labels = self.full
        units = list(self.d["units"])
        chosen = (
            rng.sample(units, min(len(units), 3))
            if rng.random() < 0.7
            else rng.sample(units, 2)
        )
        self.units = [chosen[i % len(chosen)] for i in range(n)]
        rng.shuffle(self.units)
        self.styles = [rng.randrange(3) for _ in range(n)]
        self.co = core.company(rng, lang, CO_SUFFIX[self.domain][lang])
        self.day = rng.choice(DAYS[lang])
        self.limit_unit = rng.choice(units)

    def value_text(self, index: int, number: float) -> str:
        return fmt_value(number, self.units[index], self.lang)

    def state(
        self,
        numbers: Sequence[float],
        limit: float | None,
        pad: tuple[list[str], list[str]],
    ) -> str:
        items = [
            self.t["items"][self.styles[i]].format(
                x=self.labels[i], v=self.value_text(i, numbers[i])
            )
            for i in range(len(self.labels))
        ]
        head = [self.t["intro"].format(co=self.co, day=self.day)]
        if limit is not None:
            head.append(
                self.t["limit"].format(v=fmt_value(limit, self.limit_unit, self.lang))
            )
        sec = core.SECTION[self.lang]
        return core.compose(
            self.t["title"].format(co=self.co),
            [
                (sec["background"], [*head, *pad[0]]),
                (sec["record"], items),
                (sec["notes"], pad[1]),
            ],
            self.lang,
        )

    def base(self, index: int, number: float) -> float:
        return number * FACTOR[self.units[index]]


def _separated(values: Sequence[float], best: int, lower_is_better: bool) -> bool:
    top = values[best]
    others = [v for i, v in enumerate(values) if i != best]
    if lower_is_better:
        return all(v >= top * MARGIN for v in others)
    return all(top >= v * MARGIN for v in others)


def _draw_values(
    rng: random.Random, sheet: _Sheet, best: int, spread: Sequence[float]
) -> list[float] | None:
    lo, hi = sheet.d["range"]
    lower = sheet.d["best"] is min
    top = rng.uniform(lo * 1.6, hi) if not lower else rng.uniform(lo, hi / 1.6)
    others = iter(spread)
    numbers = []
    for i in range(len(sheet.labels)):
        target = (
            top
            if i == best
            else (top / next(others) if not lower else top * next(others))
        )
        numbers.append(render_number(target, sheet.units[i]))
    bases = [sheet.base(i, x) for i, x in enumerate(numbers)]
    if any(x <= 0 for x in numbers) or not _separated(bases, best, lower):
        return None
    return numbers


def _group(rng: random.Random, lang: str, task: str) -> Group | None:
    n = rng.choice((3, 4, 5)) if task == "choice" else rng.choice((3, 4))
    sheet = _Sheet(rng, lang, n)
    spread = [rng.uniform(1.05, 1.6) for _ in range(n - 1)]
    specs = []
    if task == "choice":
        order = rng.sample(range(n), n)
        for label, item in enumerate(order):
            numbers = _draw_values(rng, sheet, item, spread)
            if numbers is None:
                return None
            facts = {
                "domain": sheet.domain,
                "labels": sheet.labels,
                "units": sheet.units,
                "numbers": numbers,
            }
            specs.append(
                (
                    label,
                    numbers,
                    None,
                    facts,
                    f"{sheet.labels[item]} given the extreme value",
                )
            )
        options = core.choice_options([sheet.labels[i] for i in order])
        instructions = sheet.t["q"]
    else:
        target = rng.randrange(n)
        if sheet.units[target] == sheet.limit_unit:
            sheet.limit_unit = rng.choice(
                [u for u in sheet.d["units"] if u != sheet.units[target]]
            )
        lo, hi = sheet.d["range"]
        limit_base = rng.uniform(lo * 1.3, hi / 1.3)
        limit = render_number(limit_base, sheet.limit_unit)
        limit_base = limit * FACTOR[sheet.limit_unit]
        others = [render_number(rng.uniform(lo, hi), u) for u in sheet.units]
        for label, factor in (
            (0, rng.uniform(0.75, 0.96)),
            (1, rng.uniform(1.04, 1.3)),
        ):
            numbers = list(others)
            numbers[target] = render_number(limit_base * factor, sheet.units[target])
            ratio = sheet.base(target, numbers[target]) / limit_base
            if (label == 1 and ratio < MARGIN) or (label == 0 and ratio > 1 / MARGIN):
                return None
            facts = {
                "domain": sheet.domain,
                "labels": sheet.labels,
                "units": sheet.units,
                "numbers": numbers,
                "limit": limit,
                "limit_unit": sheet.limit_unit,
                "target": sheet.labels[target],
            }
            specs.append(
                (
                    label,
                    numbers,
                    limit,
                    facts,
                    f"{sheet.labels[target]} measured at {numbers[target]} {sheet.units[target]}",
                )
            )
        options = core.noul_options(lang)
        instructions = sheet.t["noul"].format(x=sheet.labels[target])
    sample = sheet.state(specs[0][1], specs[0][2], ([], []))
    pad = core.filler(rng, lang, core.words(sample, lang), [*sheet.full, sheet.co])
    variants = [
        Variant(sheet.state(nums, lim, pad), label, facts, edit)
        for label, nums, lim, facts, edit in specs
    ]
    return Group(
        task, instructions, options, variants, f"{sheet.domain}_{task}", sheet.domain
    )


def build_group(
    rng: random.Random, lang: str, task_type: str, levels: int | None = None
) -> Group:
    for _ in range(300):
        group = _group(rng, lang, task_type)
        if group is not None and len({v.state for v in group.variants}) == len(
            group.variants
        ):
            core.check_group(group)
            return group
    raise RuntimeError("units group construction failed")


def a4_scenario(rng: random.Random, lang: str) -> A4Scenario:
    n = 7
    for _ in range(300):
        sheet = _Sheet(rng, lang, n)
        spread = sorted(
            rng.uniform(1.03, 1.12) if i < 3 else rng.uniform(1.15, 1.8)
            for i in range(n - 1)
        )
        numbers = _draw_values(rng, sheet, 0, spread)
        if numbers is not None:
            break
    else:
        raise RuntimeError("units a4 construction failed")
    lower = sheet.d["best"] is min
    bases = [sheet.base(i, x) for i, x in enumerate(numbers)]
    others = sorted(range(1, n), key=lambda i: bases[i], reverse=not lower)
    naive = max(range(1, n), key=lambda i: numbers[i])
    near = [others[0], others[1]] + (
        [naive] if naive not in others[:2] else [others[2]]
    )
    near += [i for i in others if i not in near]
    labels = sheet.labels
    pad = core.filler(
        rng,
        lang,
        core.words(sheet.state(numbers, None, ([], [])), lang),
        [*sheet.full, sheet.co],
    )
    state = sheet.state(numbers, None, pad)
    facts = {
        "domain": sheet.domain,
        "labels": labels,
        "units": sheet.units,
        "numbers": numbers,
    }
    ask = sheet.t["is_best"]
    return A4Scenario(
        state,
        facts,
        sheet.domain,
        f"{sheet.domain}_a4",
        sheet.t["q"],
        labels[0],
        [],
        [labels[i] for i in near],
        [],
        [labels[i] for i in range(1, n)],
        False,
        False,
        ask.format(x=labels[0]),
        ask.format(x=labels[others[0]]),
        [ask.format(x=labels[i]) for i in others[-2:]],
    )


def a4v2_plan(rng: random.Random, lang: str, turn: int) -> core.A4v2Plan:
    """Any of seven items may become the extreme one; near misses are its three runner-ups."""
    n = 7
    sheet = _Sheet(rng, lang, n)
    labels = sheet.labels
    ask = sheet.t["is_best"]

    def render(best: int, near: list[int]) -> core.A4v2Render | None:
        for _ in range(100):
            close = iter(sorted(rng.uniform(1.03, 1.12) for _ in near))
            spread = [
                next(close) if i in near else rng.uniform(1.18, 1.8)
                for i in range(n)
                if i != best
            ]
            numbers = _draw_values(rng, sheet, best, spread)
            if numbers is not None:
                break
        else:
            return None
        pad = core.filler(
            rng,
            lang,
            core.words(sheet.state(numbers, None, ([], [])), lang),
            [*sheet.full, sheet.co],
        )
        facts = {
            "domain": sheet.domain,
            "labels": labels,
            "units": sheet.units,
            "numbers": numbers,
        }
        return core.A4v2Render(
            sheet.state(numbers, None, pad),
            facts,
            sheet.domain,
            f"{sheet.domain}_v2",
            sheet.t["q"],
            lambda v: ask.format(x=v),
        )

    alternatives = []
    for best in rng.sample(range(n), n):
        others = [i for i in range(n) if i != best]
        near = rng.sample(others, 3)
        rand = rng.sample(others, 3)
        while set(rand) == set(near):
            rand = rng.sample(others, 3)
        alternatives.append(
            core.A4v2Alternative(
                best,
                labels[best],
                [labels[i] for i in near],
                [labels[i] for i in rand],
                lambda b=best, s=near: render(b, s),
            )
        )
    return core.A4v2Plan(False, alternatives)


# ---------------------------------------------------------------- oracle 2

_UNIT_EN = r"kg|g|lb|oz|km|mi|ft|m|hours?|minutes?|seconds?|mL|L|gal"
_UNIT_ZH = "千克|克|磅|盎司|千米|英里|英尺|米|小时|分钟|秒|毫升|升|加仑"
_ZH_BACK = {v: k for k, v in ZH_UNIT.items()}
_EN_BACK = {
    "hour": "h",
    "hours": "h",
    "minute": "min",
    "minutes": "min",
    "second": "s",
    "seconds": "s",
}
_ITEM = {
    "parcels": {
        "en": (
            r"({x})'s parcel weighs {v}\.",
            r"The parcel from ({x}) came in at {v}\.",
            r"({x}) sent a parcel weighing {v}\.",
        ),
        "zh": (
            r"({x})的包裹重{v}。",
            r"({x})寄出的包裹称重为{v}。",
            r"({x})的包裹有{v}重。",
        ),
    },
    "trails": {
        "en": (
            r"The ({x}) is {v} long\.",
            r"The ({x}) runs for {v}\.",
            r"Walkers on the ({x}) cover {v}\.",
        ),
        "zh": (r"({x})全长{v}。", r"({x})的长度为{v}。", r"走完({x})要走{v}。"),
    },
    "couriers": {
        "en": (
            r"({x}) completed the route in {v}\.",
            r"({x})'s run took {v}\.",
            r"({x}) needed {v} to finish the route\.",
        ),
        "zh": (
            r"({x})跑完这条线路用了{v}。",
            r"({x})的用时为{v}。",
            r"({x})花了{v}才跑完全程。",
        ),
    },
    "containers": {
        "en": (
            r"The ({x}) holds {v}\.",
            r"The ({x}) can take {v}\.",
            r"The ({x}) has a capacity of {v}\.",
        ),
        "zh": (r"({x})能装{v}。", r"({x})的容量为{v}。", r"({x})最多可装{v}。"),
    },
}
_LIMIT = {
    "parcels": {"en": r"accepts parcels of up to {v} each", "zh": r"单个包裹最重{v}"},
    "trails": {"en": r"may be at most {v} long", "zh": r"全长不得超过{v}"},
    "couriers": {"en": r"target time for the route is {v}\.", "zh": r"目标用时为{v}。"},
    "containers": {
        "en": r"cap any single container at {v}\.",
        "zh": r"容量不得超过{v}。",
    },
}
_NOUL = {
    "parcels": {"en": r"Does ({x})'s parcel exceed", "zh": r"^({x})的包裹超过"},
    "trails": {"en": r"Does the ({x}) exceed", "zh": r"^({x})超过"},
    "couriers": {"en": r"Did ({x}) take longer than", "zh": r"^({x})的用时超过"},
    "containers": {"en": r"Does the ({x}) hold more than", "zh": r"^({x})的容量超过"},
}
_BEST = {
    "parcels": {
        "en": r"Is ({x})'s parcel the heaviest",
        "zh": r"最重的包裹来自({x})吗",
    },
    "trails": {"en": r"Is the ({x}) the longest", "zh": r"最长的步道为({x})吗"},
    "couriers": {"en": r"Was ({x}) the fastest", "zh": r"用时最短的骑手为({x})吗"},
    "containers": {
        "en": r"Does the ({x}) hold the most",
        "zh": r"装得最多的容器为({x})吗",
    },
}


def _to_base(number: str, unit: str, lang: str) -> float:
    key = _ZH_BACK[unit] if lang == "zh" else _EN_BACK.get(unit, unit)
    return float(number.replace(",", "")) * FACTOR[key]


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    domain = next(
        (
            d
            for d, spec in DOMAINS.items()
            if spec[lang]["title"].split("{")[0] in state
        ),
        None,
    )
    if domain is None:
        return None
    spec = DOMAINS[domain]
    if spec["labels"] == "people":
        label_rx = core.EN_NAME_RX if lang == "en" else core.ZH_NAME_RX
    else:
        label_rx = "|".join(re.escape(x) for x in LABELS[spec["labels"]][lang])
    value_rx = (
        rf"(\d[\d,]*(?:\.\d+)?) ({_UNIT_EN})"
        if lang == "en"
        else rf"(\d+(?:\.\d+)?)({_UNIT_ZH})"
    )
    items: dict[str, float] = {}
    for pattern in _ITEM[domain][lang]:
        for m in re.finditer(pattern.format(x=label_rx, v=value_rx), state):
            items[m.group(1)] = _to_base(m.group(2), m.group(3), lang)
    if len(items) < 2:
        return None
    lower = spec["best"] is min
    ranked = sorted(items, key=items.get, reverse=not lower)
    if core.is_noul(options):
        best_q = re.search(_BEST[domain][lang].format(x=label_rx), instructions)
        if best_q:
            return int(ranked[0] == best_q.group(1))
        target = re.search(_NOUL[domain][lang].format(x=label_rx), instructions)
        limit = re.search(_LIMIT[domain][lang].format(v=value_rx), state)
        if not target or not limit or target.group(1) not in items:
            return None
        return int(
            items[target.group(1)] > _to_base(limit.group(1), limit.group(2), lang)
        )
    return core.match_option(options, lambda text: text == ranked[0])
