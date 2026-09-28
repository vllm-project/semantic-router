"""a6_checklist: level = number of listed criteria that the neutrally stated facts satisfy."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import Group, Variant

FAMILY = "a6_checklist"


@dataclass(frozen=True)
class Criterion:
    key: str
    kind: str  # "max" | "min" | "by" | "is"
    lo: int
    hi: int
    step: int
    en: tuple[str, str]
    zh: tuple[str, str]
    tenths: bool = False
    cats: tuple[tuple[str, str], ...] = ()


C = Criterion
DOMAINS: dict[str, dict[str, Any]] = {
    "flat": {
        "title": {"en": "Listing summary — {co}", "zh": "房源摘要——{co}"},
        "ask": {
            "en": "{p} is looking for a flat and has {n} requirements: {items}. How many of these requirements does the listed flat meet?",
            "zh": "{p}在找房子，提出了{n}项要求：{items}。这套房子满足其中几项要求？",
        },
        "suffix": {"en": ("Lettings", "Homes"), "zh": ("房产", "置业")},
        "criteria": (
            C(
                "rent",
                "max",
                900,
                2400,
                10,
                ("a monthly rent of at most €{T}", "The monthly rent is €{v}."),
                ("月租不超过{T}元", "月租为{v}元。"),
            ),
            C(
                "size",
                "min",
                35,
                120,
                1,
                ("at least {T} m² of floor space", "The flat measures {v} m²."),
                ("面积不少于{T}平方米", "房屋面积为{v}平方米。"),
            ),
            C(
                "metro",
                "max",
                150,
                2000,
                10,
                (
                    "no more than {T} m from a metro station",
                    "The nearest metro station is {v} m away.",
                ),
                ("距地铁站不超过{T}米", "离最近的地铁站有{v}米。"),
            ),
            C(
                "floor",
                "min",
                1,
                20,
                1,
                ("a flat on floor {T} or higher", "The flat is on floor {v}."),
                ("位于{T}楼或以上", "房子位于{v}楼。"),
            ),
            C(
                "deposit",
                "max",
                2,
                5,
                1,
                (
                    "a deposit of at most {T} months' rent",
                    "The deposit is {v} months' rent.",
                ),
                ("押金不超过{T}个月房租", "押金为{v}个月房租。"),
            ),
            C(
                "move",
                "by",
                0,
                90,
                1,
                ("availability on or before {T}", "It becomes available on {v}."),
                ("{T}或之前可以入住", "可入住日期为{v}。"),
            ),
            C(
                "built",
                "min",
                1965,
                2023,
                1,
                (
                    "a building completed in {T} or later",
                    "The building was completed in {v}.",
                ),
                ("{T}年或之后建成", "楼房建成于{v}年。"),
            ),
            C(
                "heat",
                "is",
                0,
                0,
                1,
                ("heating type: {T}", "Heating type: {v}."),
                ("供暖方式为{T}", "供暖方式：{v}。"),
                cats=(
                    ("gas boiler", "燃气锅炉"),
                    ("district heating", "集中供暖"),
                    ("electric radiators", "电暖器"),
                    ("heat pump", "热泵"),
                ),
            ),
            C(
                "balcony",
                "min",
                2,
                15,
                1,
                ("a balcony of at least {T} m²", "The balcony measures {v} m²."),
                ("阳台不小于{T}平方米", "阳台面积为{v}平方米。"),
            ),
            C(
                "service",
                "max",
                40,
                300,
                5,
                (
                    "service charges of at most €{T} a month",
                    "Service charges are €{v} a month.",
                ),
                ("物业费每月不超过{T}元", "物业费为每月{v}元。"),
            ),
        ),
    },
    "bid": {
        "title": {"en": "Supplier bid — {co}", "zh": "供应商报价——{co}"},
        "ask": {
            "en": "The purchasing team at {p2} has {n} requirements for this bid: {items}. How many of these requirements does the bid meet?",
            "zh": "{p2}的采购部对这份报价提出了{n}项要求：{items}。这份报价满足其中几项要求？",
        },
        "suffix": {"en": ("Components", "Manufacturing"), "zh": ("零部件", "制造")},
        "criteria": (
            C(
                "price",
                "max",
                5,
                80,
                1,
                ("a unit price of at most ${T}", "The quoted unit price is ${v}."),
                ("单价不超过{T}元", "报价单价为{v}元。"),
            ),
            C(
                "lead",
                "max",
                5,
                60,
                1,
                ("a lead time of at most {T} days", "The lead time is {v} days."),
                ("交货周期不超过{T}天", "交货周期为{v}天。"),
            ),
            C(
                "moq",
                "max",
                50,
                2000,
                50,
                (
                    "a minimum order of no more than {T} units",
                    "The minimum order is {v} units.",
                ),
                ("起订量不超过{T}件", "起订量为{v}件。"),
            ),
            C(
                "defect",
                "max",
                1,
                50,
                1,
                ("a defect rate of at most {T}%", "Last year's defect rate was {v}%."),
                ("次品率不高于{T}%", "去年的次品率为{v}%。"),
                tenths=True,
            ),
            C(
                "warranty",
                "min",
                6,
                48,
                1,
                ("a warranty of at least {T} months", "The warranty lasts {v} months."),
                ("质保期不少于{T}个月", "质保期为{v}个月。"),
            ),
            C(
                "first",
                "by",
                0,
                90,
                1,
                (
                    "first delivery on or before {T}",
                    "First delivery is scheduled for {v}.",
                ),
                ("{T}或之前完成首批交货", "首批交货日期定在{v}。"),
            ),
            C(
                "cert",
                "is",
                0,
                0,
                1,
                ("certification held: {T}", "Certification held: {v}."),
                ("持有{T}认证", "持有的认证：{v}。"),
                cats=(
                    ("ISO 9001", "ISO 9001"),
                    ("ISO 14001", "ISO 14001"),
                    ("ISO 45001", "ISO 45001"),
                ),
            ),
            C(
                "terms",
                "min",
                15,
                90,
                5,
                ("payment terms of at least {T} days", "Payment terms are {v} days."),
                ("账期不少于{T}天", "账期为{v}天。"),
            ),
            C(
                "years",
                "min",
                2,
                40,
                1,
                (
                    "at least {T} years in business",
                    "The company has been trading for {v} years.",
                ),
                ("成立不少于{T}年", "公司已经营业{v}年。"),
            ),
            C(
                "ontime",
                "min",
                70,
                99,
                1,
                (
                    "an on-time delivery rate of at least {T}%",
                    "Its on-time delivery rate last year was {v}%.",
                ),
                ("准时交货率不低于{T}%", "去年的准时交货率为{v}%。"),
            ),
        ),
    },
    "venue": {
        "title": {"en": "Venue fact sheet — {co}", "zh": "场地资料——{co}"},
        "ask": {
            "en": "{p} is booking a venue for a company dinner and has {n} requirements: {items}. How many of these requirements does this venue meet?",
            "zh": "{p}要为公司晚宴预订场地，提出了{n}项要求：{items}。这个场地满足其中几项要求？",
        },
        "suffix": {"en": ("Hall", "Rooms"), "zh": ("会馆", "宴会厅")},
        "criteria": (
            C(
                "seats",
                "min",
                40,
                400,
                5,
                ("room for at least {T} guests", "The hall seats {v} guests."),
                ("可容纳至少{T}位来宾", "大厅可容纳{v}位来宾。"),
            ),
            C(
                "fee",
                "max",
                500,
                6000,
                50,
                ("a rental fee of at most £{T}", "The rental fee is £{v}."),
                ("场租不超过{T}元", "场租为{v}元。"),
            ),
            C(
                "station",
                "max",
                2,
                80,
                1,
                (
                    "a location within {T} km of the train station",
                    "It is {v} km from the train station.",
                ),
                ("距火车站不超过{T}公里", "距火车站{v}公里。"),
                tenths=True,
            ),
            C(
                "parking",
                "min",
                5,
                200,
                5,
                ("at least {T} parking spaces", "There are {v} parking spaces."),
                ("至少有{T}个停车位", "共有{v}个停车位。"),
            ),
            C(
                "free",
                "by",
                0,
                90,
                1,
                ("a free date on or before {T}", "The earliest free date is {v}."),
                ("{T}或之前有空档", "最早可预订日期为{v}。"),
            ),
            C(
                "catering",
                "is",
                0,
                0,
                1,
                ("catering: {T}", "Catering: {v}."),
                ("餐饮安排为{T}", "餐饮安排：{v}。"),
                cats=(
                    ("in-house kitchen", "场地自营厨房"),
                    ("outside caterers only", "仅限外请餐饮"),
                    ("no catering", "不提供餐饮"),
                ),
            ),
            C(
                "setup",
                "min",
                2,
                8,
                1,
                (
                    "at least {T} hours of setup access",
                    "Setup access starts {v} hours before the event.",
                ),
                ("布场时间不少于{T}小时", "活动开始前{v}小时可以进场布置。"),
            ),
            C(
                "music",
                "min",
                20,
                23,
                1,
                ("music allowed until at least {T}:00", "Music must stop at {v}:00."),
                ("音乐至少可以放到{T}点", "音乐须在{v}点前停止。"),
            ),
            C(
                "access",
                "is",
                0,
                0,
                1,
                ("step-free access: {T}", "Step-free access: {v}."),
                ("无障碍通道：{T}", "无障碍通道：{v}。"),
                cats=(("available", "有"), ("not available", "无")),
            ),
            C(
                "deposit",
                "max",
                10,
                50,
                5,
                ("a deposit of at most {T}%", "A deposit of {v}% is required."),
                ("定金比例不超过{T}%", "需支付{v}%的定金。"),
            ),
        ),
    },
}


def _fmt(c: Criterion, value: int, lang: str, anchor: date) -> str:
    if c.kind == "by":
        return core.fmt_date(anchor + timedelta(days=value), lang)
    if c.kind == "is":
        return c.cats[value][0 if lang == "en" else 1]
    if c.tenths:
        return f"{value / 10:.1f}"
    return core.fmt_int(value, lang) if c.key not in ("built",) else str(value)


def _met(c: Criterion, value: int, threshold: int) -> bool:
    if c.kind in ("max", "by"):
        return value <= threshold
    if c.kind == "min":
        return value >= threshold
    return value == threshold


def _draw(rng: random.Random, c: Criterion, threshold: int, met: bool) -> int:
    if c.kind == "is":
        if met:
            return threshold
        return rng.choice([i for i in range(len(c.cats)) if i != threshold])
    values = [v for v in range(c.lo, c.hi + 1, c.step) if _met(c, v, threshold) == met]
    edge = (
        threshold
        if met
        else (threshold + c.step if c.kind in ("max", "by") else threshold - c.step)
    )
    if rng.random() < 0.25 and edge in values:
        return edge
    return rng.choice(values)


def _threshold(rng: random.Random, c: Criterion) -> int:
    if c.kind == "is":
        return rng.randrange(len(c.cats))
    span = list(range(c.lo, c.hi + 1, c.step))
    return rng.choice(span[1:-1])


def _options(n: int, lang: str) -> list[str]:
    if n == 1:
        return (
            ["Does not meet the requirement", "Meets the requirement"]
            if lang == "en"
            else ["不满足这项要求", "满足这项要求"]
        )
    if lang == "en":
        top = "Meets both requirements" if n == 2 else f"Meets all {n} requirements"
        return (
            [f"Meets none of the {n} requirements"]
            + [f"Meets exactly {k} of the {n} requirements" for k in range(1, n)]
            + [top]
        )
    top = "两项要求都满足" if n == 2 else f"{n}项要求全部满足"
    return ["一项要求都不满足"] + [f"恰好满足{k}项要求" for k in range(1, n)] + [top]


def _ask(template: str, lang: str, n: int, items: str, **names: str) -> str:
    text = template.format(n=n, items=items, **names)
    if n > 1:
        return text
    if lang == "en":
        text = text.replace("has 1 requirements", "has one requirement").replace(
            "(1) ", ""
        )
        return re.sub(
            r"How many of these requirements does (.+) meet\?",
            r"Does \1 meet this requirement?",
            text,
        )
    text = text.replace("提出了1项要求", "提出了一项要求").replace("（1）", "")
    return re.sub(r"满足其中几项要求？$", "能达到上面这项要求吗？", text)


def build_group(
    rng: random.Random, lang: str, task_type: str = "score", levels: int | None = None
) -> Group:
    levels = levels or 4
    n = levels - 1
    name = rng.choice(sorted(DOMAINS))
    spec = DOMAINS[name]
    criteria = list(spec["criteria"])
    rng.shuffle(criteria)
    chosen, extras = criteria[:n], criteria[n:]
    anchor = core.random_date(rng)
    thresholds = {c.key: _threshold(rng, c) for c in chosen}
    extra_values = {
        c.key: _draw(rng, c, _threshold(rng, c), rng.random() < 0.5) for c in extras
    }
    person, other = core.people(rng, lang, 2)
    co = core.company(rng, lang, spec["suffix"][lang])
    buyer = core.company(
        rng, lang, ("Retail", "Foods") if lang == "en" else ("零售", "食品")
    )
    fact_order = rng.sample([c.key for c in spec["criteria"]], len(spec["criteria"]))
    by_key = {c.key: c for c in spec["criteria"]}
    listed = [
        (f"({i}) " if lang == "en" else f"（{i}）")
        + (c.en if lang == "en" else c.zh)[0].format(
            T=_fmt(c, thresholds[c.key], lang, anchor)
        )
        for i, c in enumerate(chosen, 1)
    ]
    items = "; ".join(listed) if lang == "en" else "；".join(listed)
    instructions = _ask(spec["ask"][lang], lang, n, items, p=person, p2=buyer)
    specs = []
    for k in range(levels):
        met_keys = set(rng.sample([c.key for c in chosen], k))
        values = dict(extra_values)
        for c in chosen:
            values[c.key] = _draw(rng, c, thresholds[c.key], c.key in met_keys)
        facts_text = [
            (by_key[key].en if lang == "en" else by_key[key].zh)[1].format(
                v=_fmt(by_key[key], values[key], lang, anchor)
            )
            for key in fact_order
        ]
        facts = {
            "domain": name,
            "thresholds": thresholds,
            "values": values,
            "met": sorted(met_keys),
            "anchor": anchor.isoformat(),
        }
        specs.append((k, facts_text, facts, f"criteria met: {sorted(met_keys)}"))
    intro = {
        "en": f"{co} provided the following details.",
        "zh": f"{co}提供了以下信息。",
    }[lang]
    pad = core.filler(
        rng,
        lang,
        core.words(" ".join(specs[0][1]), lang) + 15,
        [person, other, co, buyer],
    )
    sec = core.SECTION[lang]
    variants = []
    for k, facts_text, facts, edit in specs:
        state = core.compose(
            spec["title"][lang].format(co=co),
            [
                (sec["background"], [intro, *pad[0]]),
                (sec["record"], facts_text),
                (sec["notes"], pad[1]),
            ],
            lang,
        )
        variants.append(Variant(state, k, facts, edit))
    group = Group(
        "score",
        instructions,
        core.score_options(_options(n, lang)),
        variants,
        f"{name}_n{n}",
        name,
        {"levels": levels},
    )
    core.check_group(group)
    return group


# ---------------------------------------------------------------- oracle 2

_NUM = r"\d[\d,]*(?:\.\d+)?"


def _parse(c: Criterion, text: str, lang: str) -> Any:
    if c.kind == "by":
        return core.to_date(text)
    if c.kind == "is":
        return text
    return float(text.replace(",", ""))


def _level_of(description: str, n: int) -> int | None:
    if re.search(r"none of|都不满足|^不满足|Does not meet", description):
        return 0
    if re.search(
        r"^Meets all|^Meets both|全部满足|都满足$|^满足这项|^Meets the requirement",
        description,
    ):
        return n
    m = re.search(r"exactly (\d+)|恰好满足(\d+)", description)
    return int(m.group(1) or m.group(2)) if m else None


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    name = next(
        (
            d
            for d, spec in DOMAINS.items()
            if spec["title"][lang].split("{")[0] in state
        ),
        None,
    )
    if name is None:
        return None
    count = listed = 0
    for c in DOMAINS[name]["criteria"]:
        crit, fact = c.en if lang == "en" else c.zh
        slot = (
            core.DATE_RX[lang]
            if c.kind == "by"
            else (
                "|".join(re.escape(x[0 if lang == "en" else 1]) for x in c.cats)
                if c.kind == "is"
                else _NUM
            )
        )
        crit_m = re.search(core.template_regex(crit, T=slot), instructions)
        if not crit_m:
            continue
        fact_m = re.search(core.template_regex(fact, v=slot), state)
        if not fact_m:
            return None
        listed += 1
        want, have = _parse(c, crit_m.group(1), lang), _parse(c, fact_m.group(1), lang)
        if c.kind in ("max", "by"):
            count += have <= want
        elif c.kind == "min":
            count += have >= want
        else:
            count += have == want
    levels = [_level_of(o["description"], listed) for o in options]
    return (
        core.score_levels(options)[levels.index(count)]
        if count in levels and levels.count(count) == 1
        else None
    )
