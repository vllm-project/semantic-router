"""a6_interpolation: level = floor(rate * L) from counts; only endpoint levels have anchors."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import Group, Variant

FAMILY = "a6_interpolation"
DOMAINS: dict[str, dict[str, Any]] = {
    "tests": {
        "en": {
            "title": "Build report — {co}",
            "facts": (
                "The nightly run executed {T} tests, and {p} of them passed.",
                "The nightly run executed {T} tests; {f} of them failed and the rest passed.",
                "In the nightly run, {p} tests passed and {f} failed.",
            ),
            "rate": "pass rate",
            "good": "passed tests",
        },
        "zh": {
            "title": "构建报告——{co}",
            "facts": (
                "夜间构建共执行了{T}个测试用例，其中{p}个通过。",
                "夜间构建共执行了{T}个测试用例，其中{f}个失败，其余全部通过。",
                "夜间构建中，{p}个测试用例通过，{f}个失败。",
            ),
            "rate": "通过率",
            "good": "通过的用例数",
        },
    },
    "inspection": {
        "en": {
            "title": "Quality inspection — {co}",
            "facts": (
                "Inspectors checked {T} items from the batch, and {p} passed inspection.",
                "Inspectors checked {T} items from the batch; {f} were rejected and the rest passed.",
                "{p} items passed inspection and {f} were rejected.",
            ),
            "rate": "pass rate",
            "good": "items that passed",
        },
        "zh": {
            "title": "质量检验——{co}",
            "facts": (
                "质检员抽检了这批货中的{T}件，其中{p}件合格。",
                "质检员抽检了这批货中的{T}件，其中{f}件不合格，其余均合格。",
                "本批抽检中有{p}件合格，{f}件不合格。",
            ),
            "rate": "合格率",
            "good": "合格件数",
        },
    },
    "survey": {
        "en": {
            "title": "Customer survey — {co}",
            "facts": (
                "{T} customers answered the survey, and {p} said they were satisfied.",
                "{T} customers answered the survey; {f} said they were not satisfied and the rest were satisfied.",
                "{p} customers said they were satisfied and {f} said they were not.",
            ),
            "rate": "satisfaction rate",
            "good": "satisfied customers",
        },
        "zh": {
            "title": "顾客满意度调查——{co}",
            "facts": (
                "共有{T}位顾客填写了问卷，其中{p}位表示满意。",
                "共有{T}位顾客填写了问卷，其中{f}位表示不满意，其余都表示满意。",
                "{p}位顾客表示满意，{f}位表示不满意。",
            ),
            "rate": "满意率",
            "good": "满意的顾客数",
        },
    },
    "deliveries": {
        "en": {
            "title": "Depot weekly summary — {co}",
            "facts": (
                "The depot made {T} deliveries last week, and {p} arrived on time.",
                "The depot made {T} deliveries last week; {f} were late and the rest arrived on time.",
                "Last week {p} deliveries arrived on time and {f} were late.",
            ),
            "rate": "on-time rate",
            "good": "on-time deliveries",
        },
        "zh": {
            "title": "配送站周报——{co}",
            "facts": (
                "上周配送站共完成{T}单配送，其中{p}单准时送达。",
                "上周配送站共完成{T}单配送，其中{f}单延误，其余都准时送达。",
                "上周有{p}单准时送达，{f}单延误。",
            ),
            "rate": "准时率",
            "good": "准时送达的单数",
        },
    },
}
CO_SUFFIX = {"en": ("Labs", "Works", "Logistics"), "zh": ("科技", "制造", "物流")}


def level_of(passed: int, total: int, levels: int) -> int:
    return levels - 1 if passed == total else passed * levels // total


def _instructions(t: dict[str, str], lang: str, levels: int) -> str:
    if lang == "en":
        return (
            f"Compute the {t['rate']} as {t['good']} divided by the total. The level is ⌊rate × {levels}⌋, "
            f"that is, the rate multiplied by {levels} and rounded down, except that a rate of exactly 100% "
            "gets the top level. Which level applies?"
        )
    return (
        f"请先计算{t['rate']}（{t['good']}÷总数）。等级为⌊比率×{levels}⌋，即比率乘以{levels}后向下取整；"
        "比率恰好为100%时取最高级。应评为哪一级？"
    )


def _descriptions(t: dict[str, str], lang: str, levels: int) -> list[str]:
    rate = t["rate"]
    if lang == "en":
        middle = [f"Level {k}" for k in range(1, levels - 1)]
        return [
            f"Lowest level — the lowest band of {rate}",
            *middle,
            f"Top level — the highest band of {rate}, including 100%",
        ]
    middle = [f"第{k}级" for k in range(1, levels - 1)]
    return [f"最低级——{rate}最低的一档", *middle, f"最高级——{rate}最高的一档（含100%）"]


def build_group(
    rng: random.Random, lang: str, task_type: str = "score", levels: int | None = None
) -> Group:
    levels = levels or 4
    name = rng.choice(sorted(DOMAINS))
    t = DOMAINS[name][lang]
    total = rng.randint(max(levels, 12), 160)
    style = rng.randrange(3)
    co = core.company(rng, lang, CO_SUFFIX[lang])
    specs = []
    for k in range(levels):
        low = math.ceil(k * total / levels)
        high = total if k == levels - 1 else math.ceil((k + 1) * total / levels) - 1
        if style != 0 and k == levels - 1 and rng.random() < 0.5:
            passed = total
        elif rng.random() < 0.3:
            passed = low
        else:
            passed = rng.randint(low, high)
        if level_of(passed, total, levels) != k:
            raise AssertionError("interpolation level construction is inconsistent")
        text = t["facts"][style].format(T=total, p=passed, f=total - passed)
        specs.append(
            (
                k,
                text,
                {"domain": name, "total": total, "passed": passed, "style": style},
            )
        )
    pad = core.filler(rng, lang, core.words(specs[0][1], lang) + 10, [co])
    sec = core.SECTION[lang]
    variants = []
    for k, text, facts in specs:
        state = core.compose(
            t["title"].format(co=co),
            [
                (sec["background"], pad[0]),
                (sec["record"], [text]),
                (sec["notes"], pad[1]),
            ],
            lang,
        )
        variants.append(
            Variant(state, k, facts, f"passed count set to {facts['passed']}")
        )
    group = Group(
        "score",
        _instructions(t, lang, levels),
        core.score_options(_descriptions(t, lang, levels)),
        variants,
        f"{name}_style{style}",
        name,
        {"levels": levels},
    )
    core.check_group(group)
    return group


# ---------------------------------------------------------------- oracle 2


def _counts(state: str, name: str, lang: str) -> tuple[int, int] | None:
    for template in DOMAINS[name][lang]["facts"]:
        rx = core.template_regex(template, T=r"\d+", p=r"\d+", f=r"\d+")
        m = re.search(rx, state)
        if not m:
            continue
        slots = re.findall(r"\{(\w)\}", template)
        got = dict(zip(slots, (int(g) for g in m.groups())))
        if "T" in got and "p" in got:
            return got["p"], got["T"]
        if "T" in got and "f" in got:
            return got["T"] - got["f"], got["T"]
        return got["p"], got["p"] + got["f"]
    return None


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    name = next(
        (
            d
            for d, spec in DOMAINS.items()
            if spec[lang]["title"].split("{")[0] in state
        ),
        None,
    )
    scale = re.search(r"⌊rate × (\d+)⌋|⌊比率×(\d+)⌋", instructions)
    if name is None or scale is None:
        return None
    levels = int(scale.group(1) or scale.group(2))
    counts = _counts(state, name, lang)
    if counts is None or len(options) != levels:
        return None
    passed, total = counts
    level = levels - 1 if passed == total else math.floor(passed * levels / total)
    for i, option in enumerate(options):
        text = option["description"]
        if text.startswith(("Lowest level", "最低级")):
            k = 0
        elif text.startswith(("Top level", "最高级")):
            k = levels - 1
        else:
            k = int(re.search(r"\d+", text).group(0))
        if k == level:
            return core.score_levels(options)[i]
    return None
