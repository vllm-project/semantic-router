"""a6_rank_position: level = how many reference entries the candidate strictly beats."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import Group, Variant

FAMILY = "a6_rank_position"
DOMAINS: dict[str, dict[str, Any]] = {
    "battery": {
        "higher": True,
        "range": (60, 300),
        "labels": "products",
        "en": {
            "title": "Battery test results — {co}",
            "refs": ("{x} lasted {v} hours.", "The {x} ran for {v} hours."),
            "cand": "The new {x} lasted {v} hours in the same test.",
            "attr": "battery life",
            "what": "reference models",
        },
        "zh": {
            "title": "电池续航测试——{co}",
            "refs": ("{x}续航{v}小时。", "{x}坚持了{v}小时。"),
            "cand": "新款{x}在同样的测试中续航{v}小时。",
            "attr": "续航时间",
            "what": "参照机型",
        },
    },
    "sprint": {
        "higher": False,
        "range": (118, 170),
        "labels": "people",
        "en": {
            "title": "Club time trial — {co}",
            "refs": (
                "{x} ran the 100 m in {v} seconds.",
                "{x} clocked {v} seconds over 100 m.",
            ),
            "cand": "{x}, the newest member, ran the 100 m in {v} seconds.",
            "attr": "100 m time",
            "what": "reference runners",
        },
        "zh": {
            "title": "俱乐部计时赛——{co}",
            "refs": ("{x}的百米成绩为{v}秒。", "{x}跑百米用了{v}秒。"),
            "cand": "新会员{x}的百米成绩为{v}秒。",
            "attr": "百米成绩",
            "what": "参照选手",
        },
    },
    "yield": {
        "higher": True,
        "range": (30, 120),
        "labels": "farms",
        "en": {
            "title": "Harvest report — {co}",
            "refs": (
                "{x} harvested {v} tonnes per hectare.",
                "The yield at {x} was {v} tonnes per hectare.",
            ),
            "cand": "The trial plot at {x} harvested {v} tonnes per hectare.",
            "attr": "yield per hectare",
            "what": "reference farms",
        },
        "zh": {
            "title": "收成报告——{co}",
            "refs": ("{x}每公顷收获{v}吨。", "{x}的产量为每公顷{v}吨。"),
            "cand": "试验田{x}每公顷收获{v}吨。",
            "attr": "每公顷产量",
            "what": "参照农场",
        },
    },
    "latency": {
        "higher": False,
        "range": (80, 900),
        "labels": "servers",
        "en": {
            "title": "Server benchmark — {co}",
            "refs": (
                "{x} answered in {v} ms on average.",
                "The average response time of {x} was {v} ms.",
            ),
            "cand": "The candidate build, {x}, answered in {v} ms on average.",
            "attr": "average response time",
            "what": "reference builds",
        },
        "zh": {
            "title": "服务器基准测试——{co}",
            "refs": ("{x}的平均响应时间为{v}毫秒。", "{x}平均{v}毫秒完成一次响应。"),
            "cand": "候选版本{x}的平均响应时间为{v}毫秒。",
            "attr": "平均响应时间",
            "what": "参照版本",
        },
    },
}
LABELS = {
    "products": {
        "en": (
            "Arden 5",
            "Brio Max",
            "Corsa Lite",
            "Dellin S",
            "Elara Go",
            "Fenix Pro",
            "Gala 3",
            "Halden X",
            "Iris Mini",
            "Juno Plus",
            "Kora 2",
        ),
        "zh": (
            "雅登5",
            "布里奥Max",
            "科尔萨Lite",
            "德林S",
            "伊拉Go",
            "菲尼Pro",
            "嘉拉3",
            "海登X",
            "艾瑞Mini",
            "朱诺Plus",
            "柯拉2",
        ),
    },
    "farms": {
        "en": (
            "Ashby Farm",
            "Brook Farm",
            "Cole Farm",
            "Dene Farm",
            "Elm Farm",
            "Fallow Farm",
            "Glen Farm",
            "Hurst Farm",
            "Ivy Farm",
            "Kerr Farm",
            "Lark Farm",
        ),
        "zh": (
            "青山农场",
            "白河农场",
            "柳湾农场",
            "东坡农场",
            "榆林农场",
            "麦田农场",
            "南谷农场",
            "石岭农场",
            "桃园农场",
            "杏林农场",
            "云雀农场",
        ),
    },
    "servers": {
        "en": (
            "Atlas-3",
            "Borealis",
            "Cygnus",
            "Draco-2",
            "Eridan",
            "Fornax",
            "Gemma-7",
            "Hydra",
            "Indus",
            "Lyra-4",
            "Mensa",
        ),
        "zh": (
            "天鹰3号",
            "北冕",
            "天鹅座",
            "天龙2号",
            "波江",
            "天炉",
            "双子7号",
            "长蛇",
            "印第安",
            "天琴4号",
            "山案",
        ),
    },
}
CO_SUFFIX = {
    "battery": {"en": ("Test Labs", "Review Lab"), "zh": ("评测实验室", "检测中心")},
    "sprint": {
        "en": ("Athletics Club", "Running Club"),
        "zh": ("田径俱乐部", "跑步俱乐部"),
    },
    "yield": {
        "en": ("Growers Cooperative", "Farmers Union"),
        "zh": ("农业合作社", "种植合作社"),
    },
    "latency": {"en": ("Systems", "Cloud"), "zh": ("系统", "云计算")},
}


def _labels(rng: random.Random, kind: str, lang: str, n: int) -> list[str]:
    if kind == "people":
        return core.people(rng, lang, n)
    return rng.sample(LABELS[kind][lang], n)


def _fmt(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10}"


def _options(n: int, lang: str) -> list[str]:
    if n == 1:
        return (
            ["Does not beat the reference", "Beats the reference"]
            if lang == "en"
            else ["没有胜过这个参照", "胜过了这个参照"]
        )
    if lang == "en":
        top = "Beats both references" if n == 2 else f"Beats all {n} references"
        return (
            [f"Beats none of the {n} references"]
            + [f"Beats exactly {k} of the {n} references" for k in range(1, n)]
            + [top]
        )
    top = "两个参照都胜过了" if n == 2 else f"胜过全部{n}个参照"
    return ["一个参照都没有胜过"] + [f"恰好胜过{k}个参照" for k in range(1, n)] + [top]


def build_group(
    rng: random.Random, lang: str, task_type: str = "score", levels: int | None = None
) -> Group:
    levels = levels or 4
    n = levels - 1
    name = rng.choice(sorted(DOMAINS))
    spec = DOMAINS[name]
    t = spec[lang]
    higher = spec["higher"]
    lo, hi = spec["range"]
    labels = _labels(rng, spec["labels"], lang, n + 1)
    cand, refs = labels[0], labels[1:]
    while True:
        values = sorted(rng.sample(range(lo, hi + 1), n))
        if all(b - a >= 3 for a, b in zip(values, values[1:])):
            break
    ref_values = rng.sample(values, n)
    styles = [rng.randrange(2) for _ in refs]
    co = core.company(rng, lang, CO_SUFFIX[name][lang])
    worst_first = sorted(values, reverse=not higher)
    margin = max(4, min(12, (hi - lo) // 8))
    specs = []
    for k in range(levels):
        if higher:
            low = worst_first[k - 1] + 1 if k > 0 else lo - margin
            high = worst_first[k] - 1 if k < n else hi + margin
        else:
            low = worst_first[k] + 1 if k < n else lo - margin
            high = worst_first[k - 1] - 1 if k > 0 else hi + margin
        value = rng.randint(max(low, 10), high)
        beaten = sum(value > r if higher else value < r for r in ref_values)
        if beaten != k:
            raise AssertionError("rank position construction is inconsistent")
        specs.append((k, value))
    ref_lines = [
        t["refs"][s].format(x=x, v=_fmt(v)) for x, v, s in zip(refs, ref_values, styles)
    ]
    cand_at = rng.randint(0, n)
    exclude = [*labels, co] if spec["labels"] == "people" else [co]
    pad = core.filler(rng, lang, core.words(" ".join(ref_lines), lang) + 20, exclude)
    sec = core.SECTION[lang]
    variants = []
    for k, value in specs:
        lines = list(ref_lines)
        lines.insert(cand_at, t["cand"].format(x=cand, v=_fmt(value)))
        state = core.compose(
            t["title"].format(co=co),
            [
                (sec["background"], pad[0]),
                (sec["record"], lines),
                (sec["notes"], pad[1]),
            ],
            lang,
        )
        facts = {
            "domain": name,
            "refs": dict(zip(refs, ref_values)),
            "candidate": cand,
            "value": value,
            "higher": higher,
        }
        variants.append(
            Variant(state, k, facts, f"candidate value set to {_fmt(value)}")
        )
    direction = (
        ("higher is better" if higher else "lower is better")
        if lang == "en"
        else ("数值越高越好" if higher else "数值越低越好")
    )
    if lang == "en":
        group_text = f"the {n} {t['what']}" if n > 1 else f"the {t['what'][:-1]}"
        instructions = (
            f"Compare {cand} with {group_text} on {t['attr']} ({direction}). "
            f"{cand} beats a reference only if its result is strictly better. Which level describes {cand}?"
        )
    else:
        instructions = (
            f"请将{cand}与{n}个{t['what']}比较{t['attr']}（{direction}）。只有当{cand}的结果严格优于某个参照时，"
            f"才算胜过该参照。{cand}属于哪一级？"
        )
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
    if name is None:
        return None
    spec = DOMAINS[name]
    t = spec[lang]
    label = (
        core.EN_NAME_RX
        if spec["labels"] == "people" and lang == "en"
        else (
            core.ZH_NAME_RX
            if spec["labels"] == "people"
            else "|".join(re.escape(x) for x in LABELS[spec["labels"]][lang])
        )
    )
    num = r"\d+\.\d"
    cand = re.search(core.template_regex(t["cand"], x=label, v=num), state)
    if not cand:
        return None
    refs = {}
    for template in t["refs"]:
        for m in re.finditer(core.template_regex(template, x=label, v=num), state):
            if m.group(1) != cand.group(1):
                refs[m.group(1)] = float(m.group(2))
    value = float(cand.group(2))
    beaten = sum(value > r if spec["higher"] else value < r for r in refs.values())
    n = len(refs)
    for i, option in enumerate(options):
        text = option["description"]
        if re.search(r"none of|都没有胜过|Does not beat|没有胜过这个", text):
            k = 0
        elif re.search(
            r"^Beats all|^Beats both|胜过全部|都胜过了|^Beats the reference|胜过了这个",
            text,
        ):
            k = n
        else:
            m = re.search(r"exactly (\d+)|恰好胜过(\d+)", text)
            k = int(m.group(1) or m.group(2))
        if k == beaten:
            return core.score_levels(options)[i]
    return None
