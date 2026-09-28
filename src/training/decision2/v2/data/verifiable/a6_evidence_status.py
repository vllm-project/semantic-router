"""a6_evidence_status (L=3): 0 record contradicts the claim, 1 does not determine it, 2 confirms it.

Every variant carries the same uncertainty wording (a date window, unreadable
labels, a smudged entry); only whether the uncertainty matters changes.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import Group, Variant

FAMILY = "a6_evidence_status"
STYLES = ("calendar", "counting", "multihop")
OPTIONS = {
    "en": (
        "The record contradicts the claim",
        "The record does not settle the claim",
        "The record confirms the claim",
    ),
    "zh": ("记录与该说法相矛盾", "记录无法确定该说法是否成立", "记录证实了该说法"),
}
ASK = {
    "en": "Claim: {claim} How does the record bear on this claim?",
    "zh": "说法：{claim}记录对这一说法的支持情况如何？",
}
TITLE = {
    "calendar": {"en": "Request log — {co}", "zh": "请求处理记录——{co}"},
    "counting": {"en": "Dispatch check — {co}", "zh": "发货核对——{co}"},
    "multihop": {"en": "Directory extract — {co}", "zh": "通讯录摘录——{co}"},
}
COLORS = {"en": ("blue", "green", "grey", "red"), "zh": ("蓝", "绿", "灰", "红")}
TEAMS = {
    "en": (
        "Payments",
        "Search",
        "Logistics",
        "Billing",
        "Research",
        "Security",
        "Design",
    ),
    "zh": ("支付组", "搜索组", "物流组", "结算组", "研究组", "安全组", "设计组"),
}
CITIES = {
    "en": ("Lisbon", "Tallinn", "Porto", "Ghent", "Lyon", "Krakow", "Osaka"),
    "zh": ("杭州", "成都", "苏州", "厦门", "青岛", "武汉", "西安"),
}
CO_SUFFIX = {"en": ("Services", "Logistics", "Systems"), "zh": ("服务", "物流", "系统")}


def _calendar(
    rng: random.Random, lang: str, cust: str
) -> tuple[str, list[tuple[int, list[str], dict[str, Any]]]]:
    received = core.random_date(rng)
    n = rng.randint(5, 30)
    due = received + timedelta(days=n)
    width = rng.randint(2, 6)
    claim_late = rng.random() < 0.4
    fmt = lambda d: core.fmt_date(d, lang)
    if lang == "en":
        head = [
            f"{cust} submitted a refund request, which was received on {fmt(received)}.",
            f"The company promises a reply no later than {n} calendar days after the day a request is received.",
        ]
        claim = (
            f"the reply to {cust}'s request was sent late."
            if claim_late
            else f"the reply to {cust}'s request was sent on time."
        )
    else:
        head = [
            f"{cust}提交了一份退款申请，公司于{fmt(received)}收到。",
            f"公司承诺在收到申请之日后{n}个自然日内回复（收到当天不计）。",
        ]
        claim = (
            f"给{cust}的回复发晚了。" if claim_late else f"给{cust}的回复是按时发出的。"
        )
    starts = {
        "ontime": due - timedelta(days=width + rng.randint(0, 3)),
        "late": due + timedelta(days=1 + rng.randint(0, 3)),
        "open": due - timedelta(days=rng.randint(0, width - 1)),
    }
    status_level = {
        "ontime": 0 if claim_late else 2,
        "late": 2 if claim_late else 0,
        "open": 1,
    }
    specs = []
    for status, start in starts.items():
        end = start + timedelta(days=width)
        if lang == "en":
            line = f"The reply went out sometime between {fmt(start)} and {fmt(end)}; the exact day was not logged."
        else:
            line = f"回复是在{fmt(start)}到{fmt(end)}之间发出的，具体日期没有记录。"
        facts = {
            "received": received.isoformat(),
            "n": n,
            "window": [start.isoformat(), end.isoformat()],
            "claim_late": claim_late,
        }
        specs.append((status_level[status], [*head, line], facts))
    return claim, specs


def _counting(
    rng: random.Random, lang: str, cust: str
) -> tuple[str, list[tuple[int, list[str], dict[str, Any]]]]:
    colors = COLORS[lang]
    color = rng.randrange(len(colors))
    threshold = rng.randint(8, 25)
    k = rng.randint(3, 7)
    n_other = rng.randint(3, 9)
    ids = sorted(rng.sample(range(100, 1000), (k - 3) + 4 + n_other + 2))
    rng.shuffle(ids)
    base_ids, switch_ids = ids[: k - 3], ids[k - 3 : k + 1]
    other_ids, unknown_ids = ids[k + 1 : k + 1 + n_other], ids[k + 1 + n_other :]
    weights = {i: rng.randint(threshold + 1, threshold + 15) for i in base_ids}
    low_weights = {i: rng.randint(max(1, threshold - 8), threshold) for i in switch_ids}
    others = {
        i: (
            rng.choice([c for c in range(len(colors)) if c != color]),
            rng.randint(3, threshold + 15),
        )
        for i in other_ids
    }
    if rng.random() < 0.5:
        others.update(
            {
                i: (color, rng.randint(max(1, threshold - 6), threshold))
                for i in other_ids[:2]
            }
        )
    plans = {0: 0, 1: rng.choice((1, 2)), 2: rng.choice((3, 4))}
    if lang == "en":
        claim = f"at least {k} {colors[color]} crates weighed more than {threshold} kg."
    else:
        claim = f"{colors[color]}色且重量超过{threshold}公斤的箱子至少有{k}个。"
    specs = []
    for level, switched in plans.items():
        lines = []
        items = {}
        for i in sorted(ids):
            if i in unknown_ids:
                lines.append(
                    f"Crate {i} was {colors[color]}; its weight label was unreadable."
                    if lang == "en"
                    else f"{i}号箱为{colors[color]}色，重量标签看不清。"
                )
                items[i] = (color, None)
                continue
            if i in base_ids:
                c, w = color, weights[i]
            elif i in switch_ids:
                on = switch_ids.index(i) < switched
                c, w = color, (
                    low_weights[i] + threshold + 1 - min(low_weights.values())
                    if on
                    else low_weights[i]
                )
            else:
                c, w = others[i]
            items[i] = (c, w)
            lines.append(
                f"Crate {i} was {colors[c]} and weighed {w} kg."
                if lang == "en"
                else f"{i}号箱为{colors[c]}色，重{w}公斤。"
            )
        known = sum(
            1
            for c, w in items.values()
            if w is not None and c == color and w > threshold
        )
        expected = 2 if known >= k else (0 if known + 2 < k else 1)
        if expected != level:
            raise AssertionError("evidence counting construction is inconsistent")
        head = [
            (
                f"{cust} checked the crates loaded for one order."
                if lang == "en"
                else f"{cust}核对了同一订单装车的箱子。"
            )
        ]
        specs.append(
            (
                level,
                head + lines,
                {
                    "items": {str(i): list(v) for i, v in items.items()},
                    "k": k,
                    "threshold": threshold,
                },
            )
        )
    return claim, specs


def _multihop(
    rng: random.Random, lang: str, cust: str
) -> tuple[str, list[tuple[int, list[str], dict[str, Any]]]]:
    teams = rng.sample(TEAMS[lang], 5)
    city, *elsewhere = rng.sample(CITIES[lang], 4)
    team_city = {
        teams[0]: city,
        teams[1]: city,
        teams[2]: elsewhere[0],
        teams[3]: elsewhere[1],
        teams[4]: rng.choice(elsewhere),
    }
    person = cust
    others = core.people(rng, lang, 3, exclude=[person])
    other_team = {p: rng.choice(teams) for p in others}
    pairs = {
        2: (teams[0], teams[1]),
        0: (teams[2], teams[3]),
        1: (teams[rng.choice((0, 1))], teams[rng.choice((2, 3, 4))]),
    }
    claim = f"{person} works in {city}." if lang == "en" else f"{person}在{city}工作。"
    order = rng.sample(teams, len(teams))
    specs = []
    for level in (0, 1, 2):
        a, b = pairs[level] if rng.random() < 0.5 else pairs[level][::-1]
        if lang == "en":
            lines = [
                f"The directory entry for {person} is smudged: it shows either the {a} team or the {b} team."
            ]
            lines += [f"{p} is on the {t} team." for p, t in other_team.items()]
            lines += [f"The {t} team is based in {team_city[t]}." for t in order]
        else:
            lines = [f"{person}的通讯录条目字迹模糊，只能看出是{a}或{b}中的一个。"]
            lines += [f"{p}隶属于{t}。" for p, t in other_team.items()]
            lines += [f"{t}设在{team_city[t]}。" for t in order]
        specs.append(
            (level, lines, {"candidates": [a, b], "team_city": team_city, "city": city})
        )
    return claim, specs


BUILDERS = {"calendar": _calendar, "counting": _counting, "multihop": _multihop}


def build_group(
    rng: random.Random, lang: str, task_type: str = "score", levels: int | None = 3
) -> Group:
    if levels not in (None, 3):
        raise ValueError("evidence status is defined for L=3 only")
    style = rng.choice(STYLES)
    cust = core.people(rng, lang, 1)[0]
    co = core.company(rng, lang, CO_SUFFIX[lang])
    claim, specs = BUILDERS[style](rng, lang, cust)
    names = [cust, co]
    pad = core.filler(rng, lang, core.words(" ".join(specs[0][1]), lang) + 10, names)
    sec = core.SECTION[lang]
    variants = []
    for level, lines, facts in sorted(specs, key=lambda s: s[0]):
        state = core.compose(
            TITLE[style][lang].format(co=co),
            [
                (sec["background"], pad[0]),
                (sec["record"], lines),
                (sec["notes"], pad[1]),
            ],
            lang,
        )
        variants.append(
            Variant(
                state,
                level,
                {"style": style, **facts},
                f"{style} evidence set to level {level}",
            )
        )
    group = Group(
        "score",
        ASK[lang].format(claim=claim),
        core.score_options(OPTIONS[lang]),
        variants,
        style,
        style,
        {"levels": 3},
    )
    core.check_group(group)
    return group


# ---------------------------------------------------------------- oracle 2


def _status(state: str, instructions: str, lang: str) -> int | None:
    d = core.DATE_RX[lang]
    received = re.search(rf"was received on ({d})|公司于({d})收到", state)
    if received:
        start = core.to_date(received.group(1) or received.group(2))
        rule = re.search(
            r"no later than (\d+) calendar days|之日后(\d+)个自然日", state
        )
        days = int(rule.group(1) or rule.group(2))
        window = re.search(rf"between ({d}) and ({d})|在({d})到({d})之间", state)
        w1, w2 = [core.to_date(g) for g in window.groups() if g]
        due = start + timedelta(days=days)
        late_claim = bool(re.search(r"was sent late|发晚了", instructions))
        if w2 <= due:
            return 0 if late_claim else 2
        if w1 > due:
            return 2 if late_claim else 0
        return 1
    if re.search(r"Crate \d+|\d+号箱", state):
        claim = re.search(
            r"at least (\d+) (\w+) crates weighed more than (\d+) kg|(.)色且重量超过(\d+)公斤的箱子至少有(\d+)个",
            instructions,
        )
        if claim.group(1):
            k, color, threshold = (
                int(claim.group(1)),
                claim.group(2),
                int(claim.group(3)),
            )
        else:
            color, threshold, k = (
                claim.group(4),
                int(claim.group(5)),
                int(claim.group(6)),
            )
        known = unknown = 0
        for m in re.finditer(
            r"Crate \d+ was (\w+) and weighed (\d+) kg|\d+号箱为(.)色，重(\d+)公斤",
            state,
        ):
            c, w = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            known += c == color and int(w) > threshold
        for m in re.finditer(
            r"Crate \d+ was (\w+); its weight label was unreadable|\d+号箱为(.)色，重量标签看不清",
            state,
        ):
            unknown += (m.group(1) or m.group(2)) == color
        return 2 if known >= k else (0 if known + unknown < k else 1)
    team = "|".join(re.escape(t) for t in TEAMS[lang])
    town = "|".join(re.escape(c) for c in CITIES[lang])
    entry = re.search(
        rf"shows either the ({team}) team or the ({team}) team\.|只能看出是({team})或({team})中的一个",
        state,
    )
    claim = re.search(rf"works in ({town})\.|在({town})工作。", instructions)
    if not entry or not claim:
        return None
    a, b = [g for g in entry.groups() if g]
    city = claim.group(1) or claim.group(2)
    pattern = (
        rf"The ({team}) team is based in ({town})\."
        if lang == "en"
        else rf"({team})设在({town})。"
    )
    located = dict(re.findall(pattern, state))
    hits = [located.get(a) == city, located.get(b) == city]
    return 2 if all(hits) else (0 if not any(hits) else 1)


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    status = _status(state, instructions, lang)
    if status is None:
        return None
    wanted = OPTIONS[lang][status]
    return core.match_option(options, lambda text: text == wanted)
