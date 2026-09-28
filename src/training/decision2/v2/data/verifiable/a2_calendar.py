"""a2_calendar: calendar/business-day/week offsets, month-end rules, deadline checks."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import A4Scenario, Group, Variant

FAMILY = "a2_calendar"
KINDS = ("cal", "bus", "week", "eom")
KIND_WEIGHTS = (3, 3, 2, 2)
N_RANGE = {"cal": (5, 60), "bus": (3, 20), "week": (1, 8), "eom": (0, 0)}

DOMAINS: dict[str, dict[str, dict[str, str]]] = {
    "claim": {
        "en": {
            "title": "Claim file — {co}",
            "intro": "{cust} filed a claim with {co} after a burst pipe damaged a storeroom.",
            "start": "The claim was received on {d}.",
            "done": "The written response went out on {d}.",
            "prefix": "Under {co}'s service standard, a written response is due",
            "day": "the day the claim is received",
            "month": "the month in which the claim is received",
            "dl": "the deadline for the written response",
            "noul": "Was the written response sent on time?",
        },
        "zh": {
            "title": "理赔档案——{co}",
            "intro": "{cust}因仓库水管爆裂，向{co}提出了理赔。",
            "start": "该申请于{d}受理。",
            "done": "书面答复于{d}发出。",
            "prefix": "按照{co}的服务标准，",
            "obl": "书面答复",
            "ev": "受理",
            "verb": "发出",
            "dl": "书面答复的截止日",
            "noul": "书面答复按时发出了吗？",
        },
    },
    "rental": {
        "en": {
            "title": "Rental record — {co}",
            "intro": "{cust} rented a portable generator from {co}.",
            "start": "The equipment was collected on {d}.",
            "done": "The equipment was brought back on {d}.",
            "prefix": "Under the rental agreement, the equipment must be returned",
            "day": "the day it is collected",
            "month": "the month in which it is collected",
            "dl": "the deadline for returning the equipment",
            "noul": "Was the equipment returned on time?",
        },
        "zh": {
            "title": "设备租赁记录——{co}",
            "intro": "{cust}从{co}租用了一台移动式发电机。",
            "start": "设备于{d}取走。",
            "done": "设备于{d}送回。",
            "prefix": "根据租赁协议，",
            "obl": "设备",
            "ev": "取走",
            "verb": "归还",
            "dl": "设备归还的截止日",
            "noul": "设备按时归还了吗？",
        },
    },
    "invoice": {
        "en": {
            "title": "Accounts payable memo — {co}",
            "intro": "{co} invoiced {cust} for repairs to a delivery van.",
            "start": "The invoice was raised on {d}.",
            "done": "Payment reached the supplier on {d}.",
            "prefix": "Under the supplier's payment terms, payment is due",
            "day": "the day the invoice is raised",
            "month": "the month in which the invoice is raised",
            "dl": "the payment deadline",
            "noul": "Was the invoice paid on time?",
        },
        "zh": {
            "title": "应付账款备忘——{co}",
            "intro": "{co}就一辆送货车的维修向{cust}开具了发票。",
            "start": "发票于{d}开具。",
            "done": "货款于{d}到账。",
            "prefix": "根据供应商的付款条件，",
            "obl": "货款",
            "ev": "开票",
            "verb": "付清",
            "dl": "付款截止日",
            "noul": "这张发票按时付清了吗？",
        },
    },
    "permit": {
        "en": {
            "title": "Planning file — {co}",
            "intro": "{co} applied for planning permission to extend its warehouse.",
            "start": "The application was submitted on {d}.",
            "done": "The planning office published its decision on {d}.",
            "prefix": "Under the local planning code, the planning office must decide",
            "day": "the day the application is submitted",
            "month": "the month in which the application is submitted",
            "dl": "the deadline for the planning decision",
            "noul": "Did the planning office decide on time?",
        },
        "zh": {
            "title": "规划审批档案——{co}",
            "intro": "{co}为仓库扩建工程递交了规划许可申请。",
            "start": "申请于{d}递交。",
            "done": "规划部门于{d}公布了决定。",
            "prefix": "根据当地规划条例，",
            "obl": "规划部门的决定",
            "ev": "递交申请",
            "verb": "作出",
            "dl": "审批决定的截止日",
            "noul": "规划部门按时作出决定了吗？",
        },
    },
    "appeal": {
        "en": {
            "title": "Appeal file — {co}",
            "intro": "{cust} disagreed with a fee charged by {co} and decided to appeal.",
            "start": "The decision letter is dated {d}.",
            "done": "The appeal was lodged on {d}.",
            "prefix": "Under the appeal rules, an appeal must be lodged",
            "day": "the date on the decision letter",
            "month": "the month of the date on the decision letter",
            "dl": "the deadline for lodging the appeal",
            "noul": "Was the appeal lodged on time?",
        },
        "zh": {
            "title": "申诉档案——{co}",
            "intro": "{cust}对{co}的一笔扣费决定有异议，准备提出申诉。",
            "start": "决定通知书的落款日期为{d}。",
            "done": "申诉于{d}提交。",
            "prefix": "根据申诉规则，",
            "obl": "申诉",
            "ev": "通知书落款",
            "verb": "提出",
            "dl": "申诉截止日",
            "noul": "申诉按时提出了吗？",
        },
    },
}
CO_SUFFIX = {
    "claim": {"en": ("Insurance", "Mutual Insurance"), "zh": ("保险", "财产保险")},
    "rental": {"en": ("Equipment Hire", "Rentals"), "zh": ("设备租赁", "机械租赁")},
    "invoice": {"en": ("Fleet Services", "Motor Works"), "zh": ("汽修", "车辆服务")},
    "permit": {"en": ("Holdings", "Property Group"), "zh": ("置业", "实业")},
    "appeal": {"en": ("Energy", "Telecom"), "zh": ("燃气", "通信")},
}
DATE_DISTRACTORS = {
    "en": (
        "The customer's account was opened on {d}.",
        "A site visit took place on {d}.",
        "The previous contract ended on {d}.",
        "An earlier, unrelated query was closed on {d}.",
    ),
    "zh": (
        "客户的账户开立于{d}。",
        "现场查看安排在{d}进行。",
        "上一份合同到{d}为止。",
        "此前一个无关的咨询已在{d}结案。",
    ),
}
HOLIDAY = {
    "en": "{d} was a public holiday, so it does not count as a business day.",
    "zh": "{d}为法定节假日，不算工作日。",
}


def deadline(kind: str, start: date, n: int, holiday: date | None) -> date:
    if kind == "cal":
        return start + timedelta(days=n)
    if kind == "week":
        return start + timedelta(days=7 * n)
    if kind == "eom":
        return core.add_months(date(start.year, start.month, 1), 2) - timedelta(days=1)
    day, count = start, 0
    while count < n:
        day += timedelta(days=1)
        if day.weekday() < 5 and day != holiday:
            count += 1
    return day


def _business(day: date, holiday: date | None) -> bool:
    return day.weekday() < 5 and day != holiday


def _bus_starts(end: date, n: int, holiday: date | None) -> list[date]:
    day, count = end, 0
    while True:
        if _business(day, holiday):
            count += 1
            if count == n:
                break
        day -= timedelta(days=1)
    starts, candidate = [], day - timedelta(days=1)
    while True:
        starts.append(candidate)
        if _business(candidate, holiday):
            return starts
        candidate -= timedelta(days=1)


def _rule(t: dict[str, str], lang: str, kind: str, n: int, co: str) -> str:
    if lang == "en":
        prefix = t["prefix"].format(co=co)
        if kind == "cal":
            return f"{prefix} no later than {n} calendar days after {t['day']}, not counting that day itself."
        if kind == "bus":
            return (
                f"{prefix} no later than the {core.ordinal_suffix(n)} business day after {t['day']}. "
                "Saturdays, Sundays and public holidays are not business days."
            )
        if kind == "week":
            return f"{prefix} no later than {n} week{'s' if n > 1 else ''} after {t['day']}."
        return (
            f"{prefix} no later than the last day of the month following {t['month']}."
        )
    head = t["prefix"].format(co=co) + t["obl"]
    ev, verb = t["ev"], t["verb"]
    if kind == "cal":
        return f"{head}应在{ev}之日后{n}个自然日内{verb}（{ev}当天不计，最后一天当天{verb}仍算按时）。"
    if kind == "bus":
        return f"{head}应在{ev}之日后的第{n}个工作日当天或之前{verb}；工作日为周一至周五，法定节假日不计。"
    if kind == "week":
        return f"{head}应在{ev}之日起{n}周内{verb}，即最晚为{n}周后的同一天。"
    return f"{head}最晚应在{ev}次月的最后一天{verb}。"


class _Scene:
    def __init__(self, rng: random.Random, lang: str) -> None:
        self.lang = lang
        self.domain = rng.choice(sorted(DOMAINS))
        self.t = DOMAINS[self.domain][lang]
        self.co = core.company(rng, lang, CO_SUFFIX[self.domain][lang])
        self.cust = core.people(rng, lang, 1)[0]
        self.style = rng.randrange(2)
        self.rule_first = rng.random() < 0.3
        self.extra = rng.sample(DATE_DISTRACTORS[lang], rng.randint(0, 2))
        self.extra_dates = [core.random_date(rng) for _ in self.extra]
        self.before: list[str] = []
        self.after: list[str] = []

    def fmt(self, value: date) -> str:
        return core.fmt_date(value, self.lang, self.style)

    def record(
        self,
        kind: str,
        n: int,
        start: date,
        holiday: date | None,
        done: date | None,
        show_wd: bool,
    ) -> list[str]:
        shown = (
            core.fmt_date_wd(start, self.lang, self.style)
            if show_wd
            else self.fmt(start)
        )
        lines = [
            self.t["start"].format(d=shown),
            _rule(self.t, self.lang, kind, n, self.co),
        ]
        if self.rule_first:
            lines.reverse()
        if kind == "bus" and holiday is not None:
            lines.append(HOLIDAY[self.lang].format(d=self.fmt(holiday)))
        if done is not None:
            lines.append(self.t["done"].format(d=self.fmt(done)))
        return lines

    def state(self, record: list[str]) -> str:
        head = self.t["intro"].format(cust=self.cust, co=self.co)
        dated = [s.format(d=self.fmt(d)) for s, d in zip(self.extra, self.extra_dates)]
        sec = core.SECTION[self.lang]
        return core.compose(
            self.t["title"].format(co=self.co),
            [
                (sec["background"], [head, *self.before, *dated]),
                (sec["record"], record),
                (sec["notes"], self.after),
            ],
            self.lang,
        )

    def pad(self, rng: random.Random, sample: list[str], avoid: Sequence[str]) -> None:
        self.before, self.after = core.filler(
            rng,
            self.lang,
            core.words(" ".join(sample), self.lang) + 25,
            [self.cust, self.co],
            avoid,
        )


def _holiday_for(rng: random.Random, start: date) -> date:
    day = start + timedelta(days=rng.randint(1, 26))
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _facts(scene: _Scene, **values: Any) -> dict[str, Any]:
    out = {"domain": scene.domain}
    for key, value in values.items():
        out[key] = value.isoformat() if isinstance(value, date) else value
    return out


def _finish(
    rng: random.Random,
    scene: _Scene,
    task: str,
    instructions: str,
    descriptions: list[str],
    specs: list[tuple[int, list[str], dict[str, Any], str]],
    template: str,
    subtype: str,
) -> Group | None:
    options = (
        core.choice_options(descriptions)
        if task == "choice"
        else core.noul_options(scene.lang)
    )
    avoid = descriptions if task == "choice" else []
    scene.pad(rng, specs[0][1], avoid)
    variants = [
        Variant(scene.state(record), label, facts, edit)
        for label, record, facts, edit in specs
    ]
    if task == "choice" and not all(
        core.presence_ok(v.state, options) for v in variants
    ):
        return None
    return Group(task, instructions, options, variants, template, subtype)


def _choice_date(rng: random.Random, lang: str) -> Group | None:
    scene = _Scene(rng, lang)
    kind = rng.choices(KINDS, weights=KIND_WEIGHTS)[0]
    k = rng.choice((3, 4)) if kind == "eom" else rng.choice((3, 4, 4, 5))
    base = core.random_date(rng)
    holiday = _holiday_for(rng, base) if kind == "bus" else None
    show_wd = kind == "bus" or rng.random() < 0.3
    lo, hi = N_RANGE[kind]
    plans: list[tuple[date, int, str]] = []
    if kind != "eom" and rng.random() < 0.5:
        for n in sorted(rng.sample(range(lo, hi + 1), k)):
            plans.append((base, n, f"period set to {n}"))
        edit_kind = "period"
    else:
        n = rng.randint(lo, hi)
        edit_kind = "start"
        if kind == "eom":
            offsets = sorted(rng.sample(range(0, 8), k))
            for off in offsets:
                month = core.add_months(date(base.year, base.month, 1), off)
                start = month + timedelta(
                    days=rng.randrange(core.month_end(month.year, month.month).day)
                )
                plans.append((start, 0, f"start moved to {start.isoformat()}"))
        else:
            ends: list[date] = []
            day = deadline(kind, base, n, holiday)
            while len(ends) < k:
                day += timedelta(days=rng.randint(1, 6))
                if kind != "bus" or _business(day, holiday):
                    ends.append(day)
            for end in ends:
                if kind == "bus":
                    start = rng.choice(_bus_starts(end, n, holiday))
                else:
                    start = end - timedelta(days=n if kind == "cal" else 7 * n)
                plans.append((start, n, f"start moved to {start.isoformat()}"))
    ends = [deadline(kind, s, n, holiday) for s, n, _ in plans]
    if len(set(ends)) != k:
        return None
    order = rng.sample(range(k), k)
    descriptions = [scene.fmt(ends[i]) for i in order]
    specs = []
    for label, i in enumerate(order):
        start, n, edit = plans[i]
        record = scene.record(kind, n, start, holiday, None, show_wd)
        facts = _facts(
            scene,
            kind=kind,
            n=n,
            start=start,
            holiday=holiday,
            deadline=ends[i],
            question="date",
        )
        specs.append((label, record, facts, edit))
    specs.sort(key=lambda spec: spec[0])
    question = {
        "en": f"Which date is {scene.t['dl']}, that is, the last day that still counts as on time?",
        "zh": f"{scene.t['dl']}是哪一天（即仍算按时的最后一天）？",
    }[lang]
    return _finish(
        rng,
        scene,
        "choice",
        question,
        descriptions,
        specs,
        f"{scene.domain}_date_{kind}_{edit_kind}",
        kind,
    )


def _choice_weekday(rng: random.Random, lang: str) -> Group | None:
    scene = _Scene(rng, lang)
    kind = rng.choice(("cal", "bus", "eom"))
    k = rng.choice((3, 4))
    base = core.random_date(rng)
    holiday = _holiday_for(rng, base) if kind == "bus" else None
    plans: list[tuple[date, int, int]] = []
    if kind == "eom":
        targets = rng.sample(range(7), k)
        for target in targets:
            for off in rng.sample(range(0, 24), 24):
                month = core.add_months(date(base.year, base.month, 1), off)
                if deadline("eom", month, 0, None).weekday() != target:
                    continue
                days = [
                    month + timedelta(days=i)
                    for i in range(core.month_end(month.year, month.month).day)
                    if (month + timedelta(days=i)).weekday() not in targets
                ]
                plans.append((rng.choice(days), 0, target))
                break
    else:
        allowed = [
            d for d in range(7) if d != base.weekday() and (kind == "cal" or d < 5)
        ]
        targets = rng.sample(allowed, min(k, len(allowed)))
        lo, hi = N_RANGE[kind]
        used: set[int] = set()
        for target in targets:
            choices = [
                n
                for n in range(lo, hi + 1)
                if n not in used
                and deadline(kind, base, n, holiday).weekday() == target
            ]
            if not choices:
                return None
            n = rng.choice(choices)
            used.add(n)
            plans.append((base, n, target))
    if len(plans) != k:
        return None
    wd = core.WEEKDAYS[lang]
    order = rng.sample(range(k), k)
    descriptions = [wd[plans[i][2]] for i in order]
    specs = []
    for label, i in enumerate(order):
        start, n, _target = plans[i]
        end = deadline(kind, start, n, holiday)
        record = scene.record(kind, n, start, holiday, None, True)
        facts = _facts(
            scene,
            kind=kind,
            n=n,
            start=start,
            holiday=holiday,
            deadline=end,
            question="weekday",
        )
        edit = (
            f"start moved to {start.isoformat()}"
            if kind == "eom"
            else f"period set to {n}"
        )
        specs.append((label, record, facts, edit))
    question = {
        "en": f"On which day of the week does {scene.t['dl']} fall?",
        "zh": f"{scene.t['dl']}是星期几？",
    }[lang]
    return _finish(
        rng,
        scene,
        "choice",
        question,
        descriptions,
        specs,
        f"{scene.domain}_weekday_{kind}",
        kind,
    )


def _noul(rng: random.Random, lang: str) -> Group | None:
    scene = _Scene(rng, lang)
    kind = rng.choices(KINDS, weights=KIND_WEIGHTS)[0]
    start = core.random_date(rng)
    holiday = _holiday_for(rng, start) if kind == "bus" else None
    show_wd = kind == "bus" or rng.random() < 0.3
    lo, hi = N_RANGE[kind]
    n = rng.randint(lo, hi)
    end = deadline(kind, start, n, holiday)
    if kind != "eom" and rng.random() < 0.3:
        done = end - timedelta(days=0 if rng.random() < 0.5 else rng.randint(1, 3))
        n_false = n - 1
        while n_false >= 1 and deadline(kind, start, n_false, holiday) >= done:
            n_false -= 1
        if n_false < 1:
            return None
        plans = [
            (0, n_false, done, f"period set to {n_false}"),
            (1, n, done, f"period set to {n}"),
        ]
        edit_kind = "period"
    else:
        on_time = end - timedelta(days=0 if rng.random() < 0.4 else rng.randint(1, 4))
        late = end + timedelta(days=1 if rng.random() < 0.4 else rng.randint(2, 5))
        plans = [
            (0, n, late, f"completion on {late.isoformat()}"),
            (1, n, on_time, f"completion on {on_time.isoformat()}"),
        ]
        edit_kind = "done"
    specs = []
    for label, n_value, done, edit in plans:
        record = scene.record(kind, n_value, start, holiday, done, show_wd)
        facts = _facts(
            scene,
            kind=kind,
            n=n_value,
            start=start,
            holiday=holiday,
            deadline=deadline(kind, start, n_value, holiday),
            done=done,
            question="on_time",
        )
        specs.append((label, record, facts, edit))
    return _finish(
        rng,
        scene,
        "noul",
        scene.t["noul"],
        [],
        specs,
        f"{scene.domain}_ontime_{kind}_{edit_kind}",
        kind,
    )


def build_group(
    rng: random.Random, lang: str, task_type: str, levels: int | None = None
) -> Group:
    for _ in range(200):
        if task_type == "noul":
            group = _noul(rng, lang)
        else:
            group = (
                _choice_weekday(rng, lang)
                if rng.random() < 0.3
                else _choice_date(rng, lang)
            )
        if group is not None:
            core.check_group(group)
            return group
    raise RuntimeError("calendar group construction failed")


# ---------------------------------------------------------------- A4


def a4_scenario(rng: random.Random, lang: str) -> A4Scenario:
    scene = _Scene(rng, lang)
    kind = rng.choices(KINDS, weights=KIND_WEIGHTS)[0]
    qtype = "weekday" if kind in ("cal", "eom") and rng.random() < 0.3 else "date"
    start = core.random_date(rng)
    holiday = _holiday_for(rng, start) if kind == "bus" else None
    lo, hi = N_RANGE[kind]
    n = rng.randint(lo, hi)
    end = deadline(kind, start, n, holiday)
    if end.weekday() == start.weekday():
        qtype = "date"
    show_wd = kind == "bus" or qtype == "weekday" or rng.random() < 0.3
    record = scene.record(kind, n, start, holiday, None, show_wd)
    scene.pad(rng, record, [])
    state = scene.state(record)
    facts = _facts(
        scene,
        kind=kind,
        n=n,
        start=start,
        holiday=holiday,
        deadline=end,
        question=qtype,
    )
    dl = scene.t["dl"]
    if qtype == "weekday":
        wd = core.WEEKDAYS[lang]
        g = end.weekday()
        below = [wd[(g - i) % 7] for i in (1, 2, 3)]
        above = [wd[(g + i) % 7] for i in (1, 2, 3)]
        question = {
            "en": f"On which day of the week does {dl} fall?",
            "zh": f"{dl}是星期几？",
        }[lang]
        template = {"en": f"Does {dl} fall on a {{v}}?", "zh": f"{dl}在{{v}}吗？"}[lang]
        return A4Scenario(
            state,
            facts,
            kind,
            f"{scene.domain}_weekday_{kind}",
            question,
            wd[g],
            below,
            above,
            below,
            above,
            True,
            False,
            template.format(v=wd[g]),
            template.format(v=rng.choice((below[0], above[0]))),
            [template.format(v=v) for v in (below[2], above[2])],
        )
    if kind == "eom":
        space = [
            core.month_end(*_ym(core.add_months(date(end.year, end.month, 1), off)))
            for off in range(-8, 9)
            if off
        ]
        below = [d for d in space if d < end][::-1]
        above = [d for d in space if d > end]
        near_below, near_above = below[:3], above[:3]
        far = [d for d in space if abs((d - end).days) > 100]
    else:
        window = [end + timedelta(days=i) for i in range(-45, 46) if i]
        if kind == "bus":
            window = [d for d in window if _business(d, holiday)]
            prev = [d for d in window if d < end][::-1]
            nxt = [d for d in window if d > end]
            near_below = _dedupe(
                [
                    prev[0],
                    prev[1],
                    _to_business(end - timedelta(days=7), holiday, -1),
                    prev[2],
                ]
            )
            near_above = _dedupe(
                [
                    nxt[0],
                    nxt[1],
                    _to_business(end + timedelta(days=7), holiday, 1),
                    nxt[2],
                ]
            )
        else:
            near_below = _dedupe(
                [
                    end - timedelta(days=1),
                    core.add_months(end, -1),
                    end - timedelta(days=7),
                    end - timedelta(days=2),
                ]
            )
            near_above = _dedupe(
                [
                    end + timedelta(days=1),
                    core.add_months(end, 1),
                    end + timedelta(days=7),
                    end + timedelta(days=2),
                ]
            )
        below = [d for d in window if d < end]
        above = [d for d in window if d > end]
        far = [d for d in window if abs((d - end).days) >= 10]
    fmt = scene.fmt
    question = {
        "en": f"Which date is {dl}, that is, the last day that still counts as on time?",
        "zh": f"{dl}是哪一天（即仍算按时的最后一天）？",
    }[lang]
    template = {"en": f"Is {dl} {{v}}?", "zh": f"{dl}为{{v}}吗？"}[lang]
    near_noul = rng.choice((near_below[0], near_above[0]))
    return A4Scenario(
        state,
        facts,
        kind,
        f"{scene.domain}_date_{kind}",
        question,
        fmt(end),
        [fmt(d) for d in near_below],
        [fmt(d) for d in near_above],
        [fmt(d) for d in below],
        [fmt(d) for d in above],
        True,
        True,
        template.format(v=fmt(end)),
        template.format(v=fmt(near_noul)),
        [template.format(v=fmt(d)) for d in far],
    )


def _ym(value: date) -> tuple[int, int]:
    return value.year, value.month


def _dedupe(values: list[date]) -> list[date]:
    out: list[date] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _to_business(day: date, holiday: date | None, step: int) -> date:
    while not _business(day, holiday):
        day += timedelta(days=step)
    return day


V2_FIRST, V2_LAST = date(2025, 3, 3), date(2027, 3, 1)
V2_WIDTH = {"cal": 24, "week": 24, "bus": 16, "eom": 6}


def _v2_space(kind: str) -> list[date]:
    if kind == "eom":
        return [core.month_end(2023 + m // 12, m % 12 + 1) for m in range(96)]
    days = [V2_FIRST + timedelta(days=i) for i in range((V2_LAST - V2_FIRST).days)]
    return [d for d in days if d.weekday() < 5] if kind == "bus" else days


def a4v2_plan(rng: random.Random, lang: str, turn: int) -> core.A4v2Plan:
    """Deadline options first (rank-balanced value windows), then a record that yields the gold."""
    scene = _Scene(rng, lang)
    kind = rng.choices(KINDS, weights=KIND_WEIGHTS)[0]
    space = _v2_space(kind)
    sets = core.rank_windows(rng, 0, len(space) - 1, 4, V2_WIDTH[kind])
    every = {space[i] for gold, near, rand in sets for i in (gold, *near, *rand)}
    dl = scene.t["dl"]
    question = {
        "en": f"Which date is {dl}, that is, the last day that still counts as on time?",
        "zh": f"{dl}是哪一天（即仍算按时的最后一天）？",
    }[lang]
    template = {"en": f"Is {dl} {{v}}?", "zh": f"{dl}为{{v}}吗？"}[lang]

    def render(end: date) -> core.A4v2Render | None:
        shown = [scene.fmt(d) for d in every]
        for _ in range(20):
            lo, hi = N_RANGE[kind]
            n = rng.randint(lo, hi)
            holiday = None
            if kind == "bus":
                pool = [end - timedelta(days=i) for i in range(1, 26)]
                pool = [d for d in pool if d.weekday() < 5 and d not in every]
                holiday = rng.choice(pool)
                start = rng.choice(_bus_starts(end, n, holiday))
            elif kind == "eom":
                first = date(end.year, end.month, 1) - timedelta(days=1)
                start = first - timedelta(days=rng.randrange(first.day))
            else:
                start = end - timedelta(days=n if kind == "cal" else 7 * n)
            if deadline(kind, start, n, holiday) != end:
                continue
            record = scene.record(
                kind, n, start, holiday, None, kind == "bus" or rng.random() < 0.3
            )
            scene.pad(rng, record, shown)
            state = scene.state(record)
            if any(core.mentions(state, text) for text in shown):
                continue
            facts = _facts(
                scene,
                kind=kind,
                n=n,
                start=start,
                holiday=holiday,
                deadline=end,
                question="date",
            )
            return core.A4v2Render(
                state,
                facts,
                kind,
                f"{scene.domain}_date_{kind}_v2",
                question,
                lambda v: template.format(v=v),
            )
        return None

    alternatives = [
        core.A4v2Alternative(
            r,
            scene.fmt(space[gold]),
            [scene.fmt(space[i]) for i in near],
            [scene.fmt(space[i]) for i in rand],
            lambda end=space[gold]: render(end),
        )
        for r, (gold, near, rand) in enumerate(sets)
    ]
    return core.A4v2Plan(True, alternatives)


# ---------------------------------------------------------------- oracle 2

_START = {
    "en": r"(?:claim was received on|equipment was collected on|invoice was raised on|"
    r"application was submitted on|decision letter is dated) (?:(?:{wd}), )?({d})",
    "zh": r"(?:于({d})(?:（星期.）)?(?:受理|取走|开具|递交)|落款日期为({d}))",
}
_DONE = {
    "en": r"(?:written response went out on|equipment was brought back on|Payment reached the supplier on|"
    r"published its decision on|appeal was lodged on) ({d})",
    "zh": r"于({d})(?:发出|送回|到账|公布了决定|提交)",
}
_HOL = {"en": r"({d}) was a public holiday", "zh": r"({d})为法定节假日"}


def _find(pattern: str, text: str, lang: str) -> list[date]:
    rx = pattern.format(d=core.DATE_RX[lang], wd=core.WEEKDAY_RX["en"])
    found = []
    for match in re.finditer(rx, text):
        found.extend(core.to_date(g) for g in match.groups() if g)
    return found


def _recompute(text: str, lang: str) -> date | None:
    starts = _find(_START[lang], text, lang)
    holidays = _find(_HOL[lang], text, lang)
    if len(starts) != 1 or len(holidays) > 1:
        return None
    start, holiday = starts[0], holidays[0] if holidays else None
    if lang == "en":
        cal = re.search(r"no later than (\d+) calendar days after", text)
        bus = re.search(
            r"no later than the (\d+)(?:st|nd|rd|th) business day after",  # codespell:ignore nd
            text,  # codespell:ignore nd
        )
        week = re.search(r"no later than (\d+) weeks? after", text)
        eom = "no later than the last day of the month following" in text
    else:
        cal = re.search(r"之日后(\d+)个自然日内", text)
        bus = re.search(r"之日后的第(\d+)个工作日", text)
        week = re.search(r"之日起(\d+)周内", text)
        eom = "次月的最后一天" in text
    if sum(bool(x) for x in (cal, bus, week, eom)) != 1:
        return None
    if cal:
        return date.fromordinal(start.toordinal() + int(cal[1]))
    if week:
        return date.fromordinal(start.toordinal() + 7 * int(week[1]))
    if eom:
        first_next = (start.replace(day=1) + timedelta(days=32)).replace(day=1)
        return (first_next + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    remaining, day = int(bus[1]), start
    while remaining:
        day = date.fromordinal(day.toordinal() + 1)
        if day.isoweekday() <= 5 and day != holiday:
            remaining -= 1
    return day


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    end = _recompute(state, lang)
    if end is None:
        return None
    wd_names = core.WEEKDAYS[lang]
    if core.is_noul(options):
        in_q = re.findall(core.DATE_RX[lang], instructions)
        if in_q:
            return int(core.to_date(in_q[0]) == end)
        named = [i for i, name in enumerate(wd_names) if name in instructions]
        if len(named) == 1:
            return int(named[0] == end.weekday())
        done = _find(_DONE[lang], state, lang)
        return int(done[0] <= end) if len(done) == 1 else None
    if all(option["description"] in wd_names for option in options):
        return core.match_option(
            options, lambda text: wd_names.index(text) == end.weekday()
        )
    return core.match_option(options, lambda text: core.to_date(text) == end)
