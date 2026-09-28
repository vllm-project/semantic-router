"""a2_multihop: follow person -> (project) -> team -> (building) -> floor/city chains."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import A4Scenario, Group, Variant

FAMILY = "a2_multihop"
HOPS = ("floor2", "city2", "city3", "floor3")

LEX = {
    "en": {
        "teams": (
            "Payments",
            "Search",
            "Logistics",
            "Billing",
            "Research",
            "Onboarding",
            "Security",
            "Mobile",
            "Data Platform",
            "Design",
            "Procurement",
            "Field Service",
        ),
        "buildings": (
            "Alder House",
            "Birch Hall",
            "Cedar Court",
            "Linden Tower",
            "Maple Yard",
            "Rowan Works",
        ),
        "cities": (
            "Lisbon",
            "Tallinn",
            "Porto",
            "Ghent",
            "Lyon",
            "Krakow",
            "Osaka",
            "Monterrey",
            "Nairobi",
            "Hobart",
            "Valparaiso",
            "Tbilisi",
        ),
        "projects": (
            "Aurora",
            "Beacon",
            "Cascade",
            "Driftwood",
            "Ember",
            "Fjord",
            "Granite",
            "Harbor",
        ),
    },
    "zh": {
        "teams": (
            "支付组",
            "搜索组",
            "物流组",
            "结算组",
            "研究组",
            "入职组",
            "安全组",
            "移动端组",
            "数据平台组",
            "设计组",
            "采购组",
            "外勤组",
        ),
        "buildings": ("枫林楼", "松涛楼", "银杏楼", "翠柏楼", "樟园楼", "梧桐楼"),
        "cities": (
            "杭州",
            "成都",
            "苏州",
            "厦门",
            "青岛",
            "武汉",
            "西安",
            "长沙",
            "大连",
            "昆明",
            "南昌",
            "贵阳",
        ),
        "projects": (
            "北极光",
            "灯塔",
            "瀑布",
            "浮木",
            "余烬",
            "峡湾",
            "花岗岩",
            "港湾",
        ),
    },
}
T = {
    "en": {
        "member": (
            "{p} is on the {t} team.",
            "{p} is a member of the {t} team.",
            "{p} works in the {t} team.",
        ),
        "floor": (
            "The {t} team sits on the {f}.",
            "The {t} team is on the {f}.",
            "You will find the {t} team on the {f}.",
        ),
        "city": (
            "The {t} team is based in {c}.",
            "The {t} team works from the {c} office.",
        ),
        "building": (
            "The {t} team works out of {b}.",
            "The {t} team is housed in {b}.",
        ),
        "bcity": ("{b} is in {c}.", "{b} stands in {c}."),
        "assign": ("{p} is assigned to Project {j}.",),
        "run": ("Project {j} is run by the {t} team.",),
        "lure_city": "{p} often travels to {c} to see clients.",
        "lure_floor": "{p} often books the meeting room on the {f}.",
        "title": "Staff directory notes — {co}",
        "intro": "The notes below come from {co}'s internal directory.",
        "q_floor": "On which floor does {p} work?",
        "q_city": "In which city does {p} work?",
        "n_floor": "Does {p} work on the {v}?",
        "n_city": "Does {p} work in {v}?",
    },
    "zh": {
        "member": ("{p}隶属于{t}。", "{p}是{t}的成员。", "{p}目前在{t}。"),
        "floor": ("{t}在{f}办公。", "{t}的办公区位于{f}。", "{t}在{f}。"),
        "city": ("{t}设在{c}。", "{t}在{c}办公。"),
        "building": ("{t}在{b}办公。", "{t}的办公地点是{b}。"),
        "bcity": ("{b}位于{c}。", "{b}在{c}。"),
        "assign": ("{p}被分配到{j}项目。",),
        "run": ("{j}项目由{t}负责。",),
        "lure_city": "{p}经常去{c}拜访客户。",
        "lure_floor": "{p}经常预订{f}的会议室。",
        "title": "员工通讯录备注——{co}",
        "intro": "以下内容摘自{co}的内部通讯录。",
        "q_floor": "{p}在几楼办公？",
        "q_city": "{p}在哪座城市工作？",
        "n_floor": "{p}在{v}办公吗？",
        "n_city": "{p}在{v}工作吗？",
    },
}
CO_SUFFIX = {
    "en": ("Analytics", "Systems", "Group", "Labs"),
    "zh": ("科技", "数据", "集团", "网络"),
}


def floor_text(value: int, lang: str) -> str:
    return f"{core.ordinal_suffix(value)} floor" if lang == "en" else f"{value}楼"


class _Directory:
    """Fixed entities of one scenario; ``target_team`` is the only operative fact."""

    def __init__(
        self, rng: random.Random, lang: str, hop: str, n_values: int, extra_values: int
    ) -> None:
        self.lang, self.hop = lang, hop
        lex = LEX[lang]
        self.co = core.company(rng, lang, CO_SUFFIX[lang])
        n_teams = n_values + extra_values
        self.teams = rng.sample(lex["teams"], n_teams)
        kind = "floor" if hop.startswith("floor") else "city"
        if kind == "floor":
            values = [
                floor_text(f, lang) for f in rng.sample(range(2, 16), n_teams + 1)
            ]
        else:
            values = rng.sample(lex["cities"], n_teams + 1)
        self.kind = kind
        self.lure_value = values.pop()
        self.value_of: dict[str, str] = {}
        self.building_of: dict[str, str] = {}
        self.city_of: dict[str, str] = {}
        if hop == "city3":
            buildings = rng.sample(
                lex["buildings"], min(len(lex["buildings"]), n_teams)
            )
            for team, value, building in zip(self.teams, values, buildings):
                self.building_of[team] = building
                self.city_of[building] = value
                self.value_of[team] = value
        else:
            self.value_of = dict(zip(self.teams, values))
        self.projects: dict[str, str] = {}
        if hop == "floor3":
            names = rng.sample(lex["projects"], n_teams)
            self.projects = dict(zip(names, self.teams))
        self.target = core.people(rng, lang, 1)[0]
        self.others = core.people(rng, lang, rng.randint(3, 7), exclude=[self.target])
        self.other_team = {p: rng.choice(self.teams) for p in self.others}
        self.other_project = (
            {p: rng.choice(sorted(self.projects)) for p in self.others}
            if self.projects
            else {}
        )
        self.styles = {
            key: rng.randrange(3)
            for key in ("member", "floor", "city", "building", "bcity")
        }
        self.lure = rng.random() < 0.6
        self.shuffle_seed = rng.getrandbits(64)

    def _pick(self, key: str) -> str:
        options = T[self.lang][key]
        return options[self.styles.get(key, 0) % len(options)]

    def sentences(self, target_team: str) -> list[str]:
        out: list[str] = []
        people_teams = [(self.target, target_team), *self.other_team.items()]
        if self.projects:
            project_of_team = {team: name for name, team in self.projects.items()}
            out.append(
                self._pick("assign").format(
                    p=self.target, j=project_of_team[target_team]
                )
            )
            out += [
                self._pick("assign").format(p=p, j=j)
                for p, j in self.other_project.items()
            ]
            out += [
                self._pick("run").format(j=j, t=t) for j, t in self.projects.items()
            ]
        else:
            out += [self._pick("member").format(p=p, t=t) for p, t in people_teams]
        for team in self.teams:
            if self.hop == "city3":
                out.append(
                    self._pick("building").format(t=team, b=self.building_of[team])
                )
            elif self.kind == "floor":
                out.append(self._pick("floor").format(t=team, f=self.value_of[team]))
            else:
                out.append(self._pick("city").format(t=team, c=self.value_of[team]))
        if self.hop == "city3":
            out += [
                self._pick("bcity").format(b=b, c=c) for b, c in self.city_of.items()
            ]
        if self.lure:
            key = "lure_floor" if self.kind == "floor" else "lure_city"
            out.append(
                T[self.lang][key].format(
                    p=self.target, f=self.lure_value, c=self.lure_value
                )
            )
        random.Random(self.shuffle_seed).shuffle(out)
        return out

    def question(self) -> str:
        return T[self.lang]["q_floor" if self.kind == "floor" else "q_city"].format(
            p=self.target
        )

    def proposition(self, value: str) -> str:
        return T[self.lang]["n_floor" if self.kind == "floor" else "n_city"].format(
            p=self.target, v=value
        )

    def state(self, target_team: str, pad: tuple[list[str], list[str]]) -> str:
        t = T[self.lang]
        sec = core.SECTION[self.lang]
        return core.compose(
            t["title"].format(co=self.co),
            [
                (sec["background"], [t["intro"].format(co=self.co), *pad[0]]),
                (sec["record"], self.sentences(target_team)),
                (sec["notes"], pad[1]),
            ],
            self.lang,
        )

    def names(self) -> list[str]:
        return [self.target, *self.others, self.co]


def _group(rng: random.Random, lang: str, task: str) -> Group | None:
    hop = rng.choice(HOPS)
    n_values = rng.choice((3, 4, 5)) if task == "choice" else rng.choice((2, 3, 4))
    d = _Directory(rng, lang, hop, n_values, rng.randint(0, 1))
    option_teams = d.teams[:n_values] if task == "choice" else rng.sample(d.teams, 2)
    base_words = core.words(" ".join(d.sentences(option_teams[0])), lang) + 20
    pad = core.filler(rng, lang, base_words, d.names())
    if task == "choice":
        order = rng.sample(option_teams, len(option_teams))
        options = core.choice_options([d.value_of[team] for team in order])
        plan = [(label, team) for label, team in enumerate(order)]
        instructions = d.question()
    else:
        options = core.noul_options(lang)
        true_team, false_team = option_teams
        plan = [(0, false_team), (1, true_team)]
        instructions = d.proposition(d.value_of[true_team])
    variants = []
    for label, team in plan:
        facts = {
            "hop": hop,
            "target": d.target,
            "target_team": team,
            "value_of": d.value_of,
            "building_of": d.building_of,
            "city_of": d.city_of,
            "projects": d.projects,
            "others": d.other_team,
            "answer": d.value_of[team],
        }
        variants.append(
            Variant(d.state(team, pad), label, facts, f"{d.target} placed in {team}")
        )
    if task == "choice" and not all(
        core.presence_ok(v.state, options) for v in variants
    ):
        return None
    return Group(task, instructions, options, variants, f"{hop}_{task}", hop)


def build_group(
    rng: random.Random, lang: str, task_type: str, levels: int | None = None
) -> Group:
    for _ in range(200):
        group = _group(rng, lang, task_type)
        if group is not None:
            core.check_group(group)
            return group
    raise RuntimeError("multihop group construction failed")


def a4_scenario(rng: random.Random, lang: str) -> A4Scenario:
    hop = rng.choice(HOPS)
    d = _Directory(rng, lang, hop, 4, rng.randint(0, 1))
    d.lure = True
    team = d.teams[0]
    base_words = core.words(" ".join(d.sentences(team)), lang) + 20
    state = d.state(team, core.filler(rng, lang, base_words, d.names()))
    gold = d.value_of[team]
    present = [d.lure_value] + [d.value_of[t] for t in d.teams[1:]]
    if d.kind == "floor":
        space = [floor_text(f, lang) for f in range(1, 21)]
    else:
        space = list(LEX[lang]["cities"])
    pool = [v for v in space if v != gold]
    absent = [v for v in pool if v not in present]
    facts = {
        "hop": hop,
        "target": d.target,
        "target_team": team,
        "value_of": d.value_of,
        "building_of": d.building_of,
        "city_of": d.city_of,
        "projects": d.projects,
        "others": d.other_team,
        "answer": gold,
    }
    return A4Scenario(
        state,
        facts,
        hop,
        f"{hop}_a4",
        d.question(),
        gold,
        [],
        present,
        [],
        pool,
        False,
        False,
        d.proposition(gold),
        d.proposition(present[0]),
        [d.proposition(v) for v in absent],
    )


V2_LURES = {
    "en": {
        "city": (
            "{p} often travels to {v} to see clients.",
            "{p} grew up in {v}.",
            "{p}'s mentor is based in {v}.",
        ),
        "floor": (
            "{p} often books the meeting room on the {v}.",
            "{p}'s mentor sits on the {v}.",
            "{p} runs a weekly workshop on the {v}.",
        ),
    },
    "zh": {
        "city": ("{p}经常去{v}拜访客户。", "{p}是在{v}长大的。", "{p}的导师常驻{v}。"),
        "floor": (
            "{p}经常预订{v}的会议室。",
            "{p}的导师在{v}办公。",
            "{p}每周在{v}主持一次培训。",
        ),
    },
}


def a4v2_plan(rng: random.Random, lang: str, turn: int) -> core.A4v2Plan:
    """Eight teams plus three person-linked lures; each candidate value appears exactly once."""
    hop = rng.choice(HOPS)
    lex, t = LEX[lang], T[lang]
    kind = "floor" if hop.startswith("floor") else "city"
    co = core.company(rng, lang, CO_SUFFIX[lang])
    teams = rng.sample(lex["teams"], 8)
    if kind == "floor":
        values = [floor_text(f, lang) for f in rng.sample(range(2, 21), 11)]
    else:
        values = rng.sample(lex["cities"], 11)
    lures, pool = values[:3], values[3:]
    building_of: dict[str, str] = {}
    city_of: dict[str, str] = {}
    if hop == "city3":
        buildings = list(lex["buildings"])
        rng.shuffle(buildings)
        city_of = dict(zip(buildings, pool))
        building_of = {
            team: buildings[i] if i < len(buildings) else rng.choice(buildings)
            for i, team in enumerate(teams)
        }
        value_of = {team: city_of[building_of[team]] for team in teams}
    else:
        value_of = dict(zip(teams, pool))
    projects = (
        dict(zip(rng.sample(lex["projects"], 8), teams)) if hop == "floor3" else {}
    )
    target = core.people(rng, lang, 1)[0]
    others = core.people(rng, lang, rng.randint(3, 6), exclude=[target])
    other_team = {p: rng.choice(teams) for p in others}
    other_project = (
        {p: rng.choice(sorted(projects)) for p in others} if projects else {}
    )
    lure_lines = [
        template.format(p=target, v=v)
        for template, v in zip(V2_LURES[lang][kind], lures)
    ]
    order_seed = rng.getrandbits(64)
    style = rng.randrange(2)
    ask = t["q_floor" if kind == "floor" else "q_city"].format(p=target)
    prop = t["n_floor" if kind == "floor" else "n_city"]

    def render(team: str) -> core.A4v2Render:
        lines: list[str] = []
        if projects:
            project_of_team = {v: k for k, v in projects.items()}
            lines.append(t["assign"][0].format(p=target, j=project_of_team[team]))
            lines += [t["assign"][0].format(p=p, j=j) for p, j in other_project.items()]
            lines += [t["run"][0].format(j=j, t=tm) for j, tm in projects.items()]
        else:
            lines += [
                t["member"][style].format(p=p, t=tm)
                for p, tm in [(target, team), *other_team.items()]
            ]
        for tm in teams:
            if hop == "city3":
                lines.append(t["building"][style].format(t=tm, b=building_of[tm]))
            elif kind == "floor":
                lines.append(t["floor"][style].format(t=tm, f=value_of[tm]))
            else:
                lines.append(t["city"][style].format(t=tm, c=value_of[tm]))
        if hop == "city3":
            lines += [
                t["bcity"][style].format(b=b, c=c)
                for b, c in city_of.items()
                if b in building_of.values()
            ]
        lines += lure_lines
        random.Random(order_seed).shuffle(lines)
        pad = core.filler(
            rng, lang, core.words(" ".join(lines), lang) + 20, [target, *others, co]
        )
        sec = core.SECTION[lang]
        state = core.compose(
            t["title"].format(co=co),
            [
                (sec["background"], [t["intro"].format(co=co), *pad[0]]),
                (sec["record"], lines),
                (sec["notes"], pad[1]),
            ],
            lang,
        )
        facts = {
            "hop": hop,
            "target": target,
            "target_team": team,
            "value_of": value_of,
            "building_of": building_of,
            "city_of": city_of,
            "projects": projects,
            "lures": lures,
            "answer": value_of[team],
        }
        return core.A4v2Render(
            state, facts, hop, f"{hop}_v2", ask, lambda v: prop.format(p=target, v=v)
        )

    present = sorted(set(value_of.values()), key=lambda v: rng.random())
    alternatives = []
    for gold in present:
        team = rng.choice([tm for tm in teams if value_of[tm] == gold])
        others_present = [v for v in present if v != gold] + lures
        rand = rng.sample(others_present, 3)
        while set(rand) == set(lures):
            rand = rng.sample(others_present, 3)
        alternatives.append(
            core.A4v2Alternative(0, gold, list(lures), rand, lambda tm=team: render(tm))
        )
    return core.A4v2Plan(False, alternatives)


# ---------------------------------------------------------------- oracle 2


def _alt(values: Sequence[str]) -> str:
    return "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    lex = LEX[lang]
    team, bld, city, proj = (
        _alt(lex[k]) for k in ("teams", "buildings", "cities", "projects")
    )
    name = core.EN_NAME_RX if lang == "en" else core.ZH_NAME_RX
    if lang == "en":
        rx = {
            "member": rf"({name}) (?:is on|is a member of|works in) the ({team}) team",
            "floor": rf"(?:The|the) ({team}) team (?:sits on|is on|on) the (\d+)(?:st|nd|rd|th) floor",
            "city": rf"The ({team}) team (?:is based in|works from the) ({city})",
            "building": rf"The ({team}) team (?:works out of|is housed in) ({bld})",
            "bcity": rf"({bld}) (?:is in|stands in) ({city})",
            "assign": rf"({name}) is assigned to Project ({proj})",
            "run": rf"Project ({proj}) is run by the ({team}) team",
        }
        q = re.search(
            rf"(?:On which floor|In which city) does ({name}) work\?", instructions
        )
        n = re.search(
            rf"Does ({name}) work (?:on the (\d+)(?:st|nd|rd|th) floor|in ({city}))\?",
            instructions,
        )
    else:
        rx = {
            "member": rf"({name})(?:隶属于|是|目前在)({team})",
            "floor": rf"({team})(?:在|的办公区位于)(\d+)楼",
            "city": rf"({team})(?:设在|在)({city})",
            "building": rf"({team})(?:在|的办公地点是)({bld})",
            "bcity": rf"({bld})(?:位于|在)({city})",
            "assign": rf"({name})被分配到({proj})项目",
            "run": rf"({proj})项目由({team})负责",
        }
        q = re.search(rf"({name})在(?:几楼办公|哪座城市工作)？", instructions)
        n = re.search(rf"({name})在(?:(\d+)楼办公|({city})工作)吗？", instructions)
    facts: dict[str, dict[str, str]] = {}
    for key, pattern in rx.items():
        pairs = [(m.group(1), m.group(2)) for m in re.finditer(pattern, state)]
        mapping = dict(pairs)
        if len(mapping) != len(pairs):
            return None
        facts[key] = mapping
    person = (q or n).group(1) if (q or n) else None
    if person is None:
        return None
    if person in facts["assign"]:
        unit = facts["run"].get(facts["assign"][person])
    else:
        unit = facts["member"].get(person)
    if unit is None:
        return None
    floor = facts["floor"].get(unit)
    town = facts["city"].get(unit) or facts["bcity"].get(
        facts["building"].get(unit, "")
    )
    if core.is_noul(options):
        if n is None:
            return None
        if n.group(2):
            return int(floor is not None and int(floor) == int(n.group(2)))
        return int(town == n.group(3))
    if floor is not None:
        return core.match_option(
            options, lambda text: int(re.match(r"\d+", text).group(0)) == int(floor)
        )
    return core.match_option(options, lambda text: text == town)
