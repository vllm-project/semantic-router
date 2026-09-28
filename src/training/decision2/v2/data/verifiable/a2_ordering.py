"""a2_ordering: positions and precedence determined by prose constraints (brute-force checked)."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from functools import cache
from itertools import permutations
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import A4Scenario, Group, Variant

FAMILY = "a2_ordering"
Constraint = tuple[str, int, int]

DOMAINS = {
    "arrive": {
        "en": {
            "v": "arrived",
            "v0": "arrive",
            "title": "Sign-in notes — {co} planning workshop",
            "ctx": "Everyone came to the planning workshop on {day}, one at a time.",
        },
        "zh": {
            "v": "到场",
            "title": "签到记录——{co}策划会",
            "ctx": "{day}的策划会上，大家是一个接一个到场的。",
        },
    },
    "race": {
        "en": {
            "v": "finished",
            "v0": "finish",
            "title": "Race notes — {co} charity 5K",
            "ctx": "The charity run took place on {day}; nobody finished at the same moment.",
        },
        "zh": {
            "v": "冲线",
            "title": "比赛记录——{co}公益跑",
            "ctx": "公益跑在{day}举行，没有人同时冲线。",
        },
    },
    "seminar": {
        "en": {
            "v": "presented",
            "v0": "present",
            "title": "Seminar notes — {co} research day",
            "ctx": "Each person gave one talk at the research day on {day}.",
        },
        "zh": {
            "v": "发言",
            "title": "发言记录——{co}研讨会",
            "ctx": "{day}的研讨会上，每人各发言一次。",
        },
    },
    "hotel": {
        "en": {
            "v": "checked in",
            "v0": "check in",
            "title": "Front desk notes — {co} Hotel",
            "ctx": "The group checked in one by one on {day}.",
        },
        "zh": {
            "v": "办理入住",
            "title": "前台记录——{co}酒店",
            "ctx": "这群客人{day}逐个办理了入住。",
        },
    },
}
PHRASES = {
    "en": {
        "before": ("{a} {v} before {b}.", "{a} {v} earlier than {b}."),
        "after": ("{a} {v} after {b}.", "{a} {v} later than {b}."),
        "imm": (
            "{a} {v} immediately after {b}, with nobody in between.",
            "{a} was the next to {v0} after {b}.",
        ),
        "notfirst": ("{a} was not the first to {v0}.", "Someone {v} before {a}."),
    },
    "zh": {
        "before": ("{a}比{b}先{v}。", "{a}{v}的时间比{b}早。"),
        "after": ("{a}比{b}晚{v}。", "{a}在{b}之后才{v}。"),
        "imm": (
            "{a}紧跟在{b}之后{v}，中间没有别人。",
            "{b}{v}之后，下一个{v}的就是{a}。",
        ),
        "notfirst": ("{a}不是第一个{v}的。", "{a}{v}之前，已经有人先{v}了。"),
    },
}
ASIDES = {
    "en": (
        "{a} wore a green scarf.",
        "{a} brought pastries for everyone.",
        "{a} had travelled in from out of town.",
        "{a} asked about the Wi-Fi password.",
        "{a} was carrying a large umbrella.",
    ),
    "zh": (
        "{a}围着一条绿色围巾。",
        "{a}给大家带了点心。",
        "{a}是从外地赶过来的。",
        "{a}问了无线网络的密码。",
        "{a}拎着一把大伞。",
    ),
}
DAYS = {
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
    "zh": ("周一", "周二", "周三", "周四", "周五", "周六"),
}
ORD_EN = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
ORD_ZH = "一二三四五六七"
TYPE_WEIGHT = {"before": 1.0, "after": 1.0, "imm": 3.0, "notfirst": 2.0}
CO_SUFFIX = {"en": ("Group", "Partners", "Collective"), "zh": ("集团", "联合", "文化")}


@cache
def _positions(n: int) -> tuple[tuple[int, ...], ...]:
    out = []
    for order in permutations(range(n)):
        pos = [0] * n
        for place, person in enumerate(order):
            pos[person] = place
        out.append(tuple(pos))
    return tuple(out)


def holds(pos: Sequence[int], c: Constraint) -> bool:
    kind, a, b = c
    if kind == "before":
        return pos[a] < pos[b]
    if kind == "after":
        return pos[a] > pos[b]
    if kind == "imm":
        return pos[a] == pos[b] + 1
    return pos[a] != 0


def solutions(n: int, constraints: Sequence[Constraint]) -> list[tuple[int, ...]]:
    return [pos for pos in _positions(n) if all(holds(pos, c) for c in constraints)]


def _true_constraints(
    pos: Sequence[int], forbidden: set[frozenset[int]]
) -> list[Constraint]:
    n = len(pos)
    out: list[Constraint] = []
    for a in range(n):
        if pos[a] != 0:
            out.append(("notfirst", a, a))
        for b in range(n):
            if a == b or frozenset((a, b)) in forbidden:
                continue
            out.append(("before", a, b) if pos[a] < pos[b] else ("after", a, b))
            if pos[a] == pos[b] + 1:
                out.append(("imm", a, b))
    return out


def _derive(
    rng: random.Random,
    n: int,
    pos: Sequence[int],
    good: Callable[[Sequence[int]], bool],
    forbidden: set[frozenset[int]],
) -> list[Constraint] | None:
    """Pick true constraints until every consistent order satisfies ``good``, then minimize."""
    candidates = _true_constraints(pos, forbidden)
    candidates.sort(key=lambda c: rng.random() ** (1 / TYPE_WEIGHT[c[0]]), reverse=True)
    chosen: list[Constraint] = []
    live = list(_positions(n))
    for c in candidates:
        narrowed = [p for p in live if holds(p, c)]
        if len(narrowed) < len(live):
            chosen.append(c)
            live = narrowed
        if all(good(p) for p in live):
            break
    if not all(good(p) for p in live):
        return None
    for c in rng.sample(chosen, len(chosen)):
        trial = [x for x in chosen if x != c]
        if not any(
            not good(p) and all(holds(p, t) for t in trial) for p in _positions(n)
        ):
            chosen = trial
    return rng.sample(chosen, len(chosen))


def _mention_gap(
    constraints: Sequence[Constraint], person: int, peer: int, n: int
) -> int:
    """Mention-count difference between the answer and a random peer (flattens a surface cue)."""
    counts = [0] * n
    for kind, a, b in constraints:
        counts[a] += 1
        if kind != "notfirst":
            counts[b] += 1
    return abs(counts[person] - counts[peer])


def _undetermined(
    rng: random.Random,
    n: int,
    pos: Sequence[int],
    x: int,
    y: int,
    size: int,
) -> list[Constraint] | None:
    candidates = _true_constraints(pos, {frozenset((x, y))})
    rng.shuffle(candidates)
    chosen: list[Constraint] = []
    for c in candidates:
        live = solutions(n, [*chosen, c])
        if (
            len(live) < len(solutions(n, chosen))
            and any(p[x] < p[y] for p in live)
            and any(p[y] < p[x] for p in live)
        ):
            chosen.append(c)
        if len(chosen) == size:
            return chosen
    return chosen if len(chosen) >= max(2, size - 1) else None


class _Scene:
    def __init__(self, rng: random.Random, lang: str, n: int) -> None:
        self.lang = lang
        self.domain = rng.choice(sorted(DOMAINS))
        self.t = DOMAINS[self.domain][lang]
        full = core.people(rng, lang, n)
        self.full = full
        self.names = [name.split()[0] for name in full] if lang == "en" else full
        self.co = core.company(rng, lang, CO_SUFFIX[lang])
        self.day = rng.choice(DAYS[lang])
        self.asides = [
            s.format(a=rng.choice(self.names))
            for s in rng.sample(ASIDES[lang], rng.randint(0, 2))
        ]
        self.styles = rng.getrandbits(32)

    def sentence(self, c: Constraint, index: int) -> str:
        kind, a, b = c
        options = PHRASES[self.lang][kind]
        template = options[(self.styles >> (index % 32)) & 1]
        return template.format(
            a=self.names[a], b=self.names[b], v=self.t["v"], v0=self.t.get("v0", "")
        )

    def roster(self) -> str:
        n = len(self.names)
        if self.lang == "en":
            listed = ", ".join(self.names[:-1]) + " and " + self.names[-1]
            return (
                f"{('Four', 'Five', 'Six', 'Seven')[n - 4]} people took part: {listed}."
            )
        listed = "、".join(self.names[:-1]) + "和" + self.names[-1]
        return f"参加者共有{core.zh_number(n)}人：{listed}。"

    def state(
        self, constraints: Sequence[Constraint], pad: tuple[list[str], list[str]]
    ) -> str:
        sec = core.SECTION[self.lang]
        facts = [self.sentence(c, i) for i, c in enumerate(constraints)]
        head = [self.roster(), self.t["ctx"].format(day=self.day), *self.asides]
        return core.compose(
            self.t["title"].format(co=self.co),
            [
                (sec["background"], [*head, *pad[0]]),
                (sec["record"], facts),
                (sec["notes"], pad[1]),
            ],
            self.lang,
        )

    def ask_position(self, k: int) -> str:
        n = len(self.names)
        if self.lang == "en":
            word = "last" if k == n - 1 else ORD_EN[k]
            return f"Based on the notes, who {self.t['v']} {word}?"
        word = "最后一个" if k == n - 1 else f"第{ORD_ZH[k]}个"
        return f"根据记录，{word}{self.t['v']}的是谁？"

    def ask_is_position(self, person: int, k: int) -> str:
        n = len(self.names)
        if self.lang == "en":
            word = "last" if k == n - 1 else ORD_EN[k]
            return f"Based on the notes, was {self.names[person]} the {word} to {self.t['v0']}?"
        word = "最后一个" if k == n - 1 else f"第{ORD_ZH[k]}个"
        return f"根据记录，{word}{self.t['v']}的人为{self.names[person]}吗？"

    def ask_before(self, x: int, y: int) -> str:
        if self.lang == "en":
            return f"Do the notes establish that {self.names[x]} {self.t['v']} before {self.names[y]}?"
        return (
            f"根据这些记录，可以确定{self.names[x]}比{self.names[y]}先{self.t['v']}吗？"
        )


def _random_pos(
    rng: random.Random, n: int, fixed: dict[int, int] | None = None
) -> tuple[int, ...]:
    while True:
        pos = rng.choice(_positions(n))
        if not fixed or all(pos[p] == k for p, k in fixed.items()):
            return pos


def _choice(rng: random.Random, lang: str) -> Group | None:
    n = rng.choice((4, 5))
    scene = _Scene(rng, lang, n)
    k = rng.choice((0, n - 1, *range(1, n - 1)))
    order = rng.sample(range(n), n)
    specs = []
    for label, person in enumerate(order):
        candidates = []
        for _ in range(4):
            pos = _random_pos(rng, n, {person: k})
            derived = _derive(rng, n, pos, lambda s, p=person: s[p] == k, set())
            if derived is not None:
                candidates.append(derived)
        if not candidates:
            return None
        peer = rng.choice([p for p in range(n) if p != person])
        constraints = min(
            candidates, key=lambda cs, p=person, q=peer: _mention_gap(cs, p, q, n)
        )
        facts = {
            "n": n,
            "names": scene.names,
            "constraints": constraints,
            "position": k,
            "answer": scene.names[person],
        }
        specs.append(
            (
                label,
                constraints,
                facts,
                f"constraints place {scene.names[person]} at position {k + 1}",
            )
        )
    options = core.choice_options([scene.names[p] for p in order])
    return _finish(
        rng,
        scene,
        "choice",
        scene.ask_position(k),
        options,
        specs,
        f"{scene.domain}_position{k}",
        f"k{k}",
    )


def _noul(rng: random.Random, lang: str) -> Group | None:
    n = rng.choice((4, 5))
    scene = _Scene(rng, lang, n)
    x, y = rng.sample(range(n), 2)
    pair = {frozenset((x, y))}
    while True:
        true_pos = _random_pos(rng, n)
        if true_pos[x] < true_pos[y]:
            break
    true_c = _derive(rng, n, true_pos, lambda s: s[x] < s[y], pair)
    if true_c is None:
        return None
    mode = "undetermined" if rng.random() < 1 / 3 else "contradicted"
    false_pos = _random_pos(rng, n)
    if mode == "contradicted":
        if false_pos[y] > false_pos[x]:
            false_pos = tuple(
                false_pos[i] if i not in (x, y) else false_pos[y if i == x else x]
                for i in range(n)
            )
        false_c = _derive(rng, n, false_pos, lambda s: s[y] < s[x], pair)
    else:
        false_c = _undetermined(rng, n, false_pos, x, y, len(true_c))
    if false_c is None:
        return None
    specs = []
    for label, cs, how in ((0, false_c, mode), (1, true_c, "established")):
        facts = {
            "n": n,
            "names": scene.names,
            "constraints": cs,
            "x": x,
            "y": y,
            "status": how,
        }
        specs.append((label, cs, facts, f"constraint set leaves the order {how}"))
    return _finish(
        rng,
        scene,
        "noul",
        scene.ask_before(x, y),
        core.noul_options(lang),
        specs,
        f"{scene.domain}_before_{mode}",
        mode,
    )


def _finish(
    rng: random.Random,
    scene: _Scene,
    task: str,
    instructions: str,
    options: list[dict[str, str]],
    specs: list[tuple[int, list[Constraint], dict[str, Any], str]],
    template: str,
    subtype: str,
) -> Group:
    longest = max(specs, key=lambda s: len(s[1]))[1]
    base = (
        core.words(
            " ".join(scene.sentence(c, i) for i, c in enumerate(longest)), scene.lang
        )
        + 30
    )
    pad = core.filler(rng, scene.lang, base, [*scene.full, scene.co])
    variants = [
        Variant(scene.state(cs, pad), label, facts, edit)
        for label, cs, facts, edit in specs
    ]
    return Group(task, instructions, options, variants, template, subtype)


def build_group(
    rng: random.Random, lang: str, task_type: str, levels: int | None = None
) -> Group:
    for _ in range(200):
        group = _choice(rng, lang) if task_type == "choice" else _noul(rng, lang)
        if group is not None and len({v.state for v in group.variants}) == len(
            group.variants
        ):
            core.check_group(group)
            return group
    raise RuntimeError("ordering group construction failed")


def a4_scenario(rng: random.Random, lang: str) -> A4Scenario:
    n = 7
    for _ in range(200):
        scene = _Scene(rng, lang, n)
        k = rng.randrange(n)
        pos = _random_pos(rng, n)
        person = pos.index(k)
        constraints = _derive(rng, n, pos, lambda s, p=person, q=k: s[p] == q, set())
        if constraints is not None:
            break
    else:
        raise RuntimeError("ordering a4 construction failed")
    base = (
        core.words(
            " ".join(scene.sentence(c, i) for i, c in enumerate(constraints)), lang
        )
        + 30
    )
    state = scene.state(
        constraints, core.filler(rng, lang, base, [*scene.full, scene.co])
    )
    others = sorted(
        (p for p in range(n) if p != person), key=lambda p: (abs(pos[p] - k), pos[p])
    )
    names = scene.names
    facts = {
        "n": n,
        "names": names,
        "constraints": constraints,
        "position": k,
        "answer": names[person],
    }
    return A4Scenario(
        state,
        facts,
        f"k{k}",
        f"{scene.domain}_a4",
        scene.ask_position(k),
        names[person],
        [],
        [names[p] for p in others],
        [],
        [names[p] for p in others],
        False,
        False,
        scene.ask_is_position(person, k),
        scene.ask_is_position(others[0], k),
        [scene.ask_is_position(p, k) for p in others[-2:]],
    )


# ---------------------------------------------------------------- oracle 2

_V = {
    "en": "arrived|finished|presented|checked in",
    "v0": "arrive|finish|present|check in",
    "zh": "到场|冲线|发言|办理入住",
}


def _parse_constraints(state: str, names: list[str], lang: str) -> list[Constraint]:
    alt = "|".join(re.escape(x) for x in sorted(names, key=len, reverse=True))
    v, v0 = _V[lang], _V["v0"]
    idx = {name: i for i, name in enumerate(names)}
    if lang == "en":
        rules = [
            (
                rf"(?<![\w])({alt}) (?:{v}) (?:before|earlier than) ({alt})\.",
                "before",
                False,
            ),
            (
                rf"(?<![\w])({alt}) (?:{v}) (?:after|later than) ({alt})\.",
                "after",
                False,
            ),
            (
                rf"(?<![\w])({alt}) (?:{v}) immediately after ({alt}), with nobody in between\.",
                "imm",
                False,
            ),
            (
                rf"(?<![\w])({alt}) was the next to (?:{v0}) after ({alt})\.",
                "imm",
                False,
            ),
            (rf"(?<![\w])({alt}) was not the first to (?:{v0})\.", "notfirst", False),
            (rf"Someone (?:{v}) before ({alt})\.", "notfirst", False),
        ]
    else:
        rules = [
            (rf"({alt})比({alt})先(?:{v})。", "before", False),
            (rf"({alt})(?:{v})的时间比({alt})早。", "before", False),
            (rf"({alt})比({alt})晚(?:{v})。", "after", False),
            (rf"({alt})在({alt})之后才(?:{v})。", "after", False),
            (rf"({alt})紧跟在({alt})之后(?:{v})，中间没有别人。", "imm", False),
            (rf"({alt})(?:{v})之后，下一个(?:{v})的就是({alt})。", "imm", True),
            (rf"({alt})不是第一个(?:{v})的。", "notfirst", False),
            (rf"({alt})(?:{v})之前，已经有人先(?:{v})了。", "notfirst", False),
        ]
    out: list[Constraint] = []
    for pattern, kind, swap in rules:
        for m in re.finditer(pattern, state):
            groups = [idx[g] for g in m.groups()]
            if kind == "notfirst":
                out.append((kind, groups[0], groups[0]))
            else:
                a, b = (groups[1], groups[0]) if swap else (groups[0], groups[1])
                out.append((kind, a, b))
    return out


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    if lang == "en":
        roster = re.search(r"people took part: ([^.]+)\.", state)
        names = re.split(r", | and ", roster.group(1)) if roster else []
    else:
        roster = re.search(r"参加者共有.人：([^。]+)。", state)
        names = re.split(r"、|和", roster.group(1)) if roster else []
    if len(names) < 3:
        return None
    n = len(names)
    live = [tuple(order) for order in permutations(range(n))]
    for kind, a, b in _parse_constraints(state, names, lang):
        live = [o for o in live if _check_order(o, kind, a, b)]
    if not live:
        return None
    alt = "|".join(re.escape(x) for x in sorted(names, key=len, reverse=True))
    ordinal = {w: i for i, w in enumerate(ORD_EN)} | {
        f"第{c}个": i for i, c in enumerate(ORD_ZH)
    }
    if core.is_noul(options):
        before = re.search(
            rf"that ({alt}) (?:{_V['en']}) before ({alt})\?|可以确定({alt})比({alt})先",
            instructions,
        )
        if before:
            x, y = [names.index(g) for g in before.groups() if g]
            return int(all(o.index(x) < o.index(y) for o in live))
        who = re.search(
            rf"was ({alt}) the (\w+) to|(最后一个|第.个)(?:{_V['zh']})的人为({alt})吗",
            instructions,
        )
        if not who:
            return None
        person = who.group(1) or who.group(4)
        word = who.group(2) or who.group(3)
        k = n - 1 if word in ("last", "最后一个") else ordinal[word]
        at_k = {o[k] for o in live}
        return int(at_k == {names.index(person)})
    word = re.search(
        r"who (?:" + _V["en"] + r") (\w+)\?|(最后一个|第.个)", instructions
    )
    if not word:
        return None
    token = word.group(1) or word.group(2)
    k = n - 1 if token in ("last", "最后一个") else ordinal[token]
    at_k = {o[k] for o in live}
    if len(at_k) != 1:
        return None
    winner = names[at_k.pop()]
    return core.match_option(options, lambda text: text == winner)


def _check_order(order: tuple[int, ...], kind: str, a: int, b: int) -> bool:
    pa, pb = order.index(a), order.index(b)
    if kind == "before":
        return pa < pb
    if kind == "after":
        return pa > pb
    if kind == "imm":
        return pa == pb + 1
    return pa != 0
