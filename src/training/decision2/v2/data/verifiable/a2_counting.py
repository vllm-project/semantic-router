"""a2_counting: count prose-log items matching a conjunction with a numeric comparison."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from v2.data.verifiable import core
from v2.data.verifiable.core import A4Scenario, Group, Variant

FAMILY = "a2_counting"
OPS = (">", ">=", "<", "<=")
OP_TEXT = {
    "en": {">": "more than", ">=": "at least", "<": "less than", "<=": "at most"},
    "zh": {">": "超过", ">=": "不少于", "<": "不足", "<=": "不超过"},
}
DAYS = {
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
    "zh": ("周一", "周二", "周三", "周四", "周五"),
}

DOMAINS: dict[str, dict[str, Any]] = {
    "crates": {
        "range": (3, 40),
        "en": {
            "cats": (
                ("blue", "blue"),
                ("green", "green"),
                ("grey", "grey"),
                ("red", "red"),
            ),
            "thirds": ("Porto", "Ghent", "Lyon"),
            "items": (
                "Crate {i} was {c}, weighed {x} kg and went to {z}.",
                "Crate {i}: {c}, {x} kg, bound for {z}.",
                "Crate {i}, a {c} one, weighed {x} kg and was shipped to {z}.",
            ),
            "title": "Dispatch log — {co}",
            "intro": "The loading dock at {co} noted every crate that left on {day}.",
            "q": "How many {c} crates weighed {op} {t} kg{z}?",
            "noul": "Were there {qual} {k} {c} crates that weighed {op} {t} kg{z}?",
            "z": " and went to {z}",
            "noun": ("crate", "crates"),
        },
        "zh": {
            "cats": ("蓝", "绿", "灰", "红"),
            "thirds": ("宁波", "天津", "厦门"),
            "items": (
                "{i}号箱为{c}色，重{x}公斤，发往{z}。",
                "{i}号箱：{c}色，{x}公斤，目的地{z}。",
                "{i}号箱是{c}色的，重{x}公斤，运往{z}。",
            ),
            "title": "发货记录——{co}",
            "intro": "{co}的装卸区记下了{day}发出的每一个箱子。",
            "pred": "{c}色、重量{op}{t}公斤{z}的箱子",
            "z": "、发往{z}",
            "cw": "个",
        },
    },
    "bakery": {
        "range": (1, 24),
        "en": {
            "cats": (
                ("sourdough", "sourdough"),
                ("rye bread", "rye bread"),
                ("croissants", "croissants"),
                ("seed buns", "seed buns"),
            ),
            "thirds": ("pickup", "delivery"),
            "items": (
                "Order {i}: {x} × {c}, for {z}.",
                "Order {i} asked for {x} units of {c}, marked for {z}.",
                "Order {i} was {c}, quantity {x}, set for {z}.",
            ),
            "title": "Order book — {co}",
            "intro": "These are the bakery orders {co} took on {day}.",
            "q": "How many orders for {c} had a quantity of {op} {t}{z}?",
            "noul": "Were there {qual} {k} orders for {c} that had a quantity of {op} {t}{z}?",
            "z": " and were marked for {z}",
            "noun": ("order", "orders"),
        },
        "zh": {
            "cats": ("酸面包", "黑麦面包", "可颂", "杂粮餐包"),
            "thirds": ("到店自取", "送货上门"),
            "items": (
                "{i}号订单：{c}{x}份，{z}。",
                "{i}号订单订了{x}份{c}，{z}。",
                "{i}号订单为{c}，数量{x}份，方式为{z}。",
            ),
            "title": "订单簿——{co}",
            "intro": "以下是{co}在{day}接到的面包订单。",
            "pred": "{c}订单中，数量{op}{t}份{z}的",
            "z": "且方式为{z}",
            "cw": "笔",
        },
    },
    "tickets": {
        "range": (5, 240),
        "en": {
            "cats": (("email", "email"), ("phone", "phone"), ("chat", "chat")),
            "thirds": ("high", "normal"),
            "items": (
                "Ticket {i} came in by {c}, was marked {z} priority and took {x} minutes to resolve.",
                "Ticket {i}: {c}, {z} priority, resolved in {x} minutes.",
                "Ticket {i} ({c}, {z} priority) was closed after {x} minutes.",
            ),
            "title": "Support desk summary — {co}",
            "intro": "The support desk at {co} logged these tickets on {day}.",
            "q": "How many {c} tickets took {op} {t} minutes to resolve{z}?",
            "noul": "Were there {qual} {k} {c} tickets that took {op} {t} minutes to resolve{z}?",
            "z": " and had {z} priority",
            "noun": ("ticket", "tickets"),
        },
        "zh": {
            "cats": ("邮件", "电话", "在线聊天"),
            "thirds": ("高", "普通"),
            "items": (
                "{i}号工单通过{c}提交，优先级为{z}，处理用时{x}分钟。",
                "{i}号工单：{c}渠道，{z}优先级，{x}分钟解决。",
                "{i}号工单来自{c}，{z}优先级，{x}分钟后关闭。",
            ),
            "title": "客服工单汇总——{co}",
            "intro": "{co}客服台在{day}登记了以下工单。",
            "pred": "通过{c}提交、处理用时{op}{t}分钟{z}的工单",
            "z": "且优先级为{z}",
            "cw": "张",
        },
    },
    "library": {
        "range": (0, 30),
        "en": {
            "cats": (
                ("mystery", "mystery"),
                ("history", "history"),
                ("poetry", "poetry"),
                ("science", "science"),
            ),
            "thirds": ("North", "Riverside"),
            "items": (
                "Book {i}, a {c} title, came back to the {z} branch {x} days overdue.",
                "Book {i}: {c}, returned to the {z} branch, {x} days overdue.",
                "Book {i} ({c}) was returned to the {z} branch {x} days late.",
            ),
            "title": "Returns log — {co} Library",
            "intro": "The returns desk recorded these books on {day}.",
            "q": "How many {c} books were {op} {t} days overdue{z}?",
            "noul": "Were there {qual} {k} {c} books that were {op} {t} days overdue{z}?",
            "z": " and returned to the {z} branch",
            "noun": ("book", "books"),
        },
        "zh": {
            "cats": ("悬疑", "历史", "诗歌", "科普"),
            "thirds": ("北区", "滨江"),
            "items": (
                "{i}号书是一本{c}类图书，归还到{z}分馆，逾期{x}天。",
                "{i}号书：{c}类，还至{z}分馆，逾期{x}天。",
                "{i}号书（{c}类）在{z}分馆归还，逾期{x}天。",
            ),
            "title": "还书登记——{co}图书馆",
            "intro": "还书处在{day}登记了以下图书。",
            "pred": "{c}类图书中，逾期{op}{t}天{z}的",
            "z": "且在{z}分馆归还",
            "cw": "本",
        },
    },
    "plants": {
        "range": (10, 180),
        "en": {
            "cats": (
                ("fern", "ferns"),
                ("olive tree", "olive trees"),
                ("cactus", "cacti"),
                ("maple", "maples"),
            ),
            "thirds": ("clay", "plastic"),
            "items": (
                "Plant {i} was a {c} in a {z} pot, {x} cm tall.",
                "Plant {i}: {c}, {x} cm, {z} pot.",
                "Plant {i}, a {c}, stood {x} cm tall in a {z} pot.",
            ),
            "title": "Nursery sales sheet — {co}",
            "intro": "{co} sold the following plants on {day}.",
            "q": "How many {c} were {op} {t} cm tall{z}?",
            "noul": "Were there {qual} {k} {c} that were {op} {t} cm tall{z}?",
            "z": " and in {z} pots",
            "noun": ("plant", "plants"),
        },
        "zh": {
            "cats": ("蕨类", "橄榄树", "仙人掌", "枫树"),
            "thirds": ("陶盆", "塑料盆"),
            "items": (
                "{i}号植株是一株{c}，高{x}厘米，种在{z}里。",
                "{i}号植株：{c}，{x}厘米，{z}。",
                "{i}号植株为{c}，株高{x}厘米，用的是{z}。",
            ),
            "title": "苗圃销售单——{co}",
            "intro": "{co}在{day}售出了以下植株。",
            "pred": "{c}中，株高{op}{t}厘米{z}的",
            "z": "且种在{z}里",
            "cw": "株",
        },
    },
}
CO_SUFFIX = {
    "crates": {"en": ("Freight", "Logistics"), "zh": ("物流", "货运")},
    "bakery": {"en": ("Bakery", "Bakehouse"), "zh": ("烘焙坊", "面包房")},
    "tickets": {"en": ("Software", "Telecom"), "zh": ("软件", "通信")},
    "library": {"en": ("Community", "Municipal"), "zh": ("社区", "市立")},
    "plants": {"en": ("Nursery", "Garden Centre"), "zh": ("苗圃", "园艺")},
}


def _holds(op: str, x: int, t: int) -> bool:
    return {">": x > t, ">=": x >= t, "<": x < t, "<=": x <= t}[op]


def satisfies(item: dict[str, Any], pred: dict[str, Any]) -> bool:
    return (
        item["c"] == pred["c"]
        and _holds(pred["op"], item["x"], pred["t"])
        and (pred["z"] is None or item["z"] == pred["z"])
    )


def _value(
    rng: random.Random, lo: int, hi: int, pred: dict[str, Any], good: bool
) -> int:
    op, t = pred["op"], pred["t"]
    edge = {
        (">", True): t + 1,
        (">", False): t,
        (">=", True): t,
        (">=", False): t - 1,
        ("<", True): t - 1,
        ("<", False): t,
        ("<=", True): t,
        ("<=", False): t + 1,
    }[(op, good)]
    if rng.random() < 0.12:
        return edge
    values = [x for x in range(lo, hi + 1) if _holds(op, x, t) == good]
    return rng.choice(values)


class _Log:
    def __init__(self, rng: random.Random, lang: str) -> None:
        self.lang = lang
        self.domain = rng.choice(sorted(DOMAINS))
        self.d = DOMAINS[self.domain]
        self.t = self.d[lang]
        lo, hi = self.d["range"]
        span = hi - lo
        self.lo, self.hi = lo, hi
        self.pred = {
            "c": rng.randrange(len(self.t["cats"])),
            "op": rng.choice(OPS),
            "t": rng.randint(lo + span // 4, hi - span // 4),
            "z": rng.randrange(len(self.t["thirds"])) if rng.random() < 0.5 else None,
        }
        self.co = core.company(rng, lang, CO_SUFFIX[self.domain][lang])
        self.day = rng.choice(DAYS[lang])

    def item(
        self, rng: random.Random, good: bool, mode: str | None = None
    ) -> dict[str, Any]:
        pred = self.pred
        n_cats, n_z = len(self.t["cats"]), len(self.t["thirds"])
        item = {
            "c": pred["c"],
            "x": _value(rng, self.lo, self.hi, pred, True),
            "z": pred["z"] if pred["z"] is not None else rng.randrange(n_z),
        }
        if good:
            return item
        modes = ["cat", "num"] + (["z"] if pred["z"] is not None else [])
        mode = mode or rng.choice([*modes, "multi"])
        if mode in ("cat", "multi"):
            item["c"] = rng.choice([c for c in range(n_cats) if c != pred["c"]])
        if mode == "num" or (mode == "multi" and rng.random() < 0.6):
            item["x"] = _value(rng, self.lo, self.hi, pred, False)
        if mode == "z" or (
            mode == "multi" and pred["z"] is not None and rng.random() < 0.5
        ):
            item["z"] = rng.choice([z for z in range(n_z) if z != pred["z"]])
        if mode == "multi" and rng.random() < 0.5:
            item["x"] = rng.randint(self.lo, self.hi)
        return item

    def sentence(self, item: dict[str, Any], ident: int, style: int) -> str:
        cat = self.t["cats"][item["c"]]
        text = self.t["items"][style].format(
            i=ident,
            c=cat[0] if self.lang == "en" else cat,
            x=item["x"],
            z=self.t["thirds"][item["z"]],
        )
        if self.lang == "en":
            text = text.replace(" a olive", " an olive")
            if item["x"] == 1:
                text = text.replace(" 1 days ", " 1 day ").replace(
                    " 1 units ", " 1 unit "
                )
        return text

    def question(self, qual: str | None = None, k: int | None = None) -> str:
        pred, t = self.pred, self.t
        op = OP_TEXT[self.lang][pred["op"]]
        z = t["z"].format(z=t["thirds"][pred["z"]]) if pred["z"] is not None else ""
        if self.lang == "en":
            cat = t["cats"][pred["c"]][1]
            if qual is None:
                return t["q"].format(c=cat, op=op, t=pred["t"], z=z)
            return t["noul"].format(qual=qual, k=k, c=cat, op=op, t=pred["t"], z=z)
        head = t["pred"].format(c=t["cats"][pred["c"]], op=op, t=pred["t"], z=z)
        if qual is None:
            return f"{head}有几{t['cw']}？"
        return f"{head}{'至少' if qual == 'at least' else '正好'}有{k}{t['cw']}吗？"

    def option(self, count: int) -> str:
        if self.lang == "zh":
            return f"{count}{self.t['cw']}"
        singular, plural = self.t["noun"]
        return f"{count} {singular if count == 1 else plural}"

    def state(
        self,
        rng: random.Random,
        items: list[dict[str, Any]],
        ids: list[int],
        styles: list[int],
        pad: tuple[list[str], list[str]],
    ) -> str:
        sec = core.SECTION[self.lang]
        log = [
            self.sentence(item, ident, style)
            for item, ident, style in zip(items, ids, styles)
        ]
        intro = self.t["intro"].format(co=self.co, day=self.day)
        return core.compose(
            self.t["title"].format(co=self.co),
            [
                (sec["background"], [intro, *pad[0]]),
                (sec["record"], log),
                (sec["notes"], pad[1]),
            ],
            self.lang,
        )


def _layout(
    rng: random.Random, log: _Log, base: int, switch: int, n_items: int
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    items = [log.item(rng, True) for _ in range(base)]
    items += [
        {
            "switch": True,
            **log.item(
                rng,
                False,
                rng.choice(
                    ["cat", "num"] + (["z"] if log.pred["z"] is not None else [])
                ),
            ),
        }
        for _ in range(switch)
    ]
    items += [log.item(rng, False) for _ in range(n_items - base - switch)]
    rng.shuffle(items)
    ids = sorted(rng.sample(range(100, 1000), n_items))
    styles = [rng.randrange(3) for _ in items]
    return items, ids, styles


def _switched(
    log: _Log, items: list[dict[str, Any]], count: int
) -> tuple[list[dict[str, Any]], list[int]]:
    out, flipped = [], []
    for position, item in enumerate(items):
        clean = {key: item[key] for key in ("c", "x", "z")}
        if item.get("switch") and len(flipped) < count:
            pred = log.pred
            clean["c"] = pred["c"]
            if pred["z"] is not None:
                clean["z"] = pred["z"]
            if not _holds(pred["op"], clean["x"], pred["t"]):
                clean["x"] = {
                    ">": pred["t"] + 1,
                    ">=": pred["t"],
                    "<": pred["t"] - 1,
                    "<=": pred["t"],
                }[pred["op"]]
            flipped.append(position)
        out.append(clean)
    return out, flipped


def _group(rng: random.Random, lang: str, task: str) -> Group | None:
    log = _Log(rng, lang)
    if task == "choice":
        low = rng.randint(1, 6)
        gaps = rng.choice(((1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 2)))
        counts = [low, low + gaps[0], low + gaps[0] + gaps[1], low + sum(gaps)]
    else:
        k = rng.randint(2, 8)
        counts = [k - 1, k]
    switch = counts[-1] - counts[0]
    n_items = rng.randint(max(6, counts[-1] + 2), min(30, counts[-1] + 18))
    items, ids, styles = _layout(rng, log, counts[0], switch, n_items)
    variants_items = []
    for count in counts:
        realized, flipped = _switched(log, items, count - counts[0])
        if sum(satisfies(item, log.pred) for item in realized) != count:
            return None
        variants_items.append((count, realized, flipped))
    sample = [
        log.sentence(item, i, s)
        for item, i, s in zip(variants_items[-1][1], ids, styles)
    ]
    base_words = core.words(" ".join(sample), lang) + 25
    if task == "choice":
        order = rng.sample(range(4), 4)
        descriptions = [log.option(counts[i]) for i in order]
        options = core.choice_options(descriptions)
        instructions = log.question()
        labels = {counts[i]: label for label, i in enumerate(order)}
    else:
        descriptions = []
        options = core.noul_options(lang)
        instructions = log.question("at least", counts[1])
        labels = {counts[0]: 0, counts[1]: 1}
    pad = core.filler(rng, lang, base_words, [log.co], avoid=descriptions)
    variants = []
    for count, realized, flipped in variants_items:
        facts = {
            "domain": log.domain,
            "pred": log.pred,
            "items": realized,
            "ids": ids,
            "count": count,
        }
        edit = (
            f"switched items {[ids[p] for p in flipped]} to match"
            if flipped
            else "base log"
        )
        variants.append(
            Variant(
                log.state(rng, realized, ids, styles, pad), labels[count], facts, edit
            )
        )
    variants.sort(key=lambda v: v.label)
    if task == "choice" and not all(
        core.presence_ok(v.state, options) for v in variants
    ):
        return None
    subtype = f"{log.pred['op']}{'_z' if log.pred['z'] is not None else ''}"
    return Group(task, instructions, options, variants, f"{log.domain}_{task}", subtype)


def build_group(
    rng: random.Random, lang: str, task_type: str, levels: int | None = None
) -> Group:
    for _ in range(200):
        group = _group(rng, lang, task_type)
        if group is not None:
            core.check_group(group)
            return group
    raise RuntimeError("counting group construction failed")


def a4_scenario(rng: random.Random, lang: str) -> A4Scenario:
    for _ in range(200):
        log = _Log(rng, lang)
        n_items = rng.randint(12, 30)
        count = rng.randint(5, min(14, n_items - 5))
        items, ids, styles = _layout(rng, log, count, 0, n_items)
        items = [{key: item[key] for key in ("c", "x", "z")} for item in items]
        if sum(satisfies(item, log.pred) for item in items) == count:
            break
    else:
        raise RuntimeError("counting a4 construction failed")
    sample = [log.sentence(item, i, s) for item, i, s in zip(items, ids, styles)]
    pad = core.filler(rng, lang, core.words(" ".join(sample), lang) + 25, [log.co])
    state = log.state(rng, items, ids, styles, pad)
    facts = {
        "domain": log.domain,
        "pred": log.pred,
        "items": items,
        "ids": ids,
        "count": count,
    }
    below = list(range(count - 1, -1, -1))
    above = list(range(count + 1, n_items + 1))
    far = [v for v in range(2, n_items + 1) if abs(v - count) >= 3]
    near_noul = rng.choice((count - 1, count + 1))
    return A4Scenario(
        state,
        facts,
        log.pred["op"],
        f"{log.domain}_a4",
        log.question(),
        log.option(count),
        [log.option(v) for v in below[:3]],
        [log.option(v) for v in [count + 1, count + 2, count + 3]],
        [log.option(v) for v in below],
        [log.option(v) for v in above],
        True,
        True,
        log.question("exactly", count),
        log.question("exactly", near_noul),
        [log.question("exactly", v) for v in far],
    )


V2_COUNTS = (2, 44)
V2_SPAN = 5


def a4v2_plan(rng: random.Random, lang: str, turn: int) -> core.A4v2Plan:
    """Count options first (rank-balanced windows), then a log whose true count is the gold.

    A wide count range keeps the random set's values (span <= ``V2_SPAN``) from
    reaching far past the range the gold itself can take.
    """
    log = _Log(rng, lang)
    sets = core.rank_windows(rng, *V2_COUNTS, 4, V2_SPAN)
    number = lambda text: int(re.match(r"\d+", text).group(0))

    def render(count: int, top: int) -> core.A4v2Render | None:
        n_items = top + rng.randint(2, 6)
        items, ids, styles = _layout(rng, log, count, 0, n_items)
        items = [{key: item[key] for key in ("c", "x", "z")} for item in items]
        if sum(satisfies(item, log.pred) for item in items) != count:
            return None
        sample = [log.sentence(item, i, s) for item, i, s in zip(items, ids, styles)]
        pad = core.filler(rng, lang, core.words(" ".join(sample), lang) + 25, [log.co])
        facts = {
            "domain": log.domain,
            "pred": log.pred,
            "items": items,
            "ids": ids,
            "count": count,
        }
        return core.A4v2Render(
            log.state(rng, items, ids, styles, pad),
            facts,
            log.pred["op"],
            f"{log.domain}_v2",
            log.question(),
            lambda v: log.question("exactly", number(v)),
        )

    top = max(v for gold, near, rand in sets for v in (gold, *near, *rand))
    alternatives = [
        core.A4v2Alternative(
            r,
            log.option(gold),
            [log.option(v) for v in near],
            [log.option(v) for v in rand],
            lambda g=gold: render(g, top),
        )
        for r, (gold, near, rand) in enumerate(sets)
    ]
    return core.A4v2Plan(True, alternatives)


# ---------------------------------------------------------------- oracle 2

_SENT = {
    "en": r"(?:Crate|Order|Ticket|Book|Plant) (\d+)[^.]*\.",
    "zh": r"(\d+)号(?:箱|订单|工单|书|植株)[^。]*。",
}
_NUM = {
    "crates": {"en": r"(\d+) kg", "zh": r"(\d+)公斤"},
    "bakery": {"en": r"(\d+) ×|(\d+) units? of|quantity (\d+)", "zh": r"(\d+)份"},
    "tickets": {"en": r"(\d+) minutes", "zh": r"(\d+)分钟"},
    "library": {"en": r"(\d+) days? (?:overdue|late)", "zh": r"逾期(\d+)天"},
    "plants": {"en": r"(\d+) cm", "zh": r"(\d+)厘米"},
}
_THIRD = {
    "crates": {"en": r"\b({z})\b", "zh": r"(?:发往|目的地|运往)({z})"},
    "bakery": {"en": r"\b({z})\b", "zh": r"({z})"},
    "tickets": {"en": r"\b({z}) priority", "zh": r"优先级为({z})|({z})优先级"},
    "library": {"en": r"the ({z}) branch", "zh": r"({z})分馆"},
    "plants": {"en": r"\b({z}) pot", "zh": r"({z})"},
}


def _lexicon_hit(text: str, words: Sequence[str], lang: str) -> int | None:
    ordered = sorted(range(len(words)), key=lambda i: -len(words[i]))
    for i in ordered:
        needle = re.escape(words[i])
        if re.search(rf"\b{needle}\b" if lang == "en" else needle, text):
            return i
    return None


def reparse(
    state: str, instructions: str, options: Sequence[dict[str, Any]], lang: str
) -> int | None:
    domain = None
    for name, spec in DOMAINS.items():
        head = spec[lang]["title"].split("{")[0]
        if head and head in state:
            domain = name
    if domain is None:
        return None
    t = DOMAINS[domain][lang]
    cats_q = [c[1] for c in t["cats"]] if lang == "en" else list(t["cats"])
    cats_s = [c[0] for c in t["cats"]] if lang == "en" else list(t["cats"])
    question = instructions
    qual = (
        re.match(r"Were there (at least|exactly) (\d+) ", question)
        if lang == "en"
        else re.search(r"(至少|正好)有(\d+)", question)
    )
    if lang == "en" and qual:
        question = question[qual.end() :]
    if lang == "en":
        ops = re.findall(r"(more than|at least|less than|at most) (\d+)", question)
        op_map = {v: k for k, v in OP_TEXT["en"].items()}
    else:
        ops = re.findall(r"(不超过|不少于|超过|不足)(\d+)", question)
        op_map = {v: k for k, v in OP_TEXT["zh"].items()}
    if len(ops) != 1:
        return None
    op, threshold = op_map[ops[0][0]], int(ops[0][1])
    cat = _lexicon_hit(question, cats_q, lang)
    third = _lexicon_hit(question, t["thirds"], lang)
    if cat is None:
        return None
    count = 0
    for match in re.finditer(_SENT[lang], state):
        sentence = match.group(0)
        nums = [
            int(g)
            for m in re.finditer(_NUM[domain][lang], sentence)
            for g in m.groups()
            if g
        ]
        if len(nums) != 1:
            return None
        c = _lexicon_hit(sentence, cats_s, lang)
        third_rx = _THIRD[domain][lang]
        zs = [
            i
            for i, z in enumerate(t["thirds"])
            if re.search(third_rx.format(z=re.escape(z)), sentence)
        ]
        if c is None or len(zs) != 1:
            return None
        ok = {
            ">": nums[0] > threshold,
            ">=": nums[0] >= threshold,
            "<": nums[0] < threshold,
            "<=": nums[0] <= threshold,
        }[op]
        if c == cat and ok and (third is None or zs[0] == third):
            count += 1
    if core.is_noul(options):
        if not qual:
            return None
        k = int(qual.group(2))
        return (
            int(count >= k)
            if qual.group(1) in ("at least", "至少")
            else int(count == k)
        )
    return core.match_option(
        options, lambda text: int(re.match(r"\d+", text).group(0)) == count
    )
