"""Program-verified multi-step arithmetic word problems as quantity graphs.

An agent does one activity over several periods; each later period is defined relative to an earlier one (times as
many, a fraction of, more / fewer, a percentage more), then one aggregate step (total, leftover after giving some
away, full containers, revenue, average or a difference). Every derived quantity is a graph node whose parents are
the quantities it is computed from. Distractors come from propagating one plausible slip (wrong operation, wrong
base period, an off-by-one count) through the graph, so they are the values a careless solver would reach.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .graph import check_problem, choice, node

NAMES = (
    "Amara",
    "Bilal",
    "Chen",
    "Dalia",
    "Emeka",
    "Freya",
    "Goran",
    "Hana",
    "Ivan",
    "Jia",
    "Kofi",
    "Lena",
    "Mateo",
    "Nadia",
    "Omar",
    "Priya",
    "Quinn",
    "Rosa",
    "Sanjay",
    "Tove",
    "Umar",
    "Vera",
    "Wen",
    "Ximena",
    "Yusuf",
    "Zoe",
    "Aiko",
    "Bruno",
    "Carmen",
    "Dmitri",
    "Elif",
    "Farid",
    "Greta",
    "Hugo",
    "Isla",
    "Jonas",
    "Kira",
    "Luis",
)
ACTIVITIES = (
    ("bakes", "baked", "cookies", "box"),
    ("picks", "picked", "apples", "basket"),
    ("sells", "sold", "tickets", "bundle"),
    ("reads", "read", "pages", None),
    ("collects", "collected", "stamps", "album page"),
    ("knits", "knitted", "scarves", "bag"),
    ("plants", "planted", "saplings", "row"),
    ("folds", "folded", "paper cranes", "jar"),
    ("packs", "packed", "parcels", "crate"),
    ("prints", "printed", "flyers", "stack"),
    ("assembles", "assembled", "chairs", "pallet"),
    ("writes", "wrote", "postcards", "envelope"),
    ("repairs", "repaired", "bicycles", None),
    ("bottles", "bottled", "jars of jam", "carton"),
)
PERIODS = (
    ("on", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")),
    ("in", ("January", "February", "March", "April", "May", "June")),
    ("in", ("the first week", "the second week", "the third week", "the fourth week")),
    ("during", ("the morning shift", "the afternoon shift", "the evening shift")),
)
FRACTIONS = ((2, "half"), (3, "a third"), (4, "a quarter"))
PERCENTS = (10, 20, 25, 50)


def _relative(
    rng: random.Random, base: int
) -> tuple[str, Callable[[int], int], str, list[Callable[[int], int]]] | None:
    """A relation of a new quantity to ``base``: (phrase template, exact op, kind, slip ops)."""
    kinds = ["times", "more", "fewer", "fraction", "percent"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "times":
            k = rng.randint(2, 4)
            word = {2: "twice", 3: "three times", 4: "four times"}[k]
            return (
                f"{word} as many {{unit}} as {{ref}}",
                (lambda x, k=k: x * k),
                kind,
                [
                    lambda x, k=k: x + k,
                    lambda x, k=k: x * (k + 1),
                    lambda x, k=k: x * k + x,
                ],
            )
        if kind == "more":
            d = rng.randint(3, 40)
            return (
                f"{d} more {{unit}} than {{ref}}",
                (lambda x, d=d: x + d),
                kind,
                [lambda x, d=d: x - d, lambda x, d=d: x * 2 + d, lambda x, d=d: d],
            )
        if kind == "fewer" and base > 8:
            d = rng.randint(2, max(2, base // 2))
            return (
                f"{d} fewer {{unit}} than {{ref}}",
                (lambda x, d=d: x - d),
                kind,
                [lambda x, d=d: x + d, lambda x, d=d: x - d + 1, lambda x, d=d: d],
            )
        if kind == "fraction":
            options = [(q, w) for q, w in FRACTIONS if base % q == 0 and base // q > 0]
            if options:
                q, w = rng.choice(options)
                return (
                    f"{w} as many {{unit}} as {{ref}}",
                    (lambda x, q=q: x // q),
                    kind,
                    [
                        lambda x, q=q: x * q,
                        lambda x, q=q: x - q,
                        lambda x, q=q: x + x // q,
                    ],
                )
        if kind == "percent":
            options = [p for p in PERCENTS if (base * p) % 100 == 0]
            if options:
                p = rng.choice(options)
                return (
                    f"{p}% more {{unit}} than {{ref}}",
                    (lambda x, p=p: x + x * p // 100),
                    kind,
                    [
                        lambda x, p=p: x * p // 100,
                        lambda x, p=p: x + p,
                        lambda x, p=p: x - x * p // 100,
                    ],
                )
    return None


def generate(rng: random.Random, pid: str) -> dict[str, Any] | None:
    name = rng.choice(NAMES)
    present, past, unit, container = rng.choice(ACTIVITIES)
    prep, period_names = rng.choice(PERIODS)
    count = rng.randint(2, min(4, len(period_names)))
    periods = list(period_names[:count])
    values: list[int] = [rng.randint(6, 90)]
    sentences = [f"{name} {past} {values[0]} {unit} {prep} {periods[0]}."]
    nodes: list[dict[str, Any]] = []
    formulas: list[tuple[int, Callable[[int], int], list[Callable[[int], int]]]] = [
        (-1, lambda x: x, [])
    ]
    for i in range(1, count):
        ref_index = i - 1 if rng.random() < 0.75 else rng.randrange(0, i)
        rel = _relative(rng, values[ref_index])
        if rel is None:
            return None
        phrase, op, _kind, slips = rel
        value = op(values[ref_index])
        if value <= 0 or value > 5000:
            return None
        values.append(value)
        formulas.append((ref_index, op, slips))
        ref = f"{prep} {periods[ref_index]}"
        sentences.append(
            f"{prep.capitalize()} {periods[i]}, {name} {past} "
            + phrase.format(unit=unit, ref=ref)
            + "."
        )
    # period nodes (period 0 is given, not a node)
    node_ids = {}
    for i in range(1, count):
        ref_index, op, slips = formulas[i]
        wrong = [str(s(values[ref_index])) for s in slips] + [
            str(values[ref_index]),
            str(values[i] + 1),
        ]
        wrong = [w for w in wrong if w.lstrip("-").isdigit() and int(w) > 0]
        options, label = choice(rng, str(values[i]), wrong[:3])
        node_id = f"n{len(nodes) + 1}"
        node_ids[i] = node_id
        deps = [node_ids[ref_index]] if ref_index in node_ids else []
        false_value = int(options[(label + 1) % len(options)]["description"])
        nodes.append(
            node(
                node_id,
                f"How many {unit} did {name} {_base(present)} {prep} {periods[i]}?",
                kind="choice",
                depends_on=deps,
                options=options,
                label=label,
                statement=f"{name} {past} {values[i]} {unit} {prep} {periods[i]}.",
                false_statement=f"{name} {past} {false_value} {unit} {prep} {periods[i]}.",
            )
        )
    total = sum(values)
    total_id = f"n{len(nodes) + 1}"
    span = (
        " and ".join([", ".join(periods[:-1]), periods[-1]])
        if count > 2
        else " and ".join(periods)
    )
    nodes.append(
        node(
            total_id,
            f"How many {unit} did {name} {_base(present)} altogether over {span}?",
            kind="choice",
            depends_on=[node_ids[i] for i in range(1, count)],
            options=(
                opt := choice(
                    rng,
                    str(total),
                    [str(total - values[-1]), str(total + values[0]), str(total - 1)],
                )
            )[0],
            label=opt[1],
            statement=f"Altogether {name} {past} {total} {unit} over {span}.",
            false_statement=f"Altogether {name} {past} {total - values[-1]} {unit} over {span}.",
        )
    )

    def slipped_totals() -> list[int]:
        """Totals a solver reaches with one slip in one relative step (propagated through later steps)."""
        out = []
        for i in range(1, count):
            for slip in formulas[i][2]:
                alt = list(values)
                alt[i] = slip(alt[formulas[i][0]])
                for j in range(i + 1, count):
                    alt[j] = formulas[j][1](alt[formulas[j][0]])
                if all(v > 0 for v in alt):
                    out.append(sum(alt))
        return out

    final_kind = rng.choice(
        ["total", "leftover", "containers", "revenue", "difference", "average"]
    )
    distractor_pool = slipped_totals() + [
        total - values[-1],
        total + values[-1],
        values[-1],
        total - 1,
        total + 1,
    ]
    if final_kind == "leftover":
        gift = rng.randint(1, max(1, total // 3))
        answer = total - gift
        sentences.append(f"Afterwards {name} gave {gift} of the {unit} to a neighbour.")
        question = f"How many {unit} does {name} have left?"
        extra = [total, total + gift, *[t - gift for t in distractor_pool]]
    elif final_kind == "containers" and container:
        size = rng.choice([s for s in (2, 3, 4, 5, 6, 8, 10, 12) if s < total] or [2])
        answer = total // size
        sentences.append(f"{name} puts the {unit} into {container}s of {size} each.")
        question = f"How many full {container}s can {name} fill?"
        extra = [
            answer + 1,
            total * size,
            total - size,
            *[t // size for t in distractor_pool],
        ]
    elif final_kind == "revenue":
        price = rng.randint(2, 15)
        answer = total * price
        sentences.append(f"Each of the {unit} brings in ${price}.")
        question = f"How many dollars do all the {unit} bring in?"
        extra = [
            total + price,
            values[-1] * price,
            *[t * price for t in distractor_pool],
        ]
    elif final_kind == "difference" and count >= 2 and values[-1] != values[0]:
        hi, lo = (count - 1, 0) if values[-1] > values[0] else (0, count - 1)
        answer = values[hi] - values[lo]
        question = f"How many more {unit} did {name} {_base(present)} {prep} {periods[hi]} than {prep} {periods[lo]}?"
        extra = [
            values[hi] + values[lo],
            values[hi],
            values[lo],
            answer + 1,
            answer - 1,
        ]
    elif final_kind == "average" and total % count == 0:
        answer = total // count
        question = f"On average, how many {unit} did {name} {_base(present)} per period over {span}?"
        extra = [
            total,
            answer + 1,
            (total - values[-1]) // max(1, count - 1),
            *[t // count for t in distractor_pool],
        ]
    else:
        final_kind = "total"
        answer = total
        question = (
            f"How many {unit} did {name} {_base(present)} altogether over {span}?"
        )
        extra = distractor_pool
        nodes.pop()  # the total node would be the final question itself
    wrong = [str(x) for x in extra if isinstance(x, int) and x > 0 and x != answer]
    n_options = 10 if rng.random() < 0.3 else 4
    pad = [answer + d for d in (1, -1, 2, -2, 10, -10, 5, -5) if answer + d > 0]
    final_options, final_label = choice(
        rng, str(answer), _unique(wrong + [str(p) for p in pad])[: n_options - 1]
    )
    story = " ".join(sentences)
    instructions, state = _final_text(rng, story, question)
    problem = {
        "pid": pid,
        "family": "reasoning_arith_graph",
        "source": "decision2_reasoning_program_v1",
        "render_template": f"arith_{final_kind}_v1",
        "language": "en",
        "licence": "program-generated",
        "state": state,
        "final": {
            "task_type": "choice",
            "instructions": instructions,
            "options": final_options,
            "label": final_label,
        },
        "nodes": nodes,
        "audit": {
            "values": values,
            "answer": answer,
            "final_kind": final_kind,
            "periods": periods,
        },
    }
    check_problem(problem)
    return problem


def _base(present: str) -> str:
    """Base form after "did": strips the third-person -s / -es."""
    if present.endswith("ies"):
        return present[:-3] + "y"
    if present.endswith(("shes", "ches", "sses", "xes")):
        return present[:-2]
    return present[:-1] if present.endswith("s") else present


def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _final_text(rng: random.Random, story: str, question: str) -> tuple[str, Any]:
    style = rng.randrange(3)
    if style == 0:
        return f"{question} Choose the numeric answer.", story
    if style == 1:
        return "Choose the number that answers the word problem in the state.", {
            "problem": f"{story} {question}"
        }
    return question, {"story": story}
