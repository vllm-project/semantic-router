"""Program-verified deductive problems with explicit derivation graphs.

Sub-families (each a different reasoning pattern; every node is a fact the derivation establishes):
- ``rules``: facts plus if-then rules over named entities; the query property is derived (or not) by forward
  chaining; nodes are the derived facts with the facts / derived facts each rule application used.
- ``truth``: a chain of speakers who say whether the previous speaker tells the truth; nodes are each speaker's
  status.
- ``order``: three to five items in a line with position clues; the unique arrangement is found by search; nodes
  are item positions and the pairwise relations the clues imply.
- ``boolean``: nested boolean expressions; nodes are sub-expression values.
- ``swaps``: people trade objects in turns; nodes are who holds what after each trade.
- ``arith``: nested integer arithmetic; nodes are sub-expression values.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from .graph import check_problem, choice, node

PEOPLE = (
    "Alma",
    "Boris",
    "Cleo",
    "Dev",
    "Esme",
    "Fynn",
    "Gia",
    "Hector",
    "Ines",
    "Jules",
    "Kemal",
    "Liv",
    "Milo",
    "Noor",
    "Otto",
    "Pia",
    "Rafa",
    "Sami",
    "Tess",
    "Ugo",
    "Vik",
    "Wren",
    "Yara",
    "Zeno",
)
ANIMALS = (
    "the cat",
    "the dog",
    "the rabbit",
    "the owl",
    "the fox",
    "the bear",
    "the mouse",
    "the eagle",
)
PROPS = (
    "big",
    "cold",
    "green",
    "kind",
    "quiet",
    "rough",
    "young",
    "round",
    "nice",
    "red",
    "blue",
    "smart",
)
OBJECTS = (
    "a red ball",
    "a blue book",
    "a green hat",
    "a yellow kite",
    "a black umbrella",
    "a white mug",
)
ITEMS = ("a vase", "a lamp", "a clock", "a plant", "a radio", "a mirror", "a globe")


def _yes_no(rng: random.Random, answer: bool) -> tuple[list[dict[str, Any]], int]:
    style = rng.choice(["letters", "option_n", "yesno"])
    if style == "yesno":
        options = [
            {"key": "yes", "description": "Yes"},
            {"key": "no", "description": "No"},
        ]
        return options, 0 if answer else 1
    return choice(
        rng, "yes" if answer else "no", ["no" if answer else "yes"], style=style
    )


def _rules(rng: random.Random) -> tuple[str, str, bool, list[dict[str, Any]]] | None:
    entities = rng.sample(ANIMALS, 2)
    props = rng.sample(PROPS, 7)
    facts = {(e, p) for e in entities for p in rng.sample(props[:4], 2)}
    rules = []
    for _ in range(rng.randint(4, 6)):
        body = rng.sample(props, rng.randint(1, 2))
        head = rng.choice([p for p in props if p not in body])
        rules.append((tuple(body), head))
    known = {f: None for f in facts}  # fact -> (rule index, premises) or None for given
    changed = True
    while changed:
        changed = False
        for index, (body, head) in enumerate(rules):
            for e in entities:
                if all((e, b) in known for b in body) and (e, head) not in known:
                    known[(e, head)] = (index, [(e, b) for b in body])
                    changed = True
    derived = [f for f, why in known.items() if why is not None]
    if not derived:
        return None
    target_entity = rng.choice(entities)
    if rng.random() < 0.55:
        candidates = [f for f in derived if f[0] == target_entity]
        if not candidates:
            return None
        target = rng.choice(candidates)
        answer = True
    else:
        missing = [p for p in props if (target_entity, p) not in known]
        if not missing:
            return None
        target = (target_entity, rng.choice(missing))
        answer = False
    text = [f"{e.capitalize()} is {p}." for e, p in sorted(facts)]
    for body, head in rules:
        text.append(f"If something is {' and '.join(body)} then it is {head}.")
    rng.shuffle(text)
    state = (
        " ".join(text)
        + " Nothing else is known, and anything not stated or derivable is false."
    )
    question = f"Is {target[0]} {target[1]}?"
    nodes: list[dict[str, Any]] = []
    ids: dict[tuple[str, str], str] = {}

    def add(fact: tuple[str, str]) -> None:
        if fact in ids or known.get(fact) is None:
            return
        _, premises = known[fact]
        for premise in premises:
            add(premise)
        node_id = f"n{len(nodes) + 1}"
        deps = [ids[p] for p in premises if p in ids]
        nodes.append(
            node(
                node_id,
                f"Can it be shown that {fact[0]} is {fact[1]}?",
                kind="noul",
                depends_on=deps,
                answer=True,
                statement=f"{fact[0].capitalize()} is {fact[1]}.",
                false_statement=f"{fact[0].capitalize()} is not {fact[1]}.",
            )
        )
        ids[fact] = node_id

    if answer:
        _, premises = known[target]
        for premise in premises:
            add(premise)
    else:
        for fact in rng.sample(derived, min(3, len(derived))):
            add(fact)
        # a blocked rule: one whose head is the target and whose body is not satisfied
        for body, head in rules:
            if head == target[1]:
                missing = [b for b in body if (target[0], b) not in known]
                if missing:
                    nodes.append(
                        node(
                            f"n{len(nodes) + 1}",
                            f"Can it be shown that {target[0]} is {missing[0]}?",
                            kind="noul",
                            answer=False,
                            statement=f"Nothing shows that {target[0]} is {missing[0]}.",
                            false_statement=f"It can be shown that {target[0]} is {missing[0]}.",
                        )
                    )
                    break
    if not nodes:
        return None
    return state, question, answer, nodes


def _truth(rng: random.Random) -> tuple[str, str, bool, list[dict[str, Any]]]:
    people = rng.sample(PEOPLE, rng.randint(4, 6))
    truthful = [rng.random() < 0.5]
    lines = [f"{people[0]} {'tells the truth' if truthful[0] else 'lies'}."]
    for i in range(1, len(people)):
        claims_truth = rng.random() < 0.5
        lines.append(
            f"{people[i]} says {people[i - 1]} {'tells the truth' if claims_truth else 'lies'}."
        )
        truthful.append(claims_truth == truthful[i - 1])
    nodes = []
    for i in range(1, len(people) - 1):
        nodes.append(
            node(
                f"n{i}",
                f"Does {people[i]} tell the truth?",
                kind="noul",
                depends_on=[f"n{i - 1}"] if i > 1 else [],
                answer=truthful[i],
                statement=f"{people[i]} {'tells the truth' if truthful[i] else 'lies'}.",
                false_statement=f"{people[i]} {'lies' if truthful[i] else 'tells the truth'}.",
            )
        )
    return " ".join(lines), f"Does {people[-1]} tell the truth?", truthful[-1], nodes


def _order(
    rng: random.Random,
) -> tuple[str, str, list[dict[str, Any]], int, list[dict[str, Any]]] | None:
    n = rng.randint(3, 5)
    items = rng.sample(ITEMS, n)
    truth = items[:]
    rng.shuffle(truth)
    pos = {item: i for i, item in enumerate(truth)}
    words = ["first", "second", "third", "fourth", "fifth"]
    pool = []
    for a, b in itertools.permutations(items, 2):
        if pos[a] < pos[b]:
            pool.append(
                (
                    f"{a.capitalize()} is somewhere to the left of {b}.",
                    lambda p, a=a, b=b: p[a] < p[b],
                )
            )
        if pos[a] == pos[b] - 1:
            pool.append(
                (
                    f"{a.capitalize()} is directly to the left of {b}.",
                    lambda p, a=a, b=b: p[a] == p[b] - 1,
                )
            )
    for a in items:
        pool.append(
            (
                f"{a.capitalize()} is in the {words[pos[a]]} position from the left.",
                lambda p, a=a, i=pos[a]: p[a] == i,
            )
        )
        if pos[a] == n - 1:
            pool.append(
                (f"{a.capitalize()} is the rightmost.", lambda p, a=a: p[a] == n - 1)
            )
    rng.shuffle(pool)
    clues: list[tuple[str, Any]] = []
    perms = [dict(zip(p, range(n))) for p in itertools.permutations(items)]
    for clue in pool:
        if any(not clue[1](p) for p in perms):
            clues.append(clue)
            perms = [p for p in perms if clue[1](p)]
        if len(perms) == 1:
            break
    if len(perms) != 1 or len(clues) > 6:
        return None
    state = (
        f"{n} objects stand on a shelf in a row from left to right: "
        + ", ".join(sorted(items))
        + ". "
        + " ".join(c for c, _ in clues)
    )
    target = rng.choice(items)
    question = f"Which object is in the {words[pos[target]]} position from the left?"
    options, label = choice(rng, target, [i for i in items if i != target])
    nodes = []
    for i, item in enumerate(sorted(items, key=lambda x: pos[x])):
        if item == target:
            continue
        opts, lab = choice(
            rng, words[pos[item]], [w for w in words[:n] if w != words[pos[item]]][:3]
        )
        nodes.append(
            node(
                f"n{len(nodes) + 1}",
                f"In which position from the left is {item}?",
                kind="choice",
                depends_on=[f"n{len(nodes)}"] if nodes else [],
                options=opts,
                label=lab,
                statement=f"{item.capitalize()} is in the {words[pos[item]]} position from the left.",
                false_statement=f"{item.capitalize()} is in the "
                f"{words[(pos[item] + 1) % n]} position from the left.",
            )
        )
    return state, question, options, label, nodes


def _boolean(rng: random.Random) -> tuple[str, str, bool, list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []

    def build(depth: int, top: bool = False) -> tuple[str, bool, str | None]:
        if depth == 0 or (not top and rng.random() < 0.25):
            value = rng.random() < 0.5
            return ("True" if value else "False"), value, None
        op = rng.choice(["and", "or", "not"])
        if op == "not":
            text, value, child = build(depth - 1)
            expr, result, deps = f"not {text}", not value, [child]
        else:
            lt, lv, lc = build(depth - 1)
            rt, rv, rc = build(depth - 1)
            expr = f"( {lt} {op} {rt} )"
            result = (lv and rv) if op == "and" else (lv or rv)
            deps = [lc, rc]
        node_id = f"n{len(nodes) + 1}"
        nodes.append(
            node(
                node_id,
                f"Does the sub-expression {expr} evaluate to True?",
                kind="noul",
                depends_on=[d for d in deps if d],
                answer=result,
                statement=f"The sub-expression {expr} evaluates to {result}.",
                false_statement=f"The sub-expression {expr} evaluates to {not result}.",
            )
        )
        return expr, result, node_id

    expr, value, _ = build(rng.randint(2, 4), top=True)
    nodes.pop()  # the whole expression is the final question
    return (
        f"Evaluate the boolean expression: {expr}",
        "Is the expression True?",
        value,
        nodes,
    )


def _swaps(
    rng: random.Random,
) -> tuple[str, str, list[dict[str, Any]], int, list[dict[str, Any]]]:
    n = rng.randint(3, 5)
    people = rng.sample(PEOPLE, n)
    objects = rng.sample(OBJECTS, n)
    holding = dict(zip(people, objects))
    lines = [f"At the start, {', '.join(f'{p} has {o}' for p, o in holding.items())}."]
    nodes = []
    last: dict[str, str] = {}
    steps = rng.randint(3, 5)
    for step in range(steps):
        a, b = rng.sample(people, 2)
        holding[a], holding[b] = holding[b], holding[a]
        lines.append(f"Then {a} and {b} swap what they hold.")
        if step < steps - 1:
            who = rng.choice([a, b])
            opts, lab = choice(
                rng, holding[who], [o for o in objects if o != holding[who]][:3]
            )
            deps = sorted({last[x] for x in (a, b) if x in last})
            node_id = f"n{len(nodes) + 1}"
            nodes.append(
                node(
                    node_id,
                    f"After swap {step + 1}, what does {who} hold?",
                    kind="choice",
                    depends_on=deps,
                    options=opts,
                    label=lab,
                    statement=f"After swap {step + 1}, {who} holds {holding[who]}.",
                    false_statement=f"After swap {step + 1}, {who} holds "
                    f"{opts[(lab + 1) % len(opts)]['description']}.",
                )
            )
            last[a] = last[b] = node_id
    target = rng.choice(people)
    options, label = choice(
        rng, holding[target], [o for o in objects if o != holding[target]]
    )
    return (
        " ".join(lines),
        f"At the end, what does {target} hold?",
        options,
        label,
        nodes,
    )


def _arith(rng: random.Random) -> tuple[str, str, int, list[str], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []

    def build(depth: int, top: bool = False) -> tuple[str, int, str | None]:
        if depth == 0 or (not top and rng.random() < 0.2):
            v = rng.randint(-9, 9)
            return (f"({v})" if v < 0 else str(v)), v, None
        op = rng.choice(["+", "-", "*"])
        lt, lv, lc = build(depth - 1)
        rt, rv, rc = build(depth - 1)
        value = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
        expr = f"({lt} {op} {rt})"
        slips = [
            lv - rv if op == "+" else lv + rv if op == "-" else lv + rv,
            -value,
            value + 1,
        ]
        opts, lab = choice(rng, str(value), [str(s) for s in slips if s != value][:3])
        node_id = f"n{len(nodes) + 1}"
        nodes.append(
            node(
                node_id,
                f"What is the value of {expr}?",
                kind="choice",
                depends_on=[d for d in (lc, rc) if d],
                options=opts,
                label=lab,
                statement=f"{expr} equals {value}.",
                false_statement=f"{expr} equals {opts[(lab + 1) % len(opts)]['description']}.",
            )
        )
        return expr, value, node_id

    expr, value, _ = build(rng.randint(2, 3), top=True)
    top = nodes.pop()
    wrong = [str(-value), str(value + 1), str(value - 1), str(value * 2)]
    wrong += [
        o["description"] for o in top["options"] if o["description"] != str(value)
    ]
    return (
        f"Compute {expr}.",
        "What is the value of the expression?",
        value,
        wrong,
        nodes,
    )


SUBFAMILIES = ("rules", "truth", "order", "boolean", "swaps", "arith")


def generate(rng: random.Random, pid: str) -> dict[str, Any] | None:
    sub = rng.choice(SUBFAMILIES)
    if sub in ("rules", "truth", "boolean"):
        made = {"rules": _rules, "truth": _truth, "boolean": _boolean}[sub](rng)
        if made is None:
            return None
        state, question, answer, nodes = made
        options, label = _yes_no(rng, answer)
        task_type = "choice"
        if rng.random() < 0.3:  # the native Noul form
            task_type = "noul"
            options = [
                {"key": "false", "description": "No"},
                {"key": "true", "description": "Yes"},
            ]
            label = 1 if answer else 0
    elif sub in ("order", "swaps"):
        made = (_order if sub == "order" else _swaps)(rng)
        if made is None:
            return None
        state, question, options, label, nodes = made
        task_type = "choice"
    else:
        state, question, value, wrong, nodes = _arith(rng)
        options, label = choice(
            rng,
            str(value),
            list(dict.fromkeys(w for w in wrong if w != str(value)))[:3],
        )
        task_type = "choice"
    if not nodes:
        return None
    if rng.random() < 0.5:
        state = {"puzzle": state}
    problem = {
        "pid": pid,
        "family": f"reasoning_logic_{sub}",
        "source": "decision2_reasoning_program_v1",
        "render_template": f"logic_{sub}_v1",
        "language": "en",
        "licence": "program-generated",
        "state": state,
        "final": {
            "task_type": task_type,
            "instructions": question,
            "options": options,
            "label": label,
        },
        "nodes": nodes,
        "audit": {"sub": sub},
    }
    check_problem(problem)
    return problem
