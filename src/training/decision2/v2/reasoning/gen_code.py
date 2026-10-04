"""Program-verified code-execution problems: predict the return value of a short Python function.

The generator is a tiny interpreter: it draws a sequence of statements over a string or an integer list, applies
each to concrete values as it goes, and renders the same statements as Python source. Every statement that changes
a variable is a graph node ("what is the value of <var> after this line?") whose parents are the nodes that last
set the variables it reads. Final and node distractors are the values reached by plausible tracing slips: skipping
a statement, applying it twice, or using the variable's value from one step earlier.
"""

from __future__ import annotations

import random
import string
from typing import Any, Callable

from .graph import check_problem, choice, node

WORDS = (
    "banana",
    "rotation",
    "mississippi",
    "level",
    "orchard",
    "cascade",
    "parallel",
    "bookkeeper",
    "ladder",
    "network",
    "harbor",
    "pepper",
    "abracadabra",
    "committee",
    "balloon",
    "satellite",
    "tomato",
    "address",
    "coffee",
    "success",
    "kayak",
    "letter",
    "assessment",
    "puzzle",
    "village",
    "occurrence",
)

Op = tuple[
    str, list[str], str, Callable[[dict[str, Any]], Any]
]  # (code, reads, writes, apply -> new value)


def _string_ops(rng: random.Random, env: dict[str, Any]) -> list[Op]:
    s: str = env["s"]
    letters = sorted(set(s)) or ["a"]
    a = rng.choice(letters)
    b = rng.choice(string.ascii_lowercase + "*#")
    i = rng.randint(0, max(0, len(s) - 2))
    j = rng.randint(i + 1, max(i + 1, len(s)))
    t = rng.choice(["x", "ab", "!", "zz", "_end", "pre_"])
    k = rng.randint(1, 3)
    ops: list[Op] = [
        ("s = s.upper()", ["s"], "s", lambda e: e["s"].upper()),
        ("s = s[::-1]", ["s"], "s", lambda e: e["s"][::-1]),
        (f"s = s.replace({a!r}, {b!r})", ["s"], "s", lambda e: e["s"].replace(a, b)),
        (f"s = s[{i}:{j}]", ["s"], "s", lambda e: e["s"][i:j]),
        (f"s = s + {t!r}", ["s"], "s", lambda e: e["s"] + t),
        (f"s = {t!r} + s", ["s"], "s", lambda e: t + e["s"]),
        (f"s = s.strip({a!r})", ["s"], "s", lambda e: e["s"].strip(a)),
        ("s = s.swapcase()", ["s"], "s", lambda e: e["s"].swapcase()),
        ("s = s.capitalize()", ["s"], "s", lambda e: e["s"].capitalize()),
        (f"n = s.count({a!r})", ["s"], "n", lambda e: e["s"].count(a)),
        (f"n = s.find({a!r})", ["s"], "n", lambda e: e["s"].find(a)),
        ("n = len(s)", ["s"], "n", lambda e: len(e["s"])),
        (f"s = s[:{k}] + s[-{k}:]", ["s"], "s", lambda e: e["s"][:k] + e["s"][-k:]),
        (
            f"s = s.center(len(s) + {2 * k}, '*')",
            ["s"],
            "s",
            lambda e: e["s"].center(len(e["s"]) + 2 * k, "*"),
        ),
        ("s = ''.join(sorted(s))", ["s"], "s", lambda e: "".join(sorted(e["s"]))),
        (
            "s = ''.join(c for c in s if c not in 'aeiou')",
            ["s"],
            "s",
            lambda e: "".join(c for c in e["s"] if c not in "aeiou"),
        ),
    ]
    if "n" in env:
        ops += [
            ("s = s + str(n)", ["s", "n"], "s", lambda e: e["s"] + str(e["n"])),
            (f"n = n * {k + 1}", ["n"], "n", lambda e: e["n"] * (k + 1)),
            (f"n = n + {k}", ["n"], "n", lambda e: e["n"] + k),
            (
                "s = s * 2 if n % 2 == 0 else s[::-1]",
                ["s", "n"],
                "s",
                lambda e: e["s"] * 2 if e["n"] % 2 == 0 else e["s"][::-1],
            ),
        ]
    return ops


def _list_ops(rng: random.Random, env: dict[str, Any]) -> list[Op]:
    lst: list[int] = env["l"]
    v = rng.randint(-5, 20)
    i = rng.randint(0, max(0, len(lst) - 1))
    j = rng.randint(i + 1, max(i + 1, len(lst)))
    k = rng.randint(2, 4)
    present = rng.choice(lst) if lst else v
    ops: list[Op] = [
        (f"l.append({v})", ["l"], "l", lambda e: e["l"] + [v]),
        (f"l.insert({i}, {v})", ["l"], "l", lambda e: e["l"][:i] + [v] + e["l"][i:]),
        ("l.reverse()", ["l"], "l", lambda e: e["l"][::-1]),
        ("l.sort()", ["l"], "l", lambda e: sorted(e["l"])),
        (
            "l = sorted(l, reverse=True)",
            ["l"],
            "l",
            lambda e: sorted(e["l"], reverse=True),
        ),
        (f"l = l[{i}:{j}]", ["l"], "l", lambda e: e["l"][i:j]),
        (f"l = [x * {k} for x in l]", ["l"], "l", lambda e: [x * k for x in e["l"]]),
        (f"l = [x + {v} for x in l]", ["l"], "l", lambda e: [x + v for x in e["l"]]),
        (
            "l = [x for x in l if x % 2 == 0]",
            ["l"],
            "l",
            lambda e: [x for x in e["l"] if x % 2 == 0],
        ),
        (
            f"l = [x for x in l if x > {v}]",
            ["l"],
            "l",
            lambda e: [x for x in e["l"] if x > v],
        ),
        ("n = sum(l)", ["l"], "n", lambda e: sum(e["l"])),
        ("n = len(l)", ["l"], "n", lambda e: len(e["l"])),
        (f"n = l.count({present})", ["l"], "n", lambda e: e["l"].count(present)),
    ]
    if lst:
        ops += [
            ("n = max(l)", ["l"], "n", lambda e: max(e["l"])),
            ("l.pop()", ["l"], "l", lambda e: e["l"][:-1]),
            (f"l[{i}] = {v}", ["l"], "l", lambda e: e["l"][:i] + [v] + e["l"][i + 1 :]),
        ]
    if "n" in env:
        ops += [
            ("l.append(n)", ["l", "n"], "l", lambda e: e["l"] + [e["n"]]),
            (f"n = n - {k}", ["n"], "n", lambda e: e["n"] - k),
            (
                "l = [x - n for x in l]",
                ["l", "n"],
                "l",
                lambda e: [x - e["n"] for x in e["l"]],
            ),
        ]
    return ops


def _run(ops: list[Op], env0: dict[str, Any]) -> dict[str, Any]:
    env = dict(env0)
    for _code, _reads, writes, apply in ops:
        env[writes] = apply(env)
    return env


def generate(rng: random.Random, pid: str) -> dict[str, Any] | None:
    kind = rng.choice(["str", "list"])
    if kind == "str":
        arg = (
            rng.choice(WORDS)
            if rng.random() < 0.7
            else "".join(rng.choice("abcde") for _ in range(rng.randint(5, 9)))
        )
        env: dict[str, Any] = {"s": arg}
        header, first = "def f(text):", "    s = text"
        call = f"f({arg!r})"
        draw = _string_ops
    else:
        arg = [rng.randint(-3, 15) for _ in range(rng.randint(3, 6))]
        env = {"l": list(arg)}
        header, first = "def f(nums):", "    l = list(nums)"
        call = f"f({arg!r})"
        draw = _list_ops
    ops: list[Op] = []
    for _ in range(rng.randint(3, 6)):
        candidates = draw(rng, env)
        code, reads, writes, apply = rng.choice(candidates)
        try:
            value = apply(env)
        except (IndexError, ValueError, TypeError):
            return None
        if (
            isinstance(value, str)
            and len(value) > 40
            or isinstance(value, list)
            and len(value) > 12
        ):
            return None
        ops.append((code, reads, writes, apply))
        env[writes] = value
    ret_var = "s" if kind == "str" else "l"
    if "n" in env and rng.random() < 0.4:
        ret_expr = f"({ret_var}, n)"
        result = (env[ret_var], env["n"])
    else:
        ret_expr = ret_var
        result = env[ret_var]
    source = "\n".join(
        [header, first, *[f"    {c}" for c, *_ in ops], f"    return {ret_expr}"]
    )

    def returned(e: dict[str, Any]) -> Any:
        return (e[ret_var], e["n"]) if ret_expr != ret_var else e[ret_var]

    # slips: skip one statement, apply one twice, swap two neighbours
    start = {"s": arg} if kind == "str" else {"l": list(arg)}
    wrong = []
    for idx in range(len(ops)):
        for variant in (
            ops[:idx] + ops[idx + 1 :],
            ops[: idx + 1] + [ops[idx]] + ops[idx + 1 :],
            ops[:idx] + ops[idx + 1 : idx + 2] + [ops[idx]] + ops[idx + 2 :],
        ):
            try:
                e = _run(variant, start)
                wrong.append(repr(returned(e)))
            except (IndexError, ValueError, TypeError, KeyError):
                continue
    final_options, final_label = choice(
        rng, repr(result), _shuffled_unique(rng, wrong, repr(result))[:3]
    )
    if len(final_options) < 3:
        return None
    # trace nodes
    nodes = []
    provenance: dict[str, set[str]] = (
        {}
    )  # variable -> node ids its current value was derived from
    trace_env = dict(start)
    for idx, (code, reads, writes, apply) in enumerate(ops):
        inherited = set().union(*(provenance.get(r, set()) for r in reads))
        line = (
            idx + 3
        )  # 1-based source line: def, the copy of the argument, then the statements
        before = trace_env.get(writes)
        trace_env[writes] = apply(trace_env)
        value = trace_env[writes]
        node_wrong = [repr(before)] if before is not None else []
        try:
            node_wrong.append(repr(apply(dict(trace_env))))  # applied twice
        except (IndexError, ValueError, TypeError):
            pass
        if idx > 0:
            try:
                prev = _run(
                    ops[: idx - 1] + [ops[idx]], start
                )  # previous statement skipped
                node_wrong.append(repr(prev[writes]))
            except (IndexError, ValueError, TypeError, KeyError):
                pass
        node_wrong = list(dict.fromkeys(w for w in node_wrong if w != repr(value)))
        if not node_wrong:
            provenance[writes] = inherited
            continue
        options, label = choice(rng, repr(value), node_wrong[:3])
        node_id = f"n{len(nodes) + 1}"
        nodes.append(
            node(
                node_id,
                f"After line {line} (`{code}`) runs, what is the value of {writes}?",
                kind="choice",
                depends_on=sorted(inherited),
                options=options,
                label=label,
                statement=f"After line {line} (`{code}`), {writes} is {value!r}.",
                false_statement=f"After line {line} (`{code}`), {writes} is {node_wrong[0]}.",
            )
        )
        provenance[writes] = {node_id}
    if not nodes:
        return None
    style = rng.randrange(3)
    if style == 0:
        state: Any = {"code": source, "input": call}
        instructions = "Choose the value that f returns for the given input. Candidates are Python literals."
    elif style == 1:
        state = f"```python\n{source}\n```\nCall: {call}"
        instructions = "What does the call return?"
    else:
        state = {"function": source, "call": call, "task": "predict the return value"}
        instructions = "Which Python value is returned by the call?"
    problem = {
        "pid": pid,
        "family": "reasoning_code_trace",
        "source": "decision2_reasoning_program_v1",
        "render_template": f"code_{kind}_v1",
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
        "audit": {"source": source, "call": call, "result": repr(result)},
    }
    check_problem(problem)
    return problem


def _shuffled_unique(rng: random.Random, items: list[str], exclude: str) -> list[str]:
    out = sorted({x for x in items if x != exclude})
    rng.shuffle(out)
    return out
