"""Program-verified causal-inference problems over small binary causal graphs.

A story gives a causal structure (confounding, mediation, chain or collision) over a treatment X, an outcome Y and a
third variable, plus conditional probabilities in words. Questions climb the three rungs: an observational
comparison, an average treatment effect (with the correct adjustment), and counterfactual-style effects (effect of
treatment on the treated, natural direct / indirect effects). Graph nodes are the structural facts, the adjustment
decision and the intermediate probabilities the exact computation uses; every answer is computed from the numbers.
"""

from __future__ import annotations

import random
from typing import Any

from .graph import check_problem, choice, node

# Each theme: (X yes, X no, Y yes, Y no, third yes, third no, population)
THEMES = (
    (
        "drinks coffee",
        "does not drink coffee",
        "sleeps poorly",
        "sleeps well",
        "has a stressful job",
        "has a calm job",
        "office workers",
    ),
    (
        "takes the new drug",
        "does not take the new drug",
        "recovers within a week",
        "does not recover within a week",
        "is over sixty",
        "is under sixty",
        "patients",
    ),
    (
        "attends the tutoring program",
        "skips the tutoring program",
        "passes the exam",
        "fails the exam",
        "has a quiet place to study",
        "has no quiet place to study",
        "students",
    ),
    (
        "uses the irrigation system",
        "does not use the irrigation system",
        "has a large harvest",
        "has a small harvest",
        "has fertile soil",
        "has poor soil",
        "farms",
    ),
    (
        "runs the ad campaign",
        "does not run the ad campaign",
        "gains new customers",
        "gains no new customers",
        "is in a big city",
        "is in a small town",
        "shops",
    ),
    (
        "wears a fitness tracker",
        "does not wear a fitness tracker",
        "loses weight",
        "does not lose weight",
        "has a gym membership",
        "has no gym membership",
        "adults",
    ),
    (
        "gets the software update",
        "does not get the software update",
        "crashes often",
        "rarely crashes",
        "is an old model",
        "is a new model",
        "phones",
    ),
    (
        "receives the fertilizer",
        "does not receive the fertilizer",
        "blooms early",
        "blooms late",
        "grows in full sun",
        "grows in shade",
        "rose bushes",
    ),
)
STRUCTURES = ("confounding", "mediation", "chain", "collision")


def _p(rng: random.Random) -> float:
    return rng.choice(range(5, 96)) / 100


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def generate(rng: random.Random, pid: str) -> dict[str, Any] | None:
    x1, x0, y1, y0, z1, z0, pop = rng.choice(THEMES)
    structure = rng.choice(STRUCTURES)
    third = {
        "confounding": "a common cause",
        "mediation": "a mediator",
        "chain": "a mediator",
        "collision": "a common effect",
    }[structure]
    edges: list[tuple[str, str]]
    if structure == "confounding":
        edges = [("Z", "X"), ("Z", "Y"), ("X", "Y")]
    elif structure == "mediation":
        edges = [("X", "Z"), ("Z", "Y"), ("X", "Y")]
    elif structure == "chain":
        edges = [("X", "Z"), ("Z", "Y")]
    else:
        edges = [("X", "Z"), ("Y", "Z")]
    names = {
        "X": f"whether a member {x1}",
        "Y": f"whether a member {y1}",
        "Z": f"whether a member {z1}",
    }
    lines = [
        f"Consider a closed population of {pop} in which only the following causal relations hold "
        f"and nothing else matters."
    ]
    for a, b in edges:
        lines.append(f"{names[a].capitalize()} directly influences {names[b]}.")
    # parameters
    pz = _p(rng)
    px_z = {1: _p(rng), 0: _p(rng)}
    px = _p(rng)
    pz_x = {1: _p(rng), 0: _p(rng)}
    py_xz = {(a, b): _p(rng) for a in (0, 1) for b in (0, 1)}
    py_x = {1: _p(rng), 0: _p(rng)}
    py = _p(rng)
    pz_xy = {(a, b): _p(rng) for a in (0, 1) for b in (0, 1)}
    facts: list[str] = []
    if structure == "confounding":
        facts += [
            f"The probability that a member {z1} is {_pct(pz)}.",
            f"For a member who {z1}, the probability that it {x1} is {_pct(px_z[1])}.",
            f"For a member who {z0}, the probability that it {x1} is {_pct(px_z[0])}.",
        ]
        for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
            facts.append(
                f"For a member who {x1 if a else x0} and {z1 if b else z0}, the probability that it {y1} "
                f"is {_pct(py_xz[(a, b)])}."
            )
    elif structure in ("mediation", "chain"):
        facts += [
            f"The probability that a member {x1} is {_pct(px)}.",
            f"For a member who {x1}, the probability that it {z1} is {_pct(pz_x[1])}.",
            f"For a member who {x0}, the probability that it {z1} is {_pct(pz_x[0])}.",
        ]
        if structure == "mediation":
            for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
                facts.append(
                    f"For a member who {x1 if a else x0} and {z1 if b else z0}, the probability that "
                    f"it {y1} is {_pct(py_xz[(a, b)])}."
                )
        else:
            for b in (1, 0):
                facts.append(
                    f"For a member who {z1 if b else z0}, the probability that it {y1} is "
                    f"{_pct(py_xz[(0, b)])}."
                )
                py_xz[(1, b)] = py_xz[(0, b)]
    else:
        facts += [
            f"The probability that a member {x1} is {_pct(px)}.",
            f"The probability that a member {y1} is {_pct(py)}.",
        ]
        for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
            facts.append(
                f"For a member who {x1 if a else x0} and {y1 if b else y0}, the probability that it {z1} "
                f"is {_pct(pz_xy[(a, b)])}."
            )
    rng.shuffle(facts)
    state_text = " ".join(lines + facts)

    nodes: list[dict[str, Any]] = []

    def add_noul(
        question: str,
        answer: bool,
        statement: str,
        false_statement: str,
        deps: list[str],
    ) -> str:
        node_id = f"n{len(nodes) + 1}"
        nodes.append(
            node(
                node_id,
                question,
                kind="noul",
                depends_on=deps,
                answer=answer,
                statement=statement,
                false_statement=false_statement,
            )
        )
        return node_id

    def add_value(
        question: str, value: float, wrong: list[float], statement: str, deps: list[str]
    ) -> str:
        node_id = f"n{len(nodes) + 1}"
        options, label = choice(
            rng, _pct(value), [_pct(w) for w in wrong if abs(w - value) > 0.0005][:3]
        )
        false = options[(label + 1) % len(options)]["description"]
        nodes.append(
            node(
                node_id,
                question,
                kind="choice",
                depends_on=deps,
                options=options,
                label=label,
                statement=statement.format(v=_pct(value)),
                false_statement=statement.format(v=false),
            )
        )
        return node_id

    has_xy = ("X", "Y") in edges
    s1 = add_noul(
        f"Does {names['Z']} play the role of {third} for {names['X']} and {names['Y']}?",
        True,
        f"{names['Z'].capitalize()} is {third} for {names['X']} and {names['Y']}.",
        f"{names['Z'].capitalize()} is not {third} for {names['X']} and {names['Y']}.",
        [],
    )
    s2 = add_noul(
        f"Does {names['X']} influence {names['Y']} directly, not only through other variables?",
        has_xy,
        f"{names['X'].capitalize()} {'does' if has_xy else 'does not'} directly influence {names['Y']}.",
        f"{names['X'].capitalize()} {'does not' if has_xy else 'does'} directly influence {names['Y']}.",
        [],
    )
    adjust = structure == "confounding"
    s3 = add_noul(
        f"To measure the effect of {names['X']} on {names['Y']}, must the analysis adjust for "
        f"{names['Z']}?",
        adjust,
        f"The analysis {'must' if adjust else 'must not'} adjust for {names['Z']}.",
        f"The analysis {'must not' if adjust else 'must'} adjust for {names['Z']}.",
        [s1],
    )

    kinds = {
        "confounding": ["ate", "ett", "correlation"],
        "mediation": ["ate", "nde", "nie"],
        "chain": ["ate", "correlation"],
        "collision": ["collider"],
    }[structure]
    query = rng.choice(kinds)
    if structure == "confounding":
        do1 = pz * py_xz[(1, 1)] + (1 - pz) * py_xz[(1, 0)]
        do0 = pz * py_xz[(0, 1)] + (1 - pz) * py_xz[(0, 0)]
        p_x1 = pz * px_z[1] + (1 - pz) * px_z[0]
        pz_given_x1 = pz * px_z[1] / p_x1
        pz_given_x0 = pz * (1 - px_z[1]) / (1 - p_x1)
        obs1 = pz_given_x1 * py_xz[(1, 1)] + (1 - pz_given_x1) * py_xz[(1, 0)]
        obs0 = pz_given_x0 * py_xz[(0, 1)] + (1 - pz_given_x0) * py_xz[(0, 0)]
        if query == "ate":
            a = add_value(
                f"If every member were made to be one that {x1}, what share of members would be ones that {y1}?",
                do1,
                [
                    obs1,
                    py_xz[(1, 1)],
                    py_xz[(1, 0)],
                    (py_xz[(1, 1)] + py_xz[(1, 0)]) / 2,
                ],
                f"Under an intervention that makes every member one that {x1}, the share that {y1} "
                f"would be {{v}}.",
                [s3],
            )
            b = add_value(
                f"If every member were made to be one that {x0}, what share of members would be ones that {y1}?",
                do0,
                [
                    obs0,
                    py_xz[(0, 1)],
                    py_xz[(0, 0)],
                    (py_xz[(0, 1)] + py_xz[(0, 0)]) / 2,
                ],
                f"Under an intervention that makes every member one that {x0}, the share that {y1} "
                f"would be {{v}}.",
                [s3],
            )
            answer = do1 > do0
            question = f"Would making every member one that {x1} raise the chance that a member {y1}?"
            deps_final = [a, b]
        elif query == "ett":
            a = add_value(
                f"For a member who {x1}, what is the probability that it {z1}?",
                pz_given_x1,
                [pz, px_z[1], 1 - pz_given_x1],
                f"For a member who {x1}, the probability that it {z1} is {{v}}.",
                [],
            )
            ett = pz_given_x1 * (py_xz[(1, 1)] - py_xz[(0, 1)]) + (1 - pz_given_x1) * (
                py_xz[(1, 0)] - py_xz[(0, 0)]
            )
            answer = ett > 0
            question = (
                f"Take a member that actually {x1}. Had it instead been one that {x0}, would it have been "
                f"less likely to be one that {y1}?"
            )
            deps_final = [a, s3]
        else:
            a = add_value(
                f"For a member who {x1}, what is the probability that it {y1}?",
                obs1,
                [do1, py_xz[(1, 1)], obs0],
                f"For a member who {x1}, the probability that it {y1} is {{v}}.",
                [],
            )
            b = add_value(
                f"For a member who {x0}, what is the probability that it {y1}?",
                obs0,
                [do0, py_xz[(0, 1)], obs1],
                f"For a member who {x0}, the probability that it {y1} is {{v}}.",
                [],
            )
            answer = obs1 > obs0
            question = f"Is a member that {x1} more likely to be one that {y1} than a member that {x0}?"
            deps_final = [a, b]
    elif structure in ("mediation", "chain"):
        do1 = pz_x[1] * py_xz[(1, 1)] + (1 - pz_x[1]) * py_xz[(1, 0)]
        do0 = pz_x[0] * py_xz[(0, 1)] + (1 - pz_x[0]) * py_xz[(0, 0)]
        if query in ("ate", "correlation"):
            a = add_value(
                f"For a member who {x1}, what is the probability that it {y1}?",
                do1,
                [py_xz[(1, 1)], py_xz[(1, 0)], do0, pz_x[1]],
                f"For a member who {x1}, the probability that it {y1} is {{v}}.",
                [s2],
            )
            b = add_value(
                f"For a member who {x0}, what is the probability that it {y1}?",
                do0,
                [py_xz[(0, 1)], py_xz[(0, 0)], do1, pz_x[0]],
                f"For a member who {x0}, the probability that it {y1} is {{v}}.",
                [s2],
            )
            answer = do1 > do0
            question = (
                f"Would making every member one that {x1} raise the chance that a member {y1}?"
                if query == "ate"
                else f"Is a member that {x1} more likely to be one that {y1} than a member that {x0}?"
            )
            deps_final = [a, b]
        elif query == "nde":
            nde = pz_x[0] * (py_xz[(1, 1)] - py_xz[(0, 1)]) + (1 - pz_x[0]) * (
                py_xz[(1, 0)] - py_xz[(0, 0)]
            )
            a = add_value(
                f"For a member who {x0}, what is the probability that it {z1}?",
                pz_x[0],
                [pz_x[1], 1 - pz_x[0], px],
                f"For a member who {x0}, the probability that it {z1} is {{v}}.",
                [],
            )
            answer = nde > 0
            question = (
                f"Keeping {names['Z']} as it would be for a member that {x0}, does being one that {x1} "
                f"raise the chance that a member {y1}?"
            )
            deps_final = [a, s2]
        else:
            nie = (pz_x[1] - pz_x[0]) * (py_xz[(0, 1)] - py_xz[(0, 0)])
            a = add_noul(
                f"Is a member that {x1} more likely to be one that {z1} than a member that {x0}?",
                pz_x[1] > pz_x[0],
                f"The probability that a member {z1} is {_pct(pz_x[1])} if it {x1} and "
                f"{_pct(pz_x[0])} if it {x0}.",
                f"The probability that a member {z1} is {_pct(pz_x[0])} if "
                f"it {x1} and {_pct(pz_x[1])} if it {x0}.",
                [],
            )
            b = add_noul(
                f"For a member that {x0}, does being one that {z1} raise the chance that it {y1}?",
                py_xz[(0, 1)] > py_xz[(0, 0)],
                f"For a member that {x0}, the probability that it {y1} is "
                f"{_pct(py_xz[(0, 1)])} if it {z1} and {_pct(py_xz[(0, 0)])} if it {z0}.",
                f"For a member that {x0}, the probability that it {y1} is {_pct(py_xz[(0, 0)])} if it "
                f"{z1} and {_pct(py_xz[(0, 1)])} if it {z0}.",
                [],
            )
            answer = nie > 0
            question = (
                f"Does being one that {x1} raise the chance that a member {y1} through its effect on "
                f"{names['Z']} alone?"
            )
            deps_final = [a, b]
        if abs(do1 - do0) < 0.005:
            return None
    else:
        # collider: X and Y are independent overall; conditioning on the common effect makes them dependent
        if rng.random() < 0.5:
            answer = False
            question = f"Overall, is a member that {x1} more likely to be one that {y1} than a member that {x0}?"
            deps_final = [s1, s2]
        else:

            def py_given(x: int) -> float:
                num = py * pz_xy[(x, 1)]
                return num / (num + (1 - py) * pz_xy[(x, 0)])

            c1, c0 = py_given(1), py_given(0)
            if abs(c1 - c0) < 0.005:
                return None
            a = add_value(
                f"For a member who {x1} and {z1}, what is the probability that it {y1}?",
                c1,
                [py, pz_xy[(1, 1)], c0],
                f"For a member who {x1} and {z1}, the probability that it {y1} is "
                f"{{v}}.",
                [s1],
            )
            b = add_value(
                f"For a member who {x0} and {z1}, what is the probability that it {y1}?",
                c0,
                [py, pz_xy[(0, 1)], c1],
                f"For a member who {x0} and {z1}, the probability that it {y1} is "
                f"{{v}}.",
                [s1],
            )
            answer = c1 > c0
            question = (
                f"Among members that {z1}, is a member that {x1} more likely to be one that {y1} than a "
                f"member that {x0}?"
            )
            deps_final = [a, b]
    final_options, final_label = choice(
        rng,
        "yes" if answer else "no",
        ["no" if answer else "yes"],
        style=rng.choice(["letters", "option_n"]),
    )
    style = rng.randrange(3)
    if style == 0:
        state: Any = state_text
        instructions = question
    elif style == 1:
        state = {"setting": state_text}
        instructions = f"{question} Answer yes or no."
    else:
        state = {"story": state_text, "question": question}
        instructions = "Answer the causal question in the state with yes or no."
    problem = {
        "pid": pid,
        "family": "reasoning_causal_graph",
        "source": "decision2_reasoning_program_v1",
        "render_template": f"causal_{structure}_{query}_v1",
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
            "structure": structure,
            "query": query,
            "answer": answer,
            "deps_final": deps_final,
        },
    }
    check_problem(problem)
    return problem
