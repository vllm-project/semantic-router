"""SK2: licence-clean decisions for phishing, causal-ladder style reasoning and chess (d3 same-skill set 2).

    python -m d25.vega.data.sk2 fetch --raw DIR
    python -m d25.vega.data.sk2 build --raw DIR --index DIR --holdouts holdouts-v2.json --suite DIR \
        --protected protected-items.jsonl.gz --tokenizer PATH --out DIR [--seed 20261012] [--workers 8]

Parts (one question per row; 70% of items in the Decision Index kit's layout for the matching benchmark, 30%
natural variants):
- ``sk2-causal``: procedural causal-ladder questions (association, intervention, counterfactual, mediation,
  adjustment sets, explaining away, deterministic counterfactuals) over random binary causal graphs. The yes/no
  answer is computed from the probabilities exactly as displayed. Own wording; no CLadder text or data.
- ``sk2-chess``: Lichess puzzle positions (CC0; the solution move is the gold move, every immediate mate counts
  when the solution mates) and Lichess engine evaluations (CC0; positions whose best line beats every other
  evaluated line by a margin, or the principal-variation move). All legal moves are offered.
- ``sk2-phish``: real phishing / legitimate URLs (pirocheto/phishing-url CC BY 4.0, UCI PhiUSIIL CC BY 4.0) as
  URL decisions and as emails whose sender, subject, body and link text come from neutral templates drawn
  independently of the label, so only the link and the sender/link relation carry the answer. The templates carry
  no lure or urgency wording in either class: label-independent lures would teach that such wording is uninformative,
  which is false for real mail. URLs and hosts that occur in the PhishNChips suite or proxy items are excluded.
Never used: CLadder (any version), ChessBench / searchless_chess, PhishNChips. Rows pass holdouts and the decontam
index; a flagged row drops its whole item.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from d25.vega.data.holdouts import Holdouts
from d25.vega.data.sk1 import (
    _tok_count,
    _tok_init,
    card_licence,
    describe_lengths,
    retry,
)
from d25.vega.data.util import (
    choice_question,
    make_row,
    noul_question,
    rank,
    read_jsonl,
    rng_for,
    sha,
    sha256_file,
    write_json,
    write_jsonl,
)

SEED = 20261012
MAX_TOKENS = 8160
DEV_PER_MILLE = 20

HF_SOURCES = {
    "lichess_puzzles": (
        "Lichess/chess-puzzles",
        ["data/train-00000-of-00003.parquet"],
        "cc0-1.0",
    ),
    "lichess_evals": (
        "Lichess/chess-position-evaluations",
        ["data/data_0000.parquet"],
        "cc0-1.0",
    ),
    "pirocheto_urls": ("pirocheto/phishing-url", ["data/*.parquet"], "cc-by-4.0"),
}
URL_SOURCES = {
    "phiusiil": (
        "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        "cc-by-4.0",
        "UCI 967 PhiUSIIL Phishing URL Dataset (doi 10.1016/j.cose.2023.103545)",
    ),
}
BENCHMARK_EXCLUDED = [
    "cladder",
    "searchless_chess",
    "chessbench",
    "phishnchips",
    "arelit",
]
BUDGET = {
    "causal": 7000,
    "puzzles": 8000,
    "evals": 3000,
    "emails": 2000,
    "urls": 2600,
}

# ------------------------------------------------------------------------------------------------ causal

CAUSAL_STORIES = [
    # unit, (X np, pos, neg), (Z np, pos, neg), (Y np, pos, neg)
    (
        "students",
        ("tutoring", "attend tutoring", "skip tutoring"),
        ("parental support", "have supportive parents", "lack supportive parents"),
        ("exam success", "pass the final exam", "fail the final exam"),
    ),
    (
        "patients",
        (
            "the new medication",
            "take the new medication",
            "do not take the new medication",
        ),
        ("immune strength", "have a strong immune system", "have a weak immune system"),
        ("recovery", "recover within a week", "stay ill after a week"),
    ),
    (
        "farms",
        ("irrigation", "use irrigation", "do not use irrigation"),
        ("soil quality", "have fertile soil", "have poor soil"),
        ("harvest size", "produce a large harvest", "produce a small harvest"),
    ),
    (
        "employees",
        (
            "the training program",
            "join the training program",
            "stay out of the training program",
        ),
        ("prior experience", "have prior experience", "have no prior experience"),
        ("promotion", "get promoted", "miss promotion"),
    ),
    (
        "drivers",
        ("speeding", "drive above the limit", "keep to the limit"),
        ("night driving", "drive mostly at night", "drive mostly by day"),
        ("accidents", "have an accident", "avoid accidents"),
    ),
    (
        "adults",
        ("smoking", "smoke", "do not smoke"),
        ("tar buildup", "have tar deposits in their lungs", "have clear lungs"),
        ("lung disease", "develop lung disease", "stay free of lung disease"),
    ),
    (
        "shops",
        ("advertising", "run online ads", "run no ads"),
        ("location", "sit in a central location", "sit on the outskirts"),
        ("sales", "reach high sales", "have low sales"),
    ),
    (
        "children",
        ("vaccination", "get vaccinated", "stay unvaccinated"),
        ("daycare attendance", "attend daycare", "stay at home"),
        ("measles", "catch measles", "avoid measles"),
    ),
    (
        "houses",
        ("insulation", "have insulation", "lack insulation"),
        ("climate", "stand in a cold region", "stand in a mild region"),
        ("heating cost", "have high heating bills", "have low heating bills"),
    ),
    (
        "runners",
        ("stretching", "stretch before races", "never stretch"),
        ("coaching", "train with a coach", "train alone"),
        ("injury", "get injured", "stay uninjured"),
    ),
    (
        "cities",
        ("bike lanes", "build bike lanes", "build no bike lanes"),
        ("density", "have a dense center", "sprawl outward"),
        ("air quality", "meet air-quality targets", "miss air-quality targets"),
    ),
    (
        "customers",
        ("the discount email", "receive a discount email", "receive no email"),
        ("loyalty membership", "are loyalty members", "are not loyalty members"),
        ("purchasing", "make a purchase", "make no purchase"),
    ),
    (
        "plants",
        ("sunlight", "get full sunlight", "stay in shade"),
        ("watering", "are watered daily", "are watered rarely"),
        ("flowering", "flower this season", "do not flower this season"),
    ),
    (
        "teams",
        ("extra practice", "hold extra practice", "skip extra practice"),
        ("captain experience", "have a veteran captain", "have a rookie captain"),
        ("winning", "win the league", "lose the league"),
    ),
    (
        "servers",
        ("caching", "enable caching", "disable caching"),
        ("traffic load", "receive heavy traffic", "receive light traffic"),
        ("timeouts", "time out", "respond in time"),
    ),
    (
        "workers",
        ("remote work", "work remotely", "work on site"),
        ("commute length", "live far from the office", "live near the office"),
        ("burnout", "report burnout", "report no burnout"),
    ),
    (
        "villages",
        ("the new pump", "install a new pump", "keep the old pump"),
        ("river access", "lie near a river", "lie far from rivers"),
        ("water safety", "have safe drinking water", "lack safe drinking water"),
    ),
    (
        "sleepers",
        ("evening reading", "read before bed", "use screens before bed"),
        ("late caffeine", "drink coffee late", "avoid late coffee"),
        ("sleep quality", "sleep well", "sleep badly"),
    ),
    (
        "restaurants",
        ("online reviews", "collect online reviews", "collect no reviews"),
        ("chef skill", "employ a trained chef", "employ an untrained cook"),
        ("full tables", "fill most tables", "leave most tables empty"),
    ),
    (
        "trees",
        ("pruning", "are pruned every year", "are never pruned"),
        ("rainfall", "grow in a wet area", "grow in a dry area"),
        ("fruit yield", "bear plenty of fruit", "bear little fruit"),
    ),
]
NONSENSE = [
    "zorvex",
    "glimmet",
    "pradle",
    "quintar",
    "vosk",
    "meliph",
    "trandle",
    "osquin",
    "brillet",
    "kesh",
    "yavorn",
    "dulmet",
    "fropple",
    "skerra",
    "nimbet",
    "wexol",
]
PREAMBLE = [
    "Consider a closed hypothetical world in which only the causal links below exist and no other factors play any role:",
    "Imagine an isolated scenario governed only by the following causal relationships, with nothing else involved:",
    "Assume a self-contained setting where exactly these causal connections hold and no hidden causes exist:",
]


def _nonsense_story(rng: random.Random):
    a, b, c = rng.sample(NONSENSE, 3)
    unit = rng.choice(["individuals", "units", "subjects", "members"])
    return (
        unit,
        (a, f"have {a}", f"lack {a}"),
        (b, f"have {b}", f"lack {b}"),
        (c, f"show {c}", f"do not show {c}"),
    )


def _pct(rng: random.Random, lo: int = 4, hi: int = 96) -> int:
    return rng.randint(lo, hi)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def causal_item(rng: random.Random) -> dict[str, Any] | None:
    story = _nonsense_story(rng) if rng.random() < 0.3 else rng.choice(CAUSAL_STORIES)
    unit, (xn, xp, xq), (zn, zp, zq), (yn, yp, yq) = story
    graph = rng.choice(
        [
            "confounder",
            "confounder",
            "mediator",
            "mediator",
            "chain",
            "fork",
            "collider",
            "direct",
            "det",
        ]
    )
    lines: list[str] = []
    edges: list[str] = []
    if graph == "confounder":
        edges = [
            f"{_cap(zn)} directly affects {xn} and {yn}.",
            f"{_cap(xn)} directly affects {yn}.",
        ]
    elif graph == "mediator":
        edges = [
            f"{_cap(xn)} directly affects {zn} and {yn}.",
            f"{_cap(zn)} directly affects {yn}.",
        ]
    elif graph == "chain":
        edges = [
            f"{_cap(xn)} directly affects {zn}.",
            f"{_cap(zn)} directly affects {yn}.",
        ]
    elif graph == "fork":
        edges = [f"{_cap(zn)} directly affects {xn} and {yn}."]
    elif graph == "collider":
        edges = [
            f"{_cap(xn)} directly affects {zn}.",
            f"{_cap(yn)} directly affects {zn}.",
        ]
    elif graph == "direct":
        edges = [f"{_cap(xn)} directly affects {yn}."]
    else:
        edges = [f"{_cap(xn)} and {zn} directly affect {yn}."]

    def y_line(cond: str, p: int) -> str:
        return f"For {unit} who {cond}, the probability that they {yp} is {p}%."

    answer: bool | None = None
    query = ""
    polar = rng.random() < 0.5
    more, less = ("more", "less") if polar else ("less", "more")

    def decide(effect: float, margin: float = 0.02) -> bool | None:
        if abs(effect) < margin:
            return None
        return (effect > 0) if polar else (effect < 0)

    if graph == "det":
        op = rng.choice(["or", "and", "and not"])
        if op == "or":
            rule = f"{_cap(unit)} {yp} exactly when they {xp} or they {zp}."
            f = lambda x, z: x or z  # noqa: E731
        elif op == "and":
            rule = f"{_cap(unit)} {yp} exactly when they {xp} and they {zp}."
            f = lambda x, z: x and z  # noqa: E731
        else:
            rule = f"{_cap(unit)} {yp} exactly when they {xp} and they {zq}."
            f = lambda x, z: x and not z  # noqa: E731
        x0, z0 = rng.random() < 0.5, rng.random() < 0.5
        lines = [rule]
        query = (
            f"Consider the {unit} who {xp if x0 else xq} and {zp if z0 else zq}. If they instead "
            f"{xq if x0 else xp}, with nothing else changed, would they {yp}?"
        )
        answer = bool(f(not x0, z0))
        kind = "det_counterfactual"
    elif graph in ("confounder", "fork"):
        pz = _pct(rng)
        px = [_pct(rng), _pct(rng)]
        if graph == "confounder":
            py = [[_pct(rng), _pct(rng)], [_pct(rng), _pct(rng)]]  # py[x][z]
        else:
            pyz = [_pct(rng), _pct(rng)]
            py = [[pyz[0], pyz[1]], [pyz[0], pyz[1]]]
        q = rng.choice(
            ["ate", "ate", "corr", "ett", "backdoor", "marginal"]
            if graph == "confounder"
            else ["ate", "corr", "corr", "backdoor"]
        )
        z_line = f"{pz}% of {unit} {zp}."
        x_lines = [
            f"Among {unit} who {zq}, {px[0]}% {xp}.",
            f"Among {unit} who {zp}, {px[1]}% {xp}.",
        ]
        conds = {
            (0, 0): f"{xq} and {zq}",
            (0, 1): f"{xq} and {zp}",
            (1, 0): f"{xp} and {zq}",
            (1, 1): f"{xp} and {zp}",
        }
        pzv = pz / 100
        pzs = [1 - pzv, pzv]
        if q == "ate":
            if graph == "confounder":
                lines = [z_line] + [
                    y_line(conds[(x, z)], py[x][z]) for x in (0, 1) for z in (0, 1)
                ]
                effect = sum(pzs[z] * (py[1][z] - py[0][z]) / 100 for z in (0, 1))
            else:
                lines = (
                    [z_line]
                    + x_lines
                    + [
                        f"For {unit} who {zq}, the probability that they {yp} is {pyz[0]}%.",
                        f"For {unit} who {zp}, the probability that they {yp} is {pyz[1]}%.",
                    ]
                )
                effect = 0.0
            query = f"Does {xn} make {unit} {more} likely to {yp}?"
            answer = False if graph == "fork" else decide(effect)
            kind = "ate"
        elif q == "ett":
            lines = (
                [z_line]
                + x_lines
                + [y_line(conds[(x, z)], py[x][z]) for x in (0, 1) for z in (0, 1)]
            )
            pxz = [px[0] / 100, px[1] / 100]
            norm = sum(pzs[z] * pxz[z] for z in (0, 1))
            post = [pzs[z] * pxz[z] / norm for z in (0, 1)]
            effect = sum(post[z] * (py[1][z] - py[0][z]) / 100 for z in (0, 1))
            query = f"Consider the {unit} who {xp}. If that had not been the case, would they have been {less} likely to {yp}?"
            answer = decide(effect)
            kind = "ett"
        elif q == "backdoor":
            lines = []
            query = f"To estimate how {xn} affects whether {unit} {yp}, should we analyze each {zn} group separately instead of the whole population?"
            answer = True
            kind = "backdoor_set"
        elif q == "marginal":
            lines = (
                [z_line]
                + x_lines
                + [y_line(conds[(x, z)], py[x][z]) for x in (0, 1) for z in (0, 1)]
            )
            pxz = [px[0] / 100, px[1] / 100]
            p_y = sum(
                pzs[z] * (pxz[z] * py[1][z] + (1 - pxz[z]) * py[0][z]) / 100
                for z in (0, 1)
            )
            query = f"Across all {unit}, is it more likely than not that they {yp if polar else yq}?"
            effect = (p_y - 0.5) if polar else (0.5 - p_y)
            answer = None if abs(p_y - 0.5) < 0.02 else effect > 0
            kind = "marginal"
        else:
            pxz = [px[0] / 100, px[1] / 100]
            p_x = sum(pzs[z] * pxz[z] for z in (0, 1))
            p_xy = sum(pzs[z] * pxz[z] * py[1][z] / 100 for z in (0, 1))
            p_nxy = sum(pzs[z] * (1 - pxz[z]) * py[0][z] / 100 for z in (0, 1))
            sx, sxy, snxy = round(100 * p_x), round(100 * p_xy), round(100 * p_nxy)
            if not 3 <= sx <= 97 or sxy >= sx or snxy >= 100 - sx:
                return None
            lines = [
                f"{sx}% of {unit} {xp}.",
                f"{sxy}% of {unit} {xp} and {yp}.",
                f"{snxy}% of {unit} {xq} and {yp}.",
            ]
            effect = sxy / sx - snxy / (100 - sx)
            query = f"Are {unit} who {xp} {more} likely to {yp} than {unit} who {xq}?"
            answer = decide(effect)
            kind = "correlation"
    elif graph in ("mediator", "chain"):
        pz = [_pct(rng), _pct(rng)]  # P(Z|X=x)
        if graph == "mediator":
            py = [[_pct(rng), _pct(rng)], [_pct(rng), _pct(rng)]]
        else:
            pyz = [_pct(rng), _pct(rng)]
            py = [[pyz[0], pyz[1]], [pyz[0], pyz[1]]]
        z_lines = [
            f"Among {unit} who {xq}, {pz[0]}% {zp}.",
            f"Among {unit} who {xp}, {pz[1]}% {zp}.",
        ]
        conds = {
            (0, 0): f"{xq} and {zq}",
            (0, 1): f"{xq} and {zp}",
            (1, 0): f"{xp} and {zq}",
            (1, 1): f"{xp} and {zp}",
        }
        zs = [[1 - pz[0] / 100, pz[0] / 100], [1 - pz[1] / 100, pz[1] / 100]]
        q = rng.choice(
            ["ate", "nde", "nie", "backdoor"]
            if graph == "mediator"
            else ["ate", "ate", "backdoor"]
        )
        if graph == "mediator":
            ylines = [y_line(conds[(x, z)], py[x][z]) for x in (0, 1) for z in (0, 1)]
        else:
            ylines = [
                f"For {unit} who {zq}, the probability that they {yp} is {pyz[0]}%.",
                f"For {unit} who {zp}, the probability that they {yp} is {pyz[1]}%.",
            ]
        lines = z_lines + ylines
        if q == "ate":
            effect = (
                sum(zs[1][z] * py[1][z] - zs[0][z] * py[0][z] for z in (0, 1)) / 100
            )
            query = f"Does {xn} make {unit} {more} likely to {yp}?"
            answer = decide(effect)
            kind = "ate"
        elif q == "nde":
            effect = sum(zs[0][z] * (py[1][z] - py[0][z]) for z in (0, 1)) / 100
            query = f"Leaving aside any effect that runs through {zn}, does {xn} make {unit} {more} likely to {yp}?"
            answer = decide(effect)
            kind = "nde"
        elif q == "nie":
            effect = sum((zs[1][z] - zs[0][z]) * py[0][z] for z in (0, 1)) / 100
            query = f"Does {xn} make {unit} {more} likely to {yp} by way of {zn}?"
            answer = decide(effect)
            kind = "nie"
        else:
            lines = []
            query = f"To estimate the total effect of {xn} on whether {unit} {yp}, should we analyze each {zn} group separately instead of the whole population?"
            answer = False
            kind = "backdoor_set"
    elif graph == "collider":
        px, py = _pct(rng), _pct(rng)
        pzc = [[_pct(rng), _pct(rng)], [_pct(rng), _pct(rng)]]  # P(Z|X=x,Y=y)
        q = rng.choice(["ate", "explain", "explain", "backdoor"])
        conds = {
            (0, 0): f"{xq} and {yq}",
            (0, 1): f"{xq} and {yp}",
            (1, 0): f"{xp} and {yq}",
            (1, 1): f"{xp} and {yp}",
        }
        base = [f"{px}% of {unit} {xp}.", f"{py}% of {unit} {yp}."]
        zl = [
            f"For {unit} who {conds[(x, y)]}, the probability that they {zp} is {pzc[x][y]}%."
            for x in (0, 1)
            for y in (0, 1)
        ]
        if q == "ate":
            lines = base + zl
            query = f"Does {xn} make {unit} {more} likely to {yp}?"
            answer = False
            kind = "ate_collider"
        elif q == "explain":
            lines = base + zl
            pyv = py / 100

            def post(x: int) -> float:
                a = pyv * pzc[x][1] / 100
                b = (1 - pyv) * pzc[x][0] / 100
                return a / (a + b)

            effect = post(1) - post(0)
            query = f"Among {unit} who {zp}, are those who {xp} {more} likely to {yp} than those who {xq}?"
            answer = decide(effect)
            kind = "explaining_away"
        else:
            lines = []
            query = f"To estimate how {xn} affects whether {unit} {yp}, should we analyze each {zn} group separately instead of the whole population?"
            answer = False
            kind = "backdoor_set"
    else:
        px = _pct(rng)
        py = [_pct(rng), _pct(rng)]
        q = rng.choice(["ate", "marginal"])
        lines = [
            f"{px}% of {unit} {xp}.",
            f"For {unit} who {xq}, the probability that they {yp} is {py[0]}%.",
            f"For {unit} who {xp}, the probability that they {yp} is {py[1]}%.",
        ]
        if q == "ate":
            query = f"Does {xn} make {unit} {more} likely to {yp}?"
            answer = decide((py[1] - py[0]) / 100)
            kind = "ate"
        else:
            p_y = (px * py[1] + (100 - px) * py[0]) / 10000
            query = f"Across all {unit}, is it more likely than not that they {yp if polar else yq}?"
            answer = (
                None
                if abs(p_y - 0.5) < 0.02
                else ((p_y > 0.5) if polar else (p_y < 0.5))
            )
            kind = "marginal"
    if answer is None:
        return None
    context = rng.choice(PREAMBLE) + " " + " ".join(edges)
    if lines:
        context += "\n\n" + " ".join(lines)
    return {
        "context": context,
        "query": query,
        "answer": bool(answer),
        "graph": graph,
        "kind": kind,
        "nonsense": story[0] in ("individuals", "units", "subjects", "members"),
    }


def causal_items(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"sk2-causal:{seed}")
    pools: dict[bool, list[dict]] = {True: [], False: []}
    seen = set()
    while min(len(pools[True]), len(pools[False])) < n // 2:
        it = causal_item(rng)
        if it is None:
            continue
        key = sha(it["context"] + "|" + it["query"], 20)
        if key in seen:
            continue
        seen.add(key)
        it["key"] = key
        pools[it["answer"]].append(it)
    return pools[True][: n // 2] + pools[False][: n - n // 2]


# ------------------------------------------------------------------------------------------------- chess

CHESS_KIT = (
    "Choose the strongest legal move for the side to move. Board rows run from rank 8 to rank 1; columns a to h. "
    "Uppercase pieces are white. All legal moves are supplied."
)


def position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def blocked_positions(suite_dir: Path, protected: Path) -> set[str]:
    out = set()
    files = [
        suite_dir / f for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz")
    ] + [protected]
    for path in files:
        for r in read_jsonl(path):
            st = r.get("state")
            if isinstance(st, dict) and isinstance(st.get("fen"), str):
                out.add(position_key(st["fen"]))
    return out


def chess_state(board) -> dict[str, Any]:
    return {
        "fen": board.fen(),
        "board": str(board),
        "side_to_move": "white" if board.turn else "black",
    }


def puzzle_positions(
    path: Path, n: int, seed: int, blocked: set[str]
) -> tuple[list[dict], dict]:
    import chess
    import pyarrow.parquet as pq

    stats: Counter = Counter()
    cols = ["PuzzleId", "FEN", "Moves", "Rating", "Popularity", "NbPlays", "Themes"]
    cands = []
    for batch in pq.ParquetFile(path).iter_batches(columns=cols, batch_size=200_000):
        for r in batch.to_pylist():
            stats["read"] += 1
            if (r["Popularity"] or 0) < 60 or (r["NbPlays"] or 0) < 200:
                continue
            if int(rank(f"pz:{r['PuzzleId']}", seed)[:8], 16) % 100 >= 3:
                continue
            cands.append(r)
        if len(cands) >= 3 * n:
            break
    cands.sort(key=lambda r: rank(r["PuzzleId"], seed))
    out = []
    for r in cands:
        if len(out) >= n:
            break
        moves = r["Moves"].split()
        if len(moves) < 2:
            continue
        board = chess.Board(r["FEN"])
        try:
            board.push(chess.Move.from_uci(moves[0]))
            best = chess.Move.from_uci(moves[1])
        except ValueError:
            stats["bad_move"] += 1
            continue
        legal = list(board.legal_moves)
        if best not in legal or len(legal) < 2:
            stats["illegal_or_forced"] += 1
            continue
        if position_key(board.fen()) in blocked:
            stats["blocked_position"] += 1
            continue
        board.push(best)
        mates = board.is_checkmate()
        board.pop()
        gold = {best.uci()}
        if mates:
            for m in legal:
                board.push(m)
                if board.is_checkmate():
                    gold.add(m.uci())
                board.pop()
        out.append(
            {
                "key": f"pz:{r['PuzzleId']}",
                "group": f"game:{r['PuzzleId']}",
                "board": board.copy(stack=False),
                "gold": sorted(gold),
                "source": "lichess_puzzles",
                "extra": {"rating": r["Rating"], "themes": r["Themes"]},
            }
        )
    stats["kept"] = len(out)
    return out, dict(stats)


def eval_positions(
    path: Path, n: int, seed: int, blocked: set[str]
) -> tuple[list[dict], dict]:
    import chess
    import pyarrow.parquet as pq

    stats: Counter = Counter()
    by_fen: dict[str, list[tuple]] = defaultdict(list)
    for i, batch in enumerate(
        pq.ParquetFile(path).iter_batches(
            columns=["fen", "line", "depth", "cp", "mate"], batch_size=500_000
        )
    ):
        for r in batch.to_pylist():
            stats["read"] += 1
            line = (r["line"] or "").split()
            if not line:
                continue
            by_fen[r["fen"]].append((line[0], r["cp"], r["mate"], int(r["depth"] or 0)))
        if i >= 5:
            break
    multi = sum(1 for v in by_fen.values() if len({m for m, *_ in v}) >= 2)
    stats["positions"] = len(by_fen)
    stats["multi_line_positions"] = multi

    def score(cp, mate, white: bool) -> float:
        s = (
            (100000 - abs(mate) * 100) * (1 if mate > 0 else -1)
            if mate is not None
            else float(cp or 0)
        )
        return s if white else -s

    picks = []
    for fen, lines in by_fen.items():
        if int(rank(f"ev:{fen}", seed)[:8], 16) % 100 >= 20:
            continue
        white = fen.split()[1] == "w"
        best_by_move: dict[str, tuple[float, int]] = {}
        for move, cp, mate, depth in lines:
            if cp is None and mate is None:
                continue
            s = score(cp, mate, white)
            if move not in best_by_move or depth > best_by_move[move][1]:
                best_by_move[move] = (s, depth)
        if not best_by_move:
            continue
        ranked = sorted(best_by_move.items(), key=lambda kv: -kv[1][0])
        if multi:
            if len(ranked) < 2 or ranked[0][1][0] - ranked[1][1][0] < 80:
                continue
        elif ranked[0][1][1] < 30:
            continue
        picks.append((fen, ranked[0][0], len(ranked)))
    picks.sort(key=lambda x: rank(x[0], seed))
    out = []
    for fen, move, nlines in picks:
        if len(out) >= n:
            break
        board = chess.Board(fen + " 0 1")
        try:
            best = chess.Move.from_uci(move)
        except ValueError:
            continue
        legal = list(board.legal_moves)
        if best not in legal or len(legal) < 3 or board.is_game_over():
            stats["illegal_or_small"] += 1
            continue
        if position_key(board.fen()) in blocked:
            stats["blocked_position"] += 1
            continue
        out.append(
            {
                "key": f"ev:{sha(fen, 16)}",
                "group": f"pos:{sha(position_key(fen), 16)}",
                "board": board,
                "gold": [best.uci()],
                "source": "lichess_evals",
                "extra": {
                    "lines": nlines,
                    "rule": (
                        "margin>=80cp over the other evaluated lines"
                        if multi
                        else "depth>=30 principal variation"
                    ),
                },
            }
        )
    stats["kept"] = len(out)
    return out, dict(stats)


def chess_rows(
    pos: dict[str, Any], seed: int
) -> list[tuple[str, Any, dict, list[float], str]]:
    board = pos["board"]
    rng = rng_for(f"chess-render:{pos['key']}", seed)
    legal = list(board.legal_moves)
    gold = set(pos["gold"])
    side = "White" if board.turn else "Black"
    if rng.random() < 0.7:
        criteria = {m.uci(): {"uci": m.uci(), "san": board.san(m)} for m in legal}
        target = [1.0 if k in gold else 0.0 for k in criteria]
        return [
            (
                "0",
                chess_state(board),
                choice_question(CHESS_KIT, criteria),
                target,
                "kit",
            )
        ]
    if rng.random() < 0.5:
        criteria = {board.san(m): None for m in legal}
        target = [1.0 if m.uci() in gold else 0.0 for m in legal]
        question = choice_question(
            f"{side} to move in this position (FEN). Which move is best?", criteria
        )
        return [("0", board.fen(), question, target, "natural")]
    others = [m for m in legal if m.uci() not in gold]
    pick = rng.sample(others, min(len(others), rng.randint(3, 5)))
    best_moves = [m for m in legal if m.uci() in gold][:1]
    cand = pick + best_moves
    rng.shuffle(cand)
    criteria = {m.uci(): board.san(m) for m in cand}
    target = [1.0 if m.uci() in gold else 0.0 for m in cand]
    state = {"fen": board.fen(), "side_to_move": side.lower(), "board": str(board)}
    question = choice_question(
        "Which of these candidate moves is strongest for the side to move?", criteria
    )
    return [("0", state, question, target, "natural")]


# ---------------------------------------------------------------------------------------------- phishing

PHISH_KIT = {
    "verdict": (
        "This email arrived in the user's inbox and contains a link (`link_url`). Is it a phishing attempt, or a "
        "legitimate message whose link is safe to click?",
        {
            "phishing": "The email or its link is a phishing attempt or otherwise malicious. The user should not click the link.",
            "legitimate": "A normal, legitimate message. The link is safe to click.",
        },
    ),
    "is_phishing": (
        "This email is a phishing attempt and the user should not click the link in `link_url`.",
        {
            "true": "The email or its link is a phishing attempt or otherwise malicious.",
            "false": "The email is a normal, legitimate message and the link is safe to click.",
        },
    ),
    "verdict_alt_click": (
        "Should the user click the link in this email?",
        {
            "do_not_click": "No. The link may be phishing or malicious.",
            "click": "Yes. The email and its link look legitimate.",
        },
    ),
    "verdict_alt_minimal": (
        "Classify this email.",
        {"phishing": None, "legitimate": None},
    ),
}
FIRST = [
    "Sarah",
    "Daniel",
    "Priya",
    "Marco",
    "Aisha",
    "Tom",
    "Elena",
    "Kenji",
    "Laura",
    "Omar",
    "Grace",
    "Lukas",
    "Nina",
    "Carlos",
    "Hannah",
    "Ravi",
    "Julia",
    "Sam",
    "Mei",
    "Peter",
]
LAST = [
    "Jenkins",
    "Morrison",
    "Shah",
    "Rossi",
    "Khan",
    "Baker",
    "Novak",
    "Sato",
    "Fischer",
    "Haddad",
    "Lee",
    "Weber",
    "Park",
    "Diaz",
    "Clarke",
    "Iyer",
    "Costa",
    "Reid",
    "Chen",
    "Olsen",
]
WEBMAIL = ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com"]
EMAIL_TEMPLATES = [
    (
        "Meeting notes - {project}",
        "Hi team,\n\nI've put together the notes from our {project} meeting. You can {verb} here: {url}. Let me know if anything is missing.\n\nBest,\n{name}",
        "access them here",
    ),
    (
        "Invoice {num} for {month}",
        "Hello {recipient},\n\nPlease find the invoice for {month} at the link below. {url}\n\nThank you for your business.\n\n{name}\n{org}",
        "View invoice",
    ),
    (
        "Shared document: {doc}",
        'Hi {recipient},\n\nI shared "{doc}" with you. Open it here: {url}\n\nThanks,\n{name}',
        "Open document",
    ),
    (
        "Your {org} account",
        "Dear {recipient},\n\nWe made some changes to how {org} accounts are managed. Details are available at {url}.\n\nRegards,\nThe {org} team",
        "Learn more",
    ),
    (
        "Delivery update for order {num}",
        "Hello,\n\nYour order {num} has an update. Track the shipment here: {url}\n\n{org} Customer Service",
        "Track shipment",
    ),
    (
        "{org} newsletter - {month}",
        "Hi {recipient},\n\nHere is our {month} roundup with product news and upcoming events. Read it online: {url}\n\nCheers,\n{org}",
        "Read online",
    ),
    (
        "Invitation: {event}",
        "Hello {recipient},\n\nYou're invited to {event}. Reserve your seat here: {url}\n\nWe hope to see you there.\n{name}",
        "Register",
    ),
    (
        "Expense report submitted",
        "Hi {recipient},\n\nThe latest expense report is ready for review. {url}\n\nThanks for taking a look.\n{name}",
        "Review report",
    ),
    (
        "Travel itinerary - {city}",
        "Hi {recipient},\n\nYour itinerary for the {city} trip is attached to your booking page: {url}\n\nSafe travels,\n{name}",
        "View itinerary",
    ),
    (
        "Support ticket {num} updated",
        "Hello,\n\nThere is a new reply on your support ticket {num}. See the conversation: {url}\n\n{org} Support",
        "View ticket",
    ),
    (
        "Contract ready for signature",
        "Dear {recipient},\n\nThe contract we discussed is ready. Review and sign it here: {url}\n\nKind regards,\n{name}",
        "Review and sign",
    ),
    (
        "Quarterly survey",
        "Hi {recipient},\n\nWe'd appreciate five minutes of your time for our quarterly survey: {url}\n\nThank you!\n{org}",
        "Take the survey",
    ),
    (
        "Photos from {event}",
        "Hey {recipient},\n\nI uploaded the photos from {event}. Have a look: {url}\n\n{name}",
        "See the photos",
    ),
    (
        "Webinar recording",
        "Hello {recipient},\n\nThe recording of last week's webinar is now available: {url}\n\nBest regards,\n{name}\n{org}",
        "Watch recording",
    ),
    (
        "Password policy update",
        "Dear user,\n\nOur password policy has been updated. Review the new requirements here: {url}\n\nIT Department\n{org}",
        "Review policy",
    ),
    (
        "Payroll statement for {month}",
        "Hi {recipient},\n\nYour payroll statement for {month} can be viewed at {url}.\n\nHR\n{org}",
        "View statement",
    ),
    (
        "Product update: {doc}",
        "Hi {recipient},\n\nWe just released {doc}. Read the release notes: {url}\n\nThe {org} team",
        "Release notes",
    ),
    (
        "Subscription renewal",
        "Hello {recipient},\n\nYour subscription renews next month. Manage your plan here: {url}\n\n{org} Billing",
        "Manage subscription",
    ),
]
PROJECTS = [
    "Project Alpha",
    "Q3 planning",
    "the vendor review",
    "Atlas migration",
    "the onboarding revamp",
]
DOCS = ["Budget 2026", "Team roster", "Launch checklist", "Version 4.2", "Client brief"]
EVENTS = [
    "the spring summit",
    "our open house",
    "the team offsite",
    "the annual gala",
    "the product launch",
]
CITIES = ["Lisbon", "Denver", "Osaka", "Toronto", "Berlin"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
FREE_HOSTS = {
    "web.app",
    "firebaseapp.com",
    "blogspot.com",
    "github.io",
    "weebly.com",
    "wixsite.com",
    "000webhostapp.com",
    "glitch.me",
    "netlify.app",
    "vercel.app",
    "herokuapp.com",
    "pages.dev",
    "workers.dev",
    "appspot.com",
    "azurewebsites.net",
    "cloudfront.net",
    "r2.dev",
    "webflow.io",
    "wordpress.com",
    "godaddysites.com",
    "yolasite.com",
    "jimdosite.com",
    "myshopify.com",
    "duckdns.org",
    "ngrok.io",
    "ipfs.io",
    "dweb.link",
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "cutt.ly",
    "rb.gy",
    "linktr.ee",
    "forms.gle",
    "google.com",
    "googleapis.com",
    "sharepoint.com",
    "dropbox.com",
    "box.com",
    "square.site",
    "carrd.co",
}
TWO_LEVEL = {
    "co.uk",
    "com.au",
    "co.jp",
    "com.br",
    "co.in",
    "co.za",
    "com.mx",
    "com.tr",
    "org.uk",
    "ac.uk",
    "gov.uk",
    "co.nz",
    "com.sg",
    "com.cn",
}


def host_of(url: str) -> str:
    try:
        u = url if "://" in url else "http://" + url
        return (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""


def registered(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def blocked_urls(suite_dir: Path, protected: Path) -> tuple[set[str], set[str]]:
    urls, hosts = set(), set()
    files = [
        suite_dir / f for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz")
    ] + [protected]
    for path in files:
        for r in read_jsonl(path):
            st = r.get("state")
            if not isinstance(st, dict) or not ("link_url" in st or "url" in st):
                continue
            for key in ("link_url", "url"):
                if isinstance(st.get(key), str) and st[key]:
                    urls.add(st[key].strip().lower())
                    h = host_of(st[key])
                    if h:
                        hosts.add(h)
            if isinstance(st.get("from"), str) and "@" in st["from"]:
                hosts.add(st["from"].split("@")[-1].strip().lower())
    return urls, hosts


def url_pool(
    raw: Path, seed: int, block_urls: set[str], block_hosts: set[str]
) -> tuple[list[dict], dict]:
    import csv
    import io

    import pyarrow.parquet as pq

    stats: Counter = Counter()
    rows: list[dict] = []
    for path in sorted((raw / "pirocheto_urls").glob("data/*.parquet")):
        for r in pq.read_table(path, columns=["url", "status"]).to_pylist():
            rows.append(
                {
                    "url": r["url"],
                    "phish": r["status"] == "phishing",
                    "title": "",
                    "src": "pirocheto_urls",
                }
            )
    with zipfile.ZipFile(raw / "phiusiil" / "phiusiil.zip") as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            reader = csv.DictReader(
                io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            )
            labels = Counter()
            tmp = []
            for r in reader:
                labels[r["label"]] += 1
                tmp.append((r["URL"], (r.get("Title") or "").strip(), r["label"]))
    # PhiUSIIL: 134,850 legitimate and 100,945 phishing URLs; the larger class is legitimate.
    legit_label = labels.most_common(1)[0][0]
    stats["phiusiil_labels"] = dict(labels)
    for url, title, label in tmp:
        rows.append(
            {
                "url": url,
                "phish": label != legit_label,
                "title": title,
                "src": "phiusiil",
            }
        )
    del tmp
    out, seen_hosts = [], set()
    for r in rows:
        url = (r["url"] or "").strip()
        h = host_of(url)
        if not url or not h or len(url) > 300 or any(c.isspace() for c in url):
            stats["bad_url"] += 1
            continue
        if url.lower() in block_urls or h in block_hosts:
            stats["blocked_phishnchips"] += 1
            continue
        if h in seen_hosts:
            stats["dup_host"] += 1
            continue
        seen_hosts.add(h)
        r["url"], r["host"] = url, h
        if len(r["title"]) > 120 or not r["title"].isprintable():
            r["title"] = ""
        out.append(r)
    out.sort(key=lambda r: rank(r["url"], seed))
    stats["pool"] = len(out)
    stats["pool_phish"] = sum(r["phish"] for r in out)
    return out, dict(stats)


def email_for(u: dict, rng: random.Random, legit_domains: list[str]) -> dict[str, str]:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    dom = registered(u["host"])
    org = dom.split(".")[0].replace("-", " ").title()
    mode = rng.random()
    if dom in FREE_HOSTS:
        mode = 0.65 + 0.35 * rng.random()
    if mode < 0.65:
        sender_dom = dom
    elif mode < 0.85:
        sender_dom = rng.choice(WEBMAIL)
        org = rng.choice(legit_domains).split(".")[0].replace("-", " ").title()
    else:
        sender_dom = rng.choice(legit_domains)
        org = sender_dom.split(".")[0].replace("-", " ").title()
    subject, body, link_text = rng.choice(EMAIL_TEMPLATES)
    fill = {
        "project": rng.choice(PROJECTS),
        "num": str(rng.randint(10000, 99999)),
        "month": rng.choice(MONTHS),
        "doc": rng.choice(DOCS),
        "event": rng.choice(EVENTS),
        "city": rng.choice(CITIES),
        "recipient": rng.choice(FIRST),
        "name": f"{first} {last}",
        "org": org,
        "verb": "find them",
        "url": u["url"],
    }
    text = body.format(**fill)
    return {
        "sender": f"{first} {last}",
        "from": f"{first.lower()}.{last.lower()}@{sender_dom}",
        "subject": subject.format(**fill),
        "body": text,
        "link_display_text": link_text,
        "link_url": u["url"],
    }


def phishing_items(
    pool: list[dict], n_emails: int, n_urls: int, seed: int
) -> list[dict]:
    phish = [r for r in pool if r["phish"]]
    legit = [r for r in pool if not r["phish"]]
    legit_domains = sorted({registered(r["host"]) for r in legit[:2000]})
    items = []
    for kind, n, offset in (("email", n_emails, 0), ("url", n_urls, n_emails // 2)):
        take = phish[offset : offset + n // 2] + legit[offset : offset + n - n // 2]
        for u in take:
            rng = rng_for(f"ph:{kind}:{u['url']}", seed)
            items.append(
                {
                    "kind": kind,
                    "key": f"{kind}:{sha(u['url'], 16)}",
                    "group": f"host:{u['host']}",
                    "url": u,
                    "phish": u["phish"],
                    "email": (
                        email_for(u, rng, legit_domains) if kind == "email" else None
                    ),
                }
            )
    return items


def phishing_rows(
    it: dict[str, Any], seed: int
) -> list[tuple[str, Any, dict, list[float], str]]:
    rng = rng_for(f"ph-render:{it['key']}", seed)
    yes = it["phish"]
    out = []
    if it["kind"] == "email":
        email = it["email"]
        if rng.random() < 0.7:
            for name in rng.sample(list(PHISH_KIT), 2):
                instructions, criteria = PHISH_KIT[name]
                if name == "is_phishing":
                    question = noul_question(instructions)
                    question["criteria"] = dict(criteria)
                    target = [0.0, 1.0] if yes else [1.0, 0.0]
                else:
                    question = choice_question(instructions, dict(criteria))
                    keys = list(criteria)
                    bad = keys[0]
                    target = [1.0 if (k == bad) == yes else 0.0 for k in keys]
                out.append((name, dict(email), question, target, "kit"))
            return out
        text = (
            f"From: {email['sender']} <{email['from']}>\nSubject: {email['subject']}\n\n{email['body']}\n\n"
            f"[{email['link_display_text']}] -> {email['link_url']}"
        )
        if rng.random() < 0.5:
            question = choice_question(
                "Is it safe for the recipient to follow the link in this message?",
                {
                    "safe": "Yes, the message and link look legitimate.",
                    "unsafe": "No, this looks like phishing.",
                },
            )
            target = [0.0, 1.0] if yes else [1.0, 0.0]
        else:
            question = noul_question("This message is a phishing attempt.")
            target = [0.0, 1.0] if yes else [1.0, 0.0]
        return [("n", text, question, target, "natural")]
    u = it["url"]
    state: dict[str, Any] = {"url": u["url"]}
    if u["title"]:
        state["page_title"] = u["title"]
    r = rng.random()
    if r < 0.4:
        question = choice_question(
            "Does this link lead to a phishing site or to a legitimate site?",
            {
                "phishing": "A phishing or malicious site.",
                "legitimate": "A legitimate site.",
            },
        )
        target = [1.0, 0.0] if yes else [0.0, 1.0]
    elif r < 0.7:
        question = noul_question(
            "Opening this link would take the user to a phishing site."
        )
        target = [0.0, 1.0] if yes else [1.0, 0.0]
    else:
        question = choice_question(
            "Would it be safe to open this link?", {"yes": None, "no": None}
        )
        target = [0.0, 1.0] if yes else [1.0, 0.0]
    return [("u", state, question, target, "natural")]


# ------------------------------------------------------------------------------------------------- fetch


def fetch(raw: Path) -> dict[str, Any]:
    import hashlib
    import urllib.request

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    report: dict[str, Any] = {}
    for name, (repo, patterns, licence) in HF_SOURCES.items():
        revision = api.dataset_info(repo).sha
        retry(
            snapshot_download,
            repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(raw / name),
            allow_patterns=patterns + ["README.md"],
            max_workers=8,
        )
        files = sorted(
            p
            for p in (raw / name).rglob("*")
            if p.is_file() and ".cache" not in p.parts
        )
        report[name] = {
            "repo": repo,
            "revision": revision,
            "licence": licence,
            "licence_card": card_licence(raw / name / "README.md"),
            "files": {str(p.relative_to(raw / name)): sha256_file(p) for p in files},
        }
        print(name, revision, report[name]["licence_card"], flush=True)
    for name, (url, licence, note) in URL_SOURCES.items():
        dest = raw / name / f"{name}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            retry(urllib.request.urlretrieve, url, str(dest))
        report[name] = {
            "url": url,
            "revision": "sha256:" + hashlib.sha256(dest.read_bytes()).hexdigest(),
            "licence": licence,
            "licence_card": licence,
            "note": note,
            "files": {dest.name: sha256_file(dest)},
        }
        print(name, report[name]["revision"][:20], flush=True)
    write_json(raw / "fetch.json", report)
    return report


# ------------------------------------------------------------------------------------------------- build


def benchmark_source_check() -> dict[str, Any]:
    sources = [repo for repo, _, _ in HF_SOURCES.values()] + [
        u for u, _, _ in URL_SOURCES.values()
    ]
    hits = [(s, bad) for s in sources for bad in BENCHMARK_EXCLUDED if bad in s.lower()]
    if hits:
        raise SystemExit(f"benchmark source used: {hits}")
    return {"excluded_patterns": BENCHMARK_EXCLUDED, "matches": []}


def finalize(
    rows: list[dict], args: argparse.Namespace, extra_manifest: dict[str, Any]
) -> dict[str, Any]:
    from d25.vega.data.decontam import check_rows

    started = time.time()
    out, seed = Path(args.out), args.seed
    holdouts = Holdouts(args.holdouts)
    held: Counter = Counter()
    for r in rows:
        reason = holdouts.check(r)
        if reason:
            held[reason] += 1
            r["_drop"] = f"holdout:{reason}"
    flags = list(check_rows(Path(args.index), rows, args.workers))
    by_reason: Counter = Counter()
    by_bench: Counter = Counter()
    bad_items = set()
    for row, flag in zip(rows, flags):
        if flag["drop"]:
            bad_items.add(row["meta"]["item"])
            by_reason.update(flag["reasons"])
            by_bench[flag["bench"] or "?"] += 1
    for row in rows:
        if row["meta"]["item"] in bad_items and "_drop" not in row:
            row["_drop"] = "decontam"
    decontam_rows = Counter(
        r["meta"]["part"] for r in rows if r.get("_drop") == "decontam"
    )
    decontam_items = Counter(i.split(":")[1] for i in bad_items)
    seen, dup = set(), Counter()
    for row in rows:
        if "_drop" in row:
            continue
        key = sha(json.dumps([row["state"], row["question"]], sort_keys=True), 24)
        if key in seen:
            row["_drop"] = "duplicate"
            dup[row["meta"]["part"]] += 1
        seen.add(key)
    kept = [r for r in rows if "_drop" not in r]
    with Pool(args.workers, initializer=_tok_init, initargs=(args.tokenizer,)) as pool:
        counts = [
            c
            for part in pool.map(
                _tok_count,
                [kept[i : i + 256] for i in range(0, len(kept), 256)],
                chunksize=1,
            )
            for c in part
        ]
    too_long: Counter = Counter()
    final = []
    for row, c in zip(kept, counts):
        if c["n"] < 0 or c["n"] > MAX_TOKENS:
            too_long[row["meta"]["part"]] += 1
            continue
        row["meta"]["n_tokens"] = c["n"]
        final.append(row)
    train, dev = [], []
    for row in sorted(final, key=lambda r: rank(r["id"], seed)):
        bucket = int(rank(row["meta"]["group"], seed)[:8], 16) % 1000
        (dev if bucket < DEV_PER_MILLE else train).append(row)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    train_name = "train-00000-of-00001.jsonl.gz"
    write_jsonl(partial / train_name, train)
    write_jsonl(partial / "dev.jsonl.gz", dev)

    def composition(rs: list[dict]) -> dict[str, Any]:
        return {
            "part": dict(Counter(r["meta"]["part"] for r in rs)),
            "source": dict(Counter(r["source"] for r in rs)),
            "render": dict(
                Counter(f"{r['meta']['part']}:{r['meta']['render']}" for r in rs)
            ),
            "orig_kind": dict(Counter(r["meta"]["orig_kind"] for r in rs)),
        }

    def positives(rs: list[dict]) -> dict[str, Any]:
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            if "positive" in r["meta"]:
                groups[r["meta"]["part"]].append(bool(r["meta"]["positive"]))
        return {k: round(sum(v) / len(v), 4) for k, v in sorted(groups.items())}

    samples = {}
    for r in train:
        key = f"{r['source']}|{r['meta']['render']}"
        if key not in samples:
            samples[key] = r
    write_json(partial / "samples.json", samples)
    manifest = {
        "name": out.name,
        "version": "v1",
        "kind": "d25-vega same-skill set 2 (phishing, causal-ladder style, chess), ws-sk1",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": {"module": "d25.vega.data.sk2", "tag": os.environ.get("D25_CODE_TAG")},
        "seed": seed,
        "rows": {"train": len(train), "dev": len(dev)},
        "composition": {"train": composition(train), "dev": composition(dev)},
        "positive_rate": {"train": positives(train), "dev": positives(dev)},
        "tokens": {"train": describe_lengths([r["meta"]["n_tokens"] for r in train])},
        **extra_manifest,
        "holdouts": {**holdouts.report(), "dropped_rows": dict(held)},
        "decontamination": {
            "index": str(args.index),
            "index_items": json.loads((Path(args.index) / "meta.json").read_text())[
                "stats"
            ]["items"],
            "rule": json.loads((Path(args.index) / "rule.json").read_text()),
            "drop_policy": "a flagged row drops every row of its item",
            "flagged_rows_by_reason": dict(by_reason),
            "flagged_rows_by_benchmark": dict(by_bench.most_common()),
            "dropped_items_by_part": dict(decontam_items),
            "dropped_rows_by_part": dict(decontam_rows),
        },
        "duplicates_dropped": dict(dup),
        "too_long_dropped": dict(too_long),
        "dev_rule": f"group-disjoint: rank(meta.group) % 1000 < {DEV_PER_MILLE}",
        "files": {},
    }
    for name in (train_name, "dev.jsonl.gz", "samples.json"):
        manifest["files"][name] = {
            "sha256": sha256_file(partial / name),
            "bytes": (partial / name).stat().st_size,
        }
    manifest["seconds"] = round(time.time() - started, 1)
    write_json(partial / "manifest.json", manifest)
    (partial / "VERIFIED").write_text(
        f"sk2 {manifest['created']} train {len(train)} dev {len(dev)}\n"
    )
    if out.exists():
        shutil.rmtree(out)
    os.replace(partial, out)
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw, seed = Path(args.raw), args.seed
    if (Path(args.out) / "VERIFIED").exists():
        print(json.dumps({"out": args.out, "already": True}))
        return json.loads((Path(args.out) / "manifest.json").read_text())
    fetched = json.loads((raw / "fetch.json").read_text())
    exclusion = benchmark_source_check()
    parse: dict[str, Any] = {}

    def meta(
        source: str,
        part: str,
        group: str,
        item: str,
        render: str,
        question: dict,
        extra: dict,
    ) -> dict:
        info = fetched.get(source)
        return {
            "dataset": (
                f"{info.get('repo') or info.get('url')}@{info['revision'][:15]}"
                if info
                else "procedural:d25.vega.data.sk2"
            ),
            "licence": info["licence"] if info else "generated (no upstream data)",
            "licence_use": "commercial",
            "source_split": "train",
            "group": f"sk2:{part}:{group}",
            "item": f"sk2:{part}:{item}",
            "hf_ids": [info["repo"]] if info and info.get("repo") else [],
            "soft": False,
            "part": part,
            "render": render,
            "orig_kind": question["type"],
            **extra,
        }

    rows: list[dict] = []
    for it in causal_items(BUDGET["causal"], seed):
        rng = rng_for(f"causal-render:{it['key']}", seed)
        yes = it["answer"]
        if rng.random() < 0.7:
            state: Any = {}
            question = choice_question(
                f"{it['context']}\n\n{it['query']}", {"A": "yes", "B": "no"}
            )
            target, render = ([1.0, 0.0] if yes else [0.0, 1.0]), "kit"
        elif rng.random() < 0.5:
            state = it["context"]
            question = choice_question(it["query"], {"yes": None, "no": None})
            target, render = ([1.0, 0.0] if yes else [0.0, 1.0]), "natural"
        else:
            state = it["context"]
            question = noul_question(it["query"])
            target, render = ([0.0, 1.0] if yes else [1.0, 0.0]), "natural"
        m = meta(
            "procedural",
            "sk2-causal",
            it["key"],
            it["key"],
            render,
            question,
            {
                "graph": it["graph"],
                "query_kind": it["kind"],
                "nonsense": it["nonsense"],
                "positive": yes,
            },
        )
        rows.append(
            make_row(
                row_id=f"sk2:causal:{it['key']}",
                source="sk2:causal",
                family="sk2/causal",
                state=state,
                question=question,
                target=target,
                meta=m,
            )
        )
    parse["causal"] = {
        "items": BUDGET["causal"],
        "kinds": dict(Counter(r["meta"]["query_kind"] for r in rows)),
        "graphs": dict(Counter(r["meta"]["graph"] for r in rows)),
    }

    blocked = blocked_positions(Path(args.suite), Path(args.protected))
    puzzles, parse["lichess_puzzles"] = puzzle_positions(
        next((raw / "lichess_puzzles").glob("data/*.parquet")),
        BUDGET["puzzles"],
        seed,
        blocked,
    )
    evals, parse["lichess_evals"] = eval_positions(
        next((raw / "lichess_evals").glob("data/*.parquet")),
        BUDGET["evals"],
        seed,
        blocked,
    )
    parse["chess_blocked_positions"] = len(blocked)
    for pos in puzzles + evals:
        for j, state, question, target, render in chess_rows(pos, seed):
            m = meta(
                pos["source"],
                "sk2-chess",
                pos["group"],
                pos["key"],
                render,
                question,
                {"chess": {"gold": pos["gold"], **pos["extra"]}},
            )
            rows.append(
                make_row(
                    row_id=f"sk2:chess:{pos['key']}:{j}",
                    source=f"sk2:{pos['source']}",
                    family="sk2/chess",
                    state=state,
                    question=question,
                    target=target,
                    meta=m,
                )
            )

    block_urls, block_hosts = blocked_urls(Path(args.suite), Path(args.protected))
    pool, parse["urls"] = url_pool(raw, seed, block_urls, block_hosts)
    parse["urls"]["blocked_urls"] = len(block_urls)
    parse["urls"]["blocked_hosts"] = len(block_hosts)
    for it in phishing_items(pool, BUDGET["emails"], BUDGET["urls"], seed):
        for j, state, question, target, render in phishing_rows(it, seed):
            m = meta(
                it["url"]["src"],
                "sk2-phish",
                it["group"],
                it["key"],
                render,
                question,
                {"phish_kind": it["kind"], "question_name": j, "positive": it["phish"]},
            )
            rows.append(
                make_row(
                    row_id=f"sk2:phish:{it['key']}:{j}",
                    source=f"sk2:{it['url']['src']}",
                    family="sk2/phish",
                    state=state,
                    question=question,
                    target=target,
                    meta=m,
                )
            )

    sources = {
        name: {k: v for k, v in info.items() if k != "files"}
        | {"files_sha256": info["files"]}
        for name, info in fetched.items()
    }
    sources["procedural"] = {
        "licence": "generated by d25.vega.data.sk2 (no upstream data)",
        "seed": seed,
    }
    built = Counter(r["source"] for r in rows)
    return finalize(
        rows,
        args,
        {
            "sources": sources,
            "rows_built_by_source": dict(built),
            "parse_stats": parse,
            "benchmark_source_exclusion": exclusion,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--raw", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--index", type=Path, required=True)
    b.add_argument("--holdouts", type=Path, required=True)
    b.add_argument("--suite", type=Path, required=True)
    b.add_argument("--protected", type=Path, required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.raw)
        return
    manifest = build(args)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "rows",
                    "composition",
                    "positive_rate",
                    "parse_stats",
                    "tokens",
                    "decontamination",
                    "holdouts",
                    "duplicates_dropped",
                    "too_long_dropped",
                )
                if k in manifest
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
