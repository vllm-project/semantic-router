"""Option-key and position leak audit for the frozen eval panels (count-only receipts).

A7 found that Decision 1.0 builders numbered Choice keys `result_<n>` in construction
order and then shuffled the options, so the key number revealed the gold. This audit
checks every eval panel for that class of cue: properties of the option surface that
predict the gold without reading the decision content.

Option-surface cues (the A7 kind): display position, the token of a non-semantic key
(numbered, letter or opaque), key number rank (numbered keys), lexicographic key rank,
key scheme, description length rank, description whitespace, shared versus unique
description text, and the key or description appearing in the instructions.
State-surface cue (typed panels only, reported separately): the position of the
option's entity in the state lists, or its top-level state role.

For every panel x group (task, family or question) and for every panel x question type,
the gold is predicted under 5-fold cross-validation (folds by source group) by:
chance; the label prior (gold frequency of the description and of semantic keys, i.e.
words, true/false/none and Score levels); each cue alone and added to the prior; and a
naive-Bayes combination of all option-surface cues (then also the state cue) with the
prior. A cue model is the better of the cue alone and the cue plus the prior; its gain
over the reference (the better of chance and the prior) gets a cluster-bootstrap 95%
interval. Pooled rows (all groups of one question type) condition the prior on the
group, so a cue must add information beyond each group's own prior (amendment 1).

Verdict: LEAK when the gain is at least 5 points and the interval is above 0; CUE when
the interval is above 0 but the gain is below 5 points; otherwise CLEAN. Deterministic checks sit next to it: numbered keys out of display order
(the A7 signature), `result_<n>` keys, distinct option orders and gold position counts.
Receipts carry counts and accuracies only, never item text or gold values.

    python3 -m v2.eval.leak_audit audit --panel-root /data/dev2/private/panels --output audit.json
    python3 -m v2.eval.leak_audit markdown --audit audit.json --output audit.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "dev2-leak-audit/1"
FOLDS = 5
REPLICATES = 2000
SEED = 20260928
LEAK_POINTS = 5.0
SMOOTHING = 1.0
SMALL_GROUP = 30
TIE = 1e-9

NUMBERED = re.compile(r"^(.*?)(\d+)$")
RESULT_KEY = re.compile(r"^result_\d+$")
OPAQUE = re.compile(r"^X[A-Z0-9]{5}$")

PRIOR_FEATURES = ("key_identity", "description_identity")
SEMANTIC_SCHEMES = {"word", "true", "false", "none"}
OPTION_CUES = (
    "position",
    "key_token",
    "key_number_rank",
    "key_lex_rank",
    "key_scheme",
    "description_length",
    "description_whitespace",
    "description_shared",
    "mentioned_in_instructions",
)
STATE_CUES = ("state_position",)

PANEL_ROLES = {
    "select": "development",
    "cal": "calibration",
    "typed-dev": "development",
    "css-pilot": "development",
    "typed-final": "formal",
    "css15": "formal",
    "public231": "formal (public subset)",
    "mlx-diag": "development diagnostic",
}


@dataclass
class Question:
    panel: str
    item_id: str
    group: str
    cluster: str
    qtype: str
    keys: list[str]
    descriptions: list[str]
    gold: int
    instructions: str = ""
    state: Any = None


# ------------------------------------------------------------------ loading


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def native_question(
    panel: str,
    item_id: str,
    group: str,
    cluster: str,
    question: dict[str, Any],
    gold_value: Any,
    state: Any,
) -> Question:
    """One System One question (choice / noul / score) with its gold as an index."""
    qtype = question["type"]
    criteria = question.get("criteria")
    if qtype == "score":
        if isinstance(criteria, dict):
            keys = [str(key) for key in criteria]
            descriptions = [str(value) for value in criteria.values()]
        else:
            keys = [str(level) for level in range(len(criteria or []))]
            descriptions = [str(text) for text in criteria or []]
        gold = int(gold_value)
    elif qtype == "noul":
        if isinstance(criteria, dict) and set(criteria) == {"true", "false"}:
            keys = list(criteria)
            descriptions = [str(criteria[key]) for key in keys]
        else:
            keys, descriptions = ["true", "false"], ["", ""]
        gold = keys.index("true" if gold_value else "false")
    elif qtype == "choice":
        keys = [str(key) for key in criteria]
        descriptions = [str(value) for value in criteria.values()]
        gold = keys.index(str(gold_value))
    else:
        raise ValueError(f"{panel}/{item_id}: unknown question type {qtype!r}")
    if not 0 <= gold < len(keys):
        raise ValueError(f"{panel}/{item_id}: gold outside the options")
    return Question(
        panel,
        item_id,
        group,
        cluster,
        qtype,
        keys,
        descriptions,
        gold,
        str(question.get("instructions") or ""),
        state,
    )


def training_question(panel: str, row: dict[str, Any]) -> Question:
    """A labelled training-format row (SELECT / CAL): options + label index."""
    options = row["options"]
    return Question(
        panel,
        str(row["id"]),
        str(row["family"]),
        str(row.get("group_id") or row["id"]),
        str(row["task_type"]),
        [str(option["key"]) for option in options],
        [str(option.get("description") or "") for option in options],
        int(row["label"]),
        str(row.get("instructions") or ""),
        row.get("state"),
    )


def public_gold(task_type: str, expected: Any) -> Any:
    if task_type == "noul":
        if isinstance(expected, bool):
            return expected
        return str(expected).strip().lower() in {"yes", "true", "1"}
    return expected


def load_panel(root: Path, panel: str) -> list[Question]:
    """Questions of one frozen panel (the gold files carry everything needed)."""
    from v2.eval import panels as registry

    if panel in ("select", "cal"):
        rows = read_jsonl(registry.path(root, panel, "gold"))
        return [training_question(panel, row) for row in rows]
    if panel in ("typed-dev", "typed-final"):
        out = []
        for row in read_jsonl(registry.path(root, panel, "gold")):
            for key, question in row["questions"].items():
                out.append(
                    native_question(
                        panel,
                        row["id"],
                        f"{row['family']}/{key}",
                        row["group_id"],
                        question,
                        row["gold"][key]["value"],
                        row["state"],
                    )
                )
        return out
    prompts = {
        row["id"]: row for row in read_jsonl(registry.path(root, panel, "prompts"))
    }
    out = []
    if panel in ("css15", "css-pilot"):
        for row in read_jsonl(registry.path(root, panel, "gold")):
            prompt = prompts[row["id"]]
            (key, question), *rest = prompt["questions"].items()
            if rest:
                raise ValueError(f"{panel}/{row['id']}: more than one question")
            out.append(
                native_question(
                    panel,
                    row["id"],
                    row["task"],
                    row["id"],
                    question,
                    row["gold"],
                    prompt["state"],
                )
            )
        return out
    if panel == "public231":
        for row in read_jsonl(registry.path(root, panel, "gold")):
            prompt = prompts[row["id"]]
            (key, question), *rest = prompt["questions"].items()
            if rest:
                raise ValueError(f"{panel}/{row['id']}: more than one question")
            out.append(
                native_question(
                    panel,
                    row["id"],
                    f"{row['family']}/{row['task_type']}",
                    str(row.get("group") or row["id"]),
                    question,
                    public_gold(row["task_type"], row["expected"]),
                    prompt["state"],
                )
            )
        return out
    if panel == "mlx-diag":
        for row in read_jsonl(registry.path(root, panel, "gold")):
            prompt = prompts[row["id"]]
            (key, question), *rest = prompt["questions"].items()
            if rest:
                raise ValueError(f"{panel}/{row['id']}: more than one question")
            out.append(
                native_question(
                    panel,
                    row["id"],
                    f"{row['source']}/{row['type']}",
                    f"{row['source']}:{row['source_id']}",
                    question,
                    row["value"],
                    prompt["state"],
                )
            )
        return out
    raise ValueError(f"unknown panel {panel!r}")


# ----------------------------------------------------------------- features


def key_scheme(key: str) -> str:
    if RESULT_KEY.match(key):
        return "result_<n>"
    if OPAQUE.match(key):
        return "opaque"
    if key in ("true", "false", "none"):
        return key
    if re.fullmatch(r"[A-Z]", key):
        return "letter"
    if re.fullmatch(r"\d+", key):
        return "digits"
    if NUMBERED.match(key):
        return "numbered"
    return "word"


def numbered_prefix(keys: list[str]) -> list[int] | None:
    """Key numbers when every key is <shared prefix><number>, else None."""
    matches = [NUMBERED.match(key) for key in keys]
    if not all(matches) or len({match.group(1) for match in matches}) != 1:
        return None
    numbers = [int(match.group(2)) for match in matches]
    return numbers if len(set(numbers)) == len(numbers) else None


def state_position(state: Any, key: str) -> str:
    """Where an option key sits in a structured state: list slot or top-level role."""
    if not isinstance(state, dict):
        return "no-structured-state"
    for field, value in state.items():
        if isinstance(value, list):
            for index, element in enumerate(value):
                inside = (
                    element == key
                    if not isinstance(element, dict)
                    else key in [v for v in element.values() if isinstance(v, str)]
                )
                if inside:
                    return f"{field}[{index}/{len(value)}]"
    for field, value in state.items():
        if value == key:
            return f"top:{field}"
    return "absent"


def rank_tag(values: list[int], index: int) -> str:
    top, bottom = max(values), min(values)
    if values[index] == top and values.count(top) == 1:
        return "longest"
    if values[index] == bottom and values.count(bottom) == 1:
        return "shortest"
    return "other"


def option_features(question: Question) -> list[dict[str, str]]:
    keys, descriptions = question.keys, question.descriptions
    count = len(keys)
    features: list[dict[str, str]] = [{} for _ in keys]
    numbers = numbered_prefix(keys)
    lexical = sorted(keys)
    lengths = [len(text) for text in descriptions]
    shared = Counter(descriptions)
    instructions = question.instructions.casefold()
    for index, (key, text) in enumerate(zip(keys, descriptions)):
        row = features[index]
        if question.qtype == "score" or key_scheme(key) in SEMANTIC_SCHEMES:
            row["key_identity"] = f"k:{key}"
        else:
            row["key_token"] = f"k:{key}"
        row["description_identity"] = (
            "d:"
            + hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()[
                :16
            ]
        )
        row["key_scheme"] = key_scheme(key)
        if question.qtype != "score":
            row["position"] = f"{index}/{count}"
            row["key_lex_rank"] = f"{lexical.index(key)}/{count}"
            if numbers is not None:
                row["key_number_rank"] = (
                    f"{sorted(numbers).index(numbers[index])}/{count}"
                )
        row["description_length"] = rank_tag(lengths, index)
        row["description_whitespace"] = (
            ("lead" if text != text.lstrip() else "")
            + ("trail" if text != text.rstrip() else "")
        ) or "none"
        row["description_shared"] = "shared" if shared[text] > 1 else "unique"
        mentions = [
            token
            for token in (key.casefold(), " ".join(text.casefold().split()))
            if len(token) >= 3 and token in instructions
        ]
        row["mentioned_in_instructions"] = "yes" if mentions else "no"
        if question.qtype == "choice" and not isinstance(question.state, str):
            row["state_position"] = state_position(question.state, key)
    return features


# -------------------------------------------------------------- estimation


def fold_of(cluster: str) -> int:
    return int(hashlib.sha256(cluster.encode("utf-8")).hexdigest(), 16) % FOLDS


def fit(
    questions: list[Question],
    features: list[list[dict[str, str]]],
    names: tuple[str, ...],
) -> dict[str, tuple[Counter, Counter]]:
    table: dict[str, tuple[Counter, Counter]] = {
        name: (Counter(), Counter()) for name in names
    }
    for question, rows in zip(questions, features):
        for index, row in enumerate(rows):
            for name in names:
                value = row.get(name)
                if value is None:
                    continue
                gold, total = table[name]
                total[value] += 1
                if index == question.gold:
                    gold[value] += 1
    return table


def expected_accuracy(
    question: Question,
    rows: list[dict[str, str]],
    table: dict[str, tuple[Counter, Counter]],
    names: tuple[str, ...],
) -> float:
    """Expected accuracy of argmax over naive-Bayes log-odds (ties split evenly)."""
    scores = []
    for row in rows:
        total_score = 0.0
        for name in names:
            value = row.get(name)
            if value is None:
                continue
            gold, total = table[name]
            hits = gold.get(value, 0)
            misses = total.get(value, 0) - hits
            total_score += math.log((hits + SMOOTHING) / (misses + SMOOTHING))
        scores.append(total_score)
    best = max(scores)
    winners = [i for i, value in enumerate(scores) if value >= best - TIE]
    return (1.0 / len(winners)) if question.gold in winners else 0.0


def cross_validated(
    questions: list[Question],
    features: list[list[dict[str, str]]],
    names: tuple[str, ...],
) -> list[float]:
    folds = [fold_of(question.cluster) for question in questions]
    out = [0.0] * len(questions)
    for fold in range(FOLDS):
        train = [i for i, value in enumerate(folds) if value != fold]
        test = [i for i, value in enumerate(folds) if value == fold]
        if not test:
            continue
        table = fit([questions[i] for i in train], [features[i] for i in train], names)
        for i in test:
            out[i] = expected_accuracy(questions[i], features[i], table, names)
    return out


def cluster_bootstrap(
    clusters: list[str], deltas: list[float], replicates: int, seed: int
) -> tuple[float, float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster, delta in zip(clusters, deltas):
        grouped[cluster].append(delta)
    keys = sorted(grouped)
    if len(keys) < 2:
        return None
    if not any(deltas):
        return (0.0, 0.0)
    sums = [sum(grouped[key]) for key in keys]
    sizes = [len(grouped[key]) for key in keys]
    indices = range(len(keys))
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        picks = rng.choices(indices, k=len(keys))
        draws.append(100.0 * sum(sums[i] for i in picks) / sum(sizes[i] for i in picks))
    draws.sort()
    return (
        draws[int(0.025 * (replicates - 1))],
        draws[int(math.ceil(0.975 * (replicates - 1)))],
    )


def verdict(gain: float, interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "INSUFFICIENT"
    if interval[0] > 0 and gain >= LEAK_POINTS:
        return "LEAK"
    if interval[0] > 0:
        return "CUE"
    return "CLEAN"


def mean(values: list[float]) -> float:
    return 100.0 * sum(values) / len(values) if values else float("nan")


def audit_group(
    questions: list[Question],
    replicates: int = REPLICATES,
    seed: int = SEED,
    pooled: bool = False,
) -> dict[str, Any]:
    """Cue models against the reference (the better of chance and the label prior).

    A cue model is the better of the cue alone and the cue added to the prior. In a
    pooled row (several groups) the prior is conditioned on the group, so a cue must
    add information beyond each group's own label prior.
    """
    features = [option_features(question) for question in questions]
    if pooled:
        for question, rows in zip(questions, features):
            for row in rows:
                for name in PRIOR_FEATURES:
                    if name in row:
                        row[name] = f"{question.group}|{row[name]}"
    clusters = [question.cluster for question in questions]
    chance = [1.0 / len(question.keys) for question in questions]
    prior = cross_validated(questions, features, PRIOR_FEATURES)
    reference_name, reference = (
        ("chance", chance) if mean(chance) >= mean(prior) else ("label_prior", prior)
    )
    present_option = tuple(
        name
        for name in OPTION_CUES
        if any(name in row for rows in features for row in rows)
    )
    present_state = tuple(
        name
        for name in STATE_CUES
        if any(name in row for rows in features for row in rows)
    )

    def assess(names: tuple[str, ...]) -> dict[str, Any]:
        alone = cross_validated(questions, features, names)
        added = cross_validated(questions, features, PRIOR_FEATURES + names)
        model, best = (
            ("alone", alone) if mean(alone) >= mean(added) else ("with_prior", added)
        )
        gain = mean(best) - mean(reference)
        interval = cluster_bootstrap(
            clusters, [b - r for b, r in zip(best, reference)], replicates, seed
        )
        return {
            "alone": mean(alone),
            "with_prior": mean(added),
            "model": model,
            "gain_over_reference": gain,
            "gain_ci95": interval,
            "verdict": verdict(gain, interval),
        }

    cues = {name: assess((name,)) for name in present_option + present_state}
    combined = {}
    if present_option:
        combined["option_surface"] = assess(present_option)
    if present_option and present_state:
        combined["option_and_state_surface"] = assess(present_option + present_state)
    return {
        "questions": len(questions),
        "clusters": len(set(clusters)),
        "small": len(set(clusters)) < SMALL_GROUP,
        "chance": mean(chance),
        "label_prior": mean(prior),
        "reference": reference_name,
        "prior_conditioned_on_group": pooled,
        "cues": cues,
        "combined": combined,
        "facts": deterministic_facts(questions),
    }


def deterministic_facts(questions: list[Question]) -> dict[str, Any]:
    schemes: Counter = Counter()
    options: Counter = Counter()
    positions: Counter = Counter()
    orders: set[tuple[str, ...]] = set()
    option_sets: set[tuple[str, ...]] = set()
    result_rows = out_of_order = 0
    out_of_order_gold_rank: Counter = Counter()
    for question in questions:
        schemes.update({key_scheme(key) for key in question.keys})
        options[len(question.keys)] += 1
        positions[f"{question.gold}/{len(question.keys)}"] += 1
        orders.add(tuple(question.keys))
        option_sets.add(tuple(sorted(question.keys)))
        if all(RESULT_KEY.match(key) for key in question.keys):
            result_rows += 1
        numbers = numbered_prefix(question.keys)
        if numbers is not None and numbers != sorted(numbers):
            out_of_order += 1
            rank = sorted(numbers).index(numbers[question.gold])
            last = len(numbers) - 1
            out_of_order_gold_rank[
                (
                    "largest"
                    if rank == last
                    else (
                        "largest_but_one"
                        if rank == last - 1
                        else "smallest" if rank == 0 else "other"
                    )
                )
            ] += 1
    return {
        "key_schemes": dict(sorted(schemes.items())),
        "option_counts": {str(k): v for k, v in sorted(options.items())},
        "gold_position_counts": dict(sorted(positions.items())),
        "distinct_option_orders": len(orders),
        "distinct_option_sets": len(option_sets),
        "result_n_key_rows": result_rows,
        "numbered_keys_out_of_display_order": out_of_order,
        "out_of_order_gold_key_rank": dict(sorted(out_of_order_gold_rank.items())),
    }


def id_mentions_gold(question: Question) -> bool:
    key = question.keys[question.gold].casefold()
    if question.qtype != "choice" or len(key) < 3:
        return False
    tokens = set(re.split(r"[^0-9a-z_]+", question.item_id.casefold()))
    return key in tokens


def worst(verdicts: Iterable[str]) -> str:
    order = ("LEAK", "CUE", "CLEAN", "INSUFFICIENT")
    present = set(verdicts)
    return next((value for value in order if value in present), "CLEAN")


def audit_panel(
    questions: list[Question], replicates: int = REPLICATES, seed: int = SEED
) -> dict[str, Any]:
    by_group: dict[str, list[Question]] = defaultdict(list)
    by_type: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_group[question.group].append(question)
        by_type[question.qtype].append(question)
    groups = {
        name: audit_group(items, replicates, seed)
        for name, items in sorted(by_group.items())
    }
    pooled = {
        name: audit_group(items, replicates, seed, pooled=True)
        for name, items in sorted(by_type.items())
    }

    def primary(entry: dict[str, Any]) -> str:
        block = entry["combined"].get("option_surface")
        return block["verdict"] if block else "CLEAN"

    def secondary(entry: dict[str, Any]) -> str:
        block = entry["combined"].get("option_and_state_surface")
        return block["verdict"] if block else primary(entry)

    return {
        "questions": len(questions),
        "items": len({q.item_id for q in questions}),
        "types": dict(Counter(q.qtype for q in questions)),
        "id_mentions_gold": sum(id_mentions_gold(q) for q in questions),
        "verdict_option_surface": worst(
            [primary(e) for e in groups.values()]
            + [primary(e) for e in pooled.values()]
        ),
        "verdict_with_state_surface": worst(
            [secondary(e) for e in groups.values()]
            + [secondary(e) for e in pooled.values()]
        ),
        "groups": groups,
        "pooled_by_type": pooled,
    }


# ---------------------------------------------------------------------- cli


def audit(args: argparse.Namespace) -> int:
    from v2.eval import panels as registry

    names = args.panel or list(PANEL_ROLES)
    verified = registry.verify(args.panel_root, names)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "rule": (
            f"LEAK if the combined surface model beats the reference (the better of "
            f"chance and the label prior; pooled rows condition the prior on the group) "
            f"by >= {LEAK_POINTS} points with the cluster-bootstrap 95% interval above 0; "
            "CUE if the interval is above 0 below that size; CLEAN otherwise "
            "(5-fold CV by source group; amendment 1)"
        ),
        "folds": FOLDS,
        "replicates": args.replicates,
        "seed": SEED,
        "panel_sha256": verified,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "panels": {},
    }
    for name in names:
        report = audit_panel(load_panel(args.panel_root, name), args.replicates, SEED)
        report["role"] = PANEL_ROLES[name]
        result["panels"][name] = report
        print(
            json.dumps(
                {
                    "panel": name,
                    "questions": report["questions"],
                    "option_surface": report["verdict_option_surface"],
                    "with_state_surface": report["verdict_with_state_surface"],
                }
            ),
            flush=True,
        )
    data = json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        target.write(data)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(data.encode()).hexdigest(),
            }
        )
    )
    return 0


def fmt_interval(interval: Any) -> str:
    if not interval:
        return "n/a"
    return f"[{interval[0]:+.1f}, {interval[1]:+.1f}]"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Panel | Role | Questions | Option-surface verdict | With state surface | Worst group (gain [95% CI]) |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for name, report in result["panels"].items():
        worst_group, worst_gain = "-", None
        for group, entry in {
            **report["groups"],
            **{f"all {k}": v for k, v in report["pooled_by_type"].items()},
        }.items():
            block = entry["combined"].get("option_surface")
            if block and (
                worst_gain is None or block["gain_over_reference"] > worst_gain[0]
            ):
                worst_gain = (block["gain_over_reference"], block["gain_ci95"])
                worst_group = group
        cell = (
            f"{worst_group}: {worst_gain[0]:+.1f} {fmt_interval(worst_gain[1])}"
            if worst_gain
            else "-"
        )
        lines.append(
            f"| {name} | {report['role']} | {report['questions']} | "
            f"{report['verdict_option_surface']} | {report['verdict_with_state_surface']} | {cell} |"
        )
    lines.append("")
    for name, report in result["panels"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            "| Group | n | Clusters | Chance | Prior | Surface only | Surface + prior | Gain over reference [95% CI] | Verdict | Strongest single cue (gain) | Out-of-order numbered keys | Orders |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |"
        )
        entries = {
            **report["groups"],
            **{f"all {k}": v for k, v in report["pooled_by_type"].items()},
        }
        for group, entry in entries.items():
            block = entry["combined"].get("option_surface") or {}
            best_cue = max(
                entry["cues"].items(),
                key=lambda item: item[1]["gain_over_reference"],
                default=(None, None),
            )
            cue_text = (
                f"{best_cue[0]} {best_cue[1]['gain_over_reference']:+.1f}"
                if best_cue[0]
                else "-"
            )
            lines.append(
                f"| {group} | {entry['questions']} | {entry['clusters']} | {entry['chance']:.1f} | "
                f"{entry['label_prior']:.1f} | {block.get('alone', float('nan')):.1f} | "
                f"{block.get('with_prior', float('nan')):.1f} | "
                f"{block.get('gain_over_reference', float('nan')):+.1f} {fmt_interval(block.get('gain_ci95'))} | "
                f"{block.get('verdict', '-')}{' (small)' if entry['small'] else ''} | {cue_text} | "
                f"{entry['facts']['numbered_keys_out_of_display_order']} | {entry['facts']['distinct_option_orders']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def markdown(args: argparse.Namespace) -> int:
    result = json.loads(args.audit.read_text(encoding="utf-8"))
    args.output.write_text(render_markdown(result), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("audit")
    run.add_argument(
        "--panel-root", type=Path, default=Path("/data/dev2/private/panels")
    )
    run.add_argument("--panel", action="append", choices=sorted(PANEL_ROLES))
    run.add_argument("--replicates", type=int, default=REPLICATES)
    run.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("markdown")
    render.add_argument("--audit", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return audit(args) if args.command == "audit" else markdown(args)


if __name__ == "__main__":
    raise SystemExit(main())
