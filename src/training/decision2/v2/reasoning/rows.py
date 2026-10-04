"""Reasoning-graph problems to System One training rows, for the graph-forcing arm and its matched controls.

Per problem: the final row plus one view per sampled graph node (at most ``cap``). A node view is asked with the
true conclusions of the node's parents written into the state ("Known intermediate results"), in one of four forms:

- native: the node's own type (Choice over candidate values, Noul yes / no), or a Score over levels 0..9 for small
  integer answers;
- statement: Noul "is this statement about the problem correct?" over the node's true or minimally false statement
  (polarity balanced);
- multi: Choice "which statement is correct?" over the node's true statement, false statements and "None of these".

Arms share rows, order and weights except for the node views:
- ``tf``: true parent conclusions, node views weighted ``aux_total`` per problem;
- ``tfm``: the same number of conclusions from other nodes of the same problem that are neither parents, ancestors
  nor descendants of the node (depth-matched where possible); nodes without a valid replacement are asked plainly;
- ``f0``: the ``tf`` rows with node weights ``placebo_weight`` (identical batches, no node signal).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from training.model.data import digest

from .graph import depths

FACTS_HEADER = "Known intermediate results"
NONE_OF_THESE = "None of these"
STATEMENT_Q = (
    "Is the following statement about this problem correct? {s}",
    "Given the problem, is this claim true? {s}",
    "Check this intermediate claim against the problem: {s} Is it correct?",
)
MULTI_Q = (
    "Which of these statements about the problem is correct?",
    "Which intermediate claim holds for this problem?",
)


def _stable(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]


def with_facts(state: Any, facts: list[str]) -> Any:
    if not facts:
        return state
    if isinstance(state, dict):
        out = dict(state)
        out["known_intermediate_results"] = list(facts)
        return out
    if isinstance(state, list):
        return [*state, {"known_intermediate_results": list(facts)}]
    return f"{state}\n\n{FACTS_HEADER}:\n" + "\n".join(f"- {f}" for f in facts)


def _row(
    problem: dict[str, Any],
    suffix: str,
    state: Any,
    task_type: str,
    instructions: Any,
    options: list[dict[str, Any]],
    label: int,
    family: str,
    template: str,
    audit: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    payload = {
        "state": state,
        "instructions": instructions,
        "options": options,
        "task_type": task_type,
    }
    return {
        "id": f"rsn_{_stable(problem['pid'], suffix)}",
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": label,
        "task_type": task_type,
        "family": family,
        "group_id": f"rsng_{_stable(problem['pid'])}",
        "language": problem.get("language", "en"),
        "split": split,
        "source": problem["source"],
        "evaluation_role": split,
        "render_template": template,
        "audit_metadata": {"pid": problem["pid"], **audit},
        "input_sha256": digest(payload),
    }


def final_row(problem: dict[str, Any], split: str = "train") -> dict[str, Any]:
    if "row" in problem:  # teacher problems keep the released row unchanged
        row = copy.deepcopy(problem["row"])
        return row
    final = problem["final"]
    return _row(
        problem,
        "final",
        problem["state"],
        final["task_type"],
        final["instructions"],
        final["options"],
        final["label"],
        problem["family"],
        problem["render_template"],
        {"view": "final"},
        split,
    )


def _score_view(item: dict[str, Any]) -> tuple[list[dict[str, Any]], int] | None:
    if item["kind"] != "choice":
        return None
    try:
        gold = int(str(item["options"][item["label"]]["description"]))
    except ValueError:
        return None
    if not 0 <= gold <= 9:
        return None
    levels = [{"key": str(i), "description": str(i)} for i in range(10)]
    return levels, gold


def node_view(
    problem: dict[str, Any],
    item: dict[str, Any],
    facts: list[str],
    rng: random.Random,
    others: list[dict[str, Any]],
    arm_tag: str,
    split: str,
) -> dict[str, Any]:
    state = with_facts(problem["state"], facts)
    roll = rng.random()
    false_ok = bool(item.get("false_statement"))
    family = f"{problem['family']}-node"
    audit = {"node": item["id"], "facts": len(facts), "arm": arm_tag}
    if roll < 0.55 or not false_ok:
        if item["kind"] == "noul":
            options = [
                {"key": "false", "description": "No"},
                {"key": "true", "description": "Yes"},
            ]
            return _row(
                problem,
                f"{item['id']}:{arm_tag}:native",
                state,
                "noul",
                item["question"],
                options,
                int(item["answer"]),
                family,
                "reasoning_node_native_v1",
                {**audit, "view": "native"},
                split,
            )
        score = _score_view(item) if rng.random() < 0.5 else None
        if score is not None:
            options, label = score
            return _row(
                problem,
                f"{item['id']}:{arm_tag}:score",
                state,
                "score",
                item["question"],
                options,
                label,
                family,
                "reasoning_node_score_v1",
                {**audit, "view": "score"},
                split,
            )
        return _row(
            problem,
            f"{item['id']}:{arm_tag}:native",
            state,
            "choice",
            item["question"],
            item["options"],
            item["label"],
            family,
            "reasoning_node_native_v1",
            {**audit, "view": "native"},
            split,
        )
    if roll < 0.85:
        truthful = rng.random() < 0.5
        claim = item["statement"] if truthful else item["false_statement"]
        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
        question = rng.choice(STATEMENT_Q).format(s=claim)
        return _row(
            problem,
            f"{item['id']}:{arm_tag}:stmt",
            state,
            "noul",
            question,
            options,
            int(truthful),
            family,
            "reasoning_node_statement_v1",
            {**audit, "view": "statement"},
            split,
        )
    pool = [
        o["false_statement"]
        for o in others
        if o.get("false_statement") and o is not item
    ]
    wrong = [item["false_statement"], *rng.sample(pool, min(2, len(pool)))]
    if rng.random() < 0.15:  # the correct option is "None of these"
        texts = list(dict.fromkeys(wrong)) + [NONE_OF_THESE]
        correct = NONE_OF_THESE
    else:
        texts = list(dict.fromkeys([item["statement"], *wrong])) + [NONE_OF_THESE]
        correct = item["statement"]
    head, tail = texts[:-1], texts[-1:]
    rng.shuffle(head)
    texts = head + tail
    keys = [chr(65 + i) for i in range(len(texts))]
    options = [{"key": k, "description": t} for k, t in zip(keys, texts)]
    return _row(
        problem,
        f"{item['id']}:{arm_tag}:multi",
        state,
        "choice",
        rng.choice(MULTI_Q),
        options,
        texts.index(correct),
        family,
        "reasoning_node_multi_v1",
        {**audit, "view": "multi"},
        split,
    )


def _relatives(
    nodes: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    parents = {n["id"]: set(n["depends_on"]) for n in nodes}
    ancestors: dict[str, set[str]] = {}

    def anc(node_id: str) -> set[str]:
        if node_id not in ancestors:
            out: set[str] = set()
            for p in parents.get(node_id, ()):
                out |= {p} | anc(p)
            ancestors[node_id] = out
        return ancestors[node_id]

    for n in nodes:
        anc(n["id"])
    descendants = {
        n["id"]: {m["id"] for m in nodes if n["id"] in ancestors[m["id"]]}
        for n in nodes
    }
    return ancestors, descendants


def problem_rows(
    problem: dict[str, Any],
    *,
    cap: int,
    aux_total: float,
    placebo_weight: float,
    seed: str,
    split: str = "train",
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    """Rows with weights for each arm: {"tf": [...], "tfm": [...], "f0": [...]}. The final row comes first."""
    rng = random.Random(_stable(seed, problem["pid"]))
    nodes = [n for n in problem["nodes"] if n.get("verified", True)]
    by_id = {n["id"]: n for n in nodes}
    chosen = nodes if len(nodes) <= cap else rng.sample(nodes, cap)
    depth = depths(nodes)
    ancestors, descendants = _relatives(nodes)
    final = final_row(problem, split)
    out: dict[str, list[tuple[dict[str, Any], float]]] = {
        "tf": [(final, 1.0)],
        "tfm": [(final, 1.0)],
        "f0": [(final, 1.0)],
    }
    if not chosen:
        return out
    weight = aux_total / len(chosen)
    for item in chosen:
        parents = [by_id[p] for p in item["depends_on"] if p in by_id]
        view_seed = _stable(seed, problem["pid"], item["id"])
        true_facts = [p["statement"] for p in parents]
        random.Random(view_seed + ":facts").shuffle(true_facts)
        tf_row = node_view(
            problem,
            item,
            true_facts,
            random.Random(view_seed + ":view"),
            nodes,
            "tf",
            split,
        )
        out["tf"].append((tf_row, weight))
        out["f0"].append((tf_row, placebo_weight))
        banned = {item["id"]} | ancestors[item["id"]] | descendants[item["id"]]
        pool = [n for n in nodes if n["id"] not in banned]
        rng_m = random.Random(view_seed + ":rewire")
        replacement = []
        for p in parents:
            if not pool:
                break
            best = min(abs(depth[n["id"]] - depth[p["id"]]) for n in pool)
            pick = rng_m.choice(
                [n for n in pool if abs(depth[n["id"]] - depth[p["id"]]) == best]
            )
            replacement.append(pick)
            pool.remove(pick)
        rewired = (
            [n["statement"] for n in replacement]
            if len(replacement) == len(parents)
            else []
        )
        random.Random(view_seed + ":facts").shuffle(rewired)
        tfm_row = node_view(
            problem,
            item,
            rewired,
            random.Random(view_seed + ":view"),
            nodes,
            "tfm",
            split,
        )
        out["tfm"].append((tfm_row, weight))
    return out


def teacher_problem(
    record: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any] | None:
    """A teacher graph record plus its released row, as a problem (verified nodes only, ids remapped)."""
    nodes = []
    for item in record.get("nodes", []):
        if not item.get("verified"):
            continue
        nodes.append(
            {
                "id": item["id"],
                "depends_on": [d for d in item["depends_on"]],
                "kind": "noul",
                "question": item["question"],
                "answer": bool(item["answer"]),
                "statement": item["statement"],
                "false_statement": item.get("false_statement"),
                "verified": True,
            }
        )
    kept = {n["id"] for n in nodes}
    for n in nodes:
        n["depends_on"] = [d for d in n["depends_on"] if d in kept]
    if not nodes:
        return None
    return {
        "pid": f"t:{row['id']}",
        "family": row["family"],
        "source": row["source"],
        "language": row["language"],
        "render_template": row["render_template"],
        "state": row["state"],
        "row": row,
        "nodes": nodes,
        "final": {
            "task_type": row["task_type"],
            "instructions": row["instructions"],
            "options": row["options"],
            "label": row["label"],
        },
    }


def dump(
    rows: list[tuple[dict[str, Any], float]], rows_path: str, weights_path: str
) -> dict[str, Any]:
    seen: set[str] = set()
    with open(rows_path, "w", encoding="utf-8") as rs, open(
        weights_path, "w", encoding="utf-8"
    ) as ws:
        for row, weight in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rs.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            ws.write(json.dumps({"id": row["id"], "weight": weight}) + "\n")
    return {"rows": len(seen)}
