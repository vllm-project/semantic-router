"""Group-disjoint answer-shortcut baselines for a Decision 2.0 data arm.

Folds: int(sha256(group_id)) % 5. Learner: per-(row, option) binary logistic
regression on 2**20 blake2b-hashed binary features, AdaGrad, L2 1e-6, 4
epochs, seeded shuffles; a row predicts its highest-scoring option (ties ->
lowest index). One model per (view, held-out fold, task type).

Gate, per task type and gated view with >= 30 rows: PASS iff accuracy <=
majority + 0.05, where majority is the cross-validated position prior on the
same rows. CJK-heavy text adds compact character bigrams to its word units,
since whitespace words are clauses there.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import heapq
import json
import math
import multiprocessing
import os
import random
import sys
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, validate_row
from v2.data.textnorm import (
    compact,
    decode_canonical_json,
    is_cjk_heavy,
    normalize,
    text_leaves,
    word_tokens,
)

SCHEMA = "decision2.v2.shortcut.v1"
FOLDS = 5
TASK_TYPES = ("choice", "noul", "score")
HYPOTHESIS_KEYS = ("hypothesis", "claim", "statement", "conclusion", "proposition")
LEARNED_VIEWS = ("state_removed", "option_only", "hypothesis_only", "full_input")
GATED_VIEWS = ("state_removed", "option_only", "hypothesis_only")
HEURISTICS = ("position_prior", "longest_option", "lexical_overlap")
MIN_GATE_ROWS = 30
MARGIN = 0.05
Z95 = 1.959963984540054
_MASK64 = (1 << 64) - 1
_WORK: dict[str, Any] = {}


@dataclasses.dataclass(frozen=True)
class LearnerParams:
    bits: int = 20
    epochs: int = 4
    l2: float = 1e-6
    learning_rate: float = 0.1
    seed: int = 0
    instruction_cross_cap: int = 64
    state_cross_cap: int = 128
    text_gram_cap: int = 1024
    state_gram_cap: int = 128
    state_cross_unigrams: int = 64


@dataclasses.dataclass(frozen=True)
class Item:
    id: str
    group_id: str
    fold: int
    task_type: str
    family: str
    source: str
    label: int
    keys: tuple[str, ...]
    options: tuple[str, ...]
    instructions: str
    state: str
    hypothesis: str | None


def fold_of(group_id: str) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest(), 16) % FOLDS


def _text(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else " ".join(text_leaves(value, decode_json=True))
    )


def hypothesis_text(state: Any) -> str | None:
    """Joined hypothesis-like fields of a dict state, or None when absent."""
    value = state
    if isinstance(state, str):
        value = decode_canonical_json(state)
    if not isinstance(value, dict) or not any(key in value for key in HYPOTHESIS_KEYS):
        return None
    return " ".join(_text(value[key]) for key in HYPOTHESIS_KEYS if key in value)


def item_of(row: Mapping[str, Any]) -> Item:
    return Item(
        id=row["id"],
        group_id=row["group_id"],
        fold=fold_of(row["group_id"]),
        task_type=row["task_type"],
        family=row["family"],
        source=row["source"],
        label=row["label"],
        keys=tuple(option["key"] for option in row["options"]),
        options=tuple(_text(option["description"]) for option in row["options"]),
        instructions=_text(row["instructions"]),
        state=_text(row["state"]),
        hypothesis=hypothesis_text(row["state"]),
    )


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Rows must satisfy the training-row contract of their own split."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{number}: blank line")
            row = json.loads(raw)
            try:
                if not isinstance(row, dict):
                    raise ValueError("every line must be a JSON object")
                validate_row(row, row.get("split"), replay="teacher_probs" in row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if row["id"] in seen:
                raise ValueError(f"{path}:{number}: duplicate id {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _unigrams(text: str) -> list[str]:
    units = word_tokens(text)
    if is_cjk_heavy(text):
        chars = compact(text)
        units += [chars[i : i + 2] for i in range(len(chars) - 1)]
    return units


def _grams12(text: str) -> list[str]:
    tokens = word_tokens(text)
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    if is_cjk_heavy(text):
        chars = compact(text)
        grams += [chars[i : i + 2] for i in range(len(chars) - 1)]
    return grams


def _blake64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little"
    )


def _mix(left: int, right: int) -> int:
    return (
        ((left * 0x9E3779B97F4A7C15) & _MASK64) ^ right
    ) * 0xBF58476D1CE4E5B9 & _MASK64


class _Hasher:
    """blake2b-64 feature hashing with a bounded memo."""

    def __init__(self, bits: int, memo: int = 500_000) -> None:
        self.mask = (1 << bits) - 1
        self.memo: dict[str, int] = {}
        self.limit = memo

    def full(self, feature: str) -> int:
        value = self.memo.get(feature)
        if value is None:
            if len(self.memo) >= self.limit:
                self.memo.clear()
            value = self.memo[feature] = _blake64(feature)
        return value

    def index(self, feature: str) -> int:
        return self.full(feature) & self.mask


def _capped(grams: Iterable[str], cap: int, hasher: _Hasher) -> list[tuple[int, str]]:
    """Distinct grams with their hashes, the `cap` smallest hashes when capped."""
    distinct = {gram: hasher.full(gram) for gram in grams}
    return heapq.nsmallest(cap, ((value, gram) for gram, value in distinct.items()))


def _crosses(
    left: Sequence[tuple[int, str]],
    right: Sequence[tuple[int, str]],
    cap: int,
    prefix: str,
) -> list[str]:
    chosen = heapq.nsmallest(
        cap, ((_mix(a, b), x, y) for a, x in left for b, y in right)
    )
    return [f"{prefix}|{x}|{y}" for _, x, y in chosen]


class _Features:
    def __init__(self, params: LearnerParams) -> None:
        self.params = params
        self.hasher = _Hasher(params.bits)
        self.option_memo: dict[str, tuple[frozenset[int], list[tuple[int, str]]]] = {}

    def option(self, description: str) -> tuple[frozenset[int], list[tuple[int, str]]]:
        """Description word 1-2grams and char 3-5grams, plus hashed unigrams."""
        memo = self.option_memo.get(description)
        if memo is None:
            index = self.hasher.index
            ids = {index(f"ow|{gram}") for gram in _grams12(description)}
            padded = f" {normalize(description)} "
            ids.update(
                index(f"oc|{padded[i : i + n]}")
                for n in (3, 4, 5)
                for i in range(len(padded) - n + 1)
            )
            unigrams = _capped(
                _unigrams(description), self.params.text_gram_cap, self.hasher
            )
            if len(self.option_memo) >= 100_000:
                self.option_memo.clear()
            memo = self.option_memo[description] = (frozenset(ids), unigrams)
        return memo

    def row(self, view: str, item: Item) -> Iterator[array]:
        params, index = self.params, self.hasher.index
        count = len(item.keys)
        instruction = state = hypothesis = ()
        instruction_units = state_units = ()
        if view in ("state_removed", "full_input"):
            instruction = _capped(
                _grams12(item.instructions), params.text_gram_cap, self.hasher
            )
            instruction_units = _capped(
                _unigrams(item.instructions), params.text_gram_cap, self.hasher
            )
        if view == "full_input":
            state = _capped(_grams12(item.state), params.state_gram_cap, self.hasher)
            state_units = _capped(
                _unigrams(item.state), params.state_cross_unigrams, self.hasher
            )
        if view == "hypothesis_only":
            hypothesis = _capped(
                _grams12(item.hypothesis or ""), params.text_gram_cap, self.hasher
            )
        for position, (key, description) in enumerate(zip(item.keys, item.options)):
            base, option_units = self.option(description)
            ids = set(base)
            ids.update(
                (
                    index("bias"),
                    index(f"pos|{position}"),
                    index(f"pos|{position}|{count}"),
                    index(f"key|{key}"),
                )
            )
            for prefix, grams in (("i", instruction), ("s", state), ("h", hypothesis)):
                for _, gram in grams:
                    ids.add(index(f"{prefix}p|{gram}|{position}"))
                    ids.add(index(f"{prefix}k|{gram}|{key}"))
            if instruction_units:
                ids.update(
                    index(feature)
                    for feature in _crosses(
                        instruction_units,
                        option_units,
                        params.instruction_cross_cap,
                        "ix",
                    )
                )
            if state_units:
                ids.update(
                    index(feature)
                    for feature in _crosses(
                        state_units, option_units, params.state_cross_cap, "sx"
                    )
                )
            yield array("I", sorted(ids))


@dataclasses.dataclass
class _ViewData:
    rows: array
    first: array
    offsets: array
    feats: array


def _view_data(view: str) -> _ViewData:
    items: list[Item] = _WORK["items"]
    features = _Features(_WORK["params"])
    rows, first, offsets, feats = (
        array("I"),
        array("Q", [0]),
        array("Q", [0]),
        array("I"),
    )
    for number, item in enumerate(items):
        if view == "hypothesis_only" and item.hypothesis is None:
            continue
        rows.append(number)
        for ids in features.row(view, item):
            feats.extend(ids)
            offsets.append(len(feats))
        first.append(len(offsets) - 1)
    return _ViewData(rows, first, offsets, feats)


def _train(
    data: _ViewData,
    members: list[int],
    items: list[Item],
    params: LearnerParams,
    seed: str,
) -> array:
    size = 1 << params.bits
    weights = array("d", bytes(8 * size))
    squares = array("d", bytes(8 * size))
    examples = []
    for member in members:
        label = items[data.rows[member]].label
        start = data.first[member]
        examples.extend(
            (start + option, 1.0 if option == label else 0.0)
            for option in range(data.first[member + 1] - start)
        )
    rng = random.Random(seed)
    rate, l2, sqrt, exp = params.learning_rate, params.l2, math.sqrt, math.exp
    feats, offsets = data.feats, data.offsets
    for _ in range(params.epochs):
        rng.shuffle(examples)
        for example, target in examples:
            ids = feats[offsets[example] : offsets[example + 1]]
            margin = max(-35.0, min(35.0, sum(map(weights.__getitem__, ids))))
            error = 1.0 / (1.0 + exp(-margin)) - target
            for feature in ids:
                weight = weights[feature]
                gradient = error + l2 * weight
                square = squares[feature] + gradient * gradient
                squares[feature] = square
                weights[feature] = weight - rate * gradient / (sqrt(square) + 1e-12)
    return weights


def _predict(data: _ViewData, member: int, weights: array) -> int:
    best, best_score = 0, -math.inf
    for option, example in enumerate(range(data.first[member], data.first[member + 1])):
        ids = data.feats[data.offsets[example] : data.offsets[example + 1]]
        score = sum(map(weights.__getitem__, ids))
        if score > best_score:
            best, best_score = option, score
    return best


def _fit_fold(task: tuple[str, int]) -> list[tuple[int, int]]:
    view, fold = task
    data: _ViewData = _WORK["views"][view]
    items: list[Item] = _WORK["items"]
    params: LearnerParams = _WORK["params"]
    predictions = []
    for task_type in TASK_TYPES:
        train, test = [], []
        for member, row in enumerate(data.rows):
            item = items[row]
            if item.task_type == task_type:
                (test if item.fold == fold else train).append(member)
        if not test:
            continue
        weights = _train(
            data, train, items, params, f"{params.seed}|{view}|{fold}|{task_type}"
        )
        predictions.extend(
            (data.rows[member], _predict(data, member, weights)) for member in test
        )
    return predictions


def _run(function: Any, tasks: list[Any], workers: int) -> list[Any]:
    if workers <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    context = multiprocessing.get_context("fork")
    with context.Pool(min(workers, len(tasks))) as pool:
        return pool.map(function, tasks, chunksize=1)


def _prior_targets(item: Item) -> list[Any]:
    """Choice priors are over positions; Noul and Score priors are over keys,
    which equal positions when options are in canonical order."""
    return (
        list(range(len(item.keys))) if item.task_type == "choice" else list(item.keys)
    )


def _prior_order(task_type: str, target: Any) -> Any:
    return int(target) if task_type == "score" else target


def position_prior(items: Sequence[Item]) -> list[int]:
    """Cross-validated most frequent gold target per (task type, option count),
    falling back to the task type; ties -> lowest position, grade or key."""
    predictions = [0] * len(items)
    for fold in range(FOLDS):
        by_count: dict[tuple[str, int], collections.Counter[Any]] = (
            collections.defaultdict(collections.Counter)
        )
        by_task: dict[str, collections.Counter[Any]] = collections.defaultdict(
            collections.Counter
        )
        for item in items:
            if item.fold != fold:
                target = _prior_targets(item)[item.label]
                by_count[(item.task_type, len(item.keys))][target] += 1
                by_task[item.task_type][target] += 1
        for number, item in enumerate(items):
            if item.fold != fold:
                continue
            valid = {
                target: option for option, target in enumerate(_prior_targets(item))
            }
            tables = (
                by_count.get((item.task_type, len(item.keys))),
                by_task.get(item.task_type),
            )
            for table in tables:
                ranked = sorted(
                    (-count, _prior_order(item.task_type, target), target)
                    for target, count in (table or {}).items()
                    if target in valid
                )
                if ranked:
                    predictions[number] = valid[ranked[0][2]]
                    break
    return predictions


def longest_option(item: Item) -> int:
    lengths = [len(normalize(description)) for description in item.options]
    return lengths.index(max(lengths))


def lexical_overlap(item: Item) -> int:
    state = set(_unigrams(item.state))
    overlaps = [
        len(state & set(_unigrams(description))) for description in item.options
    ]
    return overlaps.index(max(overlaps))


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + Z95 * Z95 / total
    centre = (p + Z95 * Z95 / (2 * total)) / denominator
    half = (
        Z95
        * math.sqrt(p * (1 - p) / total + Z95 * Z95 / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def _summary(
    members: Sequence[int], correct: Sequence[bool], prior: Sequence[bool]
) -> dict[str, Any]:
    total = len(members)
    hits = sum(correct[row] for row in members)
    majority = sum(prior[row] for row in members)
    return {
        "n": total,
        "correct": hits,
        "accuracy": round(hits / total, 6),
        "majority": round(majority / total, 6),
        "delta": round((hits - majority) / total, 6),
        "wilson95": wilson(hits, total),
        "exceeds_margin": hits - majority > MARGIN * total + 1e-9,
    }


def run_baselines(
    rows: Iterable[Mapping[str, Any]],
    *,
    params: LearnerParams = LearnerParams(),
    workers: int = 1,
) -> dict[str, Any]:
    items = [item_of(row) for row in rows]
    if len({item.id for item in items}) != len(items):
        raise ValueError("Duplicate row id")
    _WORK.clear()
    _WORK.update(items=items, params=params)
    try:
        built = _run(_view_data, list(LEARNED_VIEWS), workers)
        _WORK["views"] = dict(zip(LEARNED_VIEWS, built))
        jobs = [(view, fold) for view in LEARNED_VIEWS for fold in range(FOLDS)]
        fitted = _run(_fit_fold, jobs, workers)
    finally:
        _WORK.clear()
    predictions: dict[str, dict[int, int]] = {view: {} for view in LEARNED_VIEWS}
    for (view, _), found in zip(jobs, fitted):
        predictions[view].update(found)
    heuristic_predictions: dict[str, list[int | None]] = {
        "position_prior": list(position_prior(items)),
        "longest_option": [longest_option(item) for item in items],
        "lexical_overlap": [
            lexical_overlap(item) if item.task_type == "choice" else None
            for item in items
        ],
    }
    prior_choice = heuristic_predictions["position_prior"]
    prior = [prior_choice[row] == item.label for row, item in enumerate(items)]
    views = {view: _report(predictions[view], items, prior) for view in LEARNED_VIEWS}
    heuristics = {
        name: _report(
            {row: choice for row, choice in enumerate(chosen) if choice is not None},
            items,
            prior,
        )
        for name, chosen in heuristic_predictions.items()
    }
    gates = []
    for task_type in TASK_TYPES:
        for view in GATED_VIEWS:
            summary = views[view].get("by_task_type", {}).get(task_type)
            if summary is None or summary["n"] < MIN_GATE_ROWS:
                continue
            families = [
                {"family": family, "n": value["n"], "delta": value["delta"]}
                for family, value in sorted(
                    views[view]["by_task_family"][task_type].items(),
                    key=lambda entry: (-entry[1]["delta"], entry[0]),
                )
                if value["exceeds_margin"]
            ]
            gates.append(
                {
                    "task_type": task_type,
                    "view": view,
                    **{
                        key: summary[key]
                        for key in ("n", "accuracy", "majority", "delta", "wilson95")
                    },
                    "threshold": round(summary["majority"] + MARGIN, 6),
                    "pass": not summary["exceeds_margin"],
                    "families_over_margin": families,
                }
            )
    verdict = (
        "INCONCLUSIVE"
        if not gates
        else ("PASS" if all(gate["pass"] for gate in gates) else "FAIL")
    )
    groups = {item.group_id for item in items}
    folds = collections.Counter(item.fold for item in items)
    group_folds = collections.Counter(fold_of(group) for group in groups)
    return {
        "schema": SCHEMA,
        "rows": len(items),
        "groups": len(groups),
        "folds": {
            "rule": "int(sha256(group_id).hexdigest(), 16) % 5",
            "rows": [folds[fold] for fold in range(FOLDS)],
            "groups": [group_folds[fold] for fold in range(FOLDS)],
        },
        "learner": dataclasses.asdict(params),
        "gate": {
            "views": list(GATED_VIEWS),
            "min_rows": MIN_GATE_ROWS,
            "margin": MARGIN,
            "rule": "PASS iff accuracy <= majority + margin; majority is the "
            "cross-validated position prior on the same rows",
        },
        "majority": {
            task: {"n": value["n"], "accuracy": value["accuracy"]}
            for task, value in heuristics["position_prior"]["by_task_type"].items()
        },
        "views": views,
        "heuristics": heuristics,
        "gates": gates,
        "verdict": verdict,
    }


def _report(
    chosen: Mapping[int, int], items: Sequence[Item], prior: Sequence[bool]
) -> dict[str, Any]:
    members = sorted(chosen)
    if not members:
        return {"n_rows": 0}
    correct = [False] * len(items)
    for row, choice in chosen.items():
        correct[row] = choice == items[row].label
    report: dict[str, Any] = {"n_rows": len(members)}
    for field in ("task_type", "family", "source"):
        grouped: dict[str, list[int]] = collections.defaultdict(list)
        for row in members:
            grouped[getattr(items[row], field)].append(row)
        report[f"by_{field}"] = {
            name: _summary(rows, correct, prior)
            for name, rows in sorted(grouped.items())
        }
    nested: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in members:
        nested[items[row].task_type][items[row].family].append(row)
    report["by_task_family"] = {
        task: {
            family: _summary(rows, correct, prior)
            for family, rows in sorted(families.items())
        }
        for task, families in sorted(nested.items())
    }
    return report


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        parser.error(f"refusing to overwrite {args.receipt}")
    receipt = run_baselines(
        load_rows(args.rows), params=LearnerParams(seed=args.seed), workers=args.workers
    )
    receipt["input"] = {"sha256": file_sha256(args.rows), "rows": receipt["rows"]}
    _write_new(args.receipt, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "gates": len(receipt["gates"])}))
    return {"PASS": 0, "FAIL": 1}.get(receipt["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
