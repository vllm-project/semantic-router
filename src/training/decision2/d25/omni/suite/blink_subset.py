"""Which seven BLINK val subtasks the board uses.

The board's BLINK has 961 rows from "seven subtasks of the validation split". 43 seven-subtask sets of
BLINK val have exactly 961 rows. Three filters narrow them before any model is run:

1. the score lattice: the board's chance sum is 340.25 (``lattice``), which leaves D, E and F;
2. single-image engines (multi-image rows count as wrong): their published skill needs a number of
   correct rows that the set's single-image rows must cover, which removes D;
3. the reference fit: per-subtask accuracies of reference models, measured on all 14 val subtasks,
   predict each model's BLINK skill under every candidate; the set with the smallest error against
   the official public skills wins (``fit``).

Named sets: A, B, C from the reconstruction note; D, E, F are the lattice-consistent ones.

    python -m d25.omni.suite.blink_subset --rows rows-blink-val.jsonl.gz --results-dir runs/ \
        --official official.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from d25.omni.suite import lattice, score

# BLINK val (BLINK-Benchmark/BLINK @ a3666eb2): rows, options per row, images per row.
SUBTASKS: dict[str, tuple[int, int, int]] = {
    "Art_Style": (117, 2, 3),
    "Counting": (120, 4, 1),
    "Forensic_Detection": (132, 4, 4),
    "Functional_Correspondence": (130, 4, 2),
    "IQ_Test": (150, 4, 1),
    "Jigsaw": (150, 2, 3),
    "Multi-view_Reasoning": (133, 2, 2),
    "Object_Localization": (122, 2, 1),
    "Relative_Depth": (124, 2, 1),
    "Relative_Reflectance": (134, 3, 1),
    "Semantic_Correspondence": (139, 4, 2),
    "Spatial_Relation": (143, 2, 1),
    "Visual_Correspondence": (172, 4, 2),
    "Visual_Similarity": (135, 2, 3),
}
BOARD_ROWS = 961
NAMED = {
    "A": (
        "Counting",
        "Forensic_Detection",
        "Multi-view_Reasoning",
        "Object_Localization",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
    ),
    "B": (
        "Counting",
        "Forensic_Detection",
        "Functional_Correspondence",
        "IQ_Test",
        "Multi-view_Reasoning",
        "Relative_Depth",
        "Visual_Correspondence",
    ),
    "C": (
        "Forensic_Detection",
        "Functional_Correspondence",
        "IQ_Test",
        "Multi-view_Reasoning",
        "Relative_Reflectance",
        "Semantic_Correspondence",
        "Spatial_Relation",
    ),
    "D": (
        "Art_Style",
        "Counting",
        "Functional_Correspondence",
        "Jigsaw",
        "Multi-view_Reasoning",
        "Semantic_Correspondence",
        "Visual_Correspondence",
    ),
    "E": (
        "Counting",
        "Functional_Correspondence",
        "Multi-view_Reasoning",
        "Relative_Depth",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
    ),
    "F": (
        "Counting",
        "Functional_Correspondence",
        "Object_Localization",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
        "Visual_Similarity",
    ),
}
DEFAULT = "E"


def name_of(subset: Sequence[str]) -> str:
    key = tuple(sorted(subset))
    for name, members in NAMED.items():
        if tuple(sorted(members)) == key:
            return name
    return "+".join(s.split("_")[0][:4] for s in key)


def candidates(
    subtasks: Mapping[str, tuple[int, int, int]] = SUBTASKS,
    size: int = 7,
    rows: int = BOARD_ROWS,
):
    return [
        tuple(c)
        for c in itertools.combinations(sorted(subtasks), size)
        if sum(subtasks[s][0] for s in c) == rows
    ]


def chance_sum(subset: Sequence[str], subtasks=SUBTASKS) -> float:
    return sum(subtasks[s][0] / subtasks[s][1] for s in subset)


def single_image_rows(subset: Sequence[str], subtasks=SUBTASKS) -> int:
    return sum(subtasks[s][0] for s in subset if subtasks[s][2] == 1)


def needed_correct(skill_value: float, n: int, chance: float) -> int:
    """Correct rows behind a published skill (nearest integer on the lattice)."""
    return round(n * (chance + skill_value / 100 * (1 - chance)))


def screen(
    board_values: Sequence[float],
    single_image_skills: Sequence[float] = (),
    subtasks=SUBTASKS,
) -> list[dict]:
    """Every 961-row candidate with its lattice and single-image verdicts."""
    out = []
    for subset in candidates(subtasks):
        s = chance_sum(subset, subtasks)
        bad = lattice.check(BOARD_ROWS, s, board_values)
        single = single_image_rows(subset, subtasks)
        need = [
            needed_correct(v, BOARD_ROWS, s / BOARD_ROWS) for v in single_image_skills
        ]
        out.append(
            {
                "name": name_of(subset),
                "subtasks": list(subset),
                "chance_sum": s,
                "lattice_ok": not bad,
                "lattice_misses": len(bad),
                "single_image_rows": single,
                "single_image_needed": need,
                "single_image_ok": all(k <= single for k in need),
            }
        )
    return sorted(
        out,
        key=lambda c: (
            not (c["lattice_ok"] and c["single_image_ok"]),
            c["lattice_misses"],
            c["name"],
        ),
    )


def predicted_skill(
    per_subtask: Mapping[str, tuple[int, int]], subset: Sequence[str], subtasks=SUBTASKS
) -> float:
    """BLINK skill of one model under a candidate set from its per-subtask (correct, rows)."""
    correct = rows = 0
    for s in subset:
        k, n = per_subtask[s]
        if n != subtasks[s][0]:
            raise ValueError(f"{s}: {n} rows scored, expected {subtasks[s][0]}")
        correct, rows = correct + k, rows + n
    return score.skill(correct / rows, chance_sum(subset, subtasks) / rows)


def fit(
    models: Mapping[str, Mapping[str, tuple[int, int]]],
    official: Mapping[str, float],
    pool: Sequence[Sequence[str]] | None = None,
    subtasks=SUBTASKS,
) -> list[dict]:
    """Rank candidate sets by RMSE between predicted and official BLINK public skill."""
    names = sorted(m for m in models if m in official)
    if not names:
        raise ValueError(
            "no model has both per-subtask results and an official BLINK skill"
        )
    ranking = []
    for subset in pool or candidates(subtasks):
        errors = {
            m: predicted_skill(models[m], subset, subtasks) - official[m] for m in names
        }
        rmse = math.sqrt(sum(e * e for e in errors.values()) / len(errors))
        ranking.append(
            {
                "name": name_of(subset),
                "subtasks": list(subset),
                "rmse": rmse,
                "max_abs": max(abs(e) for e in errors.values()),
                "errors": errors,
            }
        )
    ranking.sort(key=lambda r: r["rmse"])
    return ranking


def per_subtask_counts(rows, answers) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        sub = row["metadata"]["subtask"]
        counts[sub][0] += score.row_correct(row, answers.get(row["id"]))
        counts[sub][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rows",
        help="BLINK val rows of all 14 subtasks (variants/blink-val-all.jsonl.gz)",
    )
    parser.add_argument(
        "--results-dir", help="one <model>.jsonl results file per reference model"
    )
    parser.add_argument("--official", help="JSON {model: official BLINK public skill}")
    args = parser.parse_args()
    values = lattice.board_values()["BLINK"]
    board = json.loads(lattice.BOARD_FIXTURE.read_text())
    single = sorted(
        {
            e["bench"]["BLINK"]["pub"]
            for s in board["snapshots"]
            for e in s["entries"]
            if e["single_image"]
        }
    )
    report = {"screen": screen(values, single)}
    if args.rows and args.results_dir and args.official:
        rows = list(score.read_jsonl(args.rows))
        official = json.loads(Path(args.official).read_text())
        models = {
            p.stem.split(".")[0]: per_subtask_counts(rows, score.load_answers(p))
            for p in sorted(Path(args.results_dir).glob("*.jsonl*"))
        }
        pool = [
            tuple(c["subtasks"])
            for c in report["screen"]
            if c["lattice_ok"] and c["single_image_ok"]
        ]
        report["fit_screened"] = fit(models, official, pool)
        report["fit_all"] = fit(models, official)[:10]
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
