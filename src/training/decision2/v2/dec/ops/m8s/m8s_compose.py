"""Decoder M8-small top-up rows (prereg dec-m8s-prereg-2026-09-30.md, "Top-up rows").

One deterministic pass per tier over the tier's released recipe (r2-clean). Groups are split by the frozen human /
typed partition (a group takes its first row's part; mixed groups are reported) and each part is sampled to its
native-token budget with the builder's sampler (`build_template_s.select_groups`: strata source x task type x
language, `sha256(seed \\0 group)` order, each stratum's share of the budget). Chosen rows keep the recipe's bytes and
order. Output: train.jsonl, train.parts.jsonl ({id, input_sha256, part}) and compose.json (report and hashes).

usage (launch.sh --cpu): python3 v2/dec/ops/m8s/m8s_compose.py --tier 2b|08b --recipe R --recipe-sha S \
    --quarantine Q --tokenizer TOK --output DIR [--human-budget N] [--typed-budget N] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from training.model.data import file_sha256, load_partition  # noqa: E402
from v2.common import eval_only  # noqa: E402
from v2.dec.build_template_s import group_rows, select_groups  # noqa: E402

SCHEMA = "dec-m8s-compose/1"
PARTS = ("human", "typed")
TYPED_SOURCES = frozenset(
    {
        "dec10:generated_stage4_v2",
        "dec10:generated_stage1_3",
        "decision2_verifiable_v2_a2",
        "decision2_verifiable_v2_a4v2h",
        "decision2_verifiable_v2_a6",
        "legacy:stage4-general-composition-v2",
        "decision2_targeted_programmatic_v1",
        "decision2_programmatic_original_v1",
    }
)
TYPED_PATTERNS = (
    re.compile(r"^decision2_.*(programmatic|verifiable)"),
    re.compile(r"^dec10:generated"),
)
REPLAY_SOURCE = "legacy:stage3_replay"
Row = dict[str, Any]


def part_of(row: Row) -> str:
    """The frozen partition: programmatic / generated sources are typed, human-labeled public sources human;
    replayed stage-3 rows count as human only with their upstream human label (the 9B M6 rule).
    """
    source = row["source"]
    if source == REPLAY_SOURCE:
        return "human" if "upstream_label" in row else "typed"
    if source in TYPED_SOURCES or any(p.search(source) for p in TYPED_PATTERNS):
        return "typed"
    return "human"


def compose(
    rows: list[Row],
    length_of: dict[str, int],
    *,
    quarantine: set[str],
    budgets: dict[str, int],
    seed: str,
) -> tuple[list[Row], dict[str, str], dict[str, Any]]:
    """Chosen rows in recipe order, their parts by id, and the report."""
    dropped = sum(1 for r in rows if r["group_id"] in quarantine)
    kept = [r for r in rows if r["group_id"] not in quarantine]
    groups = group_rows(kept)
    group_part = {g: part_of(members[0]) for g, members in groups.items()}
    mixed = sorted(
        g for g, members in groups.items() if len({part_of(r) for r in members}) > 1
    )
    chosen: set[str] = set()
    per_part: dict[str, Any] = {}
    for part in PARTS:
        pool = {g: m for g, m in groups.items() if group_part[g] == part}
        gtok = {g: sum(length_of[r["id"]] for r in m) for g, m in pool.items()}
        available = sum(gtok.values())
        if budgets[part] > available:
            raise ValueError(
                f"{part} budget {budgets[part]} exceeds the recipe's {available} tokens"
            )
        picked = select_groups(pool, gtok, budgets[part], f"{seed}:{part}")
        chosen.update(picked)
        per_part[part] = {
            "budget": budgets[part],
            "available_tokens": available,
            "groups": len(picked),
            "tokens": sum(gtok[g] for g in picked),
        }
    out = [r for r in kept if r["group_id"] in chosen]
    parts = {r["id"]: group_part[r["group_id"]] for r in out}
    report = {
        "recipe_rows": len(rows),
        "quarantined_rows": dropped,
        "mixed_groups": {"count": len(mixed), "first": mixed[:20]},
        "parts": per_part,
        "rows": len(out),
        "tokens": sum(length_of[r["id"]] for r in out),
        "rows_by_part": dict(sorted(Counter(parts.values()).items())),
        "tokens_by_part": {
            part: sum(length_of[r["id"]] for r in out if parts[r["id"]] == part)
            for part in PARTS
        },
        "rows_by_type": dict(sorted(Counter(r["task_type"] for r in out).items())),
        "rows_by_type_and_part": {
            part: dict(
                sorted(
                    Counter(
                        r["task_type"] for r in out if parts[r["id"]] == part
                    ).items()
                )
            )
            for part in PARTS
        },
        "rows_by_language": dict(Counter(r["language"] for r in out).most_common(15)),
        "sources": {
            part: dict(
                Counter(r["source"] for r in out if parts[r["id"]] == part).most_common(
                    40
                )
            )
            for part in PARTS
        },
    }
    return out, parts, report


def token_budget_summary(rows: list[Row], length_of: dict[str, int]) -> dict[str, int]:
    tokens: dict[str, int] = defaultdict(int)
    for r in rows:
        tokens[part_of(r)] += length_of[r["id"]]
    return dict(sorted(tokens.items()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=("2b", "08b"), required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--recipe-sha", required=True)
    p.add_argument(
        "--quarantine", type=Path, required=True, help='{"group_ids": [...]}'
    )
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--human-budget", type=int, default=3_000_000)
    p.add_argument("--typed-budget", type=int, default=3_000_000)
    p.add_argument("--seed", help="default dec-m8s-topup-<tier>-v1")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    eval_only.guard(a)
    from v2.dec.build_mixture import token_lengths

    actual = file_sha256(a.recipe)
    if actual != a.recipe_sha:
        raise ValueError(f"{a.recipe}: sha256 {actual} != {a.recipe_sha}")
    rows = load_partition(a.recipe, "train")
    quarantine = set(json.loads(a.quarantine.read_text())["group_ids"])
    length_of = dict(
        zip((r["id"] for r in rows), token_lengths(rows, a.tokenizer, a.workers))
    )
    seed = a.seed or f"dec-m8s-topup-{a.tier}-v1"
    out, parts, report = compose(
        rows,
        length_of,
        quarantine=quarantine,
        budgets={"human": a.human_budget, "typed": a.typed_budget},
        seed=seed,
    )
    eval_only.check_rows(out)
    a.output.mkdir(parents=True)
    with (a.output / "train.jsonl").open("x", encoding="utf-8") as stream:
        for r in out:
            stream.write(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    with (a.output / "train.parts.jsonl").open("x", encoding="utf-8") as stream:
        for r in out:
            stream.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "input_sha256": r["input_sha256"],
                        "part": parts[r["id"]],
                    }
                )
                + "\n"
            )
    doc = {
        "schema": SCHEMA,
        "tier": a.tier,
        "seed": seed,
        "recipe": {"path": str(a.recipe), "sha256": actual},
        "quarantine": {"path": str(a.quarantine), "sha256": file_sha256(a.quarantine)},
        "recipe_tokens_by_part": token_budget_summary(rows, length_of),
        "tokenizer_files_sha256": {
            n: file_sha256(a.tokenizer / n)
            for n in ("tokenizer.json", "tokenizer_config.json")
            if (a.tokenizer / n).is_file()
        },
        "report": report,
        "files": {
            "train.jsonl": file_sha256(a.output / "train.jsonl"),
            "train.parts.jsonl": file_sha256(a.output / "train.parts.jsonl"),
        },
    }
    (a.output / "compose.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "tier": a.tier,
                "rows": report["rows"],
                "tokens_by_part": report["tokens_by_part"],
                "train_sha256": doc["files"]["train.jsonl"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
