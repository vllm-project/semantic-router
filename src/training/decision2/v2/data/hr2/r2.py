"""HR2-r2 (prereg amendment 4): round-1 error analysis and the PRM800K boundary list.

    python3 -m v2.data.hr2.r2 analyze --review R1/review.private.json --reviewed R1/pass1b.train.jsonl \
        --train R1/final.train.jsonl --dev R1/final.dev.jsonl [--tokens TOK ...] --raw RAW \
        --out PUBLIC.json --private PRIVATE.json
    python3 -m v2.data.hr2.r2 boundary --cand CAND --raw RAW --out-dir DIR

``boundary`` lists the ``prm_step`` yes rows of the candidates (TRAIN and DEV) that, in some PRM800K
record that produced them, are the last +1 step on the walked path before that path's first -1 step
(construction drop C2). ``analyze`` classifies the round-1 gold errors by family, construction and
agreement statistics, counts for every candidate filter the errors, reviewed rows and published rows
it removes, and simulates the r2 sizes (the round-1 final files minus the r2 drops, re-balanced as
``finalize`` does). Public receipts hold counts only; ids stay in the private file on the node.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.dq.blind_review import proportion, weighted_error
from v2.data.hr2 import build
from v2.data.hr2 import families as fam
from v2.data.sources.common import sha

LICENCE_DROPS = ("vitc", "allegro")
CONSTRUCTION_DROPS = ("hs3_help",)
R2_FAMILIES = ("hs3_pref", "eth_util", "eth_cs", "eth_deon", "eth_just", "prm_step")
PRM_FILES = ("prm800k/phase1_train.jsonl", "prm800k/phase2_train.jsonl")
VITC_FILE = "tals_vitaminc/train.jsonl"
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
DESERT = re.compile(r"\b(deserve[ds]?|deserving|entitled|justified)\b", re.IGNORECASE)
Row = Mapping[str, Any]


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line]


def verify_pins(raw: Path, names: Iterable[str]) -> dict[str, str]:
    pins = {}
    for name in names:
        actual = build.file_sha256(raw / name)
        if actual != build.PINS[name]:
            raise ValueError(f"{name}: sha256 {actual} != pinned {build.PINS[name]}")
        pins[name] = actual
    return pins


# --------------------------------------------------------------------------- PRM800K boundary


def prm_records(raw: Path) -> Iterator[dict[str, Any]]:
    for name in PRM_FILES:
        yield from fam.read_jsonl(raw / name)


def boundary_keys(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """Hash keys of the last +1 step before the first -1 on each walked path (``prm_step`` rules)."""
    keys: set[str] = set()
    for record in records:
        if record.get("is_quality_control_question") or record.get(
            "is_initial_screening_question"
        ):
            continue
        if record["label"].get("finish_reason") in ("bad_problem", "give_up"):
            continue
        yes, no, _ = fam.prm_walk(record)
        if no is None or not yes:
            continue
        index, previous, text = max(yes, key=lambda item: item[0])
        problem = record["question"]["problem"].strip()
        keys.add(sha(canonical(fam.prm_state(problem, previous, index, text)))[:24])
    return keys


def is_boundary(row: Row, keys: set[str]) -> bool:
    return (
        row["family"] == "prm_step"
        and row["label"] == 1
        and row["audit_metadata"]["hr2"]["hash_key"] in keys
    )


def boundary(args: argparse.Namespace) -> int:
    pins = verify_pins(args.raw, PRM_FILES)
    keys = boundary_keys(prm_records(args.raw))
    ids: list[str] = []
    counts: dict[str, Any] = {}
    for split, name in (
        ("train", "hr2.train.cand.jsonl"),
        ("dev", "hr2.dev.cand.jsonl"),
    ):
        rows = read_rows(args.cand / name)
        yes = [r for r in rows if r["family"] == "prm_step" and r["label"] == 1]
        flagged = [r for r in yes if is_boundary(r, keys)]
        buckets = collections.Counter(
            r["audit_metadata"]["hr2"]["bucket"] for r in flagged
        )
        counts[split] = {
            "prm_yes_rows": len(yes),
            "boundary_rows": len(flagged),
            "by_bucket": dict(sorted(buckets.items())),
        }
        ids += [r["id"] for r in flagged]
    args.out_dir.mkdir(mode=0o700)
    text = "".join(i + "\n" for i in sorted(ids))
    receipt = {
        "schema": "decision2.hr2.r2-boundary.v1",
        "rule": "prm_step yes rows that are the last +1 step before the first -1 on a walked path",
        "raw": pins,
        "candidates": {
            split: build.file_sha256(args.cand / f"hr2.{split}.cand.jsonl")
            for split in ("train", "dev")
        },
        "boundary_states": len(keys),
        "rows": counts,
        "ids_sha256": build.write_new(
            args.out_dir / "construction-ids.txt", text.encode("utf-8")
        ),
    }
    build.write_json(args.out_dir / "boundary.public.json", receipt)
    print(json.dumps({split: c["boundary_rows"] for split, c in counts.items()}))
    return 0


# --------------------------------------------------------------------------- analysis


def vitc_revisions(raw: Path, cases: set[str]) -> dict[str, list[str]]:
    """Distinct evidence sentences of each wanted VitaminC case (both sides of the revision)."""
    found: dict[str, set[str]] = collections.defaultdict(set)
    for record in fam.read_jsonl(raw / VITC_FILE):
        if record["case_id"] in cases and record.get("revision_type") == "real":
            found[record["case_id"]].add(record["evidence"].strip())
    return {case: sorted(texts) for case, texts in found.items()}


def revision_class(row: Row, revisions: Mapping[str, Sequence[str]]) -> str:
    """``numeric`` when the two revision sentences differ in their numbers, else ``wording``."""
    texts = revisions.get(row["audit_metadata"]["hr2"].get("case_id"), ())
    if len(texts) != 2:
        return "other"
    numbers = [sorted(NUMBER.findall(text)) for text in texts]
    return "numeric" if numbers[0] != numbers[1] else "wording"


def construction(
    row: Row, keys: set[str], revisions: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Construction class and agreement statistics of one row (counts and categories only)."""
    hr2 = row["audit_metadata"]["hr2"]
    family = row["family"]
    if family == "hs3_pref":
        scores = hr2["individual_scores"]
        return {
            "margin": abs(hr2["overall_preference"]),
            "annotators": len(scores),
            "any_slight_annotator": any(abs(s) < 2 for s in scores),
            "domain": hr2["domain"],
        }
    if family == "hs3_help":
        levels = hr2["annotator_levels"]
        return {
            "agreement": "unanimous" if max(levels) == min(levels) else "2-1",
            "domain": hr2["domain"],
        }
    if family == "prm_step":
        return {"boundary": is_boundary(row, keys), "bucket": hr2["bucket"]}
    if family == "vitc":
        return {
            "pair": "cross (refutes)" if row["label"] == 0 else "own (supports)",
            "revision": revision_class(row, revisions),
        }
    if family == "eth_just":
        desert = DESERT.search(row["state"]["scenario"])
        return {"template": "desert" if desert else "impartiality"}
    return {}


def filters(
    keys: set[str], revisions: Mapping[str, Sequence[str]]
) -> dict[str, Callable[[Row], bool]]:
    """Every construction filter considered; L1, L2, C1 and C2 are the r2 drops."""

    def hr2(row: Row) -> Mapping[str, Any]:
        return row["audit_metadata"]["hr2"]

    return {
        "L1_drop_vitc": lambda r: r["family"] == "vitc",
        "L2_drop_allegro": lambda r: r["family"] == "allegro",
        "C1_drop_hs3_help": lambda r: r["family"] == "hs3_help",
        "C2_prm_boundary_yes": lambda r: is_boundary(r, keys),
        "alt_hs3_help_split_ratings_only": lambda r: r["family"] == "hs3_help"
        and max(hr2(r)["annotator_levels"]) != min(hr2(r)["annotator_levels"]),
        "alt_hs3_pref_margin_2": lambda r: r["family"] == "hs3_pref"
        and abs(hr2(r)["overall_preference"]) == 2,
        "alt_hs3_pref_any_slight_annotator": lambda r: r["family"] == "hs3_pref"
        and any(abs(s) < 2 for s in hr2(r)["individual_scores"]),
        "alt_vitc_wording_revisions": lambda r: r["family"] == "vitc"
        and revision_class(r, revisions) == "wording",
        "alt_eth_just_desert": lambda r: r["family"] == "eth_just"
        and bool(DESERT.search(r["state"]["scenario"])),
    }


R2_DROPS = (
    "L1_drop_vitc",
    "L2_drop_allegro",
    "C1_drop_hs3_help",
    "C2_prm_boundary_yes",
)


def agrees(item: Mapping[str, Any], who: str) -> bool:
    if item["task_type"] == "score":
        return abs(int(item[who]) - int(item["gold"])) <= 1
    return item[who] == item["gold"]


def family_stats(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in items:
        by_family[item["family"]].append(item)
    for name, group in sorted(by_family.items()):
        confidence = [(i["r1_confidence"], i["r2_confidence"]) for i in group]
        out[name] = {
            "n": len(group),
            "errors": sum(i["error"] for i in group),
            "r1_agrees": sum(agrees(i, "r1") for i in group),
            "r2_agrees": sum(agrees(i, "r2") for i in group),
            "splits": sum(i["r3"] is not None for i in group),
            "any_low_confidence": sum("low" in c for c in confidence),
            "both_high_confidence": sum(c == ("high", "high") for c in confidence),
        }
    return out


def class_table(
    items: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Row],
    keys: set[str],
    revisions: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in items:
        info = construction(rows[item["id"]], keys, revisions)
        for field, value in info.items():
            cell = f"{item['family']}|{field}={value}|gold={item['gold']}"
            table[cell]["reviewed"] += 1
            table[cell]["errors"] += item["error"]
    return {cell: dict(counts) for cell, counts in sorted(table.items())}


def native_tokens(rows: Iterable[Row], tokens: Mapping[str, int]) -> int | None:
    if not tokens:
        return None
    return sum(tokens.get(row["id"], 0) for row in rows)


def simulate(
    rows: Sequence[Row], drop: Callable[[Row], bool], tokens: Mapping[str, int]
) -> dict[str, Any]:
    kept = build.rebalance([dict(r) for r in rows if not drop(r)])
    families = collections.Counter(r["family"] for r in kept)
    return {
        "rows": len(kept),
        "groups": len({r["group_id"] for r in kept}),
        "families": dict(sorted(families.items())),
        "task_type": dict(
            sorted(collections.Counter(r["task_type"] for r in kept).items())
        ),
        "native_tokens": native_tokens(kept, tokens),
    }


def analyze(
    review: Mapping[str, Any],
    reviewed_rows: Mapping[str, Row],
    train: Sequence[Row],
    dev: Sequence[Row],
    keys: set[str],
    revisions: Mapping[str, Sequence[str]],
    tokens: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    items = review["items"]
    errors = [i for i in items if i["error"]]
    rules = filters(keys, revisions)
    table = {}
    error_ids: dict[str, list[str]] = {}
    for name, rule in rules.items():
        hit_items = [i for i in items if rule(reviewed_rows[i["id"]])]
        hit_train = [r for r in train if rule(r)]
        error_ids[name] = sorted(i["id"] for i in hit_items if i["error"])
        table[name] = {
            "errors_removed": len(error_ids[name]),
            "reviewed_removed": len(hit_items),
            "train_rows": len(hit_train),
            "dev_rows": sum(rule(r) for r in dev),
            "train_native_tokens": native_tokens(hit_train, tokens),
        }

    def r2_drop(row: Row) -> bool:
        return any(rules[name](row) for name in R2_DROPS)

    kept = [i for i in items if not r2_drop(reviewed_rows[i["id"]])]
    expected = {
        "train": simulate(train, r2_drop, tokens),
        "dev": simulate(dev, r2_drop, tokens),
    }
    by_family: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in kept:
        by_family[item["family"]].append(item)
    population = expected["train"]["families"]
    estimate = {
        "pooled": proportion(sum(i["error"] for i in kept), len(kept)),
        "population_weighted": round(weighted_error(by_family, population), 6),
        "by_family": {
            name: proportion(sum(i["error"] for i in group), len(group))
            for name, group in sorted(by_family.items())
        },
    }
    described = []
    for item in sorted(errors, key=lambda i: (i["family"], i["rid"])):
        row = reviewed_rows[item["id"]]
        entry = {
            "family": item["family"],
            "task_type": item["task_type"],
            "gold": item["gold"],
            "reviewers": [item["r1"], item["r2"]]
            + ([item["r3"]] if item["r3"] else []),
            "confidence": [item["r1_confidence"], item["r2_confidence"]],
            "construction": construction(row, keys, revisions),
        }
        if item["task_type"] == "score":
            entry["level_distance"] = [
                abs(int(item[who]) - int(item["gold"])) for who in ("r1", "r2")
            ]
        described.append(entry)
    public = {
        "schema": "decision2.hr2.r2-analysis.v1",
        "round1": {
            "n": len(items),
            "errors": len(errors),
            "by_family": family_stats(items),
            "by_class": class_table(items, reviewed_rows, keys, revisions),
        },
        "errors": described,
        "filters": table,
        "r2_drops": list(R2_DROPS),
        "r2_expected": expected,
        "r2_round1_estimate": estimate,
    }
    private = {
        "schema": "decision2.hr2.r2-analysis-private.v1",
        "errors": [
            {"id": i["id"], "rid": i["rid"], **d}
            for i, d in zip(
                sorted(errors, key=lambda i: (i["family"], i["rid"])), described
            )
        ],
        "filter_error_ids": error_ids,
    }
    return public, private


def analyze_cli(args: argparse.Namespace) -> int:
    pins = verify_pins(args.raw, PRM_FILES + (VITC_FILE,))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    reviewed = {row["id"]: row for row in read_rows(args.reviewed)}
    missing = [i["id"] for i in review["items"] if i["id"] not in reviewed]
    if missing:
        raise ValueError(f"{len(missing)} reviewed ids are not in {args.reviewed}")
    train, dev = read_rows(args.train), read_rows(args.dev)
    tokens = {
        item["id"]: item["native"] for path in args.tokens for item in read_rows(path)
    }
    cases = {
        row["audit_metadata"]["hr2"]["case_id"]
        for row in [*reviewed.values(), *train, *dev]
        if row["family"] == "vitc"
    }
    keys = boundary_keys(prm_records(args.raw))
    public, private = analyze(
        review, reviewed, train, dev, keys, vitc_revisions(args.raw, cases), tokens
    )
    public["inputs"] = {
        "raw": pins,
        "review_private": build.file_sha256(args.review),
        "reviewed_rows": build.file_sha256(args.reviewed),
        "train": build.file_sha256(args.train),
        "dev": build.file_sha256(args.dev),
        "tokens": [build.file_sha256(path) for path in args.tokens],
    }
    build.write_json(args.out, public)
    build.write_json(args.private, private)
    print(
        json.dumps(
            {
                "errors_removed": {
                    k: v["errors_removed"] for k, v in public["filters"].items()
                },
                "r2_train_rows": public["r2_expected"]["train"]["rows"],
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("analyze")
    one.add_argument("--review", required=True, type=Path)
    one.add_argument("--reviewed", required=True, type=Path)
    one.add_argument("--train", required=True, type=Path)
    one.add_argument("--dev", required=True, type=Path)
    one.add_argument("--tokens", action="append", default=[], type=Path)
    one.add_argument("--raw", required=True, type=Path)
    one.add_argument("--out", required=True, type=Path)
    one.add_argument("--private", required=True, type=Path)
    two = sub.add_parser("boundary")
    two.add_argument("--cand", required=True, type=Path)
    two.add_argument("--raw", required=True, type=Path)
    two.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return analyze_cli(args) if args.command == "analyze" else boundary(args)


if __name__ == "__main__":
    raise SystemExit(main())
