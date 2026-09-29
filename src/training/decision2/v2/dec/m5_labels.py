"""Decoder Milestone 5 own-Nox replay labels: inputs, checks, target subsets, data diagnostic.

``rows``: the multilingual-block rows (non-English A7q / A7k / A7s / H5 / H8)
of one or more mixtures as a TRAIN file for ``v2.dec.teacher_label``,
optionally without ids an earlier label file already covers.

``check-sample``: the preregistered re-label sample, made of whole
``teacher_label`` batches (length-sorted, the same token budget) chosen in
seed-keyed hash order until at least N rows, so the re-label sees exactly the
batches (rows, order, padding) of the full labeling run and must reproduce it
bitwise. ``bitwise`` compares the two label files on the sample.

``overlap``: argmax agreement of new labels with a reference own-Nox file on
rows with the same id and input hash (M3 ``nox-teacher.jsonl``).

``subset``: target records restricted to a mixture's rows of one component
(optionally English / non-English only).

``diag``: the training-data diagnostic on the block's Noul rows per language:
each teacher's predicted-yes rate (argmax) and gold agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

from .m5_block import BLOCK_COMPONENTS, component_slices, read_lines

# v2.dec.teacher_label.TOKEN_BUDGET (not imported: that module needs torch).
TOKEN_BUDGET = 24_000


def mixture(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
    manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text())
    return [
        (name, line, row)
        for name, items in component_slices(read_lines(path), manifest)
        for line, row in items
    ]


def is_block(component: str, row: dict[str, Any]) -> bool:
    return component in BLOCK_COMPONENTS and row["language"] != "en"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
    pending.replace(path)


def label_batches(lengths: list[int], budget: int = TOKEN_BUDGET) -> list[list[int]]:
    """``teacher_label``'s batching: stable length sort, greedy padded-token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        width = max([lengths[i] for i in current] + [lengths[index]])
        if current and width * (len(current) + 1) > budget:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def whole_batch_sample(
    ids: list[str], lengths: list[int], minimum: int, seed: str
) -> list[int]:
    batches = label_batches(lengths)
    ranked = sorted(
        range(len(batches)),
        key=lambda b: hashlib.sha256(
            f"{seed}\0{ids[batches[b][0]]}".encode()
        ).hexdigest(),
    )
    chosen: list[int] = []
    for b in ranked:
        if len(chosen) >= minimum:
            break
        chosen += batches[b]
    return sorted(chosen)


def read_targets(path: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["id"] in out:
                raise ValueError(f"{path}: repeated id {record['id']}")
            out[record["id"]] = (line, record)
    return out


def argmax(probs: dict[str, float]) -> str:
    return max(probs, key=probs.__getitem__)


def cmd_rows(args: argparse.Namespace) -> dict[str, Any]:
    done = set()
    for path in args.exclude_labeled:
        done |= set(read_targets(path))
    seen: dict[str, str] = {}
    lines: list[str] = []
    counts: Counter = Counter()
    for path in args.train:
        for component, line, row in mixture(path):
            if not is_block(component, row):
                continue
            if row["id"] in seen:
                if seen[row["id"]] != line:
                    raise ValueError(
                        f"{row['id']}: rendered differently across mixtures"
                    )
                continue
            seen[row["id"]] = line
            if row["id"] in done:
                counts["already_labeled"] += 1
                continue
            lines.append(line)
            counts[f"{component}/{row['task_type']}"] += 1
    write_lines(args.output, lines)
    return {
        "rows": len(lines),
        "by_component_type": dict(counts),
        "output_sha256": file_sha256(args.output),
    }


def cmd_check_sample(args: argparse.Namespace) -> dict[str, Any]:
    from .build_mixture import token_lengths

    items = read_lines(args.rows)
    rows = [row for _, row in items]
    lengths = token_lengths(rows, args.tokenizer, args.workers)
    chosen = whole_batch_sample(
        [r["id"] for r in rows], lengths, args.minimum, args.seed
    )
    write_lines(args.output, [items[i][0] for i in chosen])
    return {
        "rows": len(chosen),
        "batches": len(
            {tuple(b) for b in label_batches(lengths) if set(b) <= set(chosen)}
        ),
        "rows_sha256": file_sha256(args.rows),
        "seed": args.seed,
        "output_sha256": file_sha256(args.output),
    }


def cmd_bitwise(args: argparse.Namespace) -> dict[str, Any]:
    full, check = read_targets(args.full), read_targets(args.check)
    identical = sum(full.get(i, ("",))[0] == line for i, (line, _) in check.items())
    worst = 0.0
    for i, (_, record) in check.items():
        other = full[i][1]["teacher_probs"]
        worst = max(
            worst, max(abs(other[k] - v) for k, v in record["teacher_probs"].items())
        )
    return {
        "rows": len(check),
        "bitwise_identical": identical,
        "pass": identical == len(check) and len(check) >= args.minimum,
        "max_abs_diff": worst,
        "full_sha256": file_sha256(args.full),
        "check_sha256": file_sha256(args.check),
    }


def cmd_overlap(args: argparse.Namespace) -> dict[str, Any]:
    new = read_targets(args.labels)
    agree = n = 0
    diff = 0.0
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with args.reference.open(encoding="utf-8") as stream:
        for line in stream:
            ref = json.loads(line)
            mine = new.get(ref["id"])
            if mine is None or mine[1]["input_sha256"] != ref["input_sha256"]:
                continue
            a, b = mine[1]["teacher_probs"], ref["teacher_probs"]
            same = int(argmax(a) == argmax(b))
            agree += same
            n += 1
            diff += math.fsum(abs(a[k] - b[k]) for k in a) / len(a)
            prefix = ref["id"].split("_")[0]
            by_source[prefix][0] += same
            by_source[prefix][1] += 1
    return {
        "overlap_rows": n,
        "argmax_agreement": agree / n if n else None,
        "mean_abs_prob_diff": diff / n if n else None,
        "pass": bool(n) and agree / n >= args.threshold,
        "threshold": args.threshold,
        "labels_sha256": file_sha256(args.labels),
        "reference_sha256": file_sha256(args.reference),
    }


def cmd_subset(args: argparse.Namespace) -> dict[str, Any]:
    wanted = {}
    for component, _, row in mixture(args.train):
        if component != args.component:
            continue
        if args.language == "en" and row["language"] != "en":
            continue
        if args.language == "non-en" and row["language"] == "en":
            continue
        wanted[row["id"]] = row["input_sha256"]
    lines = []
    with args.targets.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["id"] in wanted:
                if record["input_sha256"] != wanted[record["id"]]:
                    raise ValueError(
                        f"{record['id']}: target input hash differs from TRAIN"
                    )
                lines.append(line)
    write_lines(args.output, lines)
    return {
        "wanted": len(wanted),
        "rows": len(lines),
        "missing": len(wanted) - len(lines),
        "targets_sha256": file_sha256(args.targets),
        "output_sha256": file_sha256(args.output),
    }


def cmd_diag(args: argparse.Namespace) -> dict[str, Any]:
    teachers: dict[str, dict[str, Any]] = {}
    for spec in args.teacher:
        name, paths = spec.split("=", 1)
        merged: dict[str, Any] = {}
        for path in paths.split(","):
            for i, (_, record) in read_targets(Path(path)).items():
                merged.setdefault(i, record)
        teachers[name] = merged
    out: dict[str, Any] = {}
    stats: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for component, _, row in mixture(args.train):
        if not is_block(component, row) or row["task_type"] != "noul":
            continue
        gold = row["options"][row["label"]]["key"]
        for scope in (row["language"], "_all", f"_{component}"):
            s = stats[scope]
            s["gold"]["n"] += 1
            s["gold"]["yes"] += gold == "true"
            for name, targets in teachers.items():
                record = targets.get(row["id"])
                if record is None:
                    s[name]["uncovered"] += 1
                    continue
                if record["input_sha256"] != row["input_sha256"]:
                    raise ValueError(f"{row['id']}: {name} input hash differs")
                pred = argmax(record["teacher_probs"])
                s[name]["n"] += 1
                s[name]["pred_yes"] += pred == "true"
                s[name]["agree"] += pred == gold
                if gold == "false":
                    s[name]["gold_no"] += 1
                    s[name]["gold_no_agree"] += pred == "false"
    for scope, s in sorted(stats.items()):
        entry: dict[str, Any] = {
            "rows": s["gold"]["n"],
            "gold_yes_rate": s["gold"]["yes"] / s["gold"]["n"],
        }
        for name in teachers:
            t = s[name]
            entry[name] = {
                "covered": t["n"],
                "uncovered": t["uncovered"],
                "pred_yes_rate": t["pred_yes"] / t["n"] if t["n"] else None,
                "gold_agreement": t["agree"] / t["n"] if t["n"] else None,
                "gold_no_recall": (
                    t["gold_no_agree"] / t["gold_no"] if t["gold_no"] else None
                ),
            }
        out[scope] = entry
    return {"train_sha256": file_sha256(args.train), "by_scope": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--train", type=Path, action="append", required=True)
    p.add_argument("--exclude-labeled", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("check-sample")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="the teacher package (its tokenizer)",
    )
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--minimum", type=int, default=300)
    p.add_argument("--seed", default="dec-m5-relabel-v1")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("bitwise")
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--check", type=Path, required=True)
    p.add_argument("--minimum", type=int, default=300)
    p = sub.add_parser("overlap")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.97)
    p = sub.add_parser("subset")
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--component", required=True)
    p.add_argument("--language", choices=("all", "en", "non-en"), default="all")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("diag")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument(
        "--teacher", action="append", required=True, metavar="NAME=FILE[,FILE...]"
    )
    for name in ("bitwise", "overlap", "diag"):
        sub.choices[name].add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    target = getattr(args, "output", None) or args.report
    if target.exists():
        raise FileExistsError(target)
    result = {
        "rows": cmd_rows,
        "check-sample": cmd_check_sample,
        "bitwise": cmd_bitwise,
        "overlap": cmd_overlap,
        "subset": cmd_subset,
        "diag": cmd_diag,
    }[args.command](args)
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if hasattr(args, "report"):
        args.report.write_text(text)
    else:
        args.output.with_name(args.output.name + ".manifest.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
