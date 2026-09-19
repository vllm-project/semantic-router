#!/usr/bin/env python3
"""Blind LLM-judge audit of the modality-routing labels (AR / DIFFUSION / BOTH).

The judge never sees the current label. Judgments go to an append-only
checkpoint.jsonl (fsync'd per write), so a crash or restart loses nothing and every
command resumes from what is already judged. Rubric: RUBRIC.md (its hash is stored on
every record, so mixed-rubric runs are detectable).

Two ways to produce judgments, same rubric, same checkpoint format:

  In a chat session (cheapest on context: big batches, one call per round):
      python judge_labels.py next --n 150            # blinded rows -> read them
      python judge_labels.py save <<'EOF'            # compact lines: "ID A|D|B [vh] [M|L] [tags]"
      12 A
      13 D
      14 B M
      EOF
      python judge_labels.py status

  Non-interactively via the Claude API (needs `pip install anthropic`, ANTHROPIC_API_KEY):
      python judge_labels.py api --model claude-sonnet-5 --batch-size 50
      python judge_labels.py api --dry-run           # show the request, call nothing

Then compare against the original labels (and optionally re-score model predictions):
      python judge_labels.py report --eval-report ../modality_candidate_eval_report.json
Self-consistency check: `next --rejudge --n 75` / `save --rejudge` judge a seeded random
sample again into checkpoint_rejudge.jsonl, and `report` prints intra-judge agreement.

Independent human check (no API needed):
      python judge_labels.py sheet                   # blinded human_review.tsv: all disagreements + sampled agreements
      python judge_labels.py report --human human_review.tsv   # after filling the last column with A/D/B
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from audit_lib.api_judge import (
    ApiJudgeSettings,
    judge_via_api,
    make_client,
    retryable_errors,
)
from audit_lib.checkpoint import (
    count_judged,
    read_checkpoint,
    records_for_split,
    select_ids,
)
from audit_lib.constants import (
    CHECKPOINT,
    DATA_DIR,
    HERE,
    LEGEND,
    MANIFEST,
    REJUDGE,
    RUBRIC,
)
from audit_lib.dataset import (
    DatasetMismatchError,
    clip,
    load_rows,
    rubric_hash,
    verify_pinned,
)
from audit_lib.human_review import (
    SheetSplitError,
    estimate_error_rates,
    load_human,
    pick_sheet_rows,
    write_sheet,
)
from audit_lib.judgment import build_user_message, parse_lines, save_judgments
from audit_lib.report import DEFAULT_COMPARE, build_report_lines, load_predictions


def load_split(args: argparse.Namespace) -> list[dict]:
    """Load the split named in args, pinning the default dataset by hash.

    A custom --data-dir (for example in tests) is not pinned.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The split's rows.

    Raises:
        DatasetMismatchError: If the default dataset differs from the pinned one.
    """
    path = Path(args.data_dir) / f"{args.split}.jsonl"
    if path.resolve().parent == DATA_DIR.resolve():
        verify_pinned(path, args.split, MANIFEST)
    return load_rows(args.data_dir, args.split)


def target_path(args: argparse.Namespace) -> Path:
    """Return the checkpoint file a command writes to.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The re-judge checkpoint if --rejudge is set, otherwise the primary one.
    """
    return Path(args.rejudge_checkpoint if args.rejudge else args.checkpoint)


def cmd_next(args: argparse.Namespace) -> int:
    """Print the next blinded batch for the judge.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    rows = load_split(args)
    primary = read_checkpoint(args.checkpoint)
    rejudged = read_checkpoint(args.rejudge_checkpoint)
    done = rejudged if args.rejudge else primary
    ids = select_ids(
        args.split,
        rows,
        primary,
        rejudged,
        args.n,
        rejudge=args.rejudge,
        seed=args.seed,
    )
    remaining = len(rows) - count_judged(done, args.split)
    if not ids:
        print("# DONE: nothing left to judge for this selection")
        return 0
    print(f"# split={args.split} remaining={remaining} showing={len(ids)} | {LEGEND}")
    for i in ids:
        print(f"{i} {clip(rows[i]['text'])[0]}")
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    """Read judgment lines from stdin and save them.

    Args:
        args: Parsed CLI arguments.

    Returns:
        1 if any line was rejected, otherwise 0.
    """
    rows = load_split(args)
    path = target_path(args)
    parsed, errors, ignored = parse_lines(
        sys.stdin.read(), allowed=set(range(len(rows)))
    )
    records = save_judgments(
        parsed,
        rows,
        split=args.split,
        judge=args.judge,
        rubric=rubric_hash(),
        path=path,
        existing=read_checkpoint(path),
        overwrite=args.overwrite,
    )
    total = count_judged(read_checkpoint(path), args.split)
    print(
        f"saved {len(records)} -> {path.name} | already judged (skipped) {len(parsed) - len(records)} | "
        f"bad lines {len(errors)} | ignored non-row lines {ignored} | "
        f"{args.split}: {total}/{len(rows)} judged"
    )
    for e in errors[:10]:
        print("  ERR", e)
    return 1 if errors else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print progress, label counts and tag counts per checkpoint and split.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    for path, name in (
        (args.checkpoint, "primary"),
        (args.rejudge_checkpoint, "rejudge"),
    ):
        checkpoint = read_checkpoint(path)
        if not checkpoint:
            continue
        print(f"== {name} ({Path(path).name}) ==")
        for split in sorted({s for s, _ in checkpoint}):
            recs = list(records_for_split(checkpoint, split).values())
            try:
                total = len(load_rows(args.data_dir, split))
            except FileNotFoundError:
                total = "?"
            print(
                f"  {split}: {len(recs)}/{total} judged | labels {dict(Counter(r['label'] for r in recs))} | "
                f"conf {dict(Counter(r['conf'] for r in recs))} | vh {sum(r['vh'] for r in recs)}"
            )
            print(
                f"    tags {dict(Counter(t for r in recs for t in r['tags']))} | "
                f"judges {dict(Counter(r['judge'] for r in recs))} | "
                f"rubric {dict(Counter(r['rubric'] for r in recs))}"
            )
    print(f"current rubric hash: {rubric_hash()}")
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    """Judge rows through the Claude API until the split is done or --limit is reached.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    rows = load_split(args)
    system_text = RUBRIC.read_text(encoding="utf-8")
    settings = ApiJudgeSettings(
        model=args.model,
        effort=args.effort,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        retries=args.retries,
        limit=args.limit,
        rejudge=args.rejudge,
        seed=args.seed,
    )
    if args.dry_run:
        ids = select_ids(
            args.split,
            rows,
            read_checkpoint(args.checkpoint),
            read_checkpoint(args.rejudge_checkpoint),
            args.batch_size,
            rejudge=args.rejudge,
            seed=args.seed,
        )
        print(
            f"model={args.model} effort={args.effort} rubric={rubric_hash()} "
            f"system_chars={len(system_text)} batch={len(ids)}"
        )
        print("---- user message (first batch) ----")
        print(build_user_message(rows, ids))
        return 0
    judge_via_api(
        make_client(),
        rows,
        split=args.split,
        system_text=system_text,
        rubric=rubric_hash(),
        settings=settings,
        checkpoint=Path(args.checkpoint),
        rejudge_checkpoint=Path(args.rejudge_checkpoint),
        target=target_path(args),
        retryable=retryable_errors(),
    )
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    """Write a blinded sheet for a human spot-check.

    It holds every row where the judge disagrees with the original label, plus a
    random sample of agreeing rows, shuffled.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    rows = load_split(args)
    records = records_for_split(read_checkpoint(args.checkpoint), args.split)
    picked, n_disagreements = pick_sheet_rows(rows, records, args.n_agree, args.seed)
    write_sheet(Path(args.out), rows, picked, args.split)
    print(
        f"wrote {len(picked)} rows to {args.out} ({n_disagreements} disagreements + "
        f"{len(picked) - n_disagreements} sampled agreements, shuffled)"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print agreement with the original labels, model re-scoring and paired tests.

    Args:
        args: Parsed CLI arguments.

    Returns:
        1 if nothing has been judged yet, otherwise 0.
    """
    rows = load_split(args)
    records = records_for_split(read_checkpoint(args.checkpoint), args.split)
    if not records:
        print("no judgments yet")
        return 1
    pairs = (
        tuple(tuple(spec.split(",", 1)) for spec in args.compare)
        if args.compare
        else DEFAULT_COMPARE
    )
    original = [r["label_name"] for r in rows]
    human = (
        estimate_error_rates(load_human(args.human, args.split), original, records)
        if args.human
        else None
    )
    lines = build_report_lines(
        args.split,
        rows,
        records,
        preds=load_predictions(args.eval_report, args.preds, len(rows)),
        pairs=pairs,
        human=human,
        rejudged=records_for_split(
            read_checkpoint(args.rejudge_checkpoint), args.split
        ),
        show=args.show,
    )
    print("\n".join(lines))
    return 0


def add_common_options() -> argparse.ArgumentParser:
    """Build the options every subcommand shares.

    Returns:
        A parent parser to pass to each subcommand.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--split", default="test", choices=["train", "validation", "test"]
    )
    common.add_argument(
        "--rejudge",
        action="store_true",
        help="operate on the self-consistency re-judge pass",
    )
    common.add_argument(
        "--seed", type=int, default=0, help="seed for the re-judge sample"
    )
    common.add_argument("--data-dir", default=str(DATA_DIR))
    common.add_argument("--checkpoint", default=str(CHECKPOINT))
    common.add_argument("--rejudge-checkpoint", default=str(REJUDGE))
    return common


def add_next_command(sub, common) -> None:
    """Register `next`.

    Args:
        sub: The subparsers action.
        common: Parent parser with the shared options.
    """
    s = sub.add_parser("next", parents=[common], help="print the next blinded batch")
    s.add_argument("--n", type=int, default=150)
    s.set_defaults(fn=cmd_next)


def add_save_command(sub, common) -> None:
    """Register `save` and `status`.

    Args:
        sub: The subparsers action.
        common: Parent parser with the shared options.
    """
    s = sub.add_parser(
        "save", parents=[common], help="read compact judgment lines from stdin"
    )
    s.add_argument(
        "--judge", default=os.environ.get("JUDGE_NAME", "claude-code-session")
    )
    s.add_argument(
        "--overwrite",
        action="store_true",
        help="re-judge rows already in the checkpoint",
    )
    s.set_defaults(fn=cmd_save)

    s = sub.add_parser("status", parents=[common], help="progress and label/tag counts")
    s.set_defaults(fn=cmd_status)


def add_api_command(sub, common) -> None:
    """Register `api`.

    Args:
        sub: The subparsers action.
        common: Parent parser with the shared options.
    """
    s = sub.add_parser(
        "api", parents=[common], help="judge via the Claude API using RUBRIC.md"
    )
    s.add_argument("--model", default="claude-sonnet-5")
    s.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"]
    )
    s.add_argument("--batch-size", type=int, default=50)
    s.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many rows (default: all)",
    )
    s.add_argument(
        "--retries", type=int, default=2, help="re-ask for ids missing from a reply"
    )
    s.add_argument("--max-tokens", type=int, default=16000)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_api)


def add_review_commands(sub, common) -> None:
    """Register `report` and `sheet`.

    Args:
        sub: The subparsers action.
        common: Parent parser with the shared options.
    """
    s = sub.add_parser(
        "report",
        parents=[common],
        help="agreement with original labels, model re-scoring",
    )
    s.add_argument(
        "--show", type=int, default=40, help="how many disagreements to list"
    )
    s.add_argument(
        "--eval-report",
        help="modality_candidate_eval_report.json (per_example_records)",
    )
    s.add_argument(
        "--preds",
        action="append",
        metavar="NAME=PATH",
        help='JSON file with a "preds" list aligned to the split (repeatable)',
    )
    s.add_argument(
        "--compare",
        action="append",
        metavar="A,B",
        help="model pair for the McNemar test (repeatable; default clean_baseline,candidate)",
    )
    s.add_argument(
        "--human",
        help="filled-in sheet from `sheet` (id<TAB>text<TAB>label); adds a human-review section",
    )
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser(
        "sheet", parents=[common], help="write a blinded human-review sheet"
    )
    s.add_argument("--out", default=str(HERE / "human_review.tsv"))
    s.add_argument(
        "--n-agree",
        type=int,
        default=44,
        help="random agreement rows to add to all disagreements",
    )
    s.set_defaults(fn=cmd_sheet)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with one subcommand per step.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = add_common_options()
    for register in (
        add_next_command,
        add_save_command,
        add_api_command,
        add_review_commands,
    ):
        register(sub, common)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments to parse, or None for sys.argv.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (DatasetMismatchError, SheetSplitError) as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
