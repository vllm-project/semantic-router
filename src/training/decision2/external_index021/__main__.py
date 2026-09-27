"""Run or score the independent Decision Index 0.2.1 port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import replay_published


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Pinned independent Decision Index 0.2.1 port"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "published-parity", help="replay 68 published rounded aggregate records"
    )
    for name in ("verify-suite", "run", "score", "sensitivity"):
        p = sub.add_parser(name)
        p.add_argument(
            "--suite-dir",
            type=Path,
            required=True,
            help="verified public-kit 0.2 suite directory",
        )
        p.add_argument(
            "--home-policy", choices=("first", "last", "explicit"), default="first"
        )
        p.add_argument(
            "--home-keep-ids",
            type=Path,
            help="JSON list of upstream-selected Home run IDs",
        )
        if name != "verify-suite":
            p.add_argument("--out", type=Path, required=True)
        if name == "run":
            p.add_argument(
                "--engine", required=True, help="public-kit Engine module:Class"
            )
            p.add_argument(
                "--option", action="append", default=[], help="Engine key=JSON-value"
            )
            p.add_argument("--limit", type=int)
            p.add_argument("--fresh", action="store_true")
        elif name in ("score", "sensitivity"):
            p.add_argument("--results", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "published-parity":
        result = replay_published()
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("edition", "rows", "max_display_delta", "evidence")
                },
                indent=2,
            )
        )
        if result["rows"] != 68 or result["max_display_delta"] > 0.01:
            raise SystemExit("published aggregate parity failed")
        return
    from .score import home_sensitivity, run, score_run, selected_rows, verified_suite

    keep = (
        set(json.loads(args.home_keep_ids.read_text())) if args.home_keep_ids else None
    )
    if args.command == "verify-suite":
        _rows, selection = selected_rows(
            verified_suite(args.suite_dir),
            home_policy=args.home_policy,
            home_keep_run_ids=keep,
        )
        print(
            json.dumps(
                {
                    "edition": "0.2.1",
                    "scoreable": selection.scoreable,
                    "source_scoreable": selection.scheduled,
                    "dropped": selection.dropped,
                    "home_policy": selection.home_policy,
                    "status": selection.status,
                    "keep_ids_sha256": selection.keep_ids_sha256,
                },
                indent=2,
            )
        )
        return
    if args.command == "run":
        options = {}
        for option in args.option:
            key, separator, value = option.partition("=")
            if not separator or not key:
                raise SystemExit(f"invalid engine option: {option!r}")
            try:
                options[key] = json.loads(value)
            except json.JSONDecodeError:
                options[key] = value
        result = run(
            args.suite_dir,
            engine=args.engine,
            engine_options=options,
            out_dir=args.out,
            home_policy=args.home_policy,
            home_keep_run_ids=keep,
            limit=args.limit,
            resume=not args.fresh,
        )
    elif args.command == "score":
        result = score_run(
            args.suite_dir,
            args.results,
            home_policy=args.home_policy,
            home_keep_run_ids=keep,
            out=args.out,
        )
    else:
        if args.home_policy != "first" or keep:
            raise SystemExit(
                "sensitivity scores both first and last; omit Home policy/keep IDs"
            )
        result = home_sensitivity(args.suite_dir, args.results, out=args.out)
    print(
        json.dumps(
            {k: result[k] for k in ("edition", "completed", "complete") if k in result},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
