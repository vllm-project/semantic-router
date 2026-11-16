#!/usr/bin/env python3
"""Calibrate vSR's thresholds from the user's own decision log and labels.

Subcommands: report, sample, label, suggest, apply, probe. Standard library only.
The options before the subcommand may also follow it (`calibrate.py report --id <id>`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# plan puts setup/scripts, where runtime and settings live, on the path.
import plain
import plan
import runtime
import settings
from decisions import APPS, ROUTER_LABELS
from fitting import GOALS
from questions import question, sample


def _common(parser: argparse.ArgumentParser, default: object) -> None:
    parser.add_argument(
        "--id",
        default=default,
        help="plugin id the settings are keyed under (default: detected)",
    )
    parser.add_argument(
        "--plugin-root",
        default=default,
        help="the running copy's ${CLAUDE_PLUGIN_ROOT}, to detect the id",
    )
    parser.add_argument(
        "--log", default=default, help="decision log (default: logPath from settings)"
    )
    parser.add_argument(
        "--labels",
        default=default,
        help="labels file (default: labels.jsonl beside the log)",
    )
    parser.add_argument(
        "--token-file",
        default=default,
        help="a file holding the runtime's Bearer token (else $SR_RUNTIME_TOKEN)",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _common(parser, None)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ("report", "sample", "label", "suggest", "apply", "probe"):
        commands[name] = sub.add_parser(name)
        # Suppressed, so an option given before the subcommand is not reset after it.
        _common(commands[name], argparse.SUPPRESS)
    for name in ("report", "sample", "suggest", "apply"):
        commands[name].add_argument(
            "--apps",
            help=f"comma-separated apps, of {', '.join(APPS)} (default: every app; enabled ones for sample and suggest)",
        )
    for name in ("report", "suggest"):
        commands[name].add_argument(
            "--detail",
            action="store_true",
            help="also print the option names, the table and the settings diff",
        )
    commands["sample"].add_argument("--per-side", type=int, default=2)
    commands["label"].add_argument("pairs", nargs="+", metavar="ID=LABEL")
    one = commands["suggest"]
    one.add_argument(
        "--goal",
        choices=tuple(GOALS),
        help="how the apps other than the router weigh false alarms against misses (default balanced; the router has no goal)",
    )
    one.add_argument(
        "--out",
        help="where to save the plan for `apply` (default: calibration-plan.json beside the log)",
    )
    commands["apply"].add_argument(
        "plan",
        nargs="?",
        help="the plan `suggest` saved (default: calibration-plan.json beside the log)",
    )
    commands["probe"].add_argument(
        "--url", help="runtime URL (default: runtimeUrl from settings)"
    )
    return parser.parse_args(argv)


def label(ctx: plan.Context, pairs: list[str], path: Path) -> str:
    """Appends the answers to the labels file, in the shape the /vsr pane writes them."""
    at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    at = at.replace("+00:00", "Z")
    by_id = {d.id: d for d in ctx.log.decisions}
    lines = []
    for pair in pairs:
        decision_id, _, value = pair.rpartition("=")
        if decision_id not in by_id:
            raise settings.SettingsError(
                f"no decision {decision_id!r} in {ctx.log_path}"
            )
        app = by_id[decision_id].app
        allowed = ROUTER_LABELS if app == "router" else ("yes", "no")
        if value not in allowed:
            raise settings.SettingsError(
                f"{app} labels are {', '.join(allowed)}, not {value!r}"
            )
        line = {"id": decision_id, "app": app, "label": value, "at": at}
        lines.append(json.dumps(line) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.writelines(lines)
    return f"saved {len(lines)} label(s) to {path}"


# The only files calibrate writes besides settings.json, by name: `label --labels` and `suggest --out` may point
# elsewhere, but never at another file (settings.json, code), since both run without a permission prompt.
LABELS_NAME = "labels.jsonl"
PLAN_NAME = "calibration-plan.json"


def writable(given: str | None, default: Path, name: str) -> Path:
    path = Path(given) if given else default
    if path.name != name:
        raise settings.SettingsError(f"this file must be named {name}, not {path.name}")
    return path


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        apps = plan.parse_apps(getattr(args, "apps", None))
        ctx = plan.context(args)
        if args.command == "report":
            print(plan.report(ctx, apps, args.detail))
        elif args.command == "sample":
            picked = sample(
                ctx.log.decisions,
                ctx.labels,
                ctx.options,
                apps or ctx.enabled,
                args.per_side,
            )
            entries = [question(d, ctx.options) for d in picked]
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        elif args.command == "label":
            path = writable(args.labels, ctx.path(LABELS_NAME), LABELS_NAME)
            print(label(ctx, args.pairs, path))
        elif args.command == "suggest":
            suggested = plan.suggestion(ctx, args.goal, apps)
            print(plain.suggestion(suggested))
            if args.detail:
                print("\nDetails\n" + plan.table(suggested))
                if suggested["changes"]:
                    print("\nsettings diff:\n" + plan.settings_diff(suggested), end="")
            out = writable(args.out, ctx.path(PLAN_NAME), PLAN_NAME)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(suggested, indent=2) + "\n")
            print(f"\nplan saved to {out}")
        elif args.command == "apply":
            saved = json.loads(Path(args.plan or ctx.path(PLAN_NAME)).read_text())
            print(plan.apply(ctx, saved, plan.runtime_model(ctx), apps))
        else:
            rows = runtime.run_probes(
                args.url or ctx.options["runtimeUrl"],
                ctx.token,
                ctx.options,
                set(ctx.enabled),
                smoke=False,
            )
            runtime.print_probes(rows)
        return 0
    except settings.SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
