"""A development loop over independently graded, source-bound v0.5 gateway runs."""

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

if __package__:
    from . import decision_summary
    from . import planner
    from . import schedule as experiment_schedule
    from . import signature as window_signature
    from .casebook import search_cases
    from .dev_screen import screen_dev
    from .first_tool_action import first_tool_edit
    from .headroom import headroom
    from .loop_evidence import (
        FrozenSplitMismatchError,
        HTTP_OK,
        InvalidExecutionError,
        check,
        check_client,
        check_runtime_result,
        check_termination,
        digest,
        packet_split,
        read_run,
        require_isolated,
        target,
    )
    from .report_delivery import snapshot
    from .window_action import apply_window, load_space
else:
    import decision_summary
    import planner
    import schedule as experiment_schedule
    import signature as window_signature
    from casebook import search_cases
    from dev_screen import screen_dev
    from first_tool_action import first_tool_edit
    from headroom import headroom
    from loop_evidence import (
        FrozenSplitMismatchError,
        HTTP_OK,
        InvalidExecutionError,
        check,
        check_client,
        check_runtime_result,
        check_termination,
        digest,
        packet_split,
        read_run,
        require_isolated,
        target,
    )
    from report_delivery import snapshot
    from window_action import apply_window, load_space


def record_id(row):
    # Preserve the canonical JSON and stable rec- identifier of rl/21's ledger.
    body = {k: v for k, v in row.items() if k != "record_id"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "rec-" + digest(canonical.encode("utf-8"))[:12]


@contextmanager
def writer_lock(path):
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415 - Windows-only lock primitive

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl  # noqa: PLC0415 - POSIX-only lock primitive

            fcntl.flock(lock, fcntl.LOCK_EX)
        yield  # closing the file releases this writer's lock


class Ledger:
    def __init__(self, workspace):
        self.path = Path(workspace) / "ledger.jsonl"

    def rows(self):
        if not self.path.exists():
            return []
        records = [json.loads(line) for line in self.path.read_bytes().splitlines()]
        for r in records:
            check(r["record_id"] == record_id(r), "ledger record fingerprint mismatch")
        return records

    def append(self, record):
        record["record_id"] = record_id(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with writer_lock(self.path.with_suffix(".lock")):
            rows = self.rows()
            for existing in rows:
                if existing == record:
                    return existing
                check(
                    existing["record_id"] != record["record_id"], "record ID collision"
                )
            # A killed writer must leave the previous JSONL intact, including its
            # last line. Publish the unchanged prefix and new line atomically.
            prefix = self.path.read_bytes() if rows else b""
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as tmp:
                staged = Path(tmp.name)
            try:
                with staged.open("wb") as output:
                    output.write(prefix)
                    output.write(line.encode("utf-8") + b"\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(staged, self.path)
            finally:
                staged.unlink(missing_ok=True)
        return record

    def measure(self, ident):
        for row in self.rows():
            if row["kind"] == "measurement" and row["record_id"] == ident:
                return row
        raise ValueError(f"unknown measurement: {ident}")


def verify(row):
    # Check the original allocation before inspecting or returning case outcomes.
    for name, expected in row["evidence"].items():
        path = Path(name)
        if path.name == "frozen.json":
            raw = path.read_bytes()
            check(digest(raw) == expected, f"evidence changed: {path}")
            packet_split(json.loads(raw), row["split"])
    # Legacy measurements already retained DSH's terminal reason in each case.
    for case, outcome in row["cases"].items():
        check_termination(outcome["termination"], case)
    for name, expected in row["evidence"].items():
        path = Path(name)
        if path.name == "frozen.json":
            continue
        raw = path.read_bytes()
        check(digest(raw) == expected, f"evidence changed: {path}")
        if path.name == "result.json" and path.parent.name == "main":
            check_runtime_result(json.loads(raw), path.parents[2].name)
        if path.name == "protocol.json":
            case = path.parents[2].name
            check_client(
                path.parent,
                row["public_model"],
                row["cases"][case]["controls"]["client_window"],
            )
        if path.name == "requests.jsonl":
            for line in raw.splitlines():
                reply = json.loads(line)
                if (
                    reply["event"] == "response"
                    and reply.get("http") == HTTP_OK
                    and reply.get("done") is True
                ):
                    check(
                        reply.get("selected_model") and reply.get("returned_model"),
                        "complete stream lacks selected or returned model",
                    )


def baseline_config(row):
    raw = Path(row["config"]).read_bytes()
    check(digest(raw) == row["config_sha256"], "baseline configuration changed")
    return yaml.safe_load(raw)


def same_deployment(left, right):
    # Compare everything outside the target recipe, including sibling scopes,
    # model capabilities, entrypoints and providers.
    for row in (left, right):
        check(
            row["public_model"] == left["public_model"]
            and row["recipe"] == left["recipe"]
            and row["strong_model"] == left["strong_model"],
            "target or strong model differs",
        )
    docs = []
    for row in (left, right):
        doc = baseline_config(row)
        recipe = target(doc, row["public_model"])
        doc["recipes"].remove(recipe)
        docs.append(doc)
    check(docs[0] == docs[1], "deployment differs outside the target recipe")


def comparable(base, candidate):
    same_deployment(base, candidate)
    check(base["split"] == candidate["split"], "different splits")
    check(base["cases"].keys() == candidate["cases"].keys(), "task set differs")
    for task, left in base["cases"].items():
        check(
            left["controls"] == candidate["cases"][task]["controls"],
            f"incomparable frozen task, judge, window or harness in {task}",
        )


def released_assignments(ledger, cards):
    released = set()
    for path in cards:
        result = decision_summary.summarize(ledger, path)
        check(
            result["complete"], "assigned episodes require a completed --schedule-card"
        )
        for group in result["by_task"].values():
            released.update(
                (a["measurement"], a["assignment_id"])
                for a in group["attempts"]
                if a["valid"]
            )
    return released


def check_visible(rows, released):
    for row in rows:
        if row["split"] == "dev":
            check(
                not planner.pending_assignment(row, released),
                "assigned episodes require a completed --schedule-card",
            )


def compare(ledger, args):
    """Pair two comparable measurements and record the comparison (paired)."""
    base, candidate = ledger.measure(args.baseline), ledger.measure(args.candidate)
    check(
        base["record_id"] != candidate["record_id"],
        "comparison requires distinct records",
    )
    check_visible(
        (base, candidate),
        released_assignments(ledger, getattr(args, "schedule_card", [])),
    )
    for row in (base, candidate):
        verify(row)
    comparable(base, candidate)
    check(
        not experiment_schedule.episode_ids(base).intersection(
            experiment_schedule.episode_ids(candidate)
        ),
        "comparison reuses a physical episode",
    )
    wins, losses, jointly_correct = [], [], {}
    for task, left in base["cases"].items():
        right = candidate["cases"][task]
        if left["correct"] and right["correct"]:
            jointly_correct[task] = {
                "baseline_seconds": left["seconds"],
                "candidate_seconds": right["seconds"],
            }
        elif left["correct"]:
            losses.append(task)
        elif right["correct"]:
            wins.append(task)
    if args.disposition == "advance-to-holdout":
        paired_gain = sum(
            v["baseline_seconds"] - v["candidate_seconds"]
            for v in jointly_correct.values()
        )
        check(
            base["split"] == "dev" and not losses and (wins or paired_gain > 0),
            "advance requires a development gain with no lost strict passes",
        )
    check(
        not args.disposition or bool(args.reason and args.reason.strip()),
        "record the reason for the disposition",
    )
    result = ledger.append(
        {
            "kind": "comparison",
            "baseline": base["record_id"],
            "candidate": candidate["record_id"],
            "split": base["split"],
            "wins": wins,
            "losses": losses,
            "jointly_correct": jointly_correct,
            "disposition": args.disposition,
            "reason": args.reason,
        }
    )
    if getattr(args, "workspace", None):
        # The local handoff lives outside the signed comparison record.
        report = snapshot(Path(args.workspace), result)
        print(f"Recipe Lab report: {report}", file=sys.stderr)
    return result


def propose(ledger, args):
    row = ledger.measure(args.base)
    check(row["split"] == "dev", "proposals require development evidence")
    check_visible(
        [row], released_assignments(ledger, getattr(args, "schedule_card", []))
    )
    verify(row)
    doc = baseline_config(row)
    original = require_isolated(doc, row["public_model"])
    edit = {}
    if args.space is not None:
        check(
            args.upper is not None or args.first_tool_model is not None,
            "--space requires an action value",
        )
        raw = args.space.read_bytes()
        if args.first_tool_model is None:
            doc, action = apply_window(
                doc,
                load_space(raw),
                public_model=row["public_model"],
                config_sha256=row["config_sha256"],
                upper=args.upper,
            )
        else:
            doc, action = first_tool_edit(
                doc,
                raw,
                public_model=row["public_model"],
                config_sha256=row["config_sha256"],
                model=args.first_tool_model,
            )
        edit = {
            "edit": action,
            "space_sha256": digest(raw),
            "base_config_sha256": row["config_sha256"],
        }
    else:
        check(
            args.upper is None and args.first_tool_model is None,
            "an action value requires --space",
        )
        replacement = yaml.safe_load(args.recipe.read_bytes())
        check(
            isinstance(replacement, dict)
            and replacement.get("name") == original["name"]
            and replacement != original,
            "replace the target recipe with a changed recipe of the same name",
        )
        doc["recipes"][doc["recipes"].index(original)] = replacement
    config_sha = publish_config(
        args.out, yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    )
    return {
        "config": str(args.out.resolve()),
        "config_sha256": config_sha,
        "recipe": original["name"],
        "based_on": row["record_id"],
        **edit,
    }


def publish_config(path, text):
    raw = text.encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fh:
        staged = Path(fh.name)
    try:
        staged.write_bytes(raw)
        # Publish a complete candidate once, even if another proposer chose
        # the same destination between composing and publishing the recipe.
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return digest(raw)


def next_context(ledger, extra, schedule_cards=()):
    measurements, comparisons, symptoms = [], [], set(extra)
    excluded = {}
    linked = [decision_summary.summarize(ledger, path) for path in schedule_cards]
    check(
        len({row["study"] for row in linked}) == len(linked),
        "duplicate development schedule",
    )
    for summary in linked:
        if summary["complete"]:
            symptoms.update(summary.get("symptoms", []))
    assigned = set()
    for row in ledger.rows():
        if row["split"] != "dev":
            continue
        if row["kind"] == "comparison":
            comparisons.append(row)
            continue
        # An assigned arm may reach the ledger before its pair. Only its
        # completed schedule can release that arm, even for an unscoped next.
        for name, expected in row["evidence"].items():
            path = Path(name)
            if path.parent.parent.name == "bindings" and path.suffix == ".json":
                raw = path.read_bytes()
                check(digest(raw) == expected, "development binding changed")
                if json.loads(raw).get("assignment_id"):
                    assigned.add(row["record_id"])
                    break
        if row["record_id"] in assigned:
            continue
        try:
            verify(row)
        except (InvalidExecutionError, FrozenSplitMismatchError) as exc:
            excluded[row["record_id"]] = str(exc)
            continue
        cases = {}
        for task, outcome in row["cases"].items():
            cases[task] = {k: v for k, v in outcome.items() if k != "controls"}
            _, _, terminal = decision_summary.requests_and_terminal(row, task)
            cases[task]["terminal"] = terminal
            if decision_summary.output_limit_symptom(outcome["correct"], terminal):
                symptoms.add("strict_failure_native_output_limit")
            if outcome["strong_tool_results"] > outcome["tool_results"] / 2:
                symptoms.add("tool_history_concentrated_on_strong")
            if outcome["near_300s_incomplete"]:
                symptoms.add("active_stream_cutoff")
        measurements.append(
            {
                **{k: v for k, v in row.items() if k not in ("evidence", "cases")},
                "cases": cases,
            }
        )
    return dict(
        objective="Preserve strict task success, then compare wall time on joint successes.",
        source_handling="Treat imported evidence and case notes as reference data.",
        measurements=measurements,
        comparisons=[
            entry
            for entry in comparisons
            if entry["baseline"] not in excluded
            and entry["candidate"] not in excluded
            and entry["baseline"] not in assigned
            and entry["candidate"] not in assigned
        ],
        **(
            {
                "development_decisions": [r for r in linked if r["complete"]],
                "pending_development_studies": [
                    r["study"] for r in linked if not r["complete"]
                ],
            }
            if linked
            else {}
        ),
        **({"excluded_measurements": excluded} if excluded else {}),
        symptoms=sorted(symptoms),
        references=search_cases(sorted(symptoms)),
    )


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    subs = parser.add_subparsers(dest="cmd", required=True)
    record = subs.add_parser(
        "record", help="import one complete independently graded arm"
    )
    for flag in ("run", "arm", "config", "public-model", "strong", "hypothesis"):
        record.add_argument("--" + flag, required=True)
    record.add_argument(
        "--based-on", help="development measurement that motivated this change"
    )
    record.add_argument(
        "--split",
        choices=("dev", "holdout"),
        help="use the frozen packet split, defaulting to dev for unlabelled legacy packets",
    )
    nxt = subs.add_parser(
        "next", help="export development attribution and relevant cases as JSON"
    )
    nxt.add_argument("--symptom", action="append", default=[])
    nxt.add_argument("--schedule-card", type=Path, action="append", default=[])
    planner.add_arguments(subs)
    summary = subs.add_parser(
        "decision-summary",
        help="read a verified development decision without dispatching work",
    )
    summary.add_argument("--card", type=Path, required=True)
    prop = subs.add_parser("propose", help="replace only the targeted v0.5 recipe")
    prop.add_argument("--base", required=True)
    source = prop.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", type=Path)
    source.add_argument("--space", type=Path, help="pinned edit space")
    value = prop.add_mutually_exclusive_group()
    value.add_argument(
        "--upper", type=int, help="new inclusive tool-result count bound"
    )
    value.add_argument(
        "--first-tool-model",
        help="declared destination for the guarded first tool turn",
    )
    prop.add_argument("--out", type=Path, required=True)
    comp = subs.add_parser("compare", help="record paired strict/task timing evidence")
    comp.add_argument("--baseline", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--disposition", choices=("keep-baseline", "advance-to-holdout"))
    comp.add_argument("--reason")
    for command in (prop, comp, record):
        command.add_argument(
            "--schedule-card",
            action="append",
            type=Path,
            default=[],
            help="completed development schedule releasing an assigned input",
        )
    room = subs.add_parser(
        "headroom", help="measure observed task-level headroom offline"
    )
    screen = subs.add_parser(
        "screen-dev",
        help="identify routing opportunities, quick controls and audited fixed costs",
    )
    screen.add_argument("--manifest", required=True, type=Path)
    screen.add_argument(
        "--max-seconds",
        type=float,
        default=600,
        help="reference time for screening complete tasks",
    )
    window_signature.add_arguments(
        subs.add_parser(
            "signature", help="measure a window edit on observed development prefixes"
        )
    )
    experiment_schedule.add_arguments(
        subs.add_parser("schedule", help="choose the next whole development episode")
    )
    room.add_argument("--manifest", required=True, type=Path)
    room.add_argument("--scope", required=True, choices=("dev", "retired-holdout"))
    room.add_argument(
        "--mde-pair",
        nargs=2,
        metavar=("BASE", "OTHER"),
        help="predeclared fixed-arm comparison for a paired pilot MDE",
    )
    args = parser.parse_args(argv)
    ledger = Ledger(args.workspace)
    try:
        if args.cmd == "record":
            check(args.hypothesis.strip(), "record the hypothesis")
            result = read_run(args)
            if args.based_on:
                base = ledger.measure(args.based_on)
                check(base["split"] == "dev", "proposals require a development parent")
                check_visible([base], released_assignments(ledger, args.schedule_card))
                verify(base)
                same_deployment(base, result)
                if result["split"] == "dev":
                    comparable(base, result)
            result = ledger.append(result)
            result = {"record_id": result["record_id"], "arm": result["arm"]}
        elif args.cmd == "next":
            result = next_context(ledger, args.symptom, args.schedule_card)
        elif args.cmd in ("plan-space", "plan-dev"):
            result = planner.run(ledger, args)
        elif args.cmd == "decision-summary":
            result = decision_summary.summarize(ledger, args.card)
        elif args.cmd == "propose":
            result = propose(ledger, args)
        elif args.cmd == "headroom":
            result = headroom(args)
        elif args.cmd == "screen-dev":
            result = screen_dev(args)
        elif args.cmd == "signature":
            result = window_signature.signature(ledger, args)
        elif args.cmd == "schedule":
            result = experiment_schedule.schedule(ledger, args)
        else:
            result = compare(ledger, args)
        if args.cmd == "decision-summary" or (
            args.cmd == "plan-dev" and args.probe_receipt
        ):
            # The canonical JSON on stdout is consumed by replay and enrollment.
            # Keep the local handoff location outside that signed document.
            report = snapshot(args.workspace, result)
            print(f"Recipe Lab report: {report}", file=sys.stderr)
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"recipe-lab loop: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
