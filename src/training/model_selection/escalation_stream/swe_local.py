"""Judge inputs for a local SWE trajectory directory.

The directory is caller-supplied. This module does not download trajectories
and has no default dataset. SWE-bench and SWE-Smith dumps stay on the machine
that already has them.

Each JSONL record needs id, task, instruction, actions (command and output),
final_message, and attempts. diff is optional. Views:

  task            instruction only
  actions-1..4    instruction plus the first n actions
  finished        commands, the last four actions, and the final message;
                  a final diff is withheld, including when it is the last
                  tool output
  finished-diff   the finished state with the diff kept

Length caps follow the Terminal-Bench judge protocol used for this study:
instruction 6000, command 300, action 600, final message 3000, state 24000.
A command list that still overflows keeps its first and last quarter.

Usage:
  python swe_local.py requests --trajectories DIR --out requests.jsonl
  python swe_local.py rows --trajectories DIR --out rows.jsonl
  python swe_local.py checkpoint --answers FILE --checkpoint FILE
  python swe_local.py scores --checkpoint FILE --out-dir DIR
  python swe_local.py auroc --rows FILE --scores FILE --cheap M --strong M
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS_PATH = HERE / "questions" / "swe_end.json"

TASK_CHARS = 6000
COMMAND_CHARS = 300
ACTION_CHARS = 600
FINAL_CHARS = 3000
STATE_CHARS = 24000
LAST_ACTIONS = 4
MIN_COMMANDS = 8
SOLVED = 0.5

OMITTED = "... commands omitted ..."
WITHHELD_DIFF = "[withheld: final diff]"
NOT_RECORDED = "[not recorded]"
REF = re.compile(r"^\$\d+$")

VIEWS = (
    "task",
    "actions-1",
    "actions-2",
    "actions-3",
    "actions-4",
    "finished",
    "finished-diff",
)

TASK_AGENT = "A low-cost coding agent will attempt this task."
PREFIX_AGENT = "A low-cost coding agent is partway through this attempt."
END_AGENT = (
    "A low-cost coding agent has finished this attempt. Its commands, the "
    "outputs it saw, and its final message are below."
)

REQUIRED = ("id", "task", "instruction", "actions", "final_message", "attempts")


def _dumps(body):
    return json.dumps(body, ensure_ascii=False)


def clean(value, limit):
    """Unresolved scrape references become a placeholder, then the cap applies."""
    value = "" if value is None else str(value)
    if REF.match(value):
        return NOT_RECORDED
    return value[:limit]


def load_questions(path=QUESTIONS_PATH):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows = []
    file_path = Path(path)
    if not file_path.exists():
        return rows
    for line_no, line in enumerate(file_path.read_text().splitlines(), 1):
        if line.strip():
            rows.append((line_no, json.loads(line)))
    return rows


def write_jsonl(path, rows):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("".join(_dumps(row) + "\n" for row in rows))


def load_trajectories(directory):
    """JSONL files in one directory, non-recursive, in file-name order."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"{directory}: expected a directory of .jsonl trajectories")
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{directory}: no .jsonl trajectories")
    records = []
    seen = set()
    for path in files:
        for line_no, record in read_jsonl(path):
            missing = [field for field in REQUIRED if field not in record]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {', '.join(missing)}")
            if not isinstance(record["actions"], list):
                raise ValueError(f"{path}:{line_no}: actions must be a list")
            row_id = record["id"]
            if row_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {row_id!r}")
            seen.add(row_id)
            records.append(record)
    return records


def _action(action, output):
    return {
        "command": clean(action.get("command"), ACTION_CHARS),
        "output": output,
    }


def _output_is_diff(action, diff):
    if diff is None:
        return False
    return str(action.get("output") if action.get("output") is not None else "") == str(
        diff
    )


def _last_actions(actions, diff, withhold_diff):
    chosen = actions[-LAST_ACTIONS:]
    rendered = []
    for index, action in enumerate(chosen):
        output = clean(action.get("output"), ACTION_CHARS)
        is_last = index == len(chosen) - 1
        if withhold_diff and is_last and _output_is_diff(action, diff):
            output = WITHHELD_DIFF
        rendered.append(_action(action, output))
    return rendered


def _shrink_commands(body):
    commands = body.get("commands")
    while (
        commands is not None
        and len(_dumps(body)) > STATE_CHARS
        and len(commands) > MIN_COMMANDS
    ):
        count = len(commands)
        commands = [
            *commands[: count // 4],
            OMITTED,
            *commands[-(count // 4) :],
        ]
        body["commands"] = commands
    return body


def _fit_diff(body):
    if "diff" not in body:
        return body
    while body["diff"] and len(_dumps(body)) > STATE_CHARS:
        nxt = body["diff"][: max(0, len(body["diff"]) // 2)]
        if nxt == body["diff"]:
            body["diff"] = ""
            break
        body["diff"] = nxt
    return body


def pack(body):
    """JSON state at the length cap. Commands keep their first and last quarter."""
    body = _fit_diff(_shrink_commands(body))
    return _dumps(body)


def prefix_state(record, count):
    actions = record["actions"][:count]
    body = {
        "task": clean(record["instruction"], TASK_CHARS),
        "agent": PREFIX_AGENT if count else TASK_AGENT,
        "actions_so_far": [
            _action(action, clean(action.get("output"), ACTION_CHARS))
            for action in actions
        ],
        "finished": False,
    }
    return pack(body)


def finished_state(record, keep_diff):
    actions = record["actions"]
    diff = record.get("diff")
    body = {
        "task": clean(record["instruction"], TASK_CHARS),
        "agent": END_AGENT,
        "commands": [clean(action.get("command"), COMMAND_CHARS) for action in actions],
        "last_actions": _last_actions(actions, diff, withhold_diff=not keep_diff),
        "final_message": clean(record.get("final_message"), FINAL_CHARS),
        "finished": True,
    }
    if keep_diff and diff not in (None, ""):
        body["diff"] = clean(diff, STATE_CHARS)
    return pack(body)


def build_state(record, view):
    if view == "task":
        return prefix_state(record, 0)
    if view.startswith("actions-"):
        return prefix_state(record, int(view.split("-", 1)[1]))
    if view == "finished":
        return finished_state(record, keep_diff=False)
    if view == "finished-diff":
        return finished_state(record, keep_diff=True)
    raise ValueError(f"unknown view {view!r}")


def view_k(view):
    """Finished views use k=99, matching the end-of-attempt judge requests."""
    if view in ("finished", "finished-diff"):
        return 99
    return 0


def build_requests(records):
    requests = []
    for record in records:
        for view in VIEWS:
            requests.append(
                {
                    "id": record["id"],
                    "view": view,
                    "k": view_k(view),
                    "state": build_state(record, view),
                }
            )
    return requests


def policy_rows(records):
    """Rows for explore.py. The judge state is not copied into them."""
    return [
        {"id": record["id"], "task": record["task"], "attempts": record["attempts"]}
        for record in records
    ]


def load_checkpoint(path):
    """Map (id, view) to probs. Conflicting duplicates name the file and line."""
    found = {}
    for line_no, record in read_jsonl(path):
        key = (record["id"], record["view"])
        probs = record["probs"]
        if key in found and found[key] != probs:
            raise ValueError(
                f"{path}:{line_no}: duplicate id {record['id']!r} "
                f"view {record['view']!r} with different probabilities"
            )
        found[key] = probs
    return found


def pending_requests(requests, done):
    return [row for row in requests if (row["id"], row["view"]) not in done]


def append_checkpoint(path, answers):
    """Append answers whose key is new. Identical keys are skipped."""
    file_path = Path(path)
    done = load_checkpoint(file_path) if file_path.exists() else {}
    fresh = []
    for answer in answers:
        key = (answer["id"], answer["view"])
        probs = answer["probs"]
        if key in done:
            if done[key] != probs:
                raise ValueError(
                    f"{path}: duplicate id {answer['id']!r} view {answer['view']!r} "
                    "with different probabilities"
                )
            continue
        done[key] = probs
        fresh.append({"id": answer["id"], "view": answer["view"], "probs": probs})
    if not fresh:
        return 0
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a") as handle:
        for row in fresh:
            handle.write(_dumps(row) + "\n")
    return len(fresh)


def _probability(value, index, key):
    if isinstance(value, dict):
        return float(value[key])
    return float(value[index])


def score_rows(answers, questions=None):
    """P(stronger_needed) and 1 - P(resolve) for each answered (id, view)."""
    spec = load_questions() if questions is None else questions
    outcome_index = list(spec["outcome"]["criteria"]).index("stronger_needed")
    stronger, failure = [], []
    for answer in answers:
        base = {"id": answer["id"], "view": answer["view"]}
        stronger.append(
            {
                **base,
                "score": _probability(
                    answer["probs"]["outcome"], outcome_index, "stronger_needed"
                ),
            }
        )
        resolve = _probability(answer["probs"]["resolve"], 1, "true")
        failure.append({**base, "score": 1.0 - resolve})
    return stronger, failure


def load_view_scores(path, view):
    """One score per id for a view. Conflicting duplicates name the file and line."""
    scores = {}
    for line_no, record in read_jsonl(path):
        if record.get("view") != view:
            continue
        row_id = record["id"]
        score = float(record["score"])
        if row_id in scores and scores[row_id] != score:
            raise ValueError(
                f"{path}:{line_no}: duplicate id {row_id!r} view {view!r} "
                f"with scores {scores[row_id]} and {score}"
            )
        scores[row_id] = score
    return scores


def auroc(positive, negative):
    """P(a positive score exceeds a negative score), ties counted half."""
    if not positive or not negative:
        return None
    wins = ties = 0
    for left in positive:
        for right in negative:
            if left > right:
                wins += 1
            elif left == right:
                ties += 1
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _labels(rows, scores, positive):
    matched_pos, matched_neg = [], []
    for row in rows:
        if row["id"] not in scores:
            continue
        score = scores[row["id"]]
        if positive(row):
            matched_pos.append((row, score))
        else:
            matched_neg.append((row, score))
    return matched_pos, matched_neg


def _within_task(pairs_pos, pairs_neg):
    by_task_pos = {}
    by_task_neg = {}
    for row, score in pairs_pos:
        by_task_pos.setdefault(row["task"], []).append(score)
    for row, score in pairs_neg:
        by_task_neg.setdefault(row["task"], []).append(score)
    wins = ties = pairs = 0
    for task, positive in by_task_pos.items():
        negative = by_task_neg.get(task, [])
        if not negative:
            continue
        for left in positive:
            for right in negative:
                if left > right:
                    wins += 1
                elif left == right:
                    ties += 1
                pairs += 1
    if not pairs:
        return None, 0
    return (wins + 0.5 * ties) / pairs, pairs


def cheap_failed(row, cheap):
    return float(row["attempts"][cheap]["ok"]) < SOLVED


def strong_rescues(row, cheap, strong):
    return cheap_failed(row, cheap) and float(row["attempts"][strong]["ok"]) >= SOLVED


def ranking(rows, scores, cheap, strong):
    """AUROC of cheap-attempt failure and of a strong model rescuing that failure."""
    fail_pos, fail_neg = _labels(rows, scores, lambda row: cheap_failed(row, cheap))
    rescue_pos, rescue_neg = _labels(
        rows, scores, lambda row: strong_rescues(row, cheap, strong)
    )
    within, pairs = _within_task(fail_pos, fail_neg)
    return {
        "failure_auroc": auroc(
            [score for _, score in fail_pos], [score for _, score in fail_neg]
        ),
        "failure_within_task": within,
        "failure_within_task_pairs": pairs,
        "rescue_auroc": auroc(
            [score for _, score in rescue_pos], [score for _, score in rescue_neg]
        ),
    }


def answers_from_checkpoint(path):
    return [
        {"id": row_id, "view": view, "probs": probs}
        for (row_id, view), probs in load_checkpoint(path).items()
    ]


def _cmd_requests(args):
    records = load_trajectories(args.trajectories)
    write_jsonl(args.out, build_requests(records))
    print(f"{len(records)} trajectories, {len(records) * len(VIEWS)} requests")


def _cmd_rows(args):
    records = load_trajectories(args.trajectories)
    write_jsonl(args.out, policy_rows(records))
    print(f"{len(records)} rows written to {args.out}")


def _cmd_checkpoint(args):
    answers = [record for _, record in read_jsonl(args.answers)]
    added = append_checkpoint(args.checkpoint, answers)
    print(f"{added} answers appended to {args.checkpoint}")


def _cmd_scores(args):
    stronger, failure = score_rows(answers_from_checkpoint(args.checkpoint))
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "stronger_needed.jsonl", stronger)
    write_jsonl(out_dir / "resolve_failure.jsonl", failure)
    print(f"{len(stronger)} scores in {out_dir}")


def _cmd_auroc(args):
    rows = [record for _, record in read_jsonl(args.rows)]
    lines = []
    for view in VIEWS:
        scores = load_view_scores(args.scores, view)
        if not scores:
            continue
        report = ranking(rows, scores, args.cheap, args.strong)
        failure = report["failure_auroc"]
        rescue = report["rescue_auroc"]
        failure_text = "n/a" if failure is None else f"{failure:.3f}"
        rescue_text = "n/a" if rescue is None else f"{rescue:.3f}"
        lines.append(f"{view:16s} failure {failure_text}  rescue {rescue_text}")
    if not lines:
        raise ValueError(f"{args.scores}: no scores")
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    requests = sub.add_parser("requests", help="write judge request JSONL")
    requests.add_argument("--trajectories", required=True)
    requests.add_argument("--out", required=True)
    rows = sub.add_parser("rows", help="write explore.py rows")
    rows.add_argument("--trajectories", required=True)
    rows.add_argument("--out", required=True)
    checkpoint = sub.add_parser("checkpoint", help="append judge answers")
    checkpoint.add_argument("--answers", required=True)
    checkpoint.add_argument("--checkpoint", required=True)
    scores = sub.add_parser("scores", help="export the two score files")
    scores.add_argument("--checkpoint", required=True)
    scores.add_argument("--out-dir", required=True)
    rank = sub.add_parser("auroc", help="rank cheap failures and rescues")
    rank.add_argument("--rows", required=True)
    rank.add_argument("--scores", required=True)
    rank.add_argument("--cheap", required=True)
    rank.add_argument("--strong", required=True)
    args = parser.parse_args(argv)
    {
        "requests": _cmd_requests,
        "rows": _cmd_rows,
        "checkpoint": _cmd_checkpoint,
        "scores": _cmd_scores,
        "auroc": _cmd_auroc,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
