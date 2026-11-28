"""A development loop over independently graded, source-bound v0.5 gateway runs."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

from .loop_evidence import (
    HTTP_OK,
    check,
    check_client,
    check_runtime_result,
    check_termination,
    digest,
    packet_split,
    target,
)
from .schedule import episode_ids


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


def compare(ledger, args):
    """Pair two comparable measurements and record the comparison (paired)."""
    base, candidate = ledger.measure(args.baseline), ledger.measure(args.candidate)
    check(
        base["record_id"] != candidate["record_id"],
        "comparison requires distinct records",
    )
    for row in (base, candidate):
        verify(row)
    comparable(base, candidate)
    check(
        not episode_ids(base).intersection(episode_ids(candidate)),
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
    return ledger.append(
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
