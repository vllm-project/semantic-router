"""Whether a coding-agent restart is a safe point to switch models.

A restart is a compaction or a cleared conversation. From the request alone
the Router can see a harness marker, or a prompt that no longer extends the
previous main request. A prefix break that the next main request continues,
and that is at least 4,096 tokens, is a continued restart. A prefix break
the next request does not continue is how a rewind or an edited history looks.

Switching at a turn loses the cached prefix a same-model stay would keep.
The SemiAnalysis traces have no task outcome, so success and cost are
reported only when paired rows are supplied. The traces are fetched at a
pinned revision and are not committed.

Usage:
  restart_switch.py download DEST_DIR
  restart_switch.py report TRACES.jsonl [--stronger rows.jsonl]
      [--cheaper rows.jsonl] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

BLOCK = 64
DATASET = "semianalysisai/cc-traces-weka-with-subagents-060526"
REVISION = "8a5df339ba3c31a48dd79a2f5ceb81e08679266b"
TRACES_FILE = "traces.jsonl"
TRACES_SHA256 = "954157040cbae47fef3bec70532bdb3406c75ed894e207e502461d70fa06639c"
NEAR_FULL_SHARE = 0.9
RESTART_MIN_TOKENS = 4096
LOW_REUSE_SHARE = 0.5
Z_95 = 1.96
MIN_INTERVAL_ROWS = 2
CHUNK_BYTES = 1024 * 1024
URL_TIMEOUT_SECONDS = 120
DEFAULT_MARKERS = frozenset({"context_compaction"})
DIRECTIONS = ("stronger", "cheaper")
MARKER_LIST_KEYS = ("items", "input", "content", "messages")


def verify(path, digest):
    """SHA-256 of path. Raises ValueError when it does not match digest."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            hasher.update(chunk)
    got = hasher.hexdigest()
    if got != digest.lower():
        raise ValueError(f"{path}: sha256 {got} != {digest}")
    return got


def download(dest):
    """Fetch the pinned traces into dest and check their SHA-256."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / TRACES_FILE
    if target.is_file():
        try:
            verify(target, TRACES_SHA256)
        except ValueError:
            target.unlink()
        else:
            return str(target)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TRACES_FILE}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "semantic-router-restart-switch"},
    )
    with (
        urllib.request.urlopen(request, timeout=URL_TIMEOUT_SECONDS) as response,
        target.open("wb") as out,
    ):
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
    verify(target, TRACES_SHA256)
    return str(target)


def request_has_marker(request, markers):
    """True when the request carries a harness compaction marker.

    Block ids are not a marker. A caller can pass extra item types.
    """
    if request.get("type") in markers or request.get("item_type") in markers:
        return True
    if request.get("compaction") is True:
        return True
    return any(_list_has_marker(request.get(key), markers) for key in MARKER_LIST_KEYS)


def _list_has_marker(value, markers):
    if not isinstance(value, list):
        return False
    return any(
        isinstance(item, dict)
        and (item.get("type") in markers or item.get("item_type") in markers)
        for item in value
    )


def lead(blocks, seen):
    """Leading run of blocks already seen. That run is the cached prefix."""
    matched = 0
    for block in blocks:
        if block not in seen:
            break
        matched += 1
    return matched


def extends_prefix(previous, current):
    """True when current starts with the whole previous request."""
    if len(current) < len(previous):
        return False
    return current[: len(previous)] == previous


def _next_continues(blocks, following):
    """True when the next request's opening is drawn from this request.

    The matched opening has to cover at least 90% of this request. This is
    the continuation check from the prefix-reuse study: membership in this
    request's block set, not strict concatenation.
    """
    if following is None:
        return False
    return lead(following, set(blocks)) >= NEAR_FULL_SHARE * len(blocks)


def _check_block_size(trace):
    """Token totals use 64. A trace that says otherwise is refused."""
    size = trace.get("block_size", BLOCK)
    if size != BLOCK:
        raise ValueError(f"block_size {size} is not {BLOCK}")


def main_requests(trace):
    """Main-agent requests. Subagent calls are a different stream."""
    requests = trace.get("requests")
    if requests is None:
        return []
    if not isinstance(requests, list):
        raise ValueError("trace requests must be a list")
    main = []
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("trace request must be an object")
        if request.get("type") == "subagent":
            continue
        main.append(request)
    return main


def session_rows(trace, markers):
    """Classify one session's main calls and count marker hits on all of them."""
    _check_block_size(trace)
    main = main_requests(trace)
    marker_turns = sum(1 for request in main if request_has_marker(request, markers))
    calls = [request for request in main if request.get("hash_ids")]
    calls.sort(key=lambda request: request.get("t", 0))
    block_rows = [tuple(request["hash_ids"]) for request in calls]
    seen = set()
    conversation_blocks = 0
    previous_model = None
    rows = []
    for index, request in enumerate(calls):
        blocks = block_rows[index]
        prompt_blocks = len(blocks)
        cached_blocks = lead(blocks, seen) if seen else 0
        following = block_rows[index + 1] if index + 1 < len(block_rows) else None
        prefix_break = index > 0 and not extends_prefix(block_rows[index - 1], blocks)
        next_continues = _next_continues(blocks, following)
        long_enough = prompt_blocks * BLOCK >= RESTART_MIN_TOKENS
        continued = prefix_break and next_continues and long_enough
        rows.append(
            {
                "kind": _kind(index, prefix_break, next_continues, long_enough),
                "marker": request_has_marker(request, markers),
                "prefix_break": prefix_break,
                "continued_restart": continued,
                "cached_tokens": cached_blocks * BLOCK,
                "prompt_tokens": prompt_blocks * BLOCK,
                "half_drop": _half_drop_restart(
                    request,
                    blocks,
                    cached_blocks,
                    prompt_blocks,
                    following,
                    seen,
                    conversation_blocks,
                    previous_model,
                ),
            }
        )
        if prompt_blocks * BLOCK >= RESTART_MIN_TOKENS:
            conversation_blocks = max(conversation_blocks, prompt_blocks)
        previous_model = request.get("model")
        seen.update(blocks)
    return rows, marker_turns


def _kind(index, prefix_break, next_continues, long_enough):
    if index == 0:
        return "first"
    if not prefix_break:
        return "ordinary"
    if next_continues and long_enough:
        return "continued_restart"
    if next_continues:
        return "short_continued"
    return "not_continued"


def _half_drop_restart(
    request,
    blocks,
    cached_blocks,
    prompt_blocks,
    following,
    seen,
    conversation_blocks,
    previous_model,
):
    """The prefix-reuse study's restart, kept as a cross-check.

    It is a low-reuse main turn whose prompt is at most half the longest
    long prompt so far, and which the next turn continues. Short calls do
    not set that length. A model change is not this kind of restart.
    """
    if not seen or cached_blocks >= LOW_REUSE_SHARE * prompt_blocks:
        return False
    if request.get("model") != previous_model:
        return False
    if prompt_blocks * BLOCK < RESTART_MIN_TOKENS or conversation_blocks == 0:
        return False
    if prompt_blocks > conversation_blocks / 2:
        return False
    return _next_continues(blocks, following)


def error_rates(flags, labels):
    """False-positive and false-negative rates of aligned boolean predictions.

    A true label means the turn is a compaction or a cleared conversation.
    """
    if len(flags) != len(labels):
        raise ValueError("flags and labels must be the same length")
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for flag, label in zip(flags, labels, strict=True):
        if flag and label:
            counts["tp"] += 1
        elif flag:
            counts["fp"] += 1
        elif label:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    negatives = counts["tn"] + counts["fp"]
    positives = counts["tp"] + counts["fn"]
    counts["false_positive_rate"] = counts["fp"] / negatives if negatives else None
    counts["false_negative_rate"] = counts["fn"] / positives if positives else None
    return counts


def _token_summary(values):
    if not values:
        return {"turns": 0, "median_cached_tokens": None, "total_cached_tokens": 0}
    return {
        "turns": len(values),
        "median_cached_tokens": statistics.median(values),
        "total_cached_tokens": int(sum(values)),
    }


def summarise(traces, markers=DEFAULT_MARKERS, outcomes=None):
    """Detection counts, cache loss, and paired outcomes for these sessions."""
    if outcomes is None:
        outcomes = {}
    counts = _empty_counts()
    ordinary_cached = []
    restart_cached = []
    half_drop_cached = []
    for trace in traces:
        counts["sessions"] += 1
        rows, marker_turns = session_rows(trace, markers)
        counts["marker_turns"] += marker_turns
        if not rows:
            continue
        counts["sessions_with_main"] += 1
        continued_here = 0
        half_drop_here = 0
        for index, row in enumerate(rows):
            _accumulate_row(
                counts,
                row,
                index,
                ordinary_cached,
                restart_cached,
                half_drop_cached,
            )
            continued_here += row["continued_restart"]
            half_drop_here += row["half_drop"]
        counts["runs"] += 1 + continued_here
        if continued_here:
            counts["sessions_with_continued_restart"] += 1
        if half_drop_here:
            counts["sessions_with_half_drop"] += 1
    return _public_report(
        counts, ordinary_cached, restart_cached, half_drop_cached, outcomes
    )


def _empty_counts():
    return {
        "sessions": 0,
        "sessions_with_main": 0,
        "sessions_with_continued_restart": 0,
        "sessions_with_half_drop": 0,
        "main_turns": 0,
        "marker_turns": 0,
        "first_turns": 0,
        "ordinary_turns": 0,
        "continued_restarts": 0,
        "short_continued_breaks": 0,
        "not_continued_breaks": 0,
        "prefix_rule_disagreements": 0,
        "runs": 0,
        "half_drop_restarts": 0,
        "main_uncached_tokens": 0,
        "half_drop_uncached_tokens": 0,
    }


def _accumulate_row(
    counts, row, index, ordinary_cached, restart_cached, half_drop_cached
):
    counts["main_turns"] += 1
    kind = row["kind"]
    if kind == "first":
        counts["first_turns"] += 1
    elif kind == "ordinary":
        counts["ordinary_turns"] += 1
        ordinary_cached.append(row["cached_tokens"])
    elif kind == "continued_restart":
        counts["continued_restarts"] += 1
        restart_cached.append(row["cached_tokens"])
    elif kind == "short_continued":
        counts["short_continued_breaks"] += 1
    elif kind == "not_continued":
        counts["not_continued_breaks"] += 1
    else:
        raise ValueError(f"unknown turn kind {kind}")
    if row["prefix_break"] != row["continued_restart"]:
        counts["prefix_rule_disagreements"] += 1
    if index > 0:
        uncached = row["prompt_tokens"] - row["cached_tokens"]
        counts["main_uncached_tokens"] += uncached
        if row["half_drop"]:
            counts["half_drop_restarts"] += 1
            counts["half_drop_uncached_tokens"] += uncached
            half_drop_cached.append(row["cached_tokens"])


def _rate(numerator, denominator):
    if not denominator:
        return None
    return numerator / denominator


def _public_report(counts, ordinary_cached, restart_cached, half_drop_cached, outcomes):
    prefix_breaks = (
        counts["continued_restarts"]
        + counts["short_continued_breaks"]
        + counts["not_continued_breaks"]
    )
    by_direction = {
        name: direction_report(outcomes.get(name, ())) for name in DIRECTIONS
    }
    return {
        "sessions": counts["sessions"],
        "sessions_with_main": counts["sessions_with_main"],
        "main_turns": counts["main_turns"],
        "first_turns": counts["first_turns"],
        "ordinary_turns": counts["ordinary_turns"],
        "marker_turns": counts["marker_turns"],
        "prefix_breaks": prefix_breaks,
        "continued_restarts": counts["continued_restarts"],
        "short_continued_breaks": counts["short_continued_breaks"],
        "not_continued_breaks": counts["not_continued_breaks"],
        "prefix_rule_disagreements": counts["prefix_rule_disagreements"],
        "sessions_with_continued_restart": counts["sessions_with_continued_restart"],
        "runs": counts["runs"],
        "restarts_per_session": _rate(counts["continued_restarts"], counts["sessions"]),
        "restarts_per_run": _rate(counts["continued_restarts"], counts["runs"]),
        "ordinary_switch_loss": _token_summary(ordinary_cached),
        "restart_switch_loss": _token_summary(restart_cached),
        "stay_switch_loss_tokens": 0,
        "half_drop": {
            "turns": counts["half_drop_restarts"],
            "sessions": counts["sessions_with_half_drop"],
            "median_cached_tokens": _token_summary(half_drop_cached)[
                "median_cached_tokens"
            ],
            "uncached_token_share": _rate(
                counts["half_drop_uncached_tokens"],
                counts["main_uncached_tokens"],
            ),
        },
        "outcomes": by_direction,
        "recommendation": recommendation(by_direction),
    }


def wilson_interval(successes, total):
    """Wilson score interval for a binomial rate. None when total is 0."""
    if total <= 0:
        return None
    proportion = successes / total
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt((proportion * (1 - proportion) + z2 / (4 * total)) / total)
        / denominator
    )
    return {
        "rate": proportion,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def _mean_interval(values):
    count = len(values)
    if count == 0:
        return None
    mean = sum(values) / count
    if count < MIN_INTERVAL_ROWS:
        return {"mean": mean, "low": None, "high": None}
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    margin = Z_95 * math.sqrt(variance / count)
    return {"mean": mean, "low": mean - margin, "high": mean + margin}


def direction_report(rows):
    """Stay-versus-switch success and cost for one direction."""
    rows = list(rows)
    count = len(rows)
    if count == 0:
        return {
            "n": 0,
            "stay_success": None,
            "switch_success": None,
            "success_effect": None,
            "cost_effect": None,
        }
    return {
        "n": count,
        "stay_success": wilson_interval(sum(row["stay_ok"] for row in rows), count),
        "switch_success": wilson_interval(sum(row["switch_ok"] for row in rows), count),
        "success_effect": _mean_interval(
            [row["switch_ok"] - row["stay_ok"] for row in rows]
        ),
        "cost_effect": _mean_interval(
            [row["switch_cost"] - row["stay_cost"] for row in rows]
        ),
    }


def recommendation(by_direction):
    """Go only when both directions exclude a success drop.

    A drop is a negative mean of switch_ok minus stay_ok. The interval has
    to exist and its lower bound has to be at least 0. Missing rows are
    no-go. Cost does not flip this gate; the report shows it beside the gate.
    """
    for name in DIRECTIONS:
        report = by_direction.get(name)
        if report is None or report["n"] == 0:
            return "no-go"
        effect = report["success_effect"]
        if effect is None or effect["low"] is None or effect["low"] < 0:
            return "no-go"
    return "go"


def load_outcomes(path):
    """Paired rows: stay_ok, switch_ok, stay_cost, switch_cost."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json") from exc
            rows.append(_outcome_row(record, path, line_no))
    return rows


def _outcome_row(record, path, line_no):
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_no}: row must be an object")
    return {
        "stay_ok": _bit(record, "stay_ok", path, line_no),
        "switch_ok": _bit(record, "switch_ok", path, line_no),
        "stay_cost": _number(record, "stay_cost", path, line_no),
        "switch_cost": _number(record, "switch_cost", path, line_no),
    }


def _bit(record, field, path, line_no):
    number = _number(record, field, path, line_no)
    if number not in (0.0, 1.0):
        raise ValueError(f"{path}:{line_no}: {field} must be 0 or 1")
    return number


def _number(record, field, path, line_no):
    try:
        return float(record[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: {field}") from exc


def iter_traces(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json") from exc
            if not isinstance(trace, dict):
                raise ValueError(f"{path}:{line_no}: trace must be an object")
            yield trace


def _print_report(report):
    loss = report["restart_switch_loss"]
    ordinary = report["ordinary_switch_loss"]
    half = report["half_drop"]
    print(f"sessions {report['sessions']} main turns {report['main_turns']}")
    print(
        f"marker turns {report['marker_turns']} "
        f"prefix breaks {report['prefix_breaks']} "
        f"continued restarts {report['continued_restarts']} "
        f"short continued {report['short_continued_breaks']} "
        f"not continued {report['not_continued_breaks']}"
    )
    print(
        f"restarts per session {report['restarts_per_session']} "
        f"per run {report['restarts_per_run']} "
        f"sessions with a continued restart {report['sessions_with_continued_restart']} "
        f"runs {report['runs']}"
    )
    print(
        "switch loss at a continued restart: "
        f"{loss['turns']} turns, median cached {loss['median_cached_tokens']}, "
        f"total cached {loss['total_cached_tokens']}"
    )
    print(
        "switch loss at an ordinary turn: "
        f"{ordinary['turns']} turns, median cached {ordinary['median_cached_tokens']}, "
        f"total cached {ordinary['total_cached_tokens']}"
    )
    print(f"stay loses {report['stay_switch_loss_tokens']} cached tokens")
    print(
        "half-drop cross-check: "
        f"{half['turns']} turns in {half['sessions']} sessions, "
        f"median cached {half['median_cached_tokens']}, "
        f"uncached share {half['uncached_token_share']}"
    )
    for name in DIRECTIONS:
        outcome = report["outcomes"][name]
        print(f"outcomes {name}: n={outcome['n']}")
    print(f"recommendation {report['recommendation']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="fetch the pinned traces and check sha256")
    fetch.add_argument("dest")
    report = sub.add_parser("report", help="detection, cache loss, and outcome gate")
    report.add_argument("traces")
    report.add_argument("--stronger", type=Path)
    report.add_argument("--cheaper", type=Path)
    report.add_argument("--marker", action="append", default=[])
    report.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.dest))
        return 0
    markers = set(DEFAULT_MARKERS)
    markers.update(args.marker)
    outcomes = {}
    if args.stronger:
        outcomes["stronger"] = load_outcomes(args.stronger)
    if args.cheaper:
        outcomes["cheaper"] = load_outcomes(args.cheaper)
    result = summarise(iter_traces(args.traces), frozenset(markers), outcomes)
    if args.json:
        json.dump(result, sys.stdout)
        print()
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
