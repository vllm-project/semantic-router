"""Prefix reuse in production coding-agent sessions with subagents.

Reads the SemiAnalysis cc-traces-weka-with-subagents traces, where every request lists its prompt as
64-token blocks whose hash ids are chained: an id never appears at two positions, so a seen id means its
whole prefix was seen. The leading run of a request's blocks already seen in a stream is therefore the
prefix a KV cache could serve. Reported:

  main       a main-agent turn against the main stream before it
  sub_first  a subagent's first call against the parent's stream and against earlier sibling subagents
  sub_later  a subagent's later calls against that subagent's own earlier calls
  gaps       requests that arrive after an idle gap longer than a cache TTL on their own stream

plus the share of tokens subagents carry and the models each stream used. No inference, standard library
only. The traces are fetched at a pinned revision and never committed.

Usage:
  prefix_reuse.py download DEST_DIR
  prefix_reuse.py report TRACES.jsonl [--ttl 300 --ttl 3600] [--json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
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


def lead(blocks, seen):
    """Length of the leading run of blocks already seen, which is the cached prefix."""
    n = 0
    for block in blocks:
        if block not in seen:
            break
        n += 1
    return n


def flatten(trace):
    """Requests in time order, each tagged with its stream: ("main", None) or ("sub", agent_id)."""
    out = []
    for request in trace["requests"]:
        if request.get("type") == "subagent":
            for inner in request["requests"]:
                out.append({**inner, "stream": ("sub", request["agent_id"])})
        else:
            out.append({**request, "stream": ("main", None)})
    return sorted(out, key=lambda r: r["t"])


def analyse(trace, ttls):
    """Per-request observations for one trace."""
    rows = collections.defaultdict(list)
    main_seen = set()
    sub_seen = collections.defaultdict(set)
    last_end = {}
    for request in flatten(trace):
        blocks = request.get("hash_ids") or []
        if not blocks:
            continue
        kind, agent = request["stream"]
        n = len(blocks)
        if kind == "main":
            if main_seen:
                rows["main"].append((lead(blocks, main_seen), n))
            main_seen.update(blocks)
            stream = "main"
        else:
            own = sub_seen[agent]
            if not own:
                siblings = set().union(
                    *(seen for other, seen in sub_seen.items() if other != agent)
                )
                rows["sub_first"].append(
                    (lead(blocks, main_seen), lead(blocks, siblings), n)
                )
            else:
                rows["sub_later"].append((lead(blocks, own), n))
            own.update(blocks)
            stream = agent
        previous = last_end.get(stream)
        for ttl in ttls:
            if previous is not None and request["t"] - previous > ttl:
                rows[("gap", ttl)].append((n, request.get("in") or 0))
        last_end[stream] = request["t"] + (request.get("api_time") or 0.0)
        rows["tokens_" + kind].append((request.get("in") or 0, request.get("out") or 0))
        rows["model_" + kind].append(request.get("model"))
    return rows


def _share(pairs, part=0, whole=-1):
    total = sum(p[whole] for p in pairs)
    return sum(p[part] for p in pairs) / total if total else 0.0


def summarise(traces, ttls=(300.0, 3600.0)):
    """Aggregate the observations of every trace into the reported numbers."""
    rows = collections.defaultdict(list)
    n_traces = with_subagents = 0
    subagent_groups = 0
    for trace in traces:
        n_traces += 1
        groups = sum(1 for r in trace["requests"] if r.get("type") == "subagent")
        with_subagents += bool(groups)
        subagent_groups += groups
        for key, values in analyse(trace, ttls).items():
            rows[key].extend(values)

    main, first, later = rows["main"], rows["sub_first"], rows["sub_later"]
    tin = {k: sum(t[0] for t in rows["tokens_" + k]) for k in ("main", "sub")}
    tout = {k: sum(t[1] for t in rows["tokens_" + k]) for k in ("main", "sub")}
    requests = len(rows["model_main"]) + len(rows["model_sub"])
    report = {
        "traces": n_traces,
        "traces_with_subagents": with_subagents,
        "subagent_groups": subagent_groups,
        "main_turns": len(main),
        "main_cached_prefix_share": _share(main),
        "main_cached_prefix_median_turn": (
            statistics.median(c / n for c, n in main) if main else 0.0
        ),
        "subagent_first_calls": len(first),
        "subagent_first_median_prompt_tokens": (
            statistics.median(n for _, _, n in first) * BLOCK if first else 0
        ),
        "subagent_first_shared_with_parent": _share(first, 0),
        "subagent_first_shared_with_sibling": _share(first, 1),
        "subagent_first_sibling_ge_90": (
            sum(s / n >= NEAR_FULL_SHARE for _, s, n in first) / len(first)
            if first
            else 0.0
        ),
        "subagent_later_calls": len(later),
        "subagent_later_cached_prefix_share": _share(later),
        "input_tokens": tin,
        "output_tokens": tout,
        "subagent_input_share": tin["sub"] / max(1, tin["main"] + tin["sub"]),
        "subagent_output_share": tout["sub"] / max(1, tout["main"] + tout["sub"]),
        "models_main": collections.Counter(rows["model_main"]).most_common(4),
        "models_sub": collections.Counter(rows["model_sub"]).most_common(4),
        "requests": requests,
        "idle_gaps": {},
    }
    for ttl in ttls:
        gaps = rows[("gap", ttl)]
        report["idle_gaps"][str(int(ttl))] = {
            "requests": len(gaps),
            "request_share": len(gaps) / max(1, requests),
            "median_prompt_tokens": (
                statistics.median(n for n, _ in gaps) * BLOCK if gaps else 0
            ),
            "input_token_share": sum(i for _, i in gaps)
            / max(1, tin["main"] + tin["sub"]),
        }
    return report


def read_traces(path):
    with open(path) as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def verify(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"{path} has sha256 {digest.hexdigest()}, expected {expected}")


def download(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / TRACES_FILE
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TRACES_FILE}"
    urllib.request.urlretrieve(url, target)
    verify(target, TRACES_SHA256)
    return target


def _print(report):
    g = report["idle_gaps"]
    print(
        f"traces {report['traces']}, with subagents {report['traces_with_subagents']}, subagent groups {report['subagent_groups']}"
    )
    print(
        f"main turns {report['main_turns']}: cached prefix {report['main_cached_prefix_share']:.3f} of prompt tokens, median turn {report['main_cached_prefix_median_turn']:.3f}"
    )
    print(
        f"subagent first calls {report['subagent_first_calls']}: median prompt {report['subagent_first_median_prompt_tokens']:.0f} tokens, "
        f"shared with the parent {report['subagent_first_shared_with_parent']:.3f}, with an earlier sibling {report['subagent_first_shared_with_sibling']:.3f}, "
        f"sharing at least 90% with a sibling {report['subagent_first_sibling_ge_90']:.3f}"
    )
    print(
        f"subagent later calls {report['subagent_later_calls']}: cached prefix {report['subagent_later_cached_prefix_share']:.3f}"
    )
    print(
        f"subagent share of input tokens {report['subagent_input_share']:.3f}, of output tokens {report['subagent_output_share']:.3f}"
    )
    print(f"models main {report['models_main']} | models sub {report['models_sub']}")
    for ttl, row in g.items():
        print(
            f"after an idle gap over {ttl} s: {row['requests']} of {report['requests']} requests ({row['request_share']:.3f}), "
            f"median prompt {row['median_prompt_tokens']:.0f} tokens, {row['input_token_share']:.3f} of input tokens"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser(
        "download",
        help="fetch the traces at the pinned revision and check their sha256",
    )
    fetch.add_argument("dest")
    rep = sub.add_parser("report", help="compute the prefix reuse report")
    rep.add_argument("traces")
    rep.add_argument(
        "--ttl",
        type=float,
        action="append",
        help="idle gap thresholds in seconds (default 300 and 3600)",
    )
    rep.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.dest))
        return 0
    report = summarise(read_traces(args.traces), tuple(args.ttl or (300.0, 3600.0)))
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
