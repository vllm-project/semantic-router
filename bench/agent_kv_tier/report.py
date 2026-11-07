"""Summarise the runs of run_trace.sh, run_placement.sh and run_prefix_ttft.sh.

`report.py arms DIR...` prints one block per arm directory. It reads the requests the tester completed
(`detailed_results.csv`), the tester's own summary (`client.log`), and every vLLM log in the directory
(`server*.log`): its GPU KV pool, its last logged prefix-cache hit rates (vLLM logs a rate over its most recent
requests), and its mean number of waiting requests.
`report.py prefix FILE` prints the median time to first token per arm and prompt length from prefix_ttft.py.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ENGINE_LINE = re.compile(
    r"Running: (\d+) reqs, Waiting: (\d+) reqs.*?Prefix cache hit rate: ([0-9.]+)%"
)
EXTERNAL_HIT = re.compile(r"External prefix cache hit rate: ([0-9.]+)%")
KV_POOL = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
TESTER_TTFT = re.compile(
    r"TTFT avg/p50/p95/max: ([0-9.]+)s / ([0-9.]+)s / ([0-9.]+)s / ([0-9.]+)s"
)
TESTER_SET = re.compile(r"Peak Working Set: ([0-9,]+) tokens")
TESTER_HIT = re.compile(r"Avg Workload Cache Hit Rate: ([0-9.]+)%")


def quantile(values, q):
    """Nearest-rank quantile of sorted values, the convention the reported tables use."""
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize_requests(rows):
    ok = [r for r in rows if r["success"] == "True"]
    ttft = sorted(float(r["ttft"]) for r in ok)
    return {
        "requests": len(ok),
        "input_tokens": sum(int(r["input_tokens"]) for r in ok),
        "sessions": len({r["trace_id"] for r in ok}),
        "ttft_p50": quantile(ttft, 0.5) if ttft else None,
        "ttft_p90": quantile(ttft, 0.9) if ttft else None,
        "ttft_max": ttft[-1] if ttft else None,
    }


def parse_server_log(lines):
    kv_pool, prefix_hit, external_hit, waiting = None, None, None, []
    for line in lines:
        if (m := KV_POOL.search(line)) and kv_pool is None:
            kv_pool = int(m.group(1).replace(",", ""))
        if m := ENGINE_LINE.search(line):
            waiting.append(int(m.group(2)))
            prefix_hit = float(m.group(3))
            ext = EXTERNAL_HIT.search(line)
            external_hit = float(ext.group(1)) if ext else None
    return {
        "kv_pool_tokens": kv_pool,
        "last_prefix_hit": prefix_hit,
        "last_external_hit": external_hit,
        "mean_waiting": sum(waiting) / len(waiting) if waiting else None,
    }


def parse_client_log(lines):
    out = {}
    for line in lines:
        if m := TESTER_TTFT.search(line):
            out["tester_ttft_avg_p50_p95_max"] = [float(x) for x in m.groups()]
        if m := TESTER_SET.search(line):
            out["peak_working_set_tokens"] = int(m.group(1).replace(",", ""))
        if m := TESTER_HIT.search(line):
            out["tester_cache_hit"] = float(m.group(1))
    return out


def arm_report(directory):
    d = Path(directory)
    with open(d / "detailed_results.csv") as f:
        report = summarize_requests(list(csv.DictReader(f)))
    if (d / "client.log").exists():
        report.update(
            parse_client_log(
                (d / "client.log").read_text(errors="replace").splitlines()
            )
        )
    report["servers"] = {
        p.name: parse_server_log(p.read_text(errors="replace").splitlines())
        for p in sorted(d.glob("server*.log"))
    }
    return report


def prefix_medians(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["tokens"]), []).append(r)
    return {
        key: (
            statistics.median(r["ttft_first"] for r in rs),
            statistics.median(r["ttft_second"] for r in rs),
        )
        for key, rs in sorted(groups.items())
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("arms").add_argument("dirs", nargs="+")
    sub.add_parser("prefix").add_argument("file")
    a = ap.parse_args(argv)
    if a.command == "arms":
        for directory in a.dirs:
            print(Path(directory).name, json.dumps(arm_report(directory), indent=1))
        return
    with open(a.file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for (arm, tokens), (first, second) in prefix_medians(rows).items():
        print(
            f"{arm:10s} {tokens:6d} tokens  first {first:.3f} s  second {second:.3f} s"
        )


if __name__ == "__main__":
    main()
