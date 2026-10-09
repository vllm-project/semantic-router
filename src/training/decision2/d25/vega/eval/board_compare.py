"""Compare a scored local run with the published board values of the same model (validation report).

    python -m d25.vega.eval.board_compare --scores RUN/scores.json --board-benchmarks benchmarks_v03_long.csv \
        --board-leaderboard leaderboard_v03_full.csv --engine <board engine id> --out validation.md [--title ...]

Tolerance (fixed before any result): |delta public index| <= 0.5 points, and |delta skill| <= 2.0 points on every
index benchmark with >= 500 requests; smaller benchmarks are reported but not gated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

INDEX_TOL = 0.5
SKILL_TOL = 2.0
MIN_REQUESTS = 500
AREAS = ("knowledge", "language", "retrieval", "tools", "arts")


def load_board(benchmarks_csv: str, leaderboard_csv: str, engine: str):
    rows = [r for r in csv.DictReader(open(benchmarks_csv)) if r["engine"] == engine]
    if not rows:
        raise SystemExit(f"engine {engine!r} not in {benchmarks_csv}")
    board = {}
    for r in rows:
        board[r["bench_id"]] = r
    lead = next(
        r for r in csv.DictReader(open(leaderboard_csv)) if r["engine"] == engine
    )
    return board, lead


def compare(scores: dict, board: dict, lead: dict) -> dict:
    out = {"benchmarks": [], "index": {}, "areas": {}}
    ours_index = scores["decision_index"]
    board_index = float(lead["public_index"])
    out["index"] = {
        "ours": ours_index,
        "board": board_index,
        "delta": round(ours_index - board_index, 2),
        "pass": abs(ours_index - board_index) <= INDEX_TOL,
    }
    for a in scores["areas"]:
        b = float(lead[f"pub_{a['id']}"])
        out["areas"][a["id"]] = {
            "ours": round(100 * a["skill"], 2),
            "board": b,
            "delta": round(100 * a["skill"] - b, 2),
        }
    gated_fail = []
    for n, v in sorted(scores["index_benchmarks"].items(), key=lambda kv: int(kv[0])):
        if not v.get("in_index"):
            continue
        b = board.get(n)
        bench = scores["benchmarks"].get(n, {})
        requests = int(b["requests"]) if b else bench.get("requests")
        ours = 100 * v["skill"]
        theirs = 100 * float(b["skill"]) if b and b["skill"] != "" else None
        delta = None if theirs is None else round(ours - theirs, 2)
        gated = requests is not None and requests >= MIN_REQUESTS
        ok = delta is not None and (not gated or abs(delta) <= SKILL_TOL)
        if gated and not ok:
            gated_fail.append(n)
        out["benchmarks"].append(
            {
                "id": int(n),
                "name": b["bench"] if b else bench.get("dataset", n),
                "area": b["area"] if b else "",
                "requests": requests,
                "ours_skill": round(ours, 2),
                "board_skill": None if theirs is None else round(theirs, 2),
                "delta": delta,
                "ours_raw": round(100 * v["raw"], 2),
                "board_raw": round(100 * float(b["raw"]), 2) if b else None,
                "ours_coverage": round(v["coverage"], 4),
                "board_coverage": float(b["coverage"]) if b else None,
                "gated": gated,
                "pass": ok,
            }
        )
    out["gated_fail"] = gated_fail
    out["pass"] = out["index"]["pass"] and not gated_fail
    return out


def render(result: dict, title: str, notes: list[str]) -> str:
    idx = result["index"]
    lines = [f"# {title}", ""]
    lines += notes + [""] if notes else []
    lines += [
        f"Tolerance (fixed before any result): |delta public index| <= {INDEX_TOL} points; |delta skill| <= "
        f"{SKILL_TOL} points on index benchmarks with >= {MIN_REQUESTS} requests (others reported, not gated).",
        "",
    ]
    verdict = "PASS" if result["pass"] else "FAIL"
    lines += [
        f"**Verdict: {verdict}.** Public index: local {idx['ours']:.2f} vs board {idx['board']:.2f} "
        f"(delta {idx['delta']:+.2f}, {'within' if idx['pass'] else 'outside'} {INDEX_TOL}). "
        f"Gated benchmarks outside {SKILL_TOL}: {', '.join(result['gated_fail']) or 'none'}.",
        "",
    ]
    lines += ["| Area | Local | Board | Delta |", "|---|---:|---:|---:|"]
    for a in AREAS:
        v = result["areas"].get(a)
        if v:
            lines.append(
                f"| {a} | {v['ours']:.2f} | {v['board']:.2f} | {v['delta']:+.2f} |"
            )
    lines += [
        "",
        "| # | Benchmark | Area | Requests | Local skill | Board skill | Delta | Local cov. | Board cov. | Gated | OK |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for b in result["benchmarks"]:
        board = "" if b["board_skill"] is None else f"{b['board_skill']:.2f}"
        delta = "" if b["delta"] is None else f"{b['delta']:+.2f}"
        bcov = "" if b["board_coverage"] is None else f"{b['board_coverage']:.4f}"
        lines.append(
            f"| {b['id']} | {b['name']} | {b['area']} | {b['requests']} | {b['ours_skill']:.2f} | {board} | {delta} | "
            f"{b['ours_coverage']:.4f} | {bcov} | {'yes' if b['gated'] else 'no'} | {'ok' if b['pass'] else 'FAIL'} |"
        )
    deltas = [abs(b["delta"]) for b in result["benchmarks"] if b["delta"] is not None]
    if deltas:
        lines += [
            "",
            f"Mean |delta skill| over {len(deltas)} index benchmarks: {sum(deltas) / len(deltas):.2f} points; "
            f"max {max(deltas):.2f}.",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--board-benchmarks", required=True)
    ap.add_argument("--board-leaderboard", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--title", default="Validation of the local public-index reproduction"
    )
    ap.add_argument("--note", action="append", default=[])
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    scores = json.loads(Path(a.scores).read_text())
    board, lead = load_board(a.board_benchmarks, a.board_leaderboard, a.engine)
    result = compare(scores, board, lead)
    Path(a.out).write_text(render(result, a.title, a.note))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "index": result["index"],
                "gated_fail": result["gated_fail"],
            }
        )
    )


if __name__ == "__main__":
    main()
