"""Paired A/B comparison of two grounded-fusion arms (grounding on vs off).

Reads ``samples_on.jsonl`` and ``samples_off.jsonl`` (written by evaluate.py),
joins by sample id, and reports the headline deltas with paired bootstrap CIs:

- mean normalized DRACO score (on - off)
- mean negative-criteria penalty (on - off)   <- the headline: should rise toward 0
- per-domain and "contested slice" breakdowns (items where grounding dropped a
  panel response in the on arm -- where it can actually do something)

Example:
    .venv-bench/bin/python -m bench.grounded_fusion.compare \
        --results-dir bench/grounded_fusion/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import paired_bootstrap_ci


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run evaluate.py for that arm first")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _contested(on: dict) -> bool:
    """An item is 'contested' if grounding dropped >=1 panel response in the on arm."""
    return any(p.get("dropped") for p in on.get("panel", []))


def _delta_block(pairs: list[tuple], key: str) -> dict:
    deltas = [o["final"][key] - f["final"][key] for o, f in pairs]
    mean, lo, hi = paired_bootstrap_ci(deltas)
    return {
        "n": len(pairs),
        "mean_delta": mean,
        "ci95": [lo, hi],
        "significant": (lo > 0) or (hi < 0),
    }


def compare(results_dir: str) -> dict:
    d = Path(results_dir)
    on = _load(d / "samples_on.jsonl")
    off = _load(d / "samples_off.jsonl")
    ids = [
        i for i in on if i in off and not on[i].get("error") and not off[i].get("error")
    ]
    pairs = [(on[i], off[i]) for i in ids]
    if not pairs:
        raise SystemExit("no overlapping successful samples between arms")

    contested = [(o, f) for o, f in pairs if _contested(o)]
    by_dom: dict[str, list[tuple]] = {}
    for o, f in pairs:
        by_dom.setdefault(o["domain"], []).append((o, f))

    report = {
        "n_paired": len(pairs),
        "n_contested": len(contested),
        "overall": {
            "normalized": _delta_block(pairs, "normalized"),
            "negative_penalty": _delta_block(pairs, "negative_penalty"),
        },
        "contested_slice": {
            "normalized": _delta_block(contested, "normalized") if contested else None,
            "negative_penalty": (
                _delta_block(contested, "negative_penalty") if contested else None
            ),
        },
        "per_domain_normalized_delta": {
            dom: _delta_block(ps, "normalized")["mean_delta"]
            for dom, ps in by_dom.items()
        },
        "level1_from_on_arm": _read_level1(d / "summary_on.json"),
    }
    return report


def _read_level1(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("summary", {}).get("level1", {})


def _print(report: dict) -> None:
    print("\n" + "=" * 64)
    print("GROUNDING-AWARE FUSION: A/B (on - off)")
    print("=" * 64)
    print(
        f"paired samples: {report['n_paired']} | contested (grounding dropped >=1): {report['n_contested']}"
    )
    for scope in ("overall", "contested_slice"):
        print(f"\n[{scope}]")
        for metric in ("normalized", "negative_penalty"):
            b = report[scope].get(metric)
            if not b:
                print(f"  {metric}: (no items)")
                continue
            sig = "SIGNIFICANT" if b["significant"] else "ns"
            print(
                f"  {metric:18} delta={b['mean_delta']:+.4f}  ci95=[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]  {sig}"
            )
    print("\n[per-domain normalized delta]")
    for dom, dv in sorted(
        report["per_domain_normalized_delta"].items(), key=lambda kv: kv[1]
    ):
        print(f"  {dom:28} {dv:+.4f}")
    l1 = report.get("level1_from_on_arm") or {}
    if l1.get("n_panel_graded"):
        print(
            f"\n[level-1 scorer validity]  spearman(score,quality)={l1.get('spearman_score_vs_quality')}"
            f"  discard_precision={l1.get('discard_precision')}  (n={l1['n_panel_graded']})"
        )
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="A/B compare grounded-fusion arms")
    ap.add_argument("--results-dir", default="bench/grounded_fusion/results")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    report = compare(args.results_dir)
    _print(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
