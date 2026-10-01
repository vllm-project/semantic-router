"""Decoder M9 results summary (node A host, stdlib only): one JSON with every number the results record cites.

Reads (whatever exists; missing pieces are null): the line readouts and scores under m9/lines/4b (typed DEV / CSS
pilot readouts, HT-DEV v2 pairs, Score5-typed-DEV, HR2 DEV, hs1-dev), the readout-path parity, the pick, the early
read of H9-s1, the formal-path parity, the formal run's report, paired comparisons, types, mlx-diag and successor
read-out, and the GPU-hours file. Writes m9/results/summary.json and prints a compact view.

usage: python3 m9_report.py [--root /data/dev2/runs/dec/m9] [--control N7C]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def r4(x):
    return None if x is None else round(x, 4)


def point_view(lines: Path, point: str) -> dict:
    diag = lines / "diag"
    ht = load(diag / f"{point}.htdev2.json")
    s5 = load(diag / f"{point}.score5t.json")
    hr2 = load(diag / f"{point}.hr2dev.json")
    hs1 = load(diag / f"{point}.hs1.json")
    out = {
        "htdev2_vs_I": (
            None
            if ht is None
            else {
                "H": r4(ht["H_dev2"][point]),
                "I": r4(ht["H_dev2"].get("4b-I")),
                "delta": r4(ht["delta"]),
                "ci95": [r4(x) for x in ht["ci95"]],
                "verdict": ht["verdict"],
                "tasks": {
                    t: {k: r4(v) for k, v in d.items()} for t, d in ht["tasks"].items()
                },
            }
        ),
        "score5t_check": (
            None
            if s5 is None
            else {
                "flags": s5["check"]["flags"],
                "top_share": s5["check"].get("top_share"),
            }
        ),
        "hr2dev": (
            None
            if hr2 is None
            else {
                "family_macro": r4(hr2["family_macro"]),
                "type_macro": {k: r4(v) for k, v in hr2["type_macro"].items()},
                "families": {k: r4(v["accuracy"]) for k, v in hr2["families"].items()},
                "delta_vs_I": r4(hr2.get("delta_family_macro")),
                "ci95": (
                    None
                    if "paired" not in hr2
                    else [r4(x) for x in hr2["paired"]["ci95"]]
                ),
            }
        ),
        "hs1": None,
    }
    if hs1 is not None:
        diag_out = {}
        for fam, cell in hs1["families"].items():
            d = cell["diagnostics"].get(point, {})
            r = cell["diagnostics"].get("4b-I", {})
            diag_out[fam] = {
                "accuracy": cell["accuracy"],
                **{f"{k}": d[k] for k in d},
                **{f"I_{k}": r[k] for k in r},
            }
        out["hs1"] = diag_out
    return out


def readout_view(lines: Path, line: str) -> dict | None:
    doc = load(lines / "readout" / f"{line}.json")
    if doc is None:
        return None
    out = {}
    for name, arm in doc["arms"].items():
        out[name] = {
            "T": r4(arm["T"]),
            "H3": r4(arm["H_mean"]),
            "H_pilot": r4(arm["H"]),
            "proxy": r4(arm["proxy"]),
            "by_type": {t: v["correct"] for t, v in arm["by_type"].items()},
            "rule_precedence": arm["by_family"]
            .get("rule_precedence", {})
            .get("correct"),
            "by_family": {f: v["correct"] for f, v in arm["by_family"].items()},
        }
    return out


def pair_view(lines: Path, name: str) -> dict | None:
    doc = load(lines / "diag" / name)
    if doc is None:
        return None
    return {
        "left": doc["left"],
        "right": doc["right"],
        "H": {k: r4(v) for k, v in doc["H_dev2"].items()},
        "delta": r4(doc["delta"]),
        "ci95": [r4(x) for x in doc["ci95"]],
        "p_le_0": doc.get("p_le_0"),
        "verdict": doc["verdict"],
        "tasks": {t: {k: r4(v) for k, v in d.items()} for t, d in doc["tasks"].items()},
    }


def formal_view(formal: Path, run: str) -> dict | None:
    rd = formal / run
    rep = load(rd / "REPORT.json")
    if rep is None:
        return None
    out = {
        "v3": r4(rep["v3"]["score"]),
        "T": r4(rep["v3"]["T"]),
        "H": r4(rep["v3"]["H"]),
        "public231": rep["panels"]["public231"]["correct"],
        "paired": {},
    }
    for f in sorted(glob.glob(str(rd / "PAIRED-vs-*.json"))):
        p = load(Path(f))
        name = os.path.basename(f)[len("PAIRED-vs-") : -len(".json")]
        out["paired"][name] = {
            "delta": r4(p["point"]["delta"]["score"]),
            "ci95": [r4(p["ci95"]["low"]), r4(p["ci95"]["high"])],
            "dH": r4(p["point"]["delta"]["H"]),
            "H_ci95": [
                r4(p["axis_ci95"]["H"]["delta"]["low"]),
                r4(p["axis_ci95"]["H"]["delta"]["high"]),
            ],
            "dT": r4(p["point"]["delta"]["T"]),
            "T_ci95": [
                r4(p["axis_ci95"]["T"]["delta"]["low"]),
                r4(p["axis_ci95"]["T"]["delta"]["high"]),
            ],
        }
    types = load(rd / "TYPES.json")
    if types is not None:
        out["types"] = types.get("types")
    for name in ("MLX-PAIRED.json", "MLX-PAIRED-vs-n4xf-nodeA-m9.json"):
        m = load(formal / f"{run}-mlx" / name)
        if m is not None:
            out[name] = m.get("summary", m) if isinstance(m, dict) else m
    succ = load(formal / "successor" / f"4b-{run}.json")
    if succ is not None:
        out["successor_readout"] = succ
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=Path("/data/dev2/runs/dec/m9"))
    p.add_argument("--control", default="N7C")
    a = p.parse_args()
    lines = a.root / "lines" / "4b"
    X = a.control
    points = [
        "4b-I",
        "4b-H9-a1",
        "4b-H9-a1_2",
        f"4b-{X}-a1",
        f"4b-{X}-a1_2",
        "4b-H9-s1",
    ]
    pick = load(a.root / "select" / "4b-pick.json")
    summary = {
        "control": X,
        "points": {pt: point_view(lines, pt) for pt in points if (lines / pt).is_dir()},
        "readouts": {line: readout_view(lines, line) for line in ("L-H9", f"L-{X}")},
        "pairs": {
            tag: pair_view(lines, f"pair-H9-{X}-{tag}.htdev2.json")
            for tag in ("a1", "a1_2")
        },
        "readout_parity": {
            Path(f).stem: load(Path(f))
            for f in sorted(glob.glob(str(lines / "parity" / "*.json")))
        },
        "pick": (
            None
            if pick is None
            else {
                "point": None if pick["pick"] is None else pick["pick"]["point"],
                "no_pick_reasons": pick.get("no_pick_reasons"),
                "lines": {
                    ln: [
                        {
                            k: r.get(k)
                            for k in (
                                "step",
                                "point",
                                "eligible",
                                "reasons",
                                "htdev2",
                                "T",
                                "G",
                                "H3",
                                "proxy",
                                "by_type",
                                "rule_precedence",
                                "score5t_check_flags",
                                "score5t_check_top_share",
                                "hr2dev_family_macro",
                            )
                        }
                        for r in rows
                    ]
                    for ln, rows in pick["lines"].items()
                },
            }
        ),
        "formal_parity": load(a.root / "formal" / "m9-ref-N4XF" / "PARITY.json"),
        "formal": {},
        "gpu_hours": load(a.root / "gpuh-node-a.json"),
    }
    if pick is not None and pick.get("pick"):
        run = f"m9-{pick['pick']['point']}"
        summary["formal"][run] = formal_view(a.root / "formal", run)
    summary["formal"]["m9-ref-N4XF"] = formal_view(a.root / "formal", "m9-ref-N4XF")
    out = a.root / "results" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    compact = {
        "pairs": {
            k: None if v is None else {kk: v[kk] for kk in ("delta", "ci95", "verdict")}
            for k, v in summary["pairs"].items()
        },
        "vs_I": {
            pt: (v["htdev2_vs_I"] or {}).get("delta")
            for pt, v in summary["points"].items()
        },
        "pick": None if summary["pick"] is None else summary["pick"]["point"],
        "formal": {
            k: (
                None
                if v is None
                else {kk: v.get(kk) for kk in ("v3", "T", "H", "public231")}
            )
            for k, v in summary["formal"].items()
        },
    }
    print(json.dumps(compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
