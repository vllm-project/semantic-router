"""Card facts of the DEV2.0-4B successor m10-4b-LH at T = 1 against the current revision and the card peers.

Node A, host python3, PYTHONPATH = the mirror's src/training/decision2:
  python3 facts_lh.py OUT.json
Reads the derived run (prep_lh.sh derive), the comparator runs and their mlx-diag scores; public 231 is paired
with v2.eval.gates public231 (written under the release inputs' gates/). Aggregates only.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

LH = Path("/data/dev2/runs/release/dev2-4b-lh-t1-derived")
GATES = Path("/data/dev2/runs/release/inputs/dev2-4b-lh/gates")
GOLD = Path("/data/dev2/private/panels/gold/typed-final.gold.jsonl")
RUNS = {
    "lh": (LH, LH.parent / "dev2-4b-lh-t1-derived-mlx/mlx-diag.score.json"),
    "dev2-4b": (
        Path("/data/dev2/runs/release/dev2-4b-t1-derived"),
        Path("/data/dev2/runs/release/dev2-4b-t1-derived-mlx/mlx-diag.score.json"),
    ),
    "adopted-1.0": (
        Path("/data/dev2/runs/eval/m1-adopt/nox1"),
        Path("/data/dev2/runs/eval/m2/mlx/x-nox1/mlx-diag.score.json"),
    ),
    "decider4b": (
        Path("/data/dev2/runs/eval/m1-adopt/decider4b"),
        Path("/data/dev2/runs/eval/m2/mlx/x-decider4b/mlx-diag.score.json"),
    ),
    "jet62": (
        Path("/data/dev2/runs/eval/m2/q5b-jet62"),
        Path("/data/dev2/runs/eval/m2/q5b-jet62/mlx-diag.score.json"),
    ),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def score_levels(run: Path) -> dict:
    gold = {}
    for line in GOLD.open():
        row = json.loads(line)
        for key, q in row["gold"].items():
            if q["type"] == "score":
                gold[(row["id"], key)] = int(q["value"])
    pred, hit = collections.Counter(), collections.Counter()
    true = collections.Counter(gold.values())
    for line in (run / "output/typed-final.predictions.jsonl").open():
        row = json.loads(line)
        for key, a in (row.get("answers") or {}).items():
            if (row["id"], key) not in gold or not a.get("probabilities"):
                continue
            level = int(max(a["probabilities"], key=a["probabilities"].get))
            pred[level] += 1
            hit[level] += level == gold[(row["id"], key)]
    return {
        "predicted": [pred[i] for i in range(5)],
        "gold": [true[i] for i in range(5)],
        "recall": [round(hit[i] / true[i], 3) if true[i] else None for i in range(5)],
    }


def facts(run: Path, mlx: Path) -> dict:
    r = load(run / "REPORT.json")
    tf, css, pub = (r["panels"][k] for k in ("typed-final", "css15", "public231"))
    m = load(mlx)
    lang = {
        t: {k: v.get("accuracy") for k, v in m["by_type"][t]["languages"].items()}
        for t in ("choice", "noul")
    }
    return {
        "v3": r.get("v3", {}).get("score"),
        "T": tf["T"],
        "H": css["H"],
        "types": {
            k: [v["correct"], v["n"], round(v["accuracy"], 3)]
            for k, v in tf["by_type"].items()
        },
        "typed_brier_ece": [round(tf["brier"], 3), round(tf["ece_10"], 3)],
        "families": {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in tf["by_family"].items()
        },
        "css_tasks_f1": {k: round(v["macro_f1"], 3) for k, v in css["tasks"].items()},
        "css_invalid": r["invalid"]["css15"]["invalid_or_missing"],
        "css_long": r["slices"]["css15"]["long"],
        "public231": [
            pub["correct"],
            {k: v.get("correct") for k, v in pub["tiers"].items()},
        ],
        "public231_long": r["slices"]["public231"]["long"],
        "mlx_non_english": {
            t: round(m["by_type"][t]["non_english_mean_accuracy"], 3)
            for t in ("choice", "noul")
        },
        "mlx_english": {
            t: round(m["by_type"][t]["english_accuracy"], 3) for t in ("choice", "noul")
        },
        "mlx_lang": {
            t: {k: round(v, 3) for k, v in d.items() if v is not None}
            for t, d in lang.items()
        },
        "score_levels": score_levels(run),
    }


def main() -> int:
    out = {name: facts(*paths) for name, paths in RUNS.items()}
    paired = {}
    for name in ("dev2-4b", "adopted-1.0", "same-limit-16k", "decider4b", "jet62"):
        p = load(LH / f"PAIRED-vs-{name}.json")
        paired[name] = {
            "v3": [
                round(p["point"]["delta"]["score"], 2),
                round(p["ci95"]["low"], 2),
                round(p["ci95"]["high"], 2),
            ],
            "T": [round(p["point"]["delta"]["T"], 3)]
            + [round(p["axis_ci95"]["T"]["delta"][k], 3) for k in ("low", "high")],
            "H": [round(p["point"]["delta"]["H"], 3)]
            + [round(p["axis_ci95"]["H"]["delta"][k], 3) for k in ("low", "high")],
            "right_v3": round(p["point"]["right"]["score"], 3),
        }
    out["paired"] = paired
    public = {}
    for name in ("adopted-1.0", "decider4b", "jet62"):
        dest = GATES / f"public231-vs-{name}.json"
        if not dest.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "v2.eval.gates",
                    "public231",
                    "--left",
                    str(LH),
                    "--right",
                    str(RUNS[name][0]),
                    "--left-name",
                    "m10-4b-LH (T = 1)",
                    "--right-name",
                    name,
                    "--output",
                    str(dest),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        g = load(dest)
        public[name] = {
            k: g.get(k)
            for k in (
                "left_correct",
                "right_correct",
                "ci95",
                "mcnemar_exact_p",
                "verdict",
            )
        }
    g = load(GATES / "public231-vs-dev2-4b.json")
    public["dev2-4b"] = {
        k: g.get(k)
        for k in ("left_correct", "right_correct", "ci95", "mcnemar_exact_p", "verdict")
    }
    out["public231_paired"] = public
    out["mlx_paired_vs_dev2_4b"] = load(GATES / "mlx-paired-vs-dev2-4b.json")["overall"]
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
