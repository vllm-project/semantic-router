"""Decoder M5 results table per artifact (prereg dec-m5-prereg-2026-09-29.md, "Goal and bar", "Selection", "Formal").

Three clearly separated column groups per arm artifact, plus the outcome category:

* DEVELOPMENT (selection panels, never release scores): typed-DEV C/N/S, T_dev, H_pilot (median), P (proxy v2),
  P_mean3, P vs the N4XF soup, MLX-DEV Noul-ML / Choice-ML / Score-ML / M_dev and their paired differences vs the
  N4XF soup and Nox 1.0 -- read from the paths m5-select.py uses (``soup/<ARM>/readout.json``,
  ``mlxdev/readouts/<name>/{score,vs-n4xf-soup,vs-nox1}.json`` under ``--dev-root``).
* POST-KEY (same-panel, node B, sealed and scored on node A; ``--formal-root/<run>``): v3, T, H, the paired v3
  difference vs the N4XF node-B reference [95% CI] with its T and H axes (H = the human-transfer difference),
  typed-FINAL C/N/S (floor 75% of the adopted Nox 1.0's), top-answer share per type (collapse >= 0.95), public 231
  easy / standard / hard, and paired v3 vs Nox 1.0 and Decider 4B.
* DIAGNOSTIC (mlx-diag, scored only after the v3 report; ``<run>-mlx/mlx-agg.json``): overall, non-English
  Choice / Noul / Score, ko, ja, non-English predicted-yes rate and gold-No recall (means over languages).

Outcome ("Goal and bar"): successor = v3 CI lower bound > 0 vs the reference AND CSS15 H difference >= 0 AND
typed-FINAL each >= 75% of Nox 1.0 with no type collapsed; card-only multilingual fix = v3 CI includes 0 with point
>= -1.0 AND mlx-diag non-English Noul >= .764 AND non-English Choice and Score each >= the node-B reference's -.01;
anything else negative (pending while an input is missing).

usage: python3 m5-results.py [--dev-root D] [--formal-root F] [--nox1-report R] --output PREFIX [ARM[=soup|sN] ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

TYPES = ("choice", "noul", "score")
SEEDS = ("s1", "s2", "s3")
REF_RUN = "m5-ref-N4XF-soup"
FLOOR_RATIO = 0.75
COLLAPSE_SHARE = 0.95
CARD_NOUL = 0.764
CARD_MARGIN = 0.01
CARD_V3_POINT = -1.0
MLX_KEYS = ("noul_ml", "choice_ml", "score_ml", "m_dev")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def pick_artifact(arms: dict, explicit: str | None = None) -> str:
    """Selection 1: the soup if its R >= the seed mean of R, else the median seed by R."""
    if explicit:
        return explicit
    seeds = [s for s in SEEDS if s in arms]
    values = {s: arms[s]["proxy"] for s in seeds}
    if arms["soup"]["proxy"] >= statistics.mean(values.values()):
        return "soup"
    return sorted(seeds, key=lambda s: (values[s], s))[len(seeds) // 2]


def paired(entry: dict | None) -> dict | None:
    if not entry:
        return None
    return {"diff": entry["diff"], "ci95": entry["ci95"]}


def development(dev_root: Path, arm: str, explicit: str | None) -> dict:
    readout = load(dev_root / "soup" / arm / "readout.json")
    if readout is None:
        return {
            "artifact": explicit or "soup",
            "missing": f"{dev_root}/soup/{arm}/readout.json",
        }
    arms = readout["arms"]
    chosen = pick_artifact(arms, explicit)
    data, ref = arms[chosen], arms["n4xf"]
    cmp = (readout.get("comparisons") or {}).get(f"{chosen}-minus-n4xf")
    proxy_ci = None
    if cmp:
        d = cmp["delta_b_minus_a"]["proxy"]
        proxy_ci = [d["lower95"], d["upper95"]]
    name = f"m5-{arm}-{chosen}"
    mlx = dev_root / "mlxdev" / "readouts" / name
    score = load(mlx / "score.json")
    vs_n4xf, vs_nox1 = load(mlx / "vs-n4xf-soup.json"), load(mlx / "vs-nox1.json")
    return {
        "artifact": chosen,
        "typed": {k: data["by_type"][k]["correct"] for k in TYPES},
        "T_dev": data["T"],
        "H_pilot": data["H"],
        "P": data["proxy"],
        "P_mean3": data["proxy_mean_H"],
        "n4xf_soup": {
            "P": ref["proxy"],
            "P_mean3": ref["proxy_mean_H"],
            "typed": {k: ref["by_type"][k]["correct"] for k in TYPES},
        },
        "P_minus_n4xf_soup": data["proxy"] - ref["proxy"],
        "P_minus_n4xf_soup_ci95": proxy_ci,
        "mlxdev": (
            {
                k: score[k]
                for k in (
                    *MLX_KEYS,
                    "noul_pred_yes_rate_macro",
                    "noul_gold_no_recall_macro",
                )
            }
            if score
            else None
        ),
        "mlxdev_vs_n4xf_soup": (
            {k: paired(vs_n4xf["metrics"].get(k)) for k in MLX_KEYS}
            if vs_n4xf
            else None
        ),
        "mlxdev_vs_nox1": (
            {k: paired(vs_nox1["metrics"].get(k)) for k in MLX_KEYS}
            if vs_nox1
            else None
        ),
    }


def answer_category(answer: dict):
    kind = answer.get("type")
    if kind == "choice":
        return kind, answer.get("choice")
    if kind == "noul":
        v = answer.get("noul")
        return kind, None if v is None or v == 0.5 else v > 0.5
    probs = answer.get("probabilities")
    return kind, max(probs, key=probs.get) if probs else answer.get("score")


def top_shares(predictions: Path) -> dict[str, float] | None:
    if not predictions.is_file():
        return None
    by_type: dict[str, Counter] = {}
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            for answer in (json.loads(line).get("answers") or {}).values():
                if isinstance(answer, dict) and answer.get("type") in TYPES:
                    kind, cat = answer_category(answer)
                    by_type.setdefault(kind, Counter())[cat] += 1
    return {k: max(c.values()) / sum(c.values()) for k, c in sorted(by_type.items())}


def pair_summary(p: dict | None) -> dict | None:
    if not p:
        return None
    ax = p["axis_ci95"]
    return {
        "delta": p["point"]["delta"]["score"],
        "ci95": [p["ci95"]["low"], p["ci95"]["high"]],
        "T_delta": p["point"]["delta"]["T"],
        "T_ci95": [ax["T"]["delta"]["low"], ax["T"]["delta"]["high"]],
        "H_delta": p["point"]["delta"]["H"],
        "H_ci95": [ax["H"]["delta"]["low"], ax["H"]["delta"]["high"]],
    }


def post_key(
    run_dir: Path, nox1_typed: dict | None, bar: str = "n4xf-ref"
) -> dict | None:
    rep = load(run_dir / "REPORT.json")
    if rep is None:
        return None
    typed = {k: rep["panels"]["typed-final"]["by_type"][k]["correct"] for k in TYPES}
    public = rep["panels"]["public231"]
    shares = top_shares(run_dir / "output" / "typed-final.predictions.jsonl")
    floors = {k: FLOOR_RATIO * nox1_typed[k] for k in TYPES} if nox1_typed else None
    return {
        "run": run_dir.name,
        "v3": rep["v3"]["score"],
        "T": rep["v3"]["T"],
        "H": rep["v3"]["H"],
        "typed_final": typed,
        "typed_floor_75pct_nox1": floors,
        "typed_floor_ok": all(typed[k] >= floors[k] for k in TYPES) if floors else None,
        "top_answer_share": shares,
        "collapsed": (
            sorted(k for k, v in (shares or {}).items() if v >= COLLAPSE_SHARE)
            if shares
            else None
        ),
        "public231": {
            "correct": public["correct"],
            "items": public["items"],
            **{t: public["tiers"][t]["correct"] for t in ("easy", "standard", "hard")},
        },
        "vs_bar": pair_summary(load(run_dir / f"PAIRED-vs-{bar}.json")),
        "vs_nox1": pair_summary(load(run_dir / "PAIRED-vs-adopted-1.0.json")),
        "vs_decider4b": pair_summary(load(run_dir / "PAIRED-vs-decider4b.json")),
    }


def diagnostic(agg: dict | None) -> dict | None:
    if agg is None:
        return None
    ne = agg["non_english_noul"]
    lang = agg["per_language_mean_accuracy"]
    return {
        "overall": agg["overall"],
        "non_english": agg["non_english_by_type"],
        "ko": lang.get("ko"),
        "ja": lang.get("ja"),
        "non_english_pred_yes": ne["pred_yes_rate_mean"],
        "non_english_gold_no_recall": ne["gold_no_recall_mean"],
    }


def outcome(
    post: dict | None, diag: dict | None, ref_diag: dict | None
) -> tuple[str, list[str]]:
    if post is None or post["vs_bar"] is None:
        return "pending (post-key)", [
            "no sealed report / comparison vs the N4XF node-B reference"
        ]
    vs = post["vs_bar"]
    lo, hi = vs["ci95"]
    checks = {
        f"v3 CI lower {lo:+.2f} > 0": lo > 0,
        f"human transfer H diff {vs['H_delta']:+.4f} >= 0": vs["H_delta"] >= 0,
        "typed-FINAL >= 75% of Nox 1.0 per type": bool(post["typed_floor_ok"]),
        "no type collapsed": post["collapsed"] == [],
    }
    if all(checks.values()):
        return "successor", list(checks)
    reasons = [f"not successor: {k}" for k, ok in checks.items() if not ok]
    if not (lo <= 0 <= hi and vs["delta"] >= CARD_V3_POINT):
        return "negative", reasons + [
            f"v3 {vs['delta']:+.2f} [{lo:+.2f}, {hi:+.2f}] outside the card-only band"
        ]
    if diag is None or ref_diag is None:
        return "pending (mlx-diag)", reasons + [
            "v3 within the card-only band; mlx-diag of artifact or reference missing"
        ]
    ne, rne = diag["non_english"], ref_diag["non_english"]
    card = {
        f"non-English Noul {ne['noul']:.3f} >= {CARD_NOUL}": ne["noul"] >= CARD_NOUL,
        f"non-English Choice {ne['choice']:.3f} >= ref {rne['choice']:.3f} - {CARD_MARGIN}": ne[
            "choice"
        ]
        >= rne["choice"] - CARD_MARGIN,
        f"non-English Score {ne['score']:.3f} >= ref {rne['score']:.3f} - {CARD_MARGIN}": ne[
            "score"
        ]
        >= rne["score"] - CARD_MARGIN,
    }
    if all(card.values()):
        return "card-only multilingual fix", reasons + list(card)
    return "negative", reasons + [
        f"not card-only: {k}" for k, ok in card.items() if not ok
    ]


def reference(formal: Path, nox1_typed: dict | None) -> dict | None:
    run = formal / REF_RUN
    post = post_key(run, nox1_typed, bar="n4xf-nodeA")
    if post is None:
        return None
    repeat = load(run / "REPEAT-vs-n4xf-nodeA.json")
    changes = (
        {k: v["category_changes"] for k, v in repeat["panels"].items()}
        if repeat
        else None
    )
    post["answer_changes_vs_nodeA"] = changes
    post["answer_changes_vs_nodeA_total"] = sum(changes.values()) if changes else None
    post["mlx_diag"] = diagnostic(load(formal / f"{REF_RUN}-mlx" / "mlx-agg.json"))
    return post


def build(dev_root: Path, formal: Path, nox1_report: Path, arms: list[str]) -> dict:
    nox = load(nox1_report)
    nox1_typed = (
        {k: nox["panels"]["typed-final"]["by_type"][k]["correct"] for k in TYPES}
        if nox
        else None
    )
    ref = reference(formal, nox1_typed)
    ref_diag = ref["mlx_diag"] if ref else None
    rows = []
    for spec in arms:
        arm, _, explicit = spec.partition("=")
        dev = development(dev_root, arm, explicit or None)
        name = f"m5-{arm}-{dev['artifact']}"
        post = post_key(formal / name, nox1_typed)
        diag = diagnostic(load(formal / f"{name}-mlx" / "mlx-agg.json"))
        category, reasons = outcome(post, diag, ref_diag)
        rows.append(
            {
                "arm": arm,
                "name": name,
                "development": dev,
                "post_key": post,
                "diagnostic": diag,
                "outcome": category,
                "outcome_reasons": reasons,
            }
        )
    return {
        "schema": "dec-m5-results/1",
        "rule": "dec-m5-prereg-2026-09-29.md Goal and bar / Selection / Formal runs",
        "labels": {
            "development": "development readouts (typed DEV, CSS pilot, MLX-DEV); selection only, never release scores",
            "post_key": "post-key same-panel (JevArena v3 typed FINAL + CSS15, public 231), node B collection, node A scoring",
            "diagnostic": "mlx-diag diagnostic, read only after the v3 report; drives no Milestone 5 decision",
        },
        "nox1_typed_final": nox1_typed,
        "reference": ref,
        "rows": rows,
    }


def f(v, d=3, sign=False) -> str:
    if v is None:
        return "–"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def ci(p: dict | None, key: str = "diff", cikey: str = "ci95", d: int = 3) -> str:
    if not p or p.get(key) is None:
        return "–"
    c = p.get(cikey)
    return (
        f"{p[key]:+.{d}f} [{c[0]:+.{d}f}, {c[1]:+.{d}f}]" if c else f"{p[key]:+.{d}f}"
    )


def markdown(result: dict) -> str:
    L = result["labels"]
    out = ["# Decoder M5 results per artifact\n"]
    ref = result["reference"]
    if ref:
        out.append(
            f"Reference (N4XF soup, node B `{REF_RUN}`): v3 {f(ref['v3'], 3)}; vs node-A formal N4XF "
            f"{ci(ref['vs_bar'], 'delta', 'ci95', 2)}, differing answers {ref['answer_changes_vs_nodeA_total']} "
            f"({ref['answer_changes_vs_nodeA']}); vs Nox 1.0 {ci(ref['vs_nox1'], 'delta', 'ci95', 2)}.\n"
        )
    else:
        out.append(f"Reference `{REF_RUN}`: not scored yet.\n")
    out.append(f"\n## DEVELOPMENT — {L['development']}\n")
    out.append(
        "| arm | artifact | typed-DEV C/N/S | T_dev | H_pilot | P | P_mean3 | P − N4XF soup [95% CI] | Noul-ML | Choice-ML | "
        "Score-ML | M_dev | Noul-ML Δ vs N4XF soup [CI] | M_dev Δ vs N4XF soup [CI] | Noul-ML Δ vs Nox 1.0 [CI] |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in result["rows"]:
        d = r["development"]
        if "missing" in d:
            out.append(
                f"| {r['arm']} | {d['artifact']} | missing {d['missing']} |"
                + " |" * 12
                + "\n"
            )
            continue
        m = d["mlxdev"] or {}
        vn, vx = d["mlxdev_vs_n4xf_soup"] or {}, d["mlxdev_vs_nox1"] or {}
        pci = d["P_minus_n4xf_soup_ci95"]
        pdiff = f"{d['P_minus_n4xf_soup']:+.2f}" + (
            f" [{pci[0]:+.2f}, {pci[1]:+.2f}]" if pci else ""
        )
        out.append(
            f"| {r['arm']} | {d['artifact']} | {'/'.join(str(d['typed'][k]) for k in TYPES)} | {f(d['T_dev'], 4)} | "
            f"{f(d['H_pilot'], 4)} | {f(d['P'], 2)} | {f(d['P_mean3'], 2)} | {pdiff} | {f(m.get('noul_ml'), 4)} | "
            f"{f(m.get('choice_ml'), 4)} | {f(m.get('score_ml'), 4)} | {f(m.get('m_dev'), 4)} | "
            f"{ci(vn.get('noul_ml'), d=4)} | {ci(vn.get('m_dev'), d=4)} | {ci(vx.get('noul_ml'), d=4)} |\n"
        )
    out.append(f"\n## POST-KEY — {L['post_key']}\n")
    out.append(
        "| arm | run | v3 | Δv3 vs N4XF ref [95% CI] | T | ΔT [CI] | H | human transfer ΔH [CI] | typed-FINAL C/N/S | "
        "≥ 75% Nox 1.0 | top-answer share C/N/S | public 231 (E/S/H) | Δv3 vs Nox 1.0 [CI] | Δv3 vs Decider 4B [CI] |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in result["rows"]:
        p = r["post_key"]
        if p is None:
            out.append(f"| {r['arm']} | {r['name']} | pending |" + " |" * 11 + "\n")
            continue
        sh = p["top_answer_share"] or {}
        pub = p["public231"]
        out.append(
            f"| {r['arm']} | {p['run']} | {f(p['v3'], 3)} | {ci(p['vs_bar'], 'delta', 'ci95', 2)} | {f(p['T'], 4)} | "
            f"{ci(p['vs_bar'], 'T_delta', 'T_ci95', 4)} | {f(p['H'], 4)} | {ci(p['vs_bar'], 'H_delta', 'H_ci95', 4)} | "
            f"{'/'.join(str(p['typed_final'][k]) for k in TYPES)} | {'yes' if p['typed_floor_ok'] else 'no'} | "
            f"{'/'.join(f(sh.get(k), 2) for k in TYPES)} | {pub['correct']} ({pub['easy']}/{pub['standard']}/{pub['hard']}) | "
            f"{ci(p['vs_nox1'], 'delta', 'ci95', 2)} | {ci(p['vs_decider4b'], 'delta', 'ci95', 2)} |\n"
        )
    out.append(f"\n## DIAGNOSTIC — {L['diagnostic']}\n")
    out.append(
        "| arm | overall | non-en Choice | non-en Noul | non-en Score | ko | ja | non-en predicted-yes | non-en gold-No recall |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    diag_rows = [(r["arm"], r["diagnostic"]) for r in result["rows"]]
    if ref:
        diag_rows.insert(0, ("N4XF ref", ref["mlx_diag"]))
    for arm, g in diag_rows:
        if g is None:
            out.append(f"| {arm} | pending |" + " |" * 7 + "\n")
            continue
        ne = g["non_english"]
        out.append(
            f"| {arm} | {f(g['overall'])} | {f(ne['choice'])} | {f(ne['noul'])} | {f(ne['score'])} | {f(g['ko'])} | "
            f"{f(g['ja'])} | {f(g['non_english_pred_yes'])} | {f(g['non_english_gold_no_recall'])} |\n"
        )
    out.append(
        '\n## Outcome (prereg "Goal and bar")\n\n| arm | outcome | checks |\n|---|---|---|\n'
    )
    for r in result["rows"]:
        out.append(
            f"| {r['arm']} | **{r['outcome']}** | {'; '.join(r['outcome_reasons'])} |\n"
        )
    out.append(
        "\nMLX-DEV Noul is answerability / relevance, not paraphrase, and in-distribution for arms that train more on "
        "these sub-arms. Collapse = one answer category on >= 95% of a type's typed-FINAL slots.\n"
    )
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("arms", nargs="*", default=["N5N", "N5B", "N5BN"])
    p.add_argument("--dev-root", type=Path, default=Path("/data/dev2/runs/dec/m5"))
    p.add_argument(
        "--formal-root", type=Path, default=Path("/data/dev2/runs/dec/formal/m5")
    )
    p.add_argument(
        "--nox1-report",
        type=Path,
        default=Path("/data/dev2/runs/eval/m1-adopt/nox1/REPORT.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="prefix; writes <prefix>.json and <prefix>.md",
    )
    args = p.parse_args()
    result = build(args.dev_root, args.formal_root, args.nox1_report, args.arms)
    Path(f"{args.output}.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    md = markdown(result)
    Path(f"{args.output}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
