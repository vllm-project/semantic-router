"""Score proxy runs (and public-suite samples) with the kit's own metric code.

S part: per benchmark the kit metric (track scorer, report scorer or added-benchmark scorer, exactly as the
0.3 index uses them), coverage-adjusted, chance-corrected, then the board's same-skill weights. ``chance``
is ``proxy`` (kit rule evaluated on the scored items, pre-registered) or ``kit`` (the kit's public chance
constants, used for the public-sample harness checks).

O part: accuracy per task (choice argmax, noul p >= 0.5, unanswered = wrong), chance = mean 1/options,
O_proxy = 100 * mean skill over tasks.

    python -m d25.vega.eval.proxy.score --build /data/d25/vega/proxy/build --results RUN_DIR [--out scores.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

from d25.vega.eval.proxy.common import NAMES, S_BENCHMARKS, read_jsonl, s_weights

KIT = os.environ.get("DECISION_INDEX_KIT", "/data/d25/shared/decision-index-kit")


def ensure_kit(path: str | None = None) -> None:
    """Make the decision-index kit importable; the scorer reuses its metrics and scoring."""
    try:
        import decision_index  # noqa: F401
    except ImportError:
        p = path or KIT
        if p not in sys.path:
            sys.path.insert(0, p)


ensure_kit()

TRACK = {25, 30, 43, 44, 31, 41, 40, 36, 37, 1, 2, 20, 21, 22, 50, 11, 23}
ADDED = {56, 57, 58, 59, 61, 62, 64}
F1_TRACK = {11, 37, 41, 40}
F1_REPORT = {4, 5, 12, 39, 42}
CLEAN_DROP = {28, 30, 61, 40, 62}
MC_DRAWS = 2000


def clip(x):
    return min(1.0, max(0.0, x))


def skill(raw, chance):
    return clip((raw - chance) / (1 - chance)) if chance < 1 else raw


FILLED = {"probabilities": 0}


def _fill(answer: dict) -> dict:
    """A choice answer without a distribution gets a one-hot one on its chosen key (counted in FILLED)."""
    if answer.get("type") != "noul" and "probabilities" not in answer:
        FILLED["probabilities"] += 1
        return {
            **answer,
            "type": "choice",
            "probabilities": collections.defaultdict(
                float, {answer["choice"]: 1.0} if answer.get("choice") else {}
            ),
        }
    return answer


def load_results(paths) -> dict:
    out = {}
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("results*.jsonl*")) if p.is_dir() else [p]
        for f in files:
            for r in read_jsonl(f):
                if r.get("status") == "ok" or r["run_id"] not in out:
                    if r.get("status") == "ok":
                        ans = (r.get("response") or {}).get("answers") or {}
                        r["response"]["answers"] = {k: _fill(v) for k, v in ans.items()}
                    out[r["run_id"]] = r
    return out


def _complete(rows, results):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["_evaluation"]["group_id"]].append(r)
    ok = lambda r: results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok"
    return [r for rs in groups.values() if all(ok(r) for r in rs) for r in rs]


def _semantic(q, x):
    from decision_index.scoring.metrics import semantic_label

    return semantic_label(q, x)


def _mc_f1(rows, kind, positive=None):
    """Expected (conservative/macro) F1 of uniform random answers over the scored questions."""
    from decision_index.scoring.metrics import conservative_f1, macro_f1

    items = [(q, r["expected"][k]) for r in rows for k, q in r["questions"].items()]
    rg = random.Random(20261009)
    vals = []
    for _ in range(MC_DRAWS):
        if kind == "macro":
            pairs = [
                (_semantic(q, g), _semantic(q, rg.choice(list(q["criteria"]))))
                for q, g in items
            ]
            vals.append(macro_f1(pairs))
        else:
            golds, preds, classes = [], [], set()
            for q, g in items:
                p = rg.choice(list(q["criteria"]))
                if positive:
                    golds.append(g)
                    preds.append(p)
                    classes.update(q["criteria"])
                else:
                    golds.append(_semantic(q, g))
                    preds.append(_semantic(q, p))
                    classes.update(_semantic(q, k) for k in q["criteria"])
            vals.append(conservative_f1(golds, preds, sorted(classes), positive))
    return statistics.mean(vals)


def _kit_chance(n):
    from decision_index.scoring.index import load_data

    spec = load_data("index-0.3.json")
    return spec["chance"][str(n)]["chance"]


def score_benchmark(n: int, rows: list, results: dict, chance: str = "proxy") -> dict:
    from decision_index.scoring import added as KA
    from decision_index.scoring import report as KR
    from decision_index.scoring.index import chance_baselines, static_score

    requests = len(rows)
    answered = sum(
        results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok" for r in rows
    )
    out = {
        "n": n,
        "name": NAMES.get(n, str(n)),
        "requests": requests,
        "answered": answered,
        "cases": len({r["_evaluation"]["group_id"] for r in rows}),
    }
    if n in TRACK:
        res = {
            r["_evaluation"]["run_id"]: {
                "status": results[r["_evaluation"]["run_id"]]["status"],
                "answers": (
                    results[r["_evaluation"]["run_id"]].get("response") or {}
                ).get("answers", {}),
            }
            for r in rows
            if r["_evaluation"]["run_id"] in results
        }
        if chance == "kit":
            base = chance_baselines()
        else:
            base = {}
            if n in (37, 41, 11):
                base[str(n)] = {
                    "primary": {
                        "value": _mc_f1(rows, "conservative"),
                        "monte_carlo_standard_error": 0.0,
                    }
                }
            if n == 40:
                tracks = sorted({r["_evaluation"]["track"] for r in rows})
                base["40"] = {
                    "tracks": {
                        t: {
                            "value": _mc_f1(
                                [r for r in rows if r["_evaluation"]["track"] == t],
                                "conservative",
                                positive="yes",
                            ),
                            "monte_carlo_standard_error": 0.0,
                        }
                        for t in tracks
                    }
                }
        v = static_score(n, rows, res, base)
        from decision_index import constants as C

        head = next((t for t in v["tracks"] if t["track"] == C.HEADLINE.get(n)), None)
        if head:
            out.update(
                raw=head["raw"],
                chance=head["random"],
                skill=skill(head["raw"], head["random"]),
                coverage=head["coverage"],
            )
        else:
            out.update(
                raw=v["raw"],
                chance=statistics.mean(t["random"] for t in v["tracks"]),
                skill=v["skill"],
                coverage=v["coverage"],
            )
        out["tracks"] = {t["track"]: round(t["skill"], 4) for t in v["tracks"]}
        return out
    cov = answered / requests if requests else 1.0
    if n in ADDED:
        rep = KA.report(n, rows, results)
        score = rep["score"] or 0.0
        raw = score * cov
        if chance == "kit":
            c = _kit_chance(n)
        elif n == 59:
            golds = [r["expected"][k] for r in rows for k in KA.scored_keys(n, r)]
            p = sum(g is True for g in golds) / len(golds)
            c = 2 * p / (1 + p)
        else:
            c = KA.chance(n, rows)
        out.update(
            raw=raw, chance=c, skill=skill(raw, c), coverage=cov, metric=rep["metric"]
        )
        return out
    complete = _complete(rows, results)
    v = KR.score(complete, results) if complete else {}
    metrics = {38: ("per-review F1", "review_f1")}
    name, value = KR.primary(n, v, metrics) if v else (None, None)
    raw = (value or 0.0) * cov
    if chance == "kit":
        c = _kit_chance(n)
    elif n in F1_REPORT:
        c = _mc_f1(rows, "macro")
    elif n == 38:
        fake = {
            r["_evaluation"]["run_id"]: {
                "status": "ok",
                "total_wall_ms": 0,
                "response": {
                    "answers": {
                        k: {
                            "type": "choice",
                            "choice": "yes",
                            "probabilities": {"yes": 1.0, "no": 0.0},
                        }
                        for k in r["questions"]
                    }
                },
            }
            for r in rows
        }
        c = KR.score(rows, fake)["review_f1"]
    elif n in (9, 33):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r["_evaluation"]["group_id"]].append(r)
        vals = []
        for rs in groups.values():
            p = 1.0
            for r in rs:
                for q in r["questions"].values():
                    p /= len(q["criteria"]) if q["type"] == "choice" else 2
            vals.append(p)
        c = statistics.mean(vals)
    else:
        c = statistics.mean(
            1 / (len(q["criteria"]) if q["type"] == "choice" else 2)
            for r in rows
            for q in r["questions"].values()
        )
    out.update(raw=raw, chance=c, skill=skill(raw, c), coverage=cov, metric=name)
    return out


def s_aggregate(per: dict, drop=()) -> float | None:
    w = s_weights()
    keep = [
        n for n in w if n in per and n not in drop and per[n].get("skill") is not None
    ]
    if not keep:
        return None
    return 100 * sum(w[n] * per[n]["skill"] for n in keep) / sum(w[n] for n in keep)


def score_o_task(rows, results) -> dict:
    hits, ks = [], []
    for r in rows:
        q = r["questions"]["q"]
        gold = r["expected"]["q"]
        res = results.get(r["_evaluation"]["run_id"], {})
        ok = res.get("status") == "ok"
        k = 2 if q["type"] == "noul" else len(q["criteria"])
        ks.append(1 / k)
        if not ok:
            hits.append(0.0)
            continue
        a = res["response"]["answers"]["q"]
        pred = a["noul"] >= 0.5 if q["type"] == "noul" else a["choice"]
        hits.append(float(pred == gold))
    acc, c = statistics.mean(hits), statistics.mean(ks)
    return {
        "questions": len(rows),
        "answered": sum(
            results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok"
            for r in rows
        ),
        "accuracy": acc,
        "chance": c,
        "skill": skill(acc, c),
        "family": rows[0]["metadata"].get("o_family"),
    }


def load_build(build: str | Path):
    build = Path(build)
    s_rows = {
        int(p.name.split(".")[0]): list(read_jsonl(p))
        for p in sorted((build / "s").glob("*.jsonl.gz"))
    }
    o_rows = {
        p.name.split(".")[0]: list(read_jsonl(p))
        for p in sorted((build / "o").glob("*.jsonl.gz"))
    }
    return s_rows, o_rows


def score_all(build, results, chance="proxy") -> dict:
    s_rows, o_rows = load_build(build)
    per_s = {
        n: score_benchmark(n, rows, results, chance)
        for n, rows in sorted(s_rows.items())
        if n in S_BENCHMARKS
    }
    per_o = {t: score_o_task(rows, results) for t, rows in sorted(o_rows.items())}
    fam = collections.defaultdict(list)
    for t, v in per_o.items():
        fam[v["family"]].append(v["skill"])
    return {
        "S_proxy": s_aggregate(per_s),
        "S_proxy_clean": s_aggregate(per_s, CLEAN_DROP),
        "S_proxy_no_apibank": s_aggregate(per_s, {3}),
        "O_proxy": (
            100 * statistics.mean(v["skill"] for v in per_o.values()) if per_o else None
        ),
        "O_families": {k: 100 * statistics.mean(v) for k, v in fam.items()},
        "filled_probabilities": FILLED["probabilities"],
        "coverage": {
            "s_requests": sum(v["requests"] for v in per_s.values()),
            "s_answered": sum(v["answered"] for v in per_s.values()),
            "o_questions": sum(v["questions"] for v in per_o.values()),
            "o_answered": sum(v["answered"] for v in per_o.values()),
        },
        "s": {
            str(n): {
                k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()
            }
            for n, v in per_s.items()
        },
        "o": {
            t: {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()}
            for t, v in per_o.items()
        },
    }


def public_sample_index(rows_by_n: dict, results: dict) -> dict:
    """Kit-chance public index on a stratified public sample (harness check)."""
    from d25.vega.eval.proxy.common import AREA_WEIGHTS, AREAS, GOLD

    per = {
        n: score_benchmark(n, rows, results, chance="kit")
        for n, rows in rows_by_n.items()
    }
    areas = {}
    for a, ids in AREAS.items():
        ids = [n for n in ids if n in per]
        tot = sum(GOLD.get(n, 1.0) for n in ids)
        areas[a] = (
            sum(GOLD.get(n, 1.0) * per[n]["skill"] for n in ids) / tot if tot else None
        )
    index = (
        100
        * sum(AREA_WEIGHTS[a] * v for a, v in areas.items() if v is not None)
        / sum(AREA_WEIGHTS[a] for a, v in areas.items() if v is not None)
    )
    return {
        "index": index,
        "areas": areas,
        "benchmarks": {str(n): v for n, v in per.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/data/d25/vega/proxy/build")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument(
        "--kit",
        default=None,
        help="decision-index kit checkout (default $DECISION_INDEX_KIT or the shared one)",
    )
    a = ap.parse_args(argv)
    if a.kit and a.kit not in sys.path:
        sys.path.insert(0, a.kit)
    sc = score_all(a.build, load_results(a.results))
    text = json.dumps(sc, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(
        json.dumps(
            {
                k: sc[k]
                for k in (
                    "S_proxy",
                    "S_proxy_clean",
                    "O_proxy",
                    "O_families",
                    "coverage",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
