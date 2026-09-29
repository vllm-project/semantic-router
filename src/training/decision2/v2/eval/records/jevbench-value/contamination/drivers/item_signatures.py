"""Public-231 item-level contamination signatures from stored score files (CPU, stdlib).

Usage (stdin driver): ssh NODE_A "cd /tmp && PYTHONPATH=$S python3 - '<json>'" < item_signatures.py
  json: {"out": DIR, "models": [{"name", "dir", "expected", "group"}]}
Reads DIR_m/scores/{public231,typed-final,css15}.score.json (per_item: tier, correct,
confidence, valid). Invalid answers count as wrong. Per-item outputs stay in DIR (0700);
stdout carries per-model / per-pair aggregates only (no item ids).

(i)   Hard-for-everyone: for model m, hard items that <= 20% of the OTHER models answer
      correctly; observed hits vs Rasch-expected hits (ability / difficulty fitted on all
      231 items x all models, ridge 0.1).
(ii)  Confidence: share of correct answers with confidence >= .95, accuracy among
      answers with confidence >= .95, on public hard vs the same model's typed FINAL and
      CSS15 answers.
(iii) Agreement: per pair, Pearson correlation of Rasch residuals over all 231 items, and
      shared correct answers on rare items (overall rate <= 0.35) vs the Rasch expectation.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

HARD_EVERYONE = 0.20
RARE = 0.35
CONF = 0.95


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def rasch(y, iters=300, ridge=0.1, lr=0.5):
    models, items = len(y), len(y[0])
    th = [0.0] * models
    b = [0.0] * items
    for _ in range(iters):
        gth = [0.0] * models
        gb = [0.0] * items
        hth = [ridge] * models
        hb = [ridge] * items
        for m in range(models):
            for i in range(items):
                p = sig(th[m] - b[i])
                r = y[m][i] - p
                w = p * (1 - p)
                gth[m] += r
                gb[i] -= r
                hth[m] += w
                hb[i] += w
        for m in range(models):
            th[m] += lr * (gth[m] - ridge * th[m]) / hth[m]
        for i in range(items):
            b[i] += lr * (gb[i] - ridge * b[i]) / hb[i]
        mean_b = statistics.mean(b)
        b = [v - mean_b for v in b]
    return th, b


GOLD = "/data/dev2/private/panels/gold/{}.gold.jsonl"
_gold_cache = {}


def _gold(panel):
    if panel not in _gold_cache:
        table = {}
        for line in open(GOLD.format(panel), encoding="utf-8"):
            row = json.loads(line)
            gold = row.get("gold")
            if panel == "typed-final":
                gold = ((gold or {}).get("decision") or {}).get("value")
                if isinstance(gold, int) and not isinstance(gold, bool):
                    gold = str(gold)
            table[row["id"]] = gold if isinstance(gold, str) else None
        _gold_cache[panel] = table
    return _gold_cache[panel]


def _pmax(answer):
    probs = answer.get("probabilities") or {}
    vals = [v for v in probs.values() if isinstance(v, (int, float))]
    return max(vals) if vals else None


_typed_gold = {}


def typed_rows(run_dir):
    """Typed FINAL scored with the program's `benchmark.score.evaluate_answer` (PYTHONPATH=$S)."""
    from benchmark.score import evaluate_answer

    path = Path(run_dir) / "output" / "typed-final.predictions.jsonl"
    if not path.exists():
        return None
    if not _typed_gold:
        for line in open(GOLD.format("typed-final"), encoding="utf-8"):
            row = json.loads(line)
            _typed_gold[row["id"]] = row
    rows = []
    for line in open(path, encoding="utf-8"):
        pred = json.loads(line)
        item = _typed_gold.get(pred["id"])
        if item is None:
            continue
        for key, question in item["questions"].items():
            result = evaluate_answer(
                question, item["gold"][key], (pred.get("answers") or {}).get(key)
            )
            ok = result.get("status") == "ok"
            rows.append(
                {
                    "valid": ok,
                    "correct": bool(ok and result.get("correct")),
                    "confidence": (result.get("probability") or {}).get("confidence"),
                }
            )
    return rows


def panel_rows(run_dir, panel):
    """Single-answer items with a string gold: {correct, confidence (pmax), valid}."""
    if panel == "typed-final":
        return typed_rows(run_dir)
    path = Path(run_dir) / "output" / f"{panel}.predictions.jsonl"
    if not path.exists():
        return None
    gold = _gold(panel)
    rows = []
    for line in open(path, encoding="utf-8"):
        pred = json.loads(line)
        g = gold.get(pred["id"])
        if g is None:
            continue
        answers = pred.get("answers") or {}
        answer = next(iter(answers.values()), None) if len(answers) == 1 else None
        if not isinstance(answer, dict) or not isinstance(answer.get("choice"), str):
            rows.append({"valid": False, "correct": False, "confidence": None})
            continue
        rows.append(
            {
                "valid": True,
                "correct": answer["choice"] == g,
                "confidence": _pmax(answer),
            }
        )
    return rows


def public_pmax(run_dir):
    path = Path(run_dir) / "output" / "public231.predictions.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        pred = json.loads(line)
        answers = pred.get("answers") or {}
        answer = next(iter(answers.values()), None) if len(answers) == 1 else None
        out[pred["id"]] = _pmax(answer) if isinstance(answer, dict) else None
    return out


def conf_profile(per_item, tier=None):
    rows = [r for r in per_item if tier is None or r.get("tier") == tier]
    if not rows:
        return None
    n = len(rows)
    ok = [r for r in rows if r.get("valid", True) and r.get("correct")]
    hi = [
        r for r in rows if r.get("valid", True) and (r.get("confidence") or 0) >= CONF
    ]
    return {
        "n": n,
        "acc": round(len(ok) / n, 4),
        "share_correct_conf95": (
            round(sum((r.get("confidence") or 0) >= CONF for r in ok) / len(ok), 4)
            if ok
            else None
        ),
        "frac_conf95": round(len(hi) / n, 4),
        "acc_given_conf95": (
            round(sum(bool(r.get("correct")) for r in hi) / len(hi), 4) if hi else None
        ),
        "mean_conf": round(
            statistics.mean((r.get("confidence") or 0) for r in rows), 4
        ),
    }


def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def main():
    args = json.loads(sys.argv[1])
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    specs = args["models"]
    pub, typed, css, meta = {}, {}, {}, []
    for s in specs:
        d = load(Path(s["dir"]) / "scores" / "public231.score.json")
        pm = public_pmax(s["dir"])
        pub[s["name"]] = {
            r["id"]: dict(r, confidence=pm.get(r["id"], r.get("confidence")))
            for r in d["per_item"]
        }
        typed[s["name"]] = panel_rows(s["dir"], "typed-final")
        css[s["name"]] = panel_rows(s["dir"], "css15")
        meta.append(
            {
                "name": s["name"],
                "group": s["group"],
                "correct": d["correct"],
                "expected": s["expected"],
                "match": d["correct"] == s["expected"],
                "prompts_sha256": d["prompts_sha256"][:8],
                "invalid": d["items"] - d["valid"],
            }
        )
    names = [s["name"] for s in specs]
    ids = sorted(next(iter(pub.values())))
    assert all(sorted(pub[n]) == ids for n in names)
    tier = {i: pub[names[0]][i]["tier"] for i in ids}
    y = [
        [
            1 if (pub[n][i].get("valid", True) and pub[n][i]["correct"]) else 0
            for i in ids
        ]
        for n in names
    ]
    th, b = rasch(y)
    P = [[sig(th[m] - b[i]) for i in range(len(ids))] for m in range(len(names))]
    M = len(names)
    rate = [sum(y[m][i] for m in range(M)) / M for i in range(len(ids))]

    per_model = []
    for m, n in enumerate(names):
        others = [k for k in range(M) if k != m]
        s_m = [
            i
            for i, iid in enumerate(ids)
            if tier[iid] == "hard"
            and sum(y[k][i] for k in others) / len(others) <= HARD_EVERYONE
        ]
        obs = sum(y[m][i] for i in s_m)
        exp = sum(P[m][i] for i in s_m)
        var = sum(P[m][i] * (1 - P[m][i]) for i in s_m)
        hard_idx = [i for i, iid in enumerate(ids) if tier[iid] == "hard"]
        per_model.append(
            {
                "name": n,
                "theta": round(th[m], 3),
                "public_by_tier": {
                    t: sum(y[m][i] for i, iid in enumerate(ids) if tier[iid] == t)
                    for t in ("easy", "standard", "hard")
                },
                "hard_everyone": {
                    "items": len(s_m),
                    "observed": obs,
                    "rasch_expected": round(exp, 2),
                    "z": round((obs - exp) / math.sqrt(var), 2) if var > 0 else None,
                },
                "hard_rasch_residual": round(
                    sum(y[m][i] - P[m][i] for i in hard_idx), 2
                ),
                "conf_public_hard": conf_profile(list(pub[n].values()), "hard"),
                "conf_public_all": conf_profile(list(pub[n].values())),
                "conf_typed_final": conf_profile(typed[n]) if typed[n] else None,
                "typed_final_scored_accuracy_all": (
                    (
                        load(
                            Path(specs[m]["dir"]) / "scores" / "typed-final.score.json"
                        )
                        or {}
                    ).get("overall")
                    or {}
                ).get("accuracy_all"),
                "conf_css15": conf_profile(css[n]) if css[n] else None,
            }
        )

    resid = [[y[m][i] - P[m][i] for i in range(len(ids))] for m in range(M)]
    rare = [i for i in range(len(ids)) if rate[i] <= RARE]
    pairs = []
    for a in range(M):
        for c in range(a + 1, M):
            shared = sum(y[a][i] * y[c][i] for i in rare)
            exp = sum(P[a][i] * P[c][i] for i in rare)
            pairs.append(
                {
                    "a": names[a],
                    "b": names[c],
                    "resid_corr": round(pearson(resid[a], resid[c]), 4),
                    "rare_shared": shared,
                    "rare_expected": round(exp, 2),
                    "rare_ratio": round(shared / exp, 3) if exp else None,
                }
            )
    corrs = sorted(p["resid_corr"] for p in pairs)
    for p in pairs:
        p["resid_corr_pctile"] = round(
            100 * sum(v <= p["resid_corr"] for v in corrs) / len(corrs), 1
        )

    (out / "item-signatures.private.json").write_text(
        json.dumps({"ids": ids, "names": names, "y": y, "difficulty": b, "rate": rate})
    )
    os.chmod(out / "item-signatures.private.json", 0o600)
    report = {
        "models": meta,
        "per_model": per_model,
        "pairs": pairs,
        "rare_items": len(rare),
        "hard_items": sum(t == "hard" for t in tier.values()),
        "params": {
            "hard_everyone": HARD_EVERYONE,
            "rare": RARE,
            "conf": CONF,
            "ridge": 0.1,
        },
        "resid_corr_all_pairs": {
            "median": statistics.median(corrs),
            "p90": corrs[int(0.9 * len(corrs))],
            "p95": corrs[int(0.95 * len(corrs))],
            "max": corrs[-1],
        },
    }
    (out / "item-signatures.aggregate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


main()
