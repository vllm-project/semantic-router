"""Item-level gap analysis for JevBench public 231 (node A, CPU, stdlib only).

Usage (from the worktree):
  ssh nodeA "cd /tmp && PYTHONPATH=$S python3 - '<json args>'" < gap_main.py > gap_raw.json

Per-item joins are written only under the private gap directory on node A.
Stdout carries aggregates only (no item ids, text, gold or answers).
"""

import collections
import json
import math
import os
import sys

ARGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
RUNS_ROOT = "/data/dev2/runs"
PRIV = "/data/dev2/private/eval/jevbench-value/gap"
GOLD = "/data/dev2/private/panels/gold/public231/targets.jsonl"
PROMPTS = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"
NB = PRIV + "/nodeB"

RUNS = {
    # 4B lineage + peers
    "nox1-8k": "dec/formal/m2/nox1-8k",
    "nox1-16k": "dec/formal/m3/nox1-16k",
    "nox1-adopt": "eval/m1-adopt/nox1",
    "X2": "dec/formal/e1-x2-nodeA",
    "X4R-s1": "dec/formal/m2/m2-X4R-s1-nodeA",
    "X4R-s2": "dec/formal/m2/m2-X4R-s2-nodeA",
    "X4K-s1": "dec/formal/m2/m2-X4K-s1-nodeA",
    "X4K-s2": "dec/formal/m2/m2-X4K-s2-nodeA",
    "N4T": "dec/formal/m3/m3-N4T-soup-nodeA",
    "N4J": "dec/formal/m3/m3-N4J-soup-nodeA",
    "N4L": "dec/formal/m3/m3-N4L-soup-nodeA",
    "N4LKr": "dec/formal/m3/m3-N4LKr-soup-nodeA",
    "N4LX": "dec/formal/m4/m4-N4LX-soup-nodeA",
    "N4XF": "dec/formal/m4/m4-N4XF-soup-nodeA",
    "rel-4b": "release/dev2-4b-t1-derived",
    "N5-ref": "dec/formal/m5/m5-ref-N4XF-soup",
    "N5B": "dec/formal/m5/m5-N5B-soup",
    "N5BN": "dec/formal/m5/m5-N5BN-soup",
    "N5N": "dec/formal/m5/m5-N5N-soup",
    "decider4b": "eval/m1-adopt/decider4b",
    "jpt4b": "eval/m1/p4-jpt4b",
    "jet62": "eval/m2/q5b-jet62",
    "hopperg": "eval/m2/q8-hopperg",
    # 9B lineage + peers
    "lux1-8k": "9b/formal/lux1-8k",
    "lux1-16k-shared": "9b/formal-m3/lux1-16k-shared",
    "lux1-adopt": "eval/m1/d1-lux1-autotune-cache",
    "L2-8k": "9b/formal/l2-8k",
    "B-s1-16k": "9b/formal-m3/B-s1-16k",
    "DW-16k": "9b/formal-m3/DW-16k",
    "K-a13": "9b/formal-m4/K-a13-16k",
    "U-a13": "9b/formal-m4/U-a13-16k",
    "KN-a12": "9b/formal-m4/KN-a12-16k",
    "rel-8b": "release/dev2-8b-t1-derived",
    "nimble2": "eval/m2/q6-nimble2",
    "jpt9b": "eval/m1-adopt/jpt9b",
    # 27B lineage (copied from node B) + peers
    "C0": NB + "/C0",
    "C1": NB + "/C1",
    "S1": NB + "/S1",
    "K1": NB + "/K1",
    "F0": NB + "/F0",
    "F1": NB + "/F1",
    "F2": NB + "/F2",
    "eikos27b": "eval/m4/nodeB-kernel/eikos27b",
    "autojev27": "eval/m4/nodeB-kernel/autojev27",
}

PAIRS = [
    ("N4XF", "nox1-adopt"),
    ("N4XF", "nox1-16k"),
    ("N4XF", "decider4b"),
    ("rel-4b", "nox1-adopt"),
    ("K-a13", "lux1-16k-shared"),
    ("K-a13", "lux1-adopt"),
    ("rel-8b", "lux1-adopt"),
    ("F1", "eikos27b"),
    ("F1", "autojev27"),
    ("F1", "C0"),
    ("jpt4b", "decider4b"),
]


def rpath(p):
    return p if p.startswith("/") else os.path.join(RUNS_ROOT, p)


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5**n for k in range(n + 1)]
    obs = pmf[b]
    return min(1.0, sum(p for p in pmf if p <= obs * (1 + 1e-9)))


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]] two-sided Fisher exact."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def hp(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    lo, hi = max(0, c1 - r2), min(r1, c1)
    obs = hp(a)
    return min(1.0, sum(hp(x) for x in range(lo, hi + 1) if hp(x) <= obs * (1 + 1e-9)))


def paired_ci(b, c, n):
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / n**2
    se = math.sqrt(max(var, 0))
    return {
        "diff_items": b - c,
        "diff_pts": round(100 * d, 2),
        "ci95_items": [round((d - 1.96 * se) * n, 1), round((d + 1.96 * se) * n, 1)],
        "mdd80_items": round((1.96 + 0.84) * se * n, 1),
    }


def length_bin(ch):
    if ch < 200:
        return "a<200"
    if ch < 1000:
        return "b200-1k"
    if ch <= 4000:
        return "c1k-4k"
    return "d>4k"


def nopt_bin(t):
    k = len(t["labels"])
    if t["task_type"] == "noul":
        return "noul2"
    if t["task_type"] == "score":
        return f"score{k}"
    return f"choice{k if k <= 5 else '6+'}"


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def main():
    targets = {t["id"]: t for t in jl(GOLD)}
    prompts = {p["id"]: p for p in jl(PROMPTS)}
    ids = [t["id"] for t in jl(GOLD)]
    feat = {}
    for i in ids:
        t, p = targets[i], prompts[i]
        st = p["state"]
        s_txt = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
        q = p["questions"]["decision"]
        full = (
            len(s_txt)
            + len(q.get("instructions") or "")
            + len(json.dumps(q.get("criteria"), ensure_ascii=False))
        )
        f = {
            "tier": t["tier"],
            "family": t["family"],
            "tf": t["tier"] + ":" + t["family"],
            "type": t["task_type"],
            "nopt": nopt_bin(t),
            "state_fmt": "str" if isinstance(st, str) else "dict",
            "chars_state": len(s_txt),
            "chars_full": full,
            "len_bin": length_bin(len(s_txt)),
            "len_bin_full": length_bin(full),
            "expected": str(t["expected"]),
            "labels": t["labels"],
        }
        if t["task_type"] == "choice":
            crit = q["criteria"]
            lens = {k: len(str(v)) for k, v in crit.items()}
            mx = max(lens.values())
            longest = [k for k, v in lens.items() if v == mx]
            f["longest"] = longest[0] if len(longest) == 1 else None
            f["gold_pos"] = t["labels"].index(str(t["expected"]))
        feat[i] = f

    panel = {
        "len_bin_state_by_tier": collections.Counter(
            f["tier"] + ":" + f["len_bin"] for f in feat.values()
        ),
        "len_bin_full_by_tier": collections.Counter(
            f["tier"] + ":" + f["len_bin_full"] for f in feat.values()
        ),
        "state_fmt_by_tier": collections.Counter(
            f["tier"] + ":" + f["state_fmt"] for f in feat.values()
        ),
        "nopt": collections.Counter(f["tier"] + ":" + f["nopt"] for f in feat.values()),
        "unique_longest_choice": sum(1 for f in feat.values() if f.get("longest")),
        "gold_is_unique_longest": sum(
            1
            for f in feat.values()
            if f.get("longest") and f["longest"] == f["expected"]
        ),
        "hard_families": collections.Counter(
            f["family"] for f in feat.values() if f["tier"] == "hard"
        ),
    }

    per = {}
    runs_out = {}
    for name, p in RUNS.items():
        base = rpath(p)
        try:
            sc = json.load(open(base + "/scores/public231.score.json"))
            preds = {
                r["id"]: r for r in jl(base + "/output/public231.predictions.jsonl")
            }
        except FileNotFoundError as e:
            runs_out[name] = {"missing": str(e).split(":")[0]}
            continue
        rep = {}
        try:
            rep = json.load(open(base + "/REPORT.json"))
        except Exception:
            pass
        items = {r["id"]: r for r in sc["per_item"]}
        rows = {}
        for i in ids:
            r = items[i]
            pr = preds.get(i, {})
            ans = (pr.get("answers") or {}).get("decision") or {}
            row = {
                "valid": bool(r.get("valid")),
                "correct": bool(r.get("correct")),
                "predicted": r.get("predicted"),
                "brier": r.get("brier"),
                "conf": r.get("confidence"),
                "pad": bool(r.get("point_disagrees_with_argmax")),
                "renorm": bool(r.get("renormalized")),
                "strict": r.get("strict_valid"),
                "reason": r.get("reason"),
                "trunc": pr.get("truncated_questions") or 0,
                "in_tok": (pr.get("usage") or {}).get("input_tokens"),
                "out_tok": (pr.get("usage") or {}).get("output_tokens"),
                "adapter_status": pr.get("adapter_status"),
            }
            if feat[i]["type"] == "noul":
                row["p_yes"] = ans.get("noul")
            else:
                probs = ans.get("probabilities")
                if isinstance(probs, dict):
                    row["probs"] = probs
                    row["psum"] = sum(
                        v for v in probs.values() if isinstance(v, (int, float))
                    )
                    sp = sorted(
                        (v for v in probs.values() if isinstance(v, (int, float))),
                        reverse=True,
                    )
                    row["tie_top"] = len(sp) > 1 and sp[0] == sp[1]
                if feat[i]["type"] == "choice":
                    row["point"] = ans.get("choice")
            rows[i] = row
        per[name] = rows
        v3 = rep.get("v3") or {}
        runs_out[name] = {
            "v3": v3.get("score"),
            "T": v3.get("T"),
            "H": v3.get("H"),
            "public_correct": sc.get("correct"),
            "tier_macro": sc.get("tier_macro_accuracy"),
            "brier_valid": sc.get("brier_valid"),
            "ece": sc.get("ece_pmax_15"),
        }
        runs_out[name].update(summarize(rows, feat, ids))

    with open(PRIV + "/per_item_matrix.jsonl", "w") as f:
        for i in ids:
            f.write(
                json.dumps(
                    {
                        "id": i,
                        **{k: v for k, v in feat[i].items() if k != "labels"},
                        "runs": {
                            n: {
                                "c": per[n][i]["correct"],
                                "v": per[n][i]["valid"],
                                "pred": per[n][i]["predicted"],
                                "conf": per[n][i]["conf"],
                            }
                            for n in per
                        },
                    }
                )
                + "\n"
            )

    pairs_out = {}
    disc_priv = {}
    for a, b in PAIRS:
        if a not in per or b not in per:
            pairs_out[f"{a}__vs__{b}"] = {"missing": True}
            continue
        pairs_out[f"{a}__vs__{b}"], disc_priv[f"{a}__vs__{b}"] = pair(
            per[a], per[b], feat, ids
        )
    with open(PRIV + "/pair_discordants.json", "w") as f:
        json.dump(disc_priv, f)
    os.chmod(PRIV + "/per_item_matrix.jsonl", 0o600)
    os.chmod(PRIV + "/pair_discordants.json", 0o600)

    out = {
        "panel": {
            k: dict(v) if isinstance(v, collections.Counter) else v
            for k, v in panel.items()
        },
        "runs": runs_out,
        "pairs": pairs_out,
    }
    json.dump(out, sys.stdout, indent=1, sort_keys=True)


def group_counts(rows, feat, ids, key):
    g = collections.defaultdict(lambda: [0, 0])
    for i in ids:
        g[feat[i][key]][0] += rows[i]["correct"]
        g[feat[i][key]][1] += 1
    return {k: v for k, v in sorted(g.items())}


def summarize(rows, feat, ids):
    s = {}
    for key in ("tier", "type", "tf", "nopt", "len_bin", "len_bin_full", "state_fmt"):
        s["by_" + key] = group_counts(rows, feat, ids, key)
    s["invalid"] = sum(not rows[i]["valid"] for i in ids)
    s["invalid_reasons"] = dict(
        collections.Counter(rows[i]["reason"] for i in ids if not rows[i]["valid"])
    )
    s["renormalized"] = sum(rows[i]["renorm"] for i in ids)
    s["point_ne_argmax"] = sum(rows[i]["pad"] for i in ids)
    s["point_ne_argmax_and_point_would_be_right"] = sum(
        1 for i in ids if rows[i]["pad"] and rows[i].get("point") == feat[i]["expected"]
    )
    s["top_tie"] = sum(1 for i in ids if rows[i].get("tie_top"))
    s["truncated"] = sum(1 for i in ids if rows[i]["trunc"])
    s["adapter_not_ok"] = sum(
        1 for i in ids if rows[i]["adapter_status"] not in (None, "ok")
    )
    ps = [rows[i]["psum"] for i in ids if "psum" in rows[i]]
    s["psum_max_abs_dev"] = round(max(abs(x - 1) for x in ps), 8) if ps else None
    toks = [
        rows[i]["in_tok"] for i in ids if isinstance(rows[i]["in_tok"], (int, float))
    ]
    s["in_tok_max"] = max(toks) if toks else None
    s["out_tok_sum"] = sum(
        rows[i]["out_tok"] or 0
        for i in ids
        if isinstance(rows[i]["out_tok"], (int, float))
    )
    right = [i for i in ids if rows[i]["valid"] and rows[i]["correct"]]
    wrong = [i for i in ids if rows[i]["valid"] and not rows[i]["correct"]]
    s["conf_right"] = mean(rows[i]["conf"] for i in right)
    s["conf_wrong"] = mean(rows[i]["conf"] for i in wrong)
    s["brier_right"] = mean(rows[i]["brier"] for i in right)
    s["brier_wrong"] = mean(rows[i]["brier"] for i in wrong)
    s["wrong_conf_ge_0.9"] = sum(1 for i in wrong if rows[i]["conf"] >= 0.9)
    hw = [i for i in wrong if feat[i]["tier"] == "hard"]
    s["hard_wrong_conf_ge_0.9"] = sum(1 for i in hw if rows[i]["conf"] >= 0.9)
    s["hard_conf_wrong"] = mean(rows[i]["conf"] for i in hw)
    # Noul
    nb = {}
    for tier in ("easy", "standard", "hard"):
        its = [
            i
            for i in ids
            if feat[i]["type"] == "noul"
            and feat[i]["tier"] == tier
            and rows[i]["valid"]
        ]
        if not its:
            continue
        gy = sum(feat[i]["expected"] == "yes" for i in its)
        py = sum(rows[i]["predicted"] == "yes" for i in its)
        acc_y = [rows[i]["correct"] for i in its if feat[i]["expected"] == "yes"]
        acc_n = [rows[i]["correct"] for i in its if feat[i]["expected"] == "no"]
        nb[tier] = {
            "n": len(its),
            "gold_yes": gy,
            "pred_yes": py,
            "mean_p_yes": mean(rows[i]["p_yes"] for i in its),
            "acc_on_gold_yes": f"{sum(acc_y)}/{len(acc_y)}",
            "acc_on_gold_no": f"{sum(acc_n)}/{len(acc_n)}",
        }
    s["noul"] = nb
    # Score
    sc = {}
    for tier in ("standard", "hard"):
        its = [
            i
            for i in ids
            if feat[i]["type"] == "score"
            and feat[i]["tier"] == tier
            and rows[i]["valid"]
        ]
        if not its:
            continue
        levels = sorted({l for i in its for l in feat[i]["labels"]}, key=int)
        mass = {
            l: mean(rows[i]["probs"].get(l, 0) for i in its if "probs" in rows[i])
            for l in levels
        }
        sc[tier] = {
            "n": len(its),
            "nlevels": dict(collections.Counter(len(feat[i]["labels"]) for i in its)),
            "gold_hist": dict(collections.Counter(feat[i]["expected"] for i in its)),
            "pred_hist": dict(collections.Counter(rows[i]["predicted"] for i in its)),
            "mean_mass": mass,
            "correct": sum(rows[i]["correct"] for i in its),
            "mean_abs_level_err": mean(
                abs(int(rows[i]["predicted"]) - int(feat[i]["expected"])) for i in its
            ),
            "off_by_1": sum(
                abs(int(rows[i]["predicted"]) - int(feat[i]["expected"])) == 1
                for i in its
            ),
            "off_by_2plus": sum(
                abs(int(rows[i]["predicted"]) - int(feat[i]["expected"])) >= 2
                for i in its
            ),
            "max_mass_on_one_level_mean": mean(
                max(rows[i]["probs"].values()) for i in its if "probs" in rows[i]
            ),
        }
    s["score"] = sc
    # Choice
    ch = [i for i in ids if feat[i]["type"] == "choice" and rows[i]["valid"]]
    pos_pred = collections.Counter(
        feat[i]["labels"].index(rows[i]["predicted"]) for i in ch
    )
    pos_gold = collections.Counter(feat[i]["gold_pos"] for i in ch)
    lg = [i for i in ch if feat[i].get("longest")]
    g_long = [i for i in lg if feat[i]["longest"] == feat[i]["expected"]]
    g_not = [i for i in lg if feat[i]["longest"] != feat[i]["expected"]]
    s["choice"] = {
        "n": len(ch),
        "correct": sum(rows[i]["correct"] for i in ch),
        "pos_pred": dict(sorted(pos_pred.items())),
        "pos_gold": dict(sorted(pos_gold.items())),
        "first_option_pick_rate": (
            round(pos_pred.get(0, 0) / len(ch), 3) if ch else None
        ),
        "longest_pick": sum(rows[i]["predicted"] == feat[i]["longest"] for i in lg),
        "longest_items": len(lg),
        "acc_gold_longest": f"{sum(rows[i]['correct'] for i in g_long)}/{len(g_long)}",
        "acc_gold_not_longest": f"{sum(rows[i]['correct'] for i in g_not)}/{len(g_not)}",
    }
    return s


def pair(ra, rb, feat, ids):
    n = len(ids)
    b_ids = [i for i in ids if ra[i]["correct"] and not rb[i]["correct"]]
    c_ids = [i for i in ids if rb[i]["correct"] and not ra[i]["correct"]]
    out = {
        "a_correct": sum(ra[i]["correct"] for i in ids),
        "b_correct": sum(rb[i]["correct"] for i in ids),
        "b_candidate_only": len(b_ids),
        "c_other_only": len(c_ids),
        "both_right": sum(ra[i]["correct"] and rb[i]["correct"] for i in ids),
        "both_wrong": sum(not ra[i]["correct"] and not rb[i]["correct"] for i in ids),
        "exact_binom_p": round(binom_two_sided(len(b_ids), len(c_ids)), 5),
    }
    out.update(paired_ci(len(b_ids), len(c_ids), n))
    for key in ("tier", "type", "tf", "nopt", "len_bin", "len_bin_full", "state_fmt"):
        g = collections.defaultdict(lambda: [0, 0, 0, 0])
        for i in ids:
            k = feat[i][key]
            g[k][0] += ra[i]["correct"]
            g[k][1] += rb[i]["correct"]
            g[k][2] += i in b_ids
            g[k][3] += i in c_ids
        res = {}
        for k, (ca, cb, bb, cc) in sorted(g.items()):
            tot = sum(1 for i in ids if feat[i][key] == k)
            res[k] = {
                "n": tot,
                "a": ca,
                "b": cb,
                "b_only": bb,
                "c_only": cc,
                "p_binom": round(binom_two_sided(bb, cc), 4),
            }
        out["by_" + key] = res
    # Fisher: is the hard-tier share of discordants different from non-hard?
    hb = sum(feat[i]["tier"] == "hard" for i in b_ids)
    hc = sum(feat[i]["tier"] == "hard" for i in c_ids)
    out["fisher_hard_vs_rest_discordant"] = round(
        fisher_two_sided(hb, len(b_ids) - hb, hc, len(c_ids) - hc), 4
    )
    # artefact attribution on the candidate's losses
    art = collections.Counter()
    for i in c_ids:
        r = ra[i]
        if not r["valid"]:
            art["invalid"] += 1
        if r["renorm"]:
            art["renormalized"] += 1
        if r["pad"]:
            art["point_ne_argmax"] += 1
            if r.get("point") == feat[i]["expected"]:
                art["point_would_be_right"] += 1
        if r["trunc"]:
            art["truncated"] += 1
        if r.get("tie_top"):
            art["tie_top"] += 1
        if (
            r["valid"]
            and r["conf"] is not None
            and r["conf"] < 0.55
            and feat[i]["type"] != "choice"
        ):
            art["near_coinflip_binary"] += 1
        if (
            feat[i]["type"] == "choice"
            and feat[i].get("longest") == feat[i]["expected"]
            and rb[i]["predicted"] == feat[i]["longest"]
        ):
            art["other_right_via_longest_gold"] += 1
    out["candidate_loss_artefacts"] = dict(art)
    out["conf_on_c_only"] = {
        "candidate": mean(ra[i]["conf"] for i in c_ids if ra[i]["valid"]),
        "other": mean(rb[i]["conf"] for i in c_ids if rb[i]["valid"]),
    }
    out["conf_on_b_only"] = {
        "candidate": mean(ra[i]["conf"] for i in b_ids if ra[i]["valid"]),
        "other": mean(rb[i]["conf"] for i in b_ids if rb[i]["valid"]),
    }
    out["c_only_candidate_conf_ge_0.9"] = sum(
        1 for i in c_ids if (ra[i]["conf"] or 0) >= 0.9
    )
    return out, {"b": b_ids, "c": c_ids}


if __name__ == "__main__":
    main()
