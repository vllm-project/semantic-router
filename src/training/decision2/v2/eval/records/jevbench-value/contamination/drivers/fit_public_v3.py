"""Public-231 vs post-key v3 residuals (and the C1 cross-check); aggregates only, runs locally.

Usage: python3 fit_public_v3.py SIGNATURES_AGGREGATE_JSON > fit.json
Totals / v3 / C1 are the post-key same-panel numbers from the brief (Lex from
m5-overlap-effects). Hard-tier counts come from the item-signature aggregate.
OLS with leave-one-out (externally studentized) residuals.
"""

import json
import math
import sys

# name: (public231 total, post-key v3, C1 or None, group)
MODELS = {
    "GLiNER2.5-Decide": (116, 42.52, 22.82, "peer"),
    "Bosun-0.6B": (133, 38.52, 35.03, "peer"),
    "Kai1": (114, 35.94, 17.77, "d1"),
    "Lex1": (113, 31.02, 20.82, "d1"),
    "Eos1": (142, 42.55, 37.94, "d1"),
    "JPT-0.8B": (171, 40.09, 39.01, "peer"),
    "Intern": (164, 43.54, 34.58, "peer"),
    "Kev": (147, 43.22, 39.06, "peer"),
    "Decider-2B": (175, 49.50, None, "peer"),
    "Sol1": (161, 45.58, None, "d1"),
    "This-That": (147, 46.11, None, "peer"),
    "Bosun-1.7B": (151, 42.12, None, "peer"),
    "Decider-4B": (192, 61.88, None, "peer"),
    "Nox1": (173, 56.47, None, "d1"),
    "JPT-4B": (203, 54.56, None, "peer"),
    "Jet-v6.2": (174, 60.38, None, "peer"),
    "Hopper-G": (194, 58.40, None, "peer"),
    "Lux1": (183, 65.81, None, "d1"),
    "JPT-9B": (197, 60.99, None, "peer"),
    "Nimble-v2": (185, 62.06, None, "peer"),
    "AutoJev-27B": (201, 72.13, None, "peer"),
    "Eikos-27B": (212, 69.29, None, "peer"),
    "Jebadiah-27B": (176, 65.47, None, "peer"),
    "DEV2.0-0.6B": (142, 43.54, 33.21, "ours"),
    "DEV2.0-0.8B": (156, 50.24, 40.24, "ours"),
    "DEV2.0-2B": (171, 53.44, None, "ours"),
    "DEV2.0-4B": (171, 63.15, None, "ours"),
    "DEV2.0-8B": (178, 67.74, None, "ours"),
    "DEV2.0-27B": (198, 67.21, None, "ours"),
}


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / (n - 2)
    lev = [1 / n + (x - mx) ** 2 / sxx for x in xs]
    stud = []
    for r, h in zip(res, lev):
        s2_i = ((n - 2) * s2 - r * r / (1 - h)) / (n - 3)
        stud.append(r / math.sqrt(s2_i * (1 - h)))
    r2 = 1 - sum(r * r for r in res) / sum((y - my) ** 2 for y in ys)
    return {"a": a, "b": b, "sigma": math.sqrt(s2), "r2": r2, "res": res, "stud": stud}


def main():
    sig = json.load(open(sys.argv[1]))
    hard = {m["name"]: m["public_by_tier"]["hard"] for m in sig["per_model"]}
    names = list(MODELS)
    v3 = [MODELS[n][1] for n in names]
    out = {"n": len(names), "fits": {}}
    for label, ys in (
        ("public_total", [MODELS[n][0] for n in names]),
        ("public_hard", [hard[n] for n in names]),
    ):
        f = ols(v3, ys)
        out["fits"][label] = {
            "intercept": round(f["a"], 2),
            "slope": round(f["b"], 3),
            "sigma_items": round(f["sigma"], 2),
            "r2": round(f["r2"], 3),
            "residuals": {
                n: {"items": round(r, 1), "stud": round(s, 2), "group": MODELS[n][3]}
                for n, r, s in zip(names, f["res"], f["stud"])
            },
        }
    c1n = [n for n in names if MODELS[n][2] is not None]
    fc = ols([MODELS[n][1] for n in c1n], [MODELS[n][2] for n in c1n])
    fp = ols([MODELS[n][0] for n in c1n], [MODELS[n][2] for n in c1n])
    out["c1"] = {
        "models": len(c1n),
        "c1_on_v3": {
            "slope": round(fc["b"], 3),
            "r2": round(fc["r2"], 3),
            "residuals": {n: round(r, 2) for n, r in zip(c1n, fc["res"])},
        },
        "c1_on_public": {
            "slope": round(fp["b"], 3),
            "r2": round(fp["r2"], 3),
            "residuals": {n: round(r, 2) for n, r in zip(c1n, fp["res"])},
        },
    }
    print(json.dumps(out, indent=1))


main()
