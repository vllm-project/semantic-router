"""Card rows (Full / Public / Same skills / New domains) for a model vs pplx on the paired path, with bootstrap SEs.

Same estimator as ``paired_boot.delta`` (pplx official Full / S / O plus the calibrated paired differences), on the
conservative T8 path and the naive path. A second paired_boot output (the release) run with the same seed and B gives
draw-for-draw paired deltas against the release (pplx cancels).

    python -m d25.vega.eval.proxy.card_rows --calibration CAL --model-dir OUT_MODEL --release-dir OUT_RELEASE \
        --tag _pv5o1 --out rows.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from d25.vega.eval.proxy.calibrate_v2 import OUR_T
from d25.vega.eval.proxy.gate import _fhat

PPLX_OFFICIAL = {"full": 62.75, "S": 61.10, "O": 55.57, "public": 62.25}


def comps(cal, m, p, T, draw=None):
    maps, o_coef, calx = None, None, cal
    if draw:
        o_coef = draw["O_coef"]
        if T:
            maps = draw["maps_T"]
        else:
            calx = dict(cal, final=dict(cal["final"], maps=draw["maps"]))
    fm, sm, om = _fhat(calx, m["public"], m["bench"], m["O_proxy"], T, maps, o_coef)
    fp, sp, op = _fhat(calx, p["public"], p["bench"], p["O_proxy"], T, maps, o_coef)
    return {
        "full": fm - fp,
        "S": sm - sp,
        "O": om - op,
        "public": m["public"] - p["public"],
        "O_proxy": m["O_proxy"] - p["O_proxy"],
        "public_abs": m["public"],
    }


def load(d, tag):
    d = Path(d)
    recs = {
        r["b"]: r
        for r in (
            json.loads(f.read_text()) for f in sorted((d / "boot").glob("b*.json"))
        )
    }
    return (
        json.loads((d / "point.json").read_text()),
        recs,
        json.loads((d / f"cal_draws{tag}.json").read_text()),
    )


def sd(x):
    return statistics.stdev(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--tag", default="_pv5o1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cal = json.loads(Path(a.calibration).read_text())
    pm, rm, dm = load(a.model_dir, a.tag)
    pr, rr, dr = load(a.release_dir, a.tag)
    if json.dumps(dm[:3], sort_keys=True) != json.dumps(dr[:3], sort_keys=True):
        raise SystemExit("calibration draws differ between the two runs")
    common = sorted(set(rm) & set(rr))
    out = {
        "calibration": Path(a.calibration).name,
        "replicates": len(common),
        "pplx_official": PPLX_OFFICIAL,
        "pplx_local_public": pm["pplx"]["public"],
    }
    for path, T in (("conservative_T8", tuple(OUR_T)), ("naive", ())):
        res = {}
        for name, point, recs in (("model", pm, rm), ("release", pr, rr)):
            c0 = comps(cal, point["ours"], point["pplx"], T)
            reps = [
                comps(cal, recs[b]["ours"], recs[b]["pplx"], T, dm[b]) for b in common
            ]
            res[name] = {
                "full": round(PPLX_OFFICIAL["full"] + c0["full"], 3),
                "full_se": round(sd([r["full"] for r in reps]), 3),
                "same_skill": round(PPLX_OFFICIAL["S"] + c0["S"], 3),
                "same_skill_se": round(sd([r["S"] for r in reps]), 3),
                "new_domain": round(PPLX_OFFICIAL["O"] + c0["O"], 3),
                "new_domain_se": round(sd([r["O"] for r in reps]), 3),
                "public": round(c0["public_abs"], 3),
                "public_se": round(sd([r["public_abs"] for r in reps]), 3),
                "d_public_vs_pplx_local": round(c0["public"], 3),
                "d_O_proxy_vs_pplx": round(c0["O_proxy"], 3),
                "dS_hat": round(c0["S"], 3),
                "dO_hat": round(c0["O"], 3),
                "d_full_hat": round(c0["full"], 3),
                "_reps": reps,
            }
        dv = {}
        for k in ("full", "S", "O", "public", "O_proxy"):
            key = {"S": "same_skill", "O": "new_domain"}.get(k, k)
            pt = (
                comps(cal, pm["ours"], pm["pplx"], T)[k]
                - comps(cal, pr["ours"], pr["pplx"], T)[k]
            )
            ds = [
                x[k] - y[k]
                for x, y in zip(res["model"]["_reps"], res["release"]["_reps"])
            ]
            dv[f"d_{key}_vs_release"] = round(pt, 3)
            dv[f"d_{key}_vs_release_se"] = round(sd(ds), 3)
            if k == "full":
                dv["share_of_replicates_model_above_release"] = round(
                    sum(x > 0 for x in ds) / len(ds), 3
                )
        for v in res.values():
            v.pop("_reps")
        out[path] = {**res, **dv}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
