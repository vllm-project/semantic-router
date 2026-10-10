"""Per-benchmark linear correction from local skill to the board's public skill.

For each benchmark, reference models give pairs (local skill, official public skill). ``fit_benchmark``
fits ``official = a + b * local`` (``mode="linear"``), a pure offset (``"offset"``, b = 1) or the
identity (``"identity"``, for exact and validated rebuilds), and reports leave-one-out residuals:
each model is predicted from a fit on the others. ``fit_all`` does this for every benchmark and the
leave-one-out error of the corrected public score, which is the public reproduction error σ_P that
the gate's margin uses. Skills are unclipped; the floor applies only when aggregating.

    python -m d25.omni.suite.offsets --pairs pairs.json [--modes modes.json] [--apply local.json]

``pairs.json``: ``{benchmark: {model: [local, official]}}``; ``modes.json``: ``{benchmark: mode}``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from d25.omni.suite import score

MODES = ("identity", "offset", "linear")
MIN_POINTS = {"identity": 1, "offset": 2, "linear": 3}


def _solve(xs: Sequence[float], ys: Sequence[float], mode: str) -> tuple[float, float]:
    n = len(xs)
    if mode == "identity" or n == 0:
        return 0.0, 1.0
    if mode == "offset" or n == 1:
        return sum(y - x for x, y in zip(xs, ys)) / n, 1.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my - mx, 1.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def fit_benchmark(pairs: Mapping[str, Sequence[float]], mode: str = "linear") -> dict:
    """Fit one benchmark; ``pairs`` maps model -> (local skill, official skill)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    models = sorted(pairs)
    if len(models) < MIN_POINTS[mode]:
        raise ValueError(
            f"{mode} fit needs at least {MIN_POINTS[mode]} models, got {len(models)}"
        )
    xs = [float(pairs[m][0]) for m in models]
    ys = [float(pairs[m][1]) for m in models]
    a, b = _solve(xs, ys, mode)
    residuals = {m: y - (a + b * x) for m, x, y in zip(models, xs, ys)}
    loo = {}
    for i, m in enumerate(models):
        rest = [j for j in range(len(models)) if j != i]
        if len(rest) < MIN_POINTS[mode] and mode != "identity":
            continue
        ai, bi = _solve([xs[j] for j in rest], [ys[j] for j in rest], mode)
        loo[m] = ys[i] - (ai + bi * xs[i])
    rms = lambda d: (
        math.sqrt(sum(v * v for v in d.values()) / len(d)) if d else None
    )  # noqa: E731
    my = sum(ys) / len(ys)
    sst = sum((y - my) ** 2 for y in ys)
    return {
        "mode": mode,
        "a": a,
        "b": b,
        "n": len(models),
        "residuals": residuals,
        "rmse": rms(residuals),
        "loo_residuals": loo,
        "loo_rmse": rms(loo),
        "r2": 1 - sum(r * r for r in residuals.values()) / sst if sst > 0 else None,
    }


def apply(fits: Mapping[str, Mapping], local: Mapping[str, float]) -> dict[str, float]:
    """Corrected per-benchmark skills; benchmarks without a fit pass through unchanged."""
    out = {}
    for bench, value in local.items():
        f = fits.get(bench)
        out[bench] = value if f is None else f["a"] + f["b"] * value
    return out


def fit_all(
    pairs: Mapping[str, Mapping[str, Sequence[float]]],
    modes: Mapping[str, str] | None = None,
    official_public: Mapping[str, float] | None = None,
) -> dict:
    """Fits for every benchmark plus the leave-one-out error of the corrected public score.

    ``official_public`` (model -> published public score) is optional; without it the target is the
    public score recomputed from the official per-benchmark skills, which matches the board to 0.01.
    """
    modes = modes or {}
    fits = {b: fit_benchmark(p, modes.get(b, "linear")) for b, p in pairs.items()}
    models = sorted({m for p in pairs.values() for m in p})
    public = {}
    for m in models:
        if not all(m in pairs.get(b, {}) for b in score.BENCHMARKS):
            continue
        loo_skills, official = {}, {}
        for b in score.BENCHMARKS:
            others = {k: v for k, v in pairs[b].items() if k != m}
            mode = modes.get(b, "linear")
            if len(others) < MIN_POINTS[mode]:
                mode = "offset" if len(others) >= 1 else "identity"
            f = fit_benchmark(others, mode) if others else {"a": 0.0, "b": 1.0}
            loo_skills[b] = f["a"] + f["b"] * pairs[b][m][0]
            official[b] = pairs[b][m][1]
        target = (official_public or {}).get(m, score.public_score(official))
        raw = score.public_score({b: pairs[b][m][0] for b in score.BENCHMARKS})
        public[m] = {
            "official": target,
            "raw": raw,
            "loo": score.public_score(loo_skills),
            "raw_error": raw - target,
            "loo_error": score.public_score(loo_skills) - target,
        }
    errors = [v["loo_error"] for v in public.values()]
    return {
        "benchmarks": fits,
        "public": public,
        "public_loo_rmse": (
            math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
        ),
        "public_loo_max_abs": max((abs(e) for e in errors), default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--modes")
    parser.add_argument("--apply", help="JSON {benchmark: local skill} of a new model")
    args = parser.parse_args()
    pairs = json.loads(Path(args.pairs).read_text())
    modes = json.loads(Path(args.modes).read_text()) if args.modes else None
    report = fit_all(pairs, modes)
    if args.apply:
        local = json.loads(Path(args.apply).read_text())
        corrected = apply(report["benchmarks"], local)
        report["applied"] = {
            "corrected": corrected,
            "public": score.public_score(corrected),
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
