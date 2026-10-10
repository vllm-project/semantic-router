"""Calibrate local public and proxy skills to the official Vision-board scale on reference models.

    python -m d25.omni.proxy.calibrate --board vision.json --measurements refs.json --out calibration.json

``refs.json`` holds local measurements of reference models, all as unclipped skill x 100::

    {"version": 1,
     "refs": {"<board engine>": {"public": {benchmark: skill}, "proxy": {private set: skill}}},
     "aggregate_refs": {"<id>": {"board": {"pub": float, "priv": float},
                                 "public": {...}, "proxy": {...}}}}

``refs`` are board entrants (or reference rows) with published per-benchmark skills. Aggregate refs
have only published pub/priv (stock bases such as Qwen3.8-27B read through the head-free autojev
path); they never enter a fit and serve as extra held-out checks of V_hat.

Model (pre-registered in ws-proxy/DESIGN.md):

- Public: ``P_hat = sum(w_b * max(0, g_b(p_b))) / 9.75``. ``g_b`` is the identity for benchmarks
  declared exact; otherwise a linear map fitted on refs replaces the identity only if its
  leave-one-out RMSE is at least 10% lower.
- Private: ``Q_hat = mean_b max(0, f_b)`` over the 9 private sets. Per set, ``f_b`` starts from the
  public baseline (official private skill on the corrected local public skill of the same benchmark),
  the proxy map replaces it if its leave-one-out RMSE is lower, and proxy + public replaces that only
  if it is at least 10% lower again with 8 or more training refs.
- ``V_hat = 0.5 P_hat + 0.5 Q_hat``. Errors ``d = V_hat - Full`` are leave-one-out: every map and
  every model choice is refitted without the held-out ref (nested selection).
- Margin: ``max(mean(d) + t(0.90, n-1) * sd(d) * sqrt(1 + 1/n), q90(d), 0)``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from d25.omni.proxy import board as vb

EXACT_PUBLIC_DEFAULT = ("CV-Bench", "Winoground", "R-Bench-M", "MMMU-Pro vision")
IMPROVEMENT = 0.10
MIN_FIT = 4
MIN_TWO_VARIABLE = 8
T90 = {
    1: 3.078,
    2: 1.886,
    3: 1.638,
    4: 1.533,
    5: 1.476,
    6: 1.440,
    7: 1.415,
    8: 1.397,
    9: 1.383,
    10: 1.372,
    11: 1.363,
    12: 1.356,
    13: 1.350,
    14: 1.345,
    15: 1.341,
    16: 1.337,
    17: 1.333,
    18: 1.330,
    19: 1.328,
    20: 1.325,
    21: 1.323,
    22: 1.321,
    23: 1.319,
    24: 1.318,
    25: 1.316,
    26: 1.315,
    27: 1.314,
    28: 1.313,
    29: 1.311,
    30: 1.310,
    40: 1.303,
    60: 1.296,
    120: 1.289,
}
Z90 = 1.2816


def t90(df: int) -> float:
    """One-sided 90% Student t quantile (table, interpolated in 1/df)."""
    if df < 1:
        raise ValueError("need at least two errors for a t bound")
    if df in T90:
        return T90[df]
    keys = sorted(T90)
    if df > keys[-1]:
        lo, hi, vlo, vhi = 1 / keys[-1], 0.0, T90[keys[-1]], Z90
    else:
        upper = min(k for k in keys if k > df)
        lower = max(k for k in keys if k < df)
        lo, hi, vlo, vhi = 1 / lower, 1 / upper, T90[lower], T90[upper]
    x = 1 / df
    return vhi + (vlo - vhi) * (x - hi) / (lo - hi)


def quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def ranks(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    data = np.asarray(values, dtype=float)[order]
    i = 0
    while i < len(data):
        j = i
        while j + 1 < len(data) and data[j + 1] == data[i]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class Linear:
    """``y = intercept + coef . x`` by ordinary least squares (equal ref weights)."""

    inputs: tuple[str, ...]
    intercept: float = 0.0
    coef: tuple[float, ...] = ()
    n: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Linear:
        A = (
            np.column_stack([np.ones(len(y)), X])
            if self.inputs
            else np.ones((len(y), 1))
        )
        solution, *_ = np.linalg.lstsq(A, y, rcond=None)
        return Linear(
            self.inputs,
            float(solution[0]),
            tuple(float(v) for v in solution[1:]),
            len(y),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.full(X.shape[0], self.intercept)
        for j, c in enumerate(self.coef):
            out = out + c * X[:, j]
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "intercept": self.intercept,
            "coef": dict(zip(self.inputs, self.coef)),
            "n": self.n,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Linear:
        inputs = tuple(data["inputs"])
        return Linear(
            inputs,
            float(data["intercept"]),
            tuple(float(data["coef"][k]) for k in inputs),
            int(data["n"]),
        )


@dataclass
class Ref:
    """One reference model: local measurements and, when known, published per-benchmark skills."""

    id: str
    public: dict[str, float]
    proxy: dict[str, float]
    board_public: dict[str, float] = field(default_factory=dict)
    board_private: dict[str, float] = field(default_factory=dict)
    pub: float = float("nan")
    priv: float = float("nan")
    aggregate_only: bool = False

    @property
    def full(self) -> float:
        return vb.full_score(self.pub, self.priv)


def load_refs(board: Mapping[str, Any], measurements: Mapping[str, Any]) -> list[Ref]:
    published = vb.by_engine(board)
    out: list[Ref] = []
    for engine, local in (measurements.get("refs") or {}).items():
        if engine not in published:
            raise KeyError(f"reference {engine!r} is not on the board")
        row = published[engine]
        public, private = vb.per_benchmark(row)
        out.append(
            Ref(
                engine,
                {k: float(v) for k, v in (local.get("public") or {}).items()},
                {k: float(v) for k, v in (local.get("proxy") or {}).items()},
                public,
                private,
                float(row["pub"]),
                float(row["priv"]),
            )
        )
    for name, local in (measurements.get("aggregate_refs") or {}).items():
        out.append(
            Ref(
                name,
                {k: float(v) for k, v in (local.get("public") or {}).items()},
                {k: float(v) for k, v in (local.get("proxy") or {}).items()},
                pub=float(local["board"]["pub"]),
                priv=float(local["board"]["priv"]),
                aggregate_only=True,
            )
        )
    return out


def loo_rmse(X: np.ndarray, y: np.ndarray, template: Linear) -> float:
    n = len(y)
    if n < MIN_FIT:
        return float("inf")
    errors = []
    for i in range(n):
        keep = np.arange(n) != i
        model = template.fit(X[keep], y[keep])
        errors.append(float(model.predict(X[i : i + 1])[0] - y[i]))
    return float(np.sqrt(np.mean(np.square(errors))))


@dataclass
class PublicMap:
    benchmark: str
    kind: str
    model: Linear
    loo: dict[str, float]

    def __call__(self, value: float) -> float:
        if self.kind == "identity":
            return float(value)
        return float(self.model.predict(np.array([[value]]))[0])

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "map": self.model.to_json(), "loo_rmse": self.loo}


def fit_public_map(benchmark: str, refs: Sequence[Ref], exact: bool) -> PublicMap:
    identity = Linear((benchmark,), 0.0, (1.0,), 0)
    usable = [r for r in refs if benchmark in r.public and benchmark in r.board_public]
    if exact or len(usable) < MIN_FIT:
        return PublicMap(benchmark, "identity", identity, {})
    X = np.array([[r.public[benchmark]] for r in usable])
    y = np.array([r.board_public[benchmark] for r in usable])
    identity_rmse = float(np.sqrt(np.mean(np.square(X[:, 0] - y))))
    linear_rmse = loo_rmse(X, y, Linear((benchmark,)))
    loo = {"identity": identity_rmse, "linear": linear_rmse}
    if linear_rmse <= (1 - IMPROVEMENT) * identity_rmse:
        return PublicMap(benchmark, "linear", Linear((benchmark,)).fit(X, y), loo)
    return PublicMap(benchmark, "identity", identity, loo)


def corrected_public(ref: Ref, maps: Mapping[str, PublicMap]) -> dict[str, float]:
    return {b: maps[b](ref.public[b]) for b in vb.PUBLIC if b in ref.public}


PRIVATE_MODELS = ("public", "proxy", "proxy+public")


def features(kind: str, benchmark: str) -> tuple[str, ...]:
    return {
        "mean": (),
        "public": (f"public:{benchmark}",),
        "proxy": (f"proxy:{benchmark}",),
        "proxy+public": (f"proxy:{benchmark}", f"public:{benchmark}"),
    }[kind]


def feature_values(ref: Ref, corrected: Mapping[str, float]) -> dict[str, float]:
    values = {f"public:{b}": v for b, v in corrected.items()}
    values.update({f"proxy:{b}": v for b, v in ref.proxy.items()})
    return values


@dataclass
class PrivateMap:
    """Selected map of one private set, plus the public-baseline map used when an input is missing."""

    benchmark: str
    kind: str
    model: Linear
    loo: dict[str, float]
    ranges: dict[str, tuple[float, float]]
    fallback: Linear

    def usable(self, values: Mapping[str, float]) -> bool:
        return all(name in values for name in self.model.inputs)

    def __call__(self, values: Mapping[str, float]) -> float:
        model = self.model if self.usable(values) else self.fallback
        missing = [name for name in model.inputs if name not in values]
        if missing:
            raise KeyError(f"{self.benchmark}: missing {missing}")
        X = (
            np.array([[values[name] for name in model.inputs]])
            if model.inputs
            else np.zeros((1, 0))
        )
        return float(model.predict(X)[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "map": self.model.to_json(),
            "loo_rmse": self.loo,
            "ranges": {k: list(v) for k, v in self.ranges.items()},
            "fallback": self.fallback.to_json(),
        }


def fit_private_map(
    benchmark: str, refs: Sequence[Ref], values: Mapping[str, Mapping[str, float]]
) -> PrivateMap:
    """Nested-selection fit of one private set on per-benchmark refs."""
    fitted = [r for r in refs if not r.aggregate_only and benchmark in r.board_private]

    def has(r: Ref, kind: str) -> bool:
        return all(n in values[r.id] for n in features(kind, benchmark))

    available = [k for k in PRIVATE_MODELS if sum(has(r, k) for r in fitted) > MIN_FIT]
    common = [r for r in fitted if all(has(r, k) for k in available)]
    if len(common) <= MIN_FIT:
        available = [k for k in available if k == "public"]
        common = [r for r in fitted if all(has(r, k) for k in available)]
    y = np.array([r.board_private[benchmark] for r in common])
    loo: dict[str, float] = {}
    for kind in ["mean", *available]:
        names = features(kind, benchmark)
        X = (
            np.array([[values[r.id][n] for n in names] for r in common])
            if names
            else np.zeros((len(common), 0))
        )
        loo[kind] = loo_rmse(X, y, Linear(names))
    chosen = "public" if "public" in available else "mean"
    if "proxy" in available and loo["proxy"] < loo[chosen]:
        chosen = "proxy"
    if (
        chosen == "proxy"
        and "proxy+public" in available
        and len(common) - 1 >= MIN_TWO_VARIABLE
        and loo["proxy+public"] <= (1 - IMPROVEMENT) * loo["proxy"]
    ):
        chosen = "proxy+public"
    names = features(chosen, benchmark)
    train = [r for r in fitted if all(n in values[r.id] for n in names)]
    X = (
        np.array([[values[r.id][n] for n in names] for r in train])
        if names
        else np.zeros((len(train), 0))
    )
    y_train = np.array([r.board_private[benchmark] for r in train])
    if len(train) == 0:
        raise ValueError(f"no reference has a published {benchmark} private skill")
    model = Linear(names).fit(X, y_train)
    ranges = {
        n: (float(X[:, j].min()), float(X[:, j].max())) for j, n in enumerate(names)
    }
    base = "public" if sum(has(r, "public") for r in fitted) > MIN_FIT else "mean"
    base_refs = [r for r in fitted if has(r, base)]
    base_names = features(base, benchmark)
    Xb = (
        np.array([[values[r.id][n] for n in base_names] for r in base_refs])
        if base_names
        else np.zeros((len(base_refs), 0))
    )
    fallback = Linear(base_names).fit(
        Xb, np.array([r.board_private[benchmark] for r in base_refs])
    )
    return PrivateMap(benchmark, chosen, model, loo, ranges, fallback)


@dataclass
class Calibration:
    public_maps: dict[str, PublicMap]
    private_maps: dict[str, PrivateMap]

    def estimate(
        self, public: Mapping[str, float], proxy: Mapping[str, float]
    ) -> dict[str, Any]:
        corrected = {b: self.public_maps[b](public[b]) for b in vb.PUBLIC}
        values = {f"public:{b}": v for b, v in corrected.items()}
        values.update({f"proxy:{b}": float(v) for b, v in proxy.items()})
        private = {b: self.private_maps[b](values) for b in vb.PRIVATE}
        P = vb.public_score(corrected)
        Q = vb.private_score(private)
        return {
            "P_hat": P,
            "Q_hat": Q,
            "V_hat": vb.full_score(P, Q),
            "public": corrected,
            "private": private,
            "fallbacks": [
                b for b, m in self.private_maps.items() if not m.usable(values)
            ],
            "extrapolation": self.extrapolation(values),
        }

    def extrapolation(
        self, values: Mapping[str, float], slack: float = 2.0
    ) -> list[str]:
        notes = []
        for b, m in self.private_maps.items():
            for name, (lo, hi) in m.ranges.items():
                v = values.get(name)
                if v is not None and not lo - slack <= v <= hi + slack:
                    notes.append(
                        f"{b}: {name}={v:.1f} outside fitted range [{lo:.1f}, {hi:.1f}]"
                    )
        return notes

    def to_json(self) -> dict[str, Any]:
        return {
            "public_maps": {b: m.to_json() for b, m in self.public_maps.items()},
            "private_maps": {b: m.to_json() for b, m in self.private_maps.items()},
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Calibration:
        public = {
            b: PublicMap(b, m["kind"], Linear.from_json(m["map"]), dict(m["loo_rmse"]))
            for b, m in data["public_maps"].items()
        }
        private = {
            b: PrivateMap(
                b,
                m["kind"],
                Linear.from_json(m["map"]),
                dict(m["loo_rmse"]),
                {k: (float(v[0]), float(v[1])) for k, v in m["ranges"].items()},
                Linear.from_json(m["fallback"]),
            )
            for b, m in data["private_maps"].items()
        }
        return Calibration(public, private)


def fit(
    refs: Sequence[Ref], exact_public: Sequence[str] = EXACT_PUBLIC_DEFAULT
) -> Calibration:
    fitted = [r for r in refs if not r.aggregate_only]
    public_maps = {b: fit_public_map(b, fitted, b in exact_public) for b in vb.PUBLIC}
    values = {r.id: feature_values(r, corrected_public(r, public_maps)) for r in fitted}
    private_maps = {b: fit_private_map(b, fitted, values) for b in vb.PRIVATE}
    return Calibration(public_maps, private_maps)


def margin_from_errors(errors: Sequence[float]) -> dict[str, float]:
    d = np.asarray(errors, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("need at least two leave-one-out errors")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    bound_t = mean + t90(n - 1) * sd * math.sqrt(1 + 1 / n)
    bound_q = quantile(d, 0.90)
    return {
        "n": n,
        "bias": mean,
        "sd": sd,
        "rmse": float(np.sqrt(np.mean(d**2))),
        "mae": float(np.mean(np.abs(d))),
        "t90": t90(n - 1),
        "margin_t": bound_t,
        "margin_q90": bound_q,
        "margin": max(bound_t, bound_q, 0.0),
    }


def leave_one_out(
    refs: Sequence[Ref],
    exact_public: Sequence[str] = EXACT_PUBLIC_DEFAULT,
    fitter: Callable[..., Calibration] = fit,
) -> list[dict[str, Any]]:
    """Held-out estimates for every ref: per-benchmark refs refit without themselves; aggregate refs
    use the calibration fitted on all per-benchmark refs."""
    full_fit = fitter(refs, exact_public)
    out = []
    for ref in refs:
        if ref.aggregate_only:
            cal = full_fit
        else:
            cal = fitter([r for r in refs if r.id != ref.id], exact_public)
        est = cal.estimate(ref.public, ref.proxy)
        record = {
            "id": ref.id,
            "aggregate_only": ref.aggregate_only,
            "P_hat": est["P_hat"],
            "Q_hat": est["Q_hat"],
            "V_hat": est["V_hat"],
            "pub": ref.pub,
            "priv": ref.priv,
            "full": ref.full,
            "e_P": est["P_hat"] - ref.pub,
            "e_Q": est["Q_hat"] - ref.priv,
            "d": est["V_hat"] - ref.full,
            "private_kinds": {b: cal.private_maps[b].kind for b in vb.PRIVATE},
            "fallbacks": est["fallbacks"],
        }
        if not ref.aggregate_only:
            record["private_errors"] = {
                b: est["private"][b] - ref.board_private[b] for b in vb.PRIVATE
            }
            record["public_errors"] = {
                b: est["public"][b] - ref.board_public[b]
                for b in vb.PUBLIC
                if b in est["public"]
            }
        out.append(record)
    return out


def summarize(loo: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    d = [r["d"] for r in loo]
    summary = margin_from_errors(d)
    summary["sigma_P"] = float(np.sqrt(np.mean([r["e_P"] ** 2 for r in loo])))
    summary["sigma_Q"] = float(np.sqrt(np.mean([r["e_Q"] ** 2 for r in loo])))
    summary["spearman_full"] = spearman(
        [r["V_hat"] for r in loo], [r["full"] for r in loo]
    )
    summary["spearman_private"] = spearman(
        [r["Q_hat"] for r in loo], [r["priv"] for r in loo]
    )
    per = [r for r in loo if "private_errors" in r]
    summary["private_rmse"] = (
        {
            b: float(np.sqrt(np.mean([r["private_errors"][b] ** 2 for r in per])))
            for b in vb.PRIVATE
        }
        if per
        else {}
    )
    sigma_v = 0.5 * math.sqrt(
        summary["sigma_P"] ** 2 + summary["sigma_Q"] ** 2 + 1.35**2
    )
    summary["margin_components"] = max(1.5, 1.645 * sigma_v)
    return summary


def calibrate(
    board: Mapping[str, Any],
    measurements: Mapping[str, Any],
    exact_public: Sequence[str] = EXACT_PUBLIC_DEFAULT,
) -> dict[str, Any]:
    refs = load_refs(board, measurements)
    per_benchmark = [r for r in refs if not r.aggregate_only]
    if len(per_benchmark) < MIN_FIT + 1:
        raise ValueError(
            f"need at least {MIN_FIT + 1} per-benchmark refs, got {len(per_benchmark)}"
        )
    cal = fit(refs, exact_public)
    loo = leave_one_out(refs, exact_public)
    return {
        "version": 1,
        "board": {
            "generated_utc": board.get("generated_utc"),
            "edition": board.get("edition"),
        },
        "exact_public": list(exact_public),
        "refs": [r.id for r in per_benchmark],
        "aggregate_refs": [r.id for r in refs if r.aggregate_only],
        **cal.to_json(),
        "loo": loo,
        "summary": summarize(loo),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--board", default=None, help="vision.json path or URL (default: live Space)"
    )
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exact-public", default=",".join(EXACT_PUBLIC_DEFAULT))
    args = parser.parse_args(argv)
    board = vb.load(args.board)
    measurements = json.loads(Path(args.measurements).read_text())
    exact = [b for b in args.exact_public.split(",") if b]
    unknown = sorted(set(exact) - set(vb.PUBLIC))
    if unknown:
        raise SystemExit(f"unknown benchmarks in --exact-public: {unknown}")
    result = calibrate(board, measurements, exact)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]
    print(
        f"refs {len(result['refs'])} (+{len(result['aggregate_refs'])} aggregate) | LOO d: bias {s['bias']:+.2f} "
        f"sd {s['sd']:.2f} rmse {s['rmse']:.2f} | margin {s['margin']:.2f} "
        f"(t {s['margin_t']:.2f}, q90 {s['margin_q90']:.2f}) | spearman {s['spearman_full']:.3f}"
    )


if __name__ == "__main__":
    main()
