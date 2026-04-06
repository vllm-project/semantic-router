"""Tokens-per-watt analysis helpers for fleet-sim optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import _calibrate, _min_gpus_analytical, _p99_wait


@dataclass
class TpwPoint:
    """Energy-efficiency snapshot for one pool at one utilisation level."""

    gpu_name: str
    n_gpus: int
    rho: float
    n_active: float
    power_per_gpu_w: float
    tokens_per_watt: float
    cost_per_mil: float
    p99_ttft_ms: float
    power_model_qual: str
    pool_label: str = ""
    slo_optimal: bool = False


@dataclass
class FleetTpwResult:
    """Fleet-level tokens-per-watt result for a multi-pool routing topology."""

    topology: str
    pools: list[TpwPoint]
    fleet_tpw: float
    fleet_cost_per_mil: float
    fleet_power_kw: float
    total_gpus: int
    worst_p99_ms: float
    power_model_notes: list[str]


def _power_model_quality(gpu) -> str:
    """Return HIGH / FAIR / LOW from the profile's own docstring labels."""
    if gpu.power_logistic_k <= 0.0 or gpu.power_idle_w <= 0.0:
        return "NONE"
    name = getattr(gpu, "name", "").upper()
    if "H100" in name or "H200" in name:
        return "HIGH"
    if "A100" in name:
        return "FAIR"
    return "LOW"


def _cdf_mean_l_out(cdf: list) -> int:
    """Return mean output-token count from a (possibly sub-)CDF."""
    total_prob = 0.0
    mean_total = 0.0
    prev_t, prev_f = 0, 0.0
    for thresh, frac in cdf:
        width = thresh - prev_t
        prob = frac - prev_f
        mean_total += (prev_t + width / 2.0) * prob
        total_prob += prob
        prev_t, prev_f = thresh, frac
    if total_prob > 0:
        mean_total /= total_prob
    return max(1, int(0.20 * mean_total))


def _split_cdf(full_cdf: list, b_short: int) -> tuple:
    """Split a full CDF into (short_sub_cdf, long_sub_cdf, alpha)."""
    alpha = 0.0
    for i, (t, f) in enumerate(full_cdf):
        if t >= b_short:
            if i == 0:
                alpha = f * (b_short / max(1, t))
            else:
                t0, f0 = full_cdf[i - 1]
                if t == t0:
                    alpha = f
                else:
                    alpha = f0 + (f - f0) * (b_short - t0) / (t - t0)
            break
    else:
        alpha = full_cdf[-1][1]

    alpha = max(1e-6, min(1.0 - 1e-6, alpha))

    short_cdf: list = []
    long_cdf: list = []
    for thresh, frac in full_cdf:
        if thresh <= b_short:
            short_cdf.append((thresh, frac / alpha))
        else:
            norm_f = max(0.0, (frac - alpha) / (1.0 - alpha))
            long_cdf.append((thresh, norm_f))

    if not short_cdf or short_cdf[-1][0] < b_short:
        short_cdf.append((b_short, 1.0))
    else:
        short_cdf[-1] = (short_cdf[-1][0], 1.0)

    if long_cdf:
        long_cdf[-1] = (long_cdf[-1][0], 1.0)
    else:
        long_cdf = [(b_short + 1, 1.0)]

    return short_cdf, long_cdf, alpha


def _tpw_one_pool(
    cdf: list,
    lam: float,
    gpu,
    t_slo_ms: float,
    max_ctx: int,
    pool_label: str = "",
    rho_override: float | None = None,
) -> TpwPoint:
    """Compute a single TpwPoint for one pool."""
    mu_gpu, cv2, n_slots, mean_pref = _calibrate(cdf, max_ctx, gpu)
    t_slo_s = t_slo_ms / 1000.0
    qual = _power_model_quality(gpu)
    mean_l_out = _cdf_mean_l_out(cdf)

    if rho_override is None:
        n_gpus = _min_gpus_analytical(lam, mu_gpu, t_slo_s, cv2, n_slots=n_slots)
        slo_optimal = True
    else:
        rho_c = max(0.05, min(0.95, rho_override))
        n_gpus = max(1, int(math.ceil(lam / (mu_gpu * rho_c))))
        slo_optimal = False

    rho = lam / (n_gpus * mu_gpu)
    n_active = rho * n_slots
    p99_ms = (
        _p99_wait(n_gpus * n_slots, lam, mu_gpu / n_slots, cv2) + mean_pref
    ) * 1000.0

    try:
        pw = gpu.power_at_concurrency(max(1, int(n_active)))
    except ValueError:
        pw = float("nan")

    tpw = mu_gpu * rho * mean_l_out / pw if (pw and pw > 0) else float("nan")
    cost = (n_gpus * gpu.cost_per_hr / 3600.0) / (lam * mean_l_out) * 1e6

    return TpwPoint(
        gpu_name=gpu.name,
        pool_label=pool_label or gpu.name,
        n_gpus=n_gpus,
        rho=rho,
        n_active=n_active,
        power_per_gpu_w=pw,
        tokens_per_watt=tpw,
        cost_per_mil=cost,
        p99_ttft_ms=p99_ms,
        power_model_qual=qual,
        slo_optimal=slo_optimal,
    )


def tpw_analysis(
    cdf: list,
    lam: float,
    gpus: list,
    t_slo_ms: float = 500.0,
    max_ctx: int = 8192,
    rho_sweep: list[float] | None = None,
) -> list[TpwPoint]:
    """Compute tokens-per-watt for a list of GPU profiles."""
    points: list[TpwPoint] = []

    for gpu in gpus:
        points.append(_tpw_one_pool(cdf, lam, gpu, t_slo_ms, max_ctx))
        if rho_sweep:
            for rho in rho_sweep:
                points.append(
                    _tpw_one_pool(cdf, lam, gpu, t_slo_ms, max_ctx, rho_override=rho)
                )

    return points


def fleet_tpw_analysis(
    pools: list,
    lam_total: float,
    t_slo_ms: float = 500.0,
    topology: str = "",
) -> FleetTpwResult:
    """Compute fleet-level tokens-per-watt across a multi-pool topology."""
    pool_points: list[TpwPoint] = []
    for spec in pools:
        pool_points.append(
            _tpw_one_pool(
                cdf=spec["cdf"],
                lam=spec["lam"],
                gpu=spec["gpu"],
                t_slo_ms=t_slo_ms,
                max_ctx=spec.get("max_ctx", 8192),
                pool_label=spec.get("label", spec["gpu"].name),
            )
        )

    total_output_tps = 0.0
    total_fleet_power_w = 0.0
    total_cost_per_s = 0.0
    total_gpus = 0
    worst_p99 = 0.0
    notes: list[str] = []

    for spec, pt in zip(pools, pool_points):
        mean_l_out = _cdf_mean_l_out(spec["cdf"])
        output_tps_pool = spec["lam"] * mean_l_out
        pool_power_w = pt.n_gpus * pt.power_per_gpu_w

        total_output_tps += output_tps_pool
        total_fleet_power_w += pool_power_w
        total_cost_per_s += pt.n_gpus * spec["gpu"].cost_per_hr / 3600.0
        total_gpus += pt.n_gpus
        worst_p99 = max(worst_p99, pt.p99_ttft_ms)

        if pt.power_model_qual == "LOW":
            notes.append(
                f"  Pool '{pt.pool_label}' ({pt.gpu_name}): LOW power-model quality "
                f"— projection only, no published batch-vs-power measurements."
            )

    fleet_tpw = (
        total_output_tps / total_fleet_power_w
        if total_fleet_power_w > 0
        else float("nan")
    )
    fleet_cost_per_mil = (
        total_cost_per_s / total_output_tps * 1e6
        if total_output_tps > 0
        else float("nan")
    )

    if not topology:
        topology = " + ".join(p.pool_label for p in pool_points)

    return FleetTpwResult(
        topology=topology,
        pools=pool_points,
        fleet_tpw=fleet_tpw,
        fleet_cost_per_mil=fleet_cost_per_mil,
        fleet_power_kw=total_fleet_power_w / 1000.0,
        total_gpus=total_gpus,
        worst_p99_ms=worst_p99,
        power_model_notes=notes,
    )


def print_tpw_table(points: list[TpwPoint], title: str = "") -> None:
    """Print a formatted tokens-per-watt comparison table."""
    if title:
        print(f"\n{title}")
    hdr = (
        f"  {'Pool / GPU':22s}  {'GPUs':>5}  {'ρ':>5}  {'n_act':>6}  "
        f"{'Tok/W':>6}  {'P/GPU (W)':>9}  {'$/1M tok':>9}  "
        f"{'P99 (ms)':>9}  {'PwrQ':>4}  {'SLO':>3}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for point in points:
        if not point.slo_optimal:
            continue
        label = point.pool_label or point.gpu_name
        flag = "★" if point.slo_optimal else " "
        qual_tag = point.power_model_qual
        if point.power_model_qual == "LOW":
            qual_tag += "*"
        print(
            f"  {label:22s}  {point.n_gpus:>5}  {point.rho:>5.2f}  "
            f"{point.n_active:>6.1f}  {point.tokens_per_watt:>6.2f}  "
            f"{point.power_per_gpu_w:>9.1f}  ${point.cost_per_mil:>8.2f}  "
            f"{point.p99_ttft_ms:>9.1f}  {qual_tag:>4}  {flag:>3}"
        )
    print(sep)
    if any(point.power_model_qual == "LOW" for point in points if point.slo_optimal):
        print(
            "  * LOW quality power model: projection only, no published measurements."
        )
        print("    Tok/W estimate for this GPU has high uncertainty.")
    print()


def print_fleet_tpw(result: FleetTpwResult, title: str = "") -> None:
    """Print a formatted fleet-level tokens-per-watt comparison."""
    if title:
        print(f"\n{title}")
    print(f"\n  Topology : {result.topology}")
    print(f"  Total GPUs     : {result.total_gpus}")
    print(f"  Fleet power    : {result.fleet_power_kw:.1f} kW")
    print(f"  Worst P99 TTFT : {result.worst_p99_ms:.1f} ms")
    print(
        f"  Fleet tok/W    : {result.fleet_tpw:.3f}  tok/J  ← correct multi-pool aggregate"
    )
    print(f"  Fleet $/1M tok : ${result.fleet_cost_per_mil:.2f}")

    hdr = (
        f"\n  {'Pool':22s}  {'GPU':18s}  {'GPUs':>5}  {'ρ':>5}  "
        f"{'Tok/W(pool)':>12}  {'P99(ms)':>9}  {'PwrQ':>4}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for point in result.pools:
        label = point.pool_label or point.gpu_name
        qual_tag = point.power_model_qual + (
            "*" if point.power_model_qual == "LOW" else ""
        )
        print(
            f"  {label:22s}  {point.gpu_name:18s}  {point.n_gpus:>5}  {point.rho:>5.2f}  "
            f"{point.tokens_per_watt:>12.3f}  {point.p99_ttft_ms:>9.1f}  {qual_tag:>4}"
        )
    print()
    if result.power_model_notes:
        print("  Power model caveats:")
        for note in result.power_model_notes:
            print(note)
        print()
