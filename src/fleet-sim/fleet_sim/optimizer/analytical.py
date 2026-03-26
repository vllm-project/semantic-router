import math

from ..gpu_profiles import GpuProfile

# ── CDF utilities ─────────────────────────────────────────────────────────────


def cdf_eval(cdf: list, t: int) -> float:
    """Evaluate CDF at token length t using linear interpolation."""
    if t <= 0:
        return 0.0
    prev_t, prev_f = 0, 0.0
    for thresh, frac in cdf:
        if t <= thresh:
            # Interpolate between (prev_t, prev_f) and (thresh, frac)
            if thresh == prev_t:
                return frac
            return prev_f + (frac - prev_f) * (t - prev_t) / (thresh - prev_t)
        prev_t, prev_f = thresh, frac
    return 1.0


# ── Analytical sizing (Erlang-C / Kimura) ────────────────────────────────────


def erlang_c(c: int, a: float) -> float:
    """Numerically stable Erlang-C P(W_q > 0)."""
    if c <= 0 or a <= 0:
        return 0.0
    rho = a / c
    if rho >= 1.0:
        return 1.0

    log_sum = 0.0
    for k in range(c):
        log_sum_k = k * math.log(a) - math.lgamma(k + 1)
        if k == 0:
            log_sum = log_sum_k
        else:
            mx = max(log_sum, log_sum_k)
            log_sum = mx + math.log(math.exp(log_sum - mx) + math.exp(log_sum_k - mx))
    log_last = c * math.log(a) - math.lgamma(c + 1) - math.log(1 - rho)
    mx = max(log_sum, log_last)
    log_denom = mx + math.log(math.exp(log_sum - mx) + math.exp(log_last - mx))
    return math.exp(log_last - log_denom)


def p99_wait(c: int, lam: float, mu: float, cv2: float = 1.0) -> float:
    """Kimura (1994) M/G/c P99 waiting time (s)."""
    if c <= 0 or lam <= 0 or mu <= 0:
        return float("inf")
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return float("inf")
    C = erlang_c(c, a)
    if C <= 0.01:
        return 0.0  # P(wait > 0) so small P99 wait ≈ 0
    decay = 2 * (c * mu - lam) / max(1e-9, 1 + cv2)
    if decay <= 0:
        return math.inf
    return math.log(C / 0.01) / decay


def calibrate(
    cdf: list, pool_max: int, gpu: GpuProfile, lo_clamp: int = 1
) -> tuple[float, float, int, float]:
    """Estimate (mu_gpu, cv2, n_slots) from CDF for a pool handled by gpu.

    Samples requests from the CDF slice, computes the raw service time
    (seq-len-aware) for each, and returns:
      mu_gpu  : GPU-level throughput = n_slots / E[service_time]  (req/s per GPU)
      cv2     : coefficient of variation squared of service_time
      n_slots : KV-cache concurrency of this GPU at pool_max context

    The caller should use n_slots to compute the correct Erlang-C server count:
    c_slots = n_gpus * n_slots, with per-slot rate mu_slot = mu_gpu / n_slots.

    lo_clamp : minimum token length to sample (pass gamma*B_short+1 for the
               long pool so short-side lengths are excluded from calibration).
    """
    import random

    rng = random.Random(42)
    n_slots = gpu.n_slots(pool_max)
    raw_samples = []
    prefill_samples = []
    for _ in range(3000):
        u = rng.random()
        prev = 0
        for thresh, frac in cdf:
            if u <= frac:
                lo = max(lo_clamp, prev + 1)
                hi = min(pool_max, thresh)
                length = rng.randint(lo, hi) if lo <= hi else pool_max
                l_in = max(1, int(length * 0.80))
                l_out = max(1, length - l_in)
                s_raw = gpu.service_time(l_in, l_out, pool_max)
                raw_samples.append(s_raw)
                # Prefill: ceil(l_in / chunk) iterations at single-sequence iter_t
                pref = math.ceil(l_in / gpu.chunk) * gpu.iter_latency(1)
                prefill_samples.append(pref)
                break
            prev = thresh
    if not raw_samples:
        return float(n_slots), 1.0, n_slots, 0.0
    n = len(raw_samples)
    e1 = sum(raw_samples) / n
    e2 = sum(s * s for s in raw_samples) / n
    cv2 = max(0.01, e2 / (e1 * e1) - 1.0)
    mu_gpu = n_slots / e1  # GPU-level throughput (n_slots parallel requests)
    mean_prefill_s = (
        sum(prefill_samples) / len(prefill_samples) if prefill_samples else 0.0
    )
    return mu_gpu, cv2, n_slots, mean_prefill_s


def min_gpus_analytical(
    lam: float,
    mu: float,
    t_slo: float,
    cv2: float = 1.0,
    rho_max: float = 0.85,
    n_slots: int = 1,
) -> int:
    """Minimum GPU count such that P99 wait ≤ t_slo AND utilisation ≤ rho_max.

    Each GPU provides n_slots concurrent KV-cache slots. The Erlang-C model
    uses c_slots = n_gpus * n_slots servers each at rate mu_slot = mu / n_slots.
    This correctly captures KV-slot-level queuing: a request waits only until
    any slot is free, not until an entire GPU is free.

    The utilisation cap (default ρ_max=0.85) guards against the Kimura
    approximation becoming inaccurate near ρ→1.
    """
    if lam <= 0:
        return 1
    mu_slot = mu / n_slots  # per-slot service rate
    a = lam * (1.0 / mu_slot)  # Erlang load in slot-units (= n_slots * lam/mu)
    # Minimum from P99 SLO constraint — iterate over GPU counts
    c_slo = max(1, math.ceil(lam / mu) + 1)
    while p99_wait(c_slo * n_slots, lam, mu_slot, cv2) > t_slo:
        c_slo += 1
        if c_slo > 5000:
            return c_slo
    # Minimum from utilisation cap constraint (GPU-level utilisation)
    c_rho = math.ceil(lam / (mu * rho_max)) if rho_max < 1.0 else c_slo
    return max(c_slo, c_rho)
