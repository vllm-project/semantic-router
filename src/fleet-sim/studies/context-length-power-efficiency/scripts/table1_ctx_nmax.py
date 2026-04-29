"""
table1_ctx_nmax.py — reproduce Table 1 of paper-f.

Table 1: n_max and tok/W vs. context window for Llama-3.1-70B
         on H100-SXM5 and B200-SXM (TP=8, fp16).

Usage:
    python scripts/table1_ctx_nmax.py
"""

from _sim_path import add_sim_to_syspath

SIM_ROOT = add_sim_to_syspath()

from profiles import B200_POWER_MODE, B200_PROFILE_QUALITY, B200_PROFILE, H100_PROFILE

CONTEXT_WINDOWS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

print("Table 1: n_max and tok/W vs. context window (Llama-3.1-70B, TP=8, fp16)")
print(f"\n{'ctx_K':>6}  "
      f"{'H100 nmax':>10}  {'H100 P(W)':>10}  {'H100 tok/W':>11}  "
      f"{'B200 nmax':>10}  {'B200 P(W)':>10}  {'B200 tok/W':>11}  "
      f"{'ratio':>6}")
print("-" * 90)

for ctx in CONTEXT_WINDOWS:
    mean = ctx // 2

    ns_h  = H100_PROFILE.n_slots(ctx)
    ph    = H100_PROFILE.power_at_concurrency(ns_h)
    tpwh  = ns_h / H100_PROFILE.iter_latency(ns_h, float(mean)) / ph

    ns_b  = B200_PROFILE.n_slots(ctx)
    pb    = B200_PROFILE.power_at_concurrency(ns_b)
    tpwb  = ns_b / B200_PROFILE.iter_latency(ns_b, float(mean)) / pb

    print(f"{ctx//1024:>5}K  "
          f"{ns_h:>10}  {ph:>9.0f}W  {tpwh:>11.2f}  "
          f"{ns_b:>10}  {pb:>9.0f}W  {tpwb:>11.2f}  "
          f"{tpwb/tpwh:>6.2f}×")

print("\nNote: The 1/W law — tok/W halves each time context window doubles.")
print("      Both H100 and B200 follow the same slope; B200 is ~1.5–1.75× higher.")
print(f"      B200 power mode: {B200_POWER_MODE} ({B200_PROFILE_QUALITY}).")
