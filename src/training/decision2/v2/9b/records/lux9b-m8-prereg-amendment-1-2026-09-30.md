# 9B Milestone 8, amendment 1: GPU3–4 only after the coordinator's GPU2 reclaim; KDX dropped (before any GPU job)

Amends the [preregistration](lux9b-m8-prereg-2026-09-30.md) (`0d5534325`). Written 2026-09-30 ~17:55 UTC+8,
before any Milestone 8 GPU job (only the CPU prompt build had run).

## What changed

- **The coordinator's 17:15 note reclaimed node A GPU2 for 27B M5's L128 arm** (a ~12-hour job,
  `d2-27b-M5-L128-s2-full`, running since 09:22Z; "never co-tenant") and assigned **9B M8 node A GPU3–4 only**.
  The launch check found that job on GPU2 (VRAM 48%), so nothing was launched there.
- **The lease check** (`lib.sh lent_ok`) now also reads JSON owner entries (the 27B entry on GPU2 is JSON); any
  status other than idle / released refuses, and an entry without a status counts as busy unless it records a
  finished job. Each job still refuses a GPU with allocated VRAM and writes only `owner.9b-m8`.

## Amendment

- **GPUs:** node A GPU3–4. GPU2 is not used. GPU6–7 still need a further amendment.
- **Chains** (same steps, re-laid for two GPUs):
  - `m8-gpu3`: teacher parity → shard 0 → C-m1 α-1 re-read → (shards 1–2 done) teacher build (CPU) → D1-m1
    (preflights) → D1-m1 α-1 readouts → early D1 → (if D1 continues) D1-m2..m5 → KD1 line.
  - `m8-gpu4`: shard 1 → (parity PASS) shard 2 → (build) D2-m1 (preflights) → readouts → early D2 → D2-m2..m5 →
    KD2 line.
  - Shard 1 starts beside the parity job; if parity fails, M8 stops and shard 1's output is never used.
- **KDX is dropped** (it needed a third GPU to finish inside the milestone): **at most two finalists**, the
  non-dropped picks of KD1 and KD2 in that priority. Everything else (teacher, arms, λ, control reuse and recipe
  check, early rule, α rule, formal, items 1–8, choice among passers, card) is unchanged.
- **Projection ≈ 6.5 GPU-h** (the KDX line and one formal run fewer); caps and stop rules unchanged.
- Code: `lux9b/m8/lib.sh`, `run_gpu.sh`, `chains/m8-g3.sh`, `chains/m8-g4.sh` and `dryrun_test.sh` (all checks
  pass, including the refusal of a JSON "running" owner entry); `chains/m8-g2.sh` removed.
