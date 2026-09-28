# T2 early arms E1: Sol 2B and Nox 4B controls plus own-Lux soft targets

Status: **pre-registered before launch** (2026-09-28). Development evidence
only. These are early 2B/4B runs on node B GPU0–2, which were otherwise idle;
they are launched before the 0.8B C1 readout, so they do not claim the matrix
v1 scale-up criterion (pass at T1 first). They provide the same-size controls
every later 2B/4B factor needs and test the factor with the strongest
size-independent rationale: own Lux 1.0 dominates both Sol and Nox on the
same-panel post-key v3 (Lux 66.27 vs Sol 45.58 and Nox 56.47, eval-track
baselines).

## Frozen configuration

Everything is C1 as amended (prereg `1a5241981`, amendments `d8b3f9137` and
`268568ad8`) except the start weights and node:

| Item | Value |
| --- | --- |
| Starts | own `Decision-1.0-Sol-2B@ce0c018a28de16d6639b1cd203b761bf643b89e6` (1,883,930,944 decision parameters; weights byte-identical to `0665a411`) and `Decision-1.0-Nox-4B@cde2a68dbaa557ea65dc458104d410a0802ee259` (4,208,383,488; weights byte-identical to `0bb83350`); both Hub-hash verified |
| Data / budget / optimizer / objective / selection / calibration / readout | identical to C1: rights-clean v2 (`61740be4…`, `32a4352d…`, `3e34f6cb…`), 466 LoRA r16 updates, LoRA LR 5e-5, head LR 2.5e-5, CE + 0.5 Brier, 8 evenly spaced SELECT checkpoints with earliest tie-break, CAL per-type temperatures, one calibrated typed DEV + CSS pilot readout |
| Runtime | node B image `sha256:ce895822…45f2fb` (package trees byte-identical to node A's `f83b1d10…` per the eval track), code `a4c6f00004af766438bb027115835431fac78e69` |
| GPUs | node B GPU0–2 (track allocation since 10:40 UTC+8) |

| Arm | Start | Factor |
| --- | --- | --- |
| E1-S0 | Sol 1.0 | control |
| E1-S2 | Sol 1.0 | + 0.5 × KL(Lux 1.0, T 2.00544) on all TRAIN rows (the same Lux label file `752b7c8f…` as C1-A2b) |
| E1-X0 | Nox 1.0 | control |
| E1-X2 | Nox 1.0 | + 0.5 × KL(Lux 1.0) as E1-S2 |

The Lux labels were produced by the C1 teacher job (TRAIN agreement Choice
3263/3908, Noul 2709/3031, Score 447/516) and pass the one-sided teacher
criterion of amendment 2; they depend only on TRAIN rows and the shared
renderer, not on the student.

## Preflight, stop and decision rules

Each arm runs zero-step and one-step smokes and must pass every
`dec-arm-preflight/2` gate before its full run; a failed gate stops the arm.
Stop on nonfinite loss/gradient or a projected exclusive time above 3.0
GPU-hours per arm. Same-runtime Sol 1.0 / Nox 1.0 dev controls on node B:
Sol proxy 43.48 (T .5919, H .3194), Nox 52.48 (T .6681, H .4123).

Factor effect = E1-S2 − E1-S0 and E1-X2 − E1-X0 on the dev proxy, 10,000-draw
paired bootstrap, confirmed only if Δ ≥ +1.0 with lower 95% bound > 0 (σ_seed
taken from the 0.8B N0 pair and disclosed as borrowed), retention floors as
matrix v1. A 2B or 4B arm becomes a formal candidate only if its dev proxy is at
least its same-runtime 1.0 control + 3.0 with a positive paired lower bound and
all answers valid; it is then frozen in a committed lock and scored once on
post-key v3 + public231 with the eval track's frozen runner against the adopted
Sol 1.0 (45.580 / 161) or Nox 1.0 (56.470 / 173) baselines.
