# M3a — AutoJev-27B ROCm runtime qualification: result (2026-09-28)

Preregistration: [`m3a-prereg-2026-09-28.md`](m3a-prereg-2026-09-28.md) (§1–4, committed at `0a4d2f1c5`,
code formatted at `244510ded`) and [amendment 1](m3a-prereg-amendment-1-2026-09-28.md) (`5ea69f0fb`, waves
only). Every process ran from the exact node-A mirror of `5ea69f0fb`. Count-only report:
[`m3a/autojev-qualification.report.json`](m3a/autojev-qualification.report.json).

## Outcome: FAILED on the preregistered frozen-autotune gate (G3) — AutoJev path stopped

| Gate | Result | Evidence |
| --- | --- | --- |
| G1 identity | PASS | All 5 processes: pinned model id, revision `6f5b557e` (attested), adapter `autojev27-native-v1`, backend `autojev-native-eager`, decision config `bacbcbb2…`; image `f83b1d10…`; FLA path (no reference fallback); 26,086,635,760 loaded parameters; package / source tree hashes `d0b1e161…` / `550ccd85…` in every process (the same values as the eval track's node-A AutoJev run). The 58 package files equal the Hub at `6f5b557e` byte for byte; source checkout `ee63c151`, clean. |
| G2 repeat determinism (≤ 1e-3, zero flips) | PASS, bitwise | P1–P2, P1–P3, P2–P3 (three GPUs) and P1–P4 (same GPU, new process): 910 / 910 answers identical, maximum drift **0.0**, zero validity or argmax mismatches |
| G3 frozen autotune | **FAIL** | Warm-up froze 6 autotune entries; the two concurrent round-1 processes P2 (GPU3) and P3 (GPU4) added 3 `l2norm_fwd_kernel` entries (6 → 9); P1 and P4 then added none |
| G4 spot check vs eval node-A predictions (517 questions) | PASS | Validity identical; argmax agreement 99.61% (2 flips, both near-ties: top-2 margins 0.021 / 0.022 and 0.000 / 0.028); p99 drift 0.0149 (bound 0.05), median 0.0, max 0.056. Typed-final 158 / 158 bitwise identical; public 231 91 / 231, CSS15 58 / 128 identical (the eval run autotuned per process without a persisted cache) |

**Cause of the G3 failure.** The FLA `l2norm_fwd_kernel` autotune key is `(D, NB, dtypes)` with
`NB = ceil(rows / 65,536)` and rows = tokens × heads, so its key changes with sequence-length buckets.
The preregistration assumed FLA autotune keys do not depend on sequence length below 65,536 tokens;
that is true of the gated-delta chunk kernels but not of `l2norm`. Round 1 met NB = 2, 3 and 4 for the
first time and autotuned them concurrently on two GPUs. The warm-up set had no long prompt: the pool
holds only 9 RP-v2 prompts with ≥ 4,096 native tokens (maximum 5,251), all drawn into R, so the
preregistered "8 long warm-up prompts" could not be met. This shortfall was visible in the set
manifest before the run and was not amended; it is recorded here as a process error of this track.

**Decision (preregistered failure rule):** the qualification is recorded as failed and the AutoJev
path stops for Milestone 3a: no AutoJev production, no rerun. No AutoJev target was produced or
published. Outputs on the spot-check set stay on node A and were never converted.

**For the coordinator.** The property the coordinator asked for — repeat-run determinism across
processes with a shared persisted autotune cache — held bitwise, and the spot check passed; what
failed is this track's stricter production-safety gate (every kernel choice frozen before
production). If AutoJev targets are still wanted, a new, separately preregistered qualification
would warm every `NB` bucket up to the 8,192-token limit (for example a length ladder of synthetic or
TRAIN prompts), then require G2 and G3 unchanged; about 0.2 GPU-h on node A, then about 3.5 GPU-h of
production (8–10 prompts/s per GPU). That is a coordinator decision.

## Sets and processes

- W: 48 RP-v2 TRAIN prompts (16 per type, 0 long; file `f9d958e8…`). R: 393 RP-v2 TRAIN prompts (128 per
  type + the 9 long ones; ids `9a80d202…`). S: 231 public 231 + 128 typed-final + 128 CSS15 items from
  the sealed gold-free prompt files. R ∪ S file `c1d6d922…` (880 prompts, 910 questions); sets
  manifest `02d91835…`.
- Processes (node A, mirror `5ea69f0fb`): warm-up GPU2 66.7 s; P2 GPU3 140.5 s and P3 GPU4 140.3 s
  (concurrent); P4 GPU2 123.5 s; P1 GPU2 123.5 s. P1 was first refused by the launcher's idle-GPU check
  (GPU2 still at 15% VRAM seconds after the warm-up container exited) and produced no output; it
  was restarted once after P4, as the preregistered infrastructure clause allows. Outputs: warm
  `63254f65…`, P1 `fe8c1ab8…`, P2 `3f0e35f5…`, P3 `c94f1118…`, P4 `0482ee70…`.
- Throughput: about 8–10 prompts/s per GPU including model load and the 49 GB package re-hash.

## GPU-hours

Qualification 0.165 (warm-up 0.019, P1–P4 0.147). Node A GPU3–4 were released at 11:43 UTC; GPU2 is
kept for the positional-key own-Lux re-derivation (item 3) and then released.
