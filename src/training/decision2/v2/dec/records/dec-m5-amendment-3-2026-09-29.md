# Decoder Milestone 5 — amendment 3 (diagnostic formal runs for arms dropped only by the Noul-ML rule)

**Timing, disclosed:** written after the N5N soup readout and the MLX-DEV baselines, before any N5B or N5BN readout and
before any Milestone 5 formal run of an arm. The preregistered selection outcome for N5N stands and is reported as
is; this amendment does not make N5N or any other arm a finalist.

## What the readouts showed

Selection (`m5-select.py`, development readouts only):

- **N5N soup:** eligible; R = 61.81 vs the N4XF soup's 61.47 (a tie); P_mean3 62.69; typed DEV 492 / 273 / 378 vs
  501 / 264 / 362.
- **N5N dropped** by selection rule 3's second criterion: Noul-ML vs the N4XF soup −0.011 [−0.017, −0.005], upper
  bound < 0.

The MLX-DEV baselines show that this criterion cannot see the failure the milestone targets:

| Model | Noul-ML | predicted-yes | gold-No recall | Choice-ML | Score-ML |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nox 1.0 | .788 | .526 | .762 | .823 | .463 |
| N4XF soup | .853 | .530 | .823 | .916 | .539 |
| N5N soup | .842 | .541 | .801 | .899 | .542 |

On the held-out answerability / relevance slices, N4XF has no yes-bias at all: predicted-yes is .530 against gold
.50. Its non-English yes-bias exists only on paraphrase-style items (`mlx-diag` .760). So Noul-ML measures the
in-distribution answerability trade that every Milestone 5 intervention makes by design, as the prereg's caveat said.
It does not measure the PAWS-style yes-bias. Applied as written, the criterion would stop the milestone from answering
its question: does any arm fix the multilingual-Noul loss at equal or better v3?

## Amendment

1. The preregistered selection outcome of every arm is computed and reported unchanged. Finalists are exactly the
   preregistered ones.
2. An arm artifact that is eligible and not ≥ 8 R points behind the N4XF soup, but dropped **only** by the Noul-ML
   criterion, gets **one diagnostic formal run**. It uses the same runner, node-B reference and frozen caches: v3 and
   public 231, then `mlx-diag` after the v3 report is sealed.
3. Diagnostic runs are labelled "diagnostic (not selected)" in every record and table. They cannot produce a successor
   or a card-only claim in this milestone, and cannot trigger the 2B port. They are reported to the coordinator as
   evidence on the milestone question only.
4. At most one formal run per arm (finalist or diagnostic), so at most three formal runs besides the reference. Added
   cost ≤ 0.5 GPU-h, inside the cap.
