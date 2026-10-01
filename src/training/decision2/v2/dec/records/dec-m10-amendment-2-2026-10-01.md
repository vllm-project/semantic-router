# Decoder Milestone 10 — amendment 2 (gates in two waves; the shared C0 reference)

Written 2026-10-01 ≈12:45 UTC+8 (04:45Z), **before any arm readout**: the only development readouts so far are
the references `4b-C0-e` (DEV2.0-4B's weights) and `4b-BASE-e` (the untrained base through the label-token readout).
No LH / FB / LT2 / NT2 point has been read. Preregistration `2dea44d6f`; amendment 1 `04088322b`.

## Changes

1. **Two gate waves.** The development gates and the pick run once for the first-wave arms (LH, FB, LT2) as soon as
   their soups are read (`m10/select/4b-finalists-w1.json`), and once more for NT2 when it is read
   (`4b-finalists-w2.json`). Finalists stay ≤ 2 in total, in the preregistered pick order: wave-1 finalists keep their
   slots and go formal at once; NT2 takes a slot only if one is left, or replaces a wave-1 finalist only if that
   finalist's formal run has not started and NT2 ranks above it by the preregistered order (HT-DEV v2 GAIN first, then
   the retention-macro Δ). Reason: NT2 is the optional second wave and trains ≈ 1 h after the others.
2. **Shared reference.** Every arm is gated against `4b-C0-e` (node E). Node F's points are compared with it only if
   `4b-C0-f` (the same weights read on node F) reproduces `4b-C0-e` exactly (0 differing decisions on every panel);
   otherwise node F's points are gated against `4b-C0-f`. Both checks are reported.

Everything else is unchanged.
