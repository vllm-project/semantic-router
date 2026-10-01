# Decoder M11 — amendment 2 (2B rules output without retention / yes-bias inputs; 2026-10-01 08:30Z)

Prereg [`dec-m11-prereg-2026-10-01.md`](dec-m11-prereg-2026-10-01.md) (`b8c3caf22`), "Development gates" and
"Probe TRAIN-overlap per tier". Written after the incident below and before any rerun of the 2B rules.

## What happened

- Node A's 2B scoring (`m11-score.sh pull / points / contrast / readout / rules`, mirror `c7d50a0c2`) ran **twice
  concurrently** at 08:18Z (one remote command block was executed twice; the scoring commands had no lock).
- In both runs the retention-probe step failed: the 2B tier probe gold drops the 100 GSM8K items that hit the 2B TRAIN
  (prereg rule), but `m10_probes.py score` looks up gold for every prediction (`KeyError` on a dropped id). The failure
  ended each `points` call before the `hs1-dev` scoring, so neither the retention nor the yes-bias inputs existed.
- The first run's `rules` step then wrote `m11/select/2b-finalists.json` (08:20:49Z) with "retention probes readout
  missing" and "yes-bias guard: hs1-dev false-yes rate missing" on both points; the second run's `rules` step was
  refused ("the rules run once"). That file is an incomplete-input artifact, not a gate outcome.

## Change

1. `m11/select/2b-finalists.json` is moved aside unchanged as `2b-finalists.invalid-incomplete-inputs.json` (kept for
   the record).
2. `m11-score.sh` (`probes`) restricts both sides' probe predictions to the tier gold's ids before scoring (and checks
   that every gold item has a prediction); `m11-score.sh` takes an exclusive lock, so a second concurrent scoring
   process is refused.
3. The 2B `points` steps are completed (retention, `hs1-dev`, `2b-BASE-e`) and the 2B rules run **once** more with all
   inputs. No gate, threshold, reference or point changes.

## Disclosure

The invalid file's other reasons are inputs that were complete and do not change: `2b-LH` fails the typed type floors
(Noul 213 vs 230 − 12; Score 175 vs 257 − 12), the `set_reconciliation` family floor, the Noul `rule_precedence` floor
and the Score5-typed-DEV check (COLLAPSE) against `2b-C0-f`; `2b-NT` had no other reason (HT-DEV v2 +.029
[+.015, +.042], GAIN, vs `2b-C0-e`). The rerun can only add retention / yes-bias reasons; it cannot remove `2b-LH`'s.
