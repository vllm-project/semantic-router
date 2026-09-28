# Eval M3 preregistration — amendment 1: audit reference (2026-09-28)

Committed after audit run v1 (code `3bea72d93`, receipt `audit-v1.json` SHA-256
`480b61f5…` on node A, kept) and before run v2. Cues, folds, bootstrap, thresholds and
actions are unchanged.

## What v1 showed about the method

1. **Below-chance priors.** In class-balanced groups the cross-validated majority prior
   falls below chance: the training folds' majority class is under-represented in the
   held-out fold (for example public 231 Noul 37.2 against 50.0). Any feature that merely
   breaks the prior's ties then shows a positive "gain" over the prior.
2. **Unconditioned pooled prior.** The pooled rows (all groups of one question type)
   used a prior pooled across tasks. In CSS15 every task has exactly one option set and
   order, so no item-level surface information exists, yet the pooled row read CUE
   (+1.9 [+0.7, +3.1]): trailing whitespace on each task's last option acted as a task
   identifier that recovered per-task label priors.

## Change

- Reference = the better of chance and the label prior (by group mean).
- A cue model = the better of the cue alone and the cue added to the prior; its gain over
  the reference gets the same cluster bootstrap.
- Pooled rows condition the prior on the group (prior features become group × key and
  group × description).

Per-group v1 rows that were CLEAN stay CLEAN under the change unless a cue beats chance
where the v1 prior was below it; both receipts are reported. New unit test: two
fixed-order tasks with different class balance and task-specific whitespace must pool to
CLEAN.

## Follow-up analyses (added, same rules)

`v2/eval/leak_effects.py`:

- `follow`: per model, from sealed predictions, the pick rate of a named cue option
  (longest or shortest description, or the option equal to a top-level state field),
  accuracy when the gold is and is not that option, and the share of wrong answers that
  went to it. Run for public 231 Choice (description length) and typed DEV
  `transition_table` (current-state option, never the gold by construction).
- `proxy`: the development proxy recomputed with a typed-DEV family excluded from T_dev,
  against v3 on the frozen 16-model calibration set and the 8 out-of-sample peers. The
  frozen proxy is kept unless the variant improves leave-one-out error and same-tier
  order agreement together.
