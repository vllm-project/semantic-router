# Gemma 4 composition coverage and the next discriminative arm

**Status: CPU-only audit and prospective design. No new optimizer run is
authorized by this note.** The completed [official Gemma development arm](gemma4-full-development-arm-result-2026-09-28.md)
is HOLD: its SELECT family macro accuracy of 0.72491 missed the frozen 0.794
threshold for the next diagnostic stage. The audit below reads only the
already authorized TRAIN and SELECT control plus the private, completed
SELECT predictions. It is not an independent model result.

## What the 60/40 composition cases actually cover

| Property | TRAIN | SELECT |
| --- | ---: | ---: |
| Rows / independent groups | 60 / 60 | 40 / 40 |
| English / Chinese | 45 / 15 | 18 / 22 |
| Two / three ordered operations | 30 / 30 | 12 / 28 |
| Distinct ordered operator sequences | 22 | 16 |
| Distinct 4-option descriptions, valid oracle label | 60 / 60 | 40 / 40 |

All cases are Choice with four alternatives, five-symbol starts and two or
three operations from `reverse`, `rotate_left`, `swap_ends` and
`replace_middle_z`. Both partitions use the same `textual-oracle-v1`
generator and `ordered_character_actions_v1` rendering template. Their IDs,
groups and exact input hashes do not overlap; the template **does** overlap.
Fourteen of the 16 SELECT operator sequences occur in TRAIN. By row,
31 of 40 SELECT cases have a seen sequence, so the existing SELECT family
mainly tests new values and option arrangements under a familiar renderer,
not transfer to a new generator.

At the selected step-456 checkpoint, only **7/40** SELECT composition cases
were correct: 5/31 with a seen operator sequence and 2/9 with an unseen
sequence. English was 4/18 and Chinese 3/22. The gold option indices were
distributed 10/5/12/13 across positions 0–3; predictions were 9/15/6/10,
suggesting some index-1 preference but not a single-position collapse.
Mean gold probability was 0.2409, mean winning probability 0.5534, and
normalized Brier 0.4719. Every gold oracle value appeared exactly once in
the candidate descriptions, and candidate descriptions were distinct.

The 40-row accuracy is noisy, and same-template exposure cannot establish
genuine cross-source composition. The very low score on 31 previously seen
operator sequences makes unseen sequence alone an insufficient explanation.
TRAIN contains only 60 composition rows among 7,455 total; SELECT assigns
that 40-row family one of six equal family weights. This mismatch makes the
development macro score sensitive to a small, weak family. It is an
actionable hypothesis, not proof that more composition examples will fix the
model. In particular, the step-456 composition accuracy fell from 0.275 at
step 256 to 0.175 while Score recovered; both outcomes must remain visible.

## One-variable prospective data intervention

Compare the frozen 456-step control to **one** new official Gemma arm whose
only treatment is a different TRAIN composition mixture. Preserve the exact
official source revision, text-only q/o LoRA plus native Decision head,
optimizer/loss, one-epoch 456-update schedule, SELECT 700, decoding,
checkpoint list and stop/selection rules from the signed control. Use a
new, separately locked roster rather than reusing the old lock. No Qwen
reference is a strict matched causal control for this arm.

1. Generate **320 new TRAIN Choice cases**, balanced 160 English/160 Chinese
   and, within each language, 80 two-step/80 three-step cases. Use a new
   `composition-v2` code path and wording, with a separately implemented
   oracle for the same four primitive operations. Vary the document setting,
   action wording and distractor construction; do not copy the v1 state or
   answer strings. Keep exactly four distinct alternatives, balanced gold
   positions and unique source/group IDs. Cases are separate situations, not
   option-order variants counted as new situations.
2. Replace **exactly 320** admitted synthetic Stage4 Choice rows from the
   six core `stage4_*` families, leaving every human-labeled row, Noul,
   Score and existing pilot row unchanged. Choose removal rows by a frozen
   minimum absolute Gemma-token-length matching rule, breaking ties by
   SHA-256 of source ID and the fixed experiment seed. Each of the six source
   families must retain at least half of its admitted rows. If the matched
   replacement changes the 3,620,578-token control budget by more than
   **0.5%**, or changes row count, type count, input length eligibility or
   456-update feasibility, abort before GPU work. The exact generator code,
   seed, new rows, removed IDs, tokenizer, token totals and dataset hash
   must be signed and locked before launch.
3. Independently generate **240 diagnostic cases** from a distinct
   `composition-v3` implementation and renderer, balanced 120/120 by
   language and 60/60 by two/three operations within each language. Keep
   the same semantic primitives but different prose, distractor mechanism
   and random stream. Freeze and independently verify the oracle, option
   uniqueness, group/input/near-duplicate isolation and semantic signature
   separation from TRAIN/SELECT. Do not use this set for checkpoint or
   temperature selection. Score the already frozen control and the chosen
   intervention checkpoint on it only after their prediction files are
   frozen, with paired group uncertainty. This is a development transfer
   diagnostic, not a substitute for JevArena or JevBench.
4. Retain the existing step-16 numeric/budget gate and step-256 futility and
   Score-collapse gates. Select checkpoints by the unchanged six-family
   SELECT macro accuracy, then lower Brier, then earlier step. Report all
   six families and Choice/Noul/Score, including lost Stage4 capability.
   A rise on the 40 old-template SELECT cases without a paired rise on the
   independent v3 renderer will be interpreted as template-local gain.

The **predefined discriminative readout** is the intervention-minus-control
change on the generator-disjoint 240-case paired accuracy, alongside the
frozen SELECT family macro and composition changes. A meaningful candidate
for later separate diagnosis would have SELECT family macro at least 0.794,
show composition improvement beyond the 40-case noise, avoid Score collapse
and show a positive paired v3 result without broad task regression. These
are prospective development criteria only. No threshold is retroactively
applied to the completed HOLD arm, and no formal or public labels are needed
to construct this comparison.

Before any GPU cell, implement and independently test both generators,
produce the exact private datasets and hashes, verify their licensing and
overlap audit, confirm the same actual admitted rows/tokens/steps, then seek
a separate resource/launch review. If CPU quality or budget gates fail,
record the failure and stop rather than increasing the sample count or
searching new templates against the same SELECT labels.
