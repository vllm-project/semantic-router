# Own-Nox 4B matched Choice screen: zero-step STOP

The [prospective protocol](nox4b-choice-paired-runtime-prereg-2026-09-27.md)
stopped **before any optimizer update**. No one-step smoke, matched step-128
training, typed DEV, CSS pilot, CAL fit, formal JevArena v3, public JevBench,
package or publication followed. The existing own-Nox clean-v2 4B candidate
remains HOLD; the high-scoring third-party-initialized 4B research model is
ineligible for Decision 2.0 release and contributes no score to this run.

## Input and execution checks

The two selected cards were atomically reserved, had zero HBM allocation,
passed one-visible-device BF16 checks, and loaded the same optimized `fla`
path and exact mirrored local trainer. All 97 own Decision-1.0 Nox source
files matched its release inventory. TRAIN, SELECT and CAL matched their
frozen byte hashes and row counts 7,455/700/700; the treatment's four named
source cohorts contained exactly 2,240 Choice rows. Every zero-step process
had 4,194,465 unpadded TRAIN tokens, maximum length 6,596, 466 planned
updates, the same model-file inventory, same code/data hashes, and exactly
`baseline` then `zero_step_only` metric events. There was no optimizer event
or checkpoint. The private input/launch preflight passed.

An initial control launch rejected its impossible source-list/weight-1.0 CLI
combination **before loading the model or data**. A concurrent treatment
zero-step process completed; neither outcome was compared or used. The
[signed pre-comparison correction](nox4b-choice-paired-runtime-prereg-2026-09-27.md#execution-correction-2026-09-27-1046-utc)
at `8cac30b7d` omitted inactive source IDs from the control. Data, loss,
initial weights, numerical gate, fixed checkpoint, token budget and selection
thresholds were unchanged. Four new independent zero-step processes were then
run: control/treatment on each reserved card. The earlier failed launch and
unused treatment output remain in the private evidence.

## Four-cell result

All four processes produced complete, ordered 700-row native SELECT outputs
with the same prompt/token identities, task types and option domains. The
predeclared gate required **zero categorical changes and maximum option
probability drift ≤1e-4 for every pair**. The four-cell comparator returned
STOP, SHA-256
`f29b54b84a34fe0e211af5e7c0afffe85988f6bb5ec5427a3dbe38c7067248a4`.

| Pair | Category changes | Maximum probability drift | Gate |
| --- | ---: | ---: | --- |
| First-card control vs first-card treatment | 1/700 | 0.01503915 | FAIL |
| Second-card control vs second-card treatment | 0/700 | 0 | PASS |
| First-card control vs second-card control | 1/700 | 0.01503915 | FAIL |
| First-card treatment vs second-card treatment | 0/700 | 0 | PASS |
| First-card control vs second-card treatment | 1/700 | 0.01503915 | FAIL |
| First-card treatment vs second-card control | 0/700 | 0 | PASS |

The first-card control's prediction file hash was
`eabd788e4e656974e3f380cdff0d32f1ad0887ddb2034235b2c43b9c3f994de3`;
the other three independent processes produced byte-identical prediction
files at
`6b8d25a0097efde6fe22610a2f36d1d2190214b09296317cc9a25945ff402707`.
One Score `targeted_quantized_median` item flipped between levels 2 and 1.
This proves that the frozen strict causal gate failed even within the two
control processes. The pattern does **not** establish a root cause; a
run-to-run numeric or kernel path difference is a hypothesis only. It would
be selective to discard the first control because the other three happened
to agree. No treatment gain can be inferred.

From each zero-step log's filesystem birth to its final metrics update, the
failed control, unused first treatment and four comparison processes used
approximately 1.16, 58.13, 59.05, 61.29, 62.39 and 58.34 GPU-seconds,
respectively: **300.35 GPU-seconds = 0.08343 GPU-hour** for these six
allocated processes. Two earlier single-device runtime probes add at most
about 20 wall-seconds, making a conservative total **below 0.090 GPU-hour**.
The estimate excludes container scheduling overhead and uses file timestamps,
so it is not hardware telemetry. It is below the predeclared 1.20 GPU-hour
cap. Both task reservations were archived and both cards were released after
confirming no task container and zero HBM allocation.

## 4B decision and next discriminating test

**HOLD.** The own-Nox clean-v2 candidate improved SELECT to 611/700 but its
typed DEV/CSS pilot proxy was 53.94 (< the retained 56.0 development
threshold), and its public 231 result was 173/231, equal to the own 1.0
control. The human-only and structured arms showed better pilot transfer but
did not establish a substantial combined first-release gain. The current
paired weighting screen supplies no usable trained candidate. A future 4B
experiment should first preregister and isolate the two observed zero-step
numeric paths (for example, a fixed reference gated-delta implementation and
independent process repeats) and require cross-process identity before any
matched training. A new official-Qwen or own-Nox initialization arm remains
eligible only with its own frozen data, budget and native scoring protocol.
Decider 4B/Hopper are strong peer **selection** targets from Decision Index,
not imported JevArena or JevBench ranks.
