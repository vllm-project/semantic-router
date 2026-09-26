# JevBench v1.4.2 and live Jev catalog delta

Status: **primary-source research update, no new model score**. Checked on
2026-09-26 UTC after the earlier 1,690-row catalog download and before any
Decision 2.0 release result. This note does not revise the frozen public231
development panel, whose version and prompt hashes remain in its own protocol.

## JevBench version boundary

The [tagged v1.4.2 release](https://github.com/fstandhartinger/jevbench/blob/v1.4.2/docs/RELEASE-v1.4.2.md)
is pinned to peeled Git tag commit
`1df665e3956d7aab7fa0208ff6c4f2d8557f9f90`. It reports 93 systems,
89 ranked, using the 534 frozen v1.2 decisions **plus 308 new sealed
decisions**. The public231 subset is part of the older item set. The
[v1.4 method](https://github.com/fstandhartinger/jevbench/blob/v1.4.2/docs/METHOD-v1.4.md)
keeps Intelligence, Calibration, Speed and Cost as separate axes, takes their
equal-weight harmonic mean, and applies Intelligence, Speed and Cost gates.
Intelligence blends 80% of the v1.3 axis with 20% chance-corrected sealed
accuracy, then penalizes a public-to-sealed accuracy gap above 25 percentage
points. Invalid answers count as wrong. No system lacking an official sealed
measurement receives a v1.4 rank.

The released top two are decider-4b v2 at composite 64.13 and Jev 1.13.0 at
63.29. Their Intelligence axes invert that ordering: 49.4 and 53.1. The
decider's reported public and sealed accuracies are 83.5% and 34.7%, a
48.8-point gap. These are **upstream measurements** on its own protocol, not
Decision 2.0 results. The release also states that Eikos 4B/27B lack official
sealed rows and that some newly measured systems move to v1.4.3. No historical
v1.2, public231, v1.4, or JevArena number can be merged into one rank table
without the exact panel, version, adapter and scoring program.

**Training implication:** the wide public-to-sealed gap is a direct warning
against optimizing visible JevBench items. Keep independent, editorially
screened originals and human transfer central to model selection and release.
The Decision 2.0 public231 chart remains an explicitly exposed development
diagnostic. An official JevBench rank requires an actual upstream sealed run;
our chart cannot infer one.

## All-about-Jev live export check

The [All about Jev](https://hanxiao.io/all-about-jev/) UI displayed 1,753
entries when checked. Its separately downloaded JSONL export at that time
contained **1,645 rows**, SHA-256
`47113f73fffa4aa3ea04667b8c071b5815c054cdcdaa23f61947d432ad622aa8`:
210 model, 172 benchmark, 52 paper, and 1,211 other entries. This differs
from the earlier 1,690-row export and from the current UI count, so the three
numbers are recorded as distinct snapshots, with no assumption that the UI
and export are synchronized. The new export remains outside source control.

Against the earlier 423-row model/benchmark/paper inventory, the export has
21 previously unlisted model or benchmark IDs and no new paper ID. Promising
**catalog leads, not verified competitors**, include a sub-0.5B decision model,
a new 4B typed model, a GLiNER adapter, an Arabic specialist, and Chinese and
Russian decision benchmarks. Their source cards, licenses, native adapters,
training overlap and question quality must be checked before any same-panel
inference or JevArena inclusion. Catalog descriptions and author scores are
not evaluation evidence.

Three newly indexed primary repositories offer narrower, usable evidence:

- [ZH-Decision-Bench v0.1](https://github.com/CodyQin/zh-decision-bench)
  reports 219 Chinese items/284 typed questions: 179 from human-labelled
  MASSIVE Chinese development utterances and 40 synthetic items adjudicated
  by its owner. It tests Choice, Noul, Score, probability calibration,
  option-order flips and simplified/traditional Chinese variation. Its small
  non-voice task sizes (25 and 15 items) make per-scenario calibration noisy.
  A future development diagnostic needs exact source-ID exclusion against our
  MASSIVE TRAIN candidate and frozen native adapters; the author's model
  comparison is not our result.
- [RuDecide v0.1](https://github.com/smolnikov-k/rudecide) lists 2,235
  Russian questions in a public source-task track and 1,543 in an applied
  track, spanning Choice, Noul and Score. The README discloses mixed upstream
  licenses, including CC BY-SA 4.0, and says some reported systems trained on
  parts of the applied track. Treat both tracks as attributed, exposed
  **evaluation-only** candidates, separate from our sealed authored panel.
  Check task/source group overlap and native language validity before use.
- The [Jev-Omni three-readout ablation](https://github.com/CondadosAI/jev-omni-eval)
  compares a trained decision head, digit-token logits from untouched Gemma 4
  12B, and digit logits from the fine-tuned weights on the same inputs. Its
  author reports head-versus-digit gains on those fine-tuned weights of
  +6.1/+4.8 points for DecisionBench medium/hard and +9.1/+9.8 on the BLINK
  and MMStar visual tasks, while the trained head's advantage over the
  untouched base is much smaller on the visual tasks. This is **author-run
  evidence**, not a Decision 2.0 comparison. It motivates a matched
  backbone/data/compute ablation of semantic heads versus token readouts,
  with the untouched source as a separate control; vision gains should not
  be projected onto text-only Decision 2.0.

## Actionable release rule

Keep a versioned external-benchmark appendix with source attribution and
upstream protocol. Report our public231 results separately from an upstream
official sealed run, if one is later obtained. Prioritize the independently
authored and reviewed release panel; the current 12-case v13 pilot cannot
substitute for the target 1,200–1,500 original items.
