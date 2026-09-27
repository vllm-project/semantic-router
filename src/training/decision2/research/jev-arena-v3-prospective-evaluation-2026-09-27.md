# JevArena v3 prospective evaluation architecture

**Status:** user-selected architecture and prospective score proposal, written
before any authored release-pool qualification or sealed FINAL evaluation.
Existing [JevArena v2](../jev_arena/arena_v2.py) remains a separate six-axis
protocol. No v2 score or publication bundle is reinterpreted as v3.

## Four named tracks, one model identity

| Track | Inputs | Decision and permitted use |
| --- | --- | --- |
| Development | TRAIN-disjoint SELECT/CAL, typed DEV 1,600 with about 400 related groups, CSS pilot 1,430 across three tasks; a small frozen daily subset may reduce cost | Select, diagnose and fit calibration. A public test repeatedly consulted here is development data, not blind confirmation. |
| JevArena v3 sealed core | Fresh typed FINAL 1,600, CSS held-out 6,547 across 15 tasks, and 1,200–1,480 independently authored, human-reviewed sealed originals | Primary same-panel 1.0/2.0/open-model release comparison. Total 9,347–9,627 original questions; variants do not increase the independent count. |
| Public cross-checks | JevBench public 231 split by difficulty; Decision Bench v4 1,041 text-evaluable plus 30 visual-only N/E | Separately labeled public rankings and task tables. They never enter v3's headline or blind denominator. |
| External Decision Index | Complete version 0.2.1: 150,759 scheduled / 150,317 scoreable Choice/Noul requests, if its exact row and scoring port passes | One broad external evaluation per frozen package. Same-edition published 1.0 rows may be reused here; not for JevArena paired intervals. No Score claim or Space-official rank from a local run. |

For each model, all four tracks bind that model's one pinned package
manifest, native inference adapter and calibration. Each track has its own
exact prompt, source,
scorer and gold identity. Unsupported, invalid and missing responses count
wrong on the full eligible denominator. No rank combines unlike benchmark
editions, native inference paths or matched-hardware conditions.

## Proposed primary score and accompanying evidence

Let `T` be typed FINAL Choice/Noul/Score family-macro accuracy, `H` the
median of the 15 CSS task macro-F1 scores, and `A` authored FINAL
Choice/Noul/Score family-macro accuracy. Each is a fraction in `[0,1]`
using the existing all-item invalid-as-wrong scorers. The proposed v3
capability score is `100 × (T × H × A)^(1/3)`; zero on any axis yields zero.
The source of every panel, formula, roster, selection and tie rule must be
hash-frozen before opening any FINAL gold. This formula is not active until
the v3 scorer and its release integration are implemented and audited.

Report the three axes alongside Choice/Noul/Score, every CSS task, every
authored mechanism and language, valid-response rate, Brier and ECE,
selective risk, long-input strata, and pairwise counterfactual/order/label
joint-correct rates. These reliability, transfer and efficiency views are
mandatory and require numeric release guardrails frozen before FINAL gold;
they are not diluted into the capability score. Latency, throughput and
cost compare only on matched hardware/service conditions. Use measured
loaded parameters for capability/size Pareto.

For 2.0-versus-1.0 intervals, resample independent synthetic groups,
human-transfer task and item structure, and authored independent groups.
Preserve the paired model predictions inside every resample. Publish both
per-axis and aggregate paired intervals and flag ties/uncertainty using a
predeclared rule. An additional perturbation is paired with its original,
never counted as a new independent question.

## Non-negotiable release gates

The authored panel needs source, oracle, ambiguity, duplicate/shortcut,
rights and independent human-review approval. Its current small DEV pilot
does not satisfy the 1,200–1,480-original release requirement. Freeze all
candidate checkpoints, CAL, full native package hashes, prediction paths
and the v3 formula before opening typed, CSS or authored FINAL labels.
Validate the release package against the scored native inference under the
same runtime and panel. A missing panel, unqualified human review, failed
parity or post-key model selection keeps the release on HOLD; public or
external results cannot substitute for the sealed core.

Decision Index 0.2.1 remains a valuable external stress test, but its large
row count measures performance on 38 mostly public benchmark projections,
not 150,317 independent private cases. Its current public 0.2 kit must be
ported and verified, including 30,419 added rows and the unresolved Home
duplicate retained-ID rule. A provisional port may report a sensitivity
analysis, not an exact same-edition rank.
