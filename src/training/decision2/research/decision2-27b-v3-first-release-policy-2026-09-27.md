# DEV2.0-27B first-release evidence policy

Status: prospective **27B candidate** protocol, written before any 27B v3
prediction or score. It is not a JevArena result or publication approval. The
existing 27B BEST368 checkpoint is a development selection, not a release
model. This document does not replace the already recorded, stricter historical
v3 gate; later decisions must identify which policy was used and when it was
fixed. Do not retroactively describe this policy as the original v3 gate.

## Weight lineage and native inference

The eligible start is the official general post-trained
`Qwen/Qwen3.8-27B` commit
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The selected LoRA and
decision head are from the completed clean-v2 `checkpoint-0000368`, with
native model fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
No third-party decision-model weight may initialize the release model.
Teachers and open controls do not change this rule. The CAL digest is
`e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78`.

Rebuild a **new** `llm-semantic-router/DEV2.0-27B` adapter-preserving package
from the exact original checkpoint, scored inference sources, immutable base
snapshot, CAL and dependency lock. The former lowercase-ID package is a
research artifact; its development scores do not transfer to the new package.
Bind the new `MODEL_MANIFEST.json` SHA-256 as the evaluation revision
`package-sha256:<digest>`. Verify every functional package and base file and
the actual **25,688,227,840** loaded text/LoRA/head parameters. Verify the
official upstream repository commit independently through the HF CLI.

Before v3 predictions, repeat the native three-type gold-free smoke and the
full matched-runtime DEV1,600 plus CSS pilot1,430 source/package parity for
the **new** package revision. The full parity receipt must pass the existing
zero categorical-change, p99 probability drift ≤0.005 and maximum drift
≤0.02 rules. Preserve the old and new receipts separately. The scorer must
consume predictions from `publication.package_native_arena` and the loader
copied inside that exact package, with all explicit criteria, no truncation,
BF16 backbone, FP32 head and frozen CAL. Do not use a CPU-merged checkpoint or
transfer scores from another physical/runtime path.

The uppercase package built from that exact source now has manifest SHA-256
`47f1e934062ec5d8f75dc6b74812c4fbfaf2d567f1a099f7f9f5cb2d41ffe782`.
Its new full DEV1,600 and CSS pilot1,430 parity receipt has SHA-256
`d53236dabc9c4e09479f6eef72aff1bb367426ecd9ee6f7c2f1603a4b95c4426`:
zero categorical changes and zero p99/maximum probability drift on both
panels. Two original over-budget CSS pilot answers remained invalid. This
closes the package-parity prerequisite only; it supplies no FINAL score.

## Comparator and evidence identity

Freeze the first-release decision comparison roster **before** any 27B v3
prediction: candidate `llm-semantic-router/DEV2.0-27B` against our native
`llm-semantic-router/Decision-1.0-Lux-9B` revision
`bd45a30aee8c84032791c245c70f86dee5389cc8` and native open
`alibiserikbay/JevK5-9B` revision
`d6521a18a86999190e9d775c915af3d6d6772fc4`, plus the same-size open
`denis-pplx/autojev-27b` revision
`6f5b557e037f5edb25c7dc92dbc6553e5a19c015` and its published native
runtime source revision `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`.
Lux and JevK5 are **nearest-size**, not equal-size: each has substantially
fewer loaded parameters than the 27B candidate. Record measured parameters
from qualified packages before inference. All controls require all-file native
attestations, a fixed
gold-free smoke with two-process repeatability, and then complete predictions
on the exact same v3 and public-subset prompt bytes. Reuse only predictions
whose panel, model, adapter, and revision digests already match this roster.
Lux and JevK5 retain the existing strict smoke maximum probability drift
`1e-6`. AutoJev uses a separately fixed external-peer rule: zero categorical
changes and maximum option probability drift ≤`0.02` across the same 32
gold-free questions in two fresh processes. Both maximum and p99 drift and
stable native invalid counts are reported. This tolerance is only a peer
qualification/sensitivity bound; it does not weaken the candidate's full
package parity. AutoJev's **first** complete v3/public run after roster
freeze is the primary run, regardless of its eventual score. No second run
may replace it by reference to gold labels. If the external-peer smoke misses
the fixed rule, AutoJev is HOLD from the launch rank, and any exploratory
score is separately labeled. Earlier comparator failures under a stricter
policy remain failures in their original protocol and are not retroactively
reclassified.
At the time of this policy, none of the three controls has verified full v3
predictions. The cited Decision Index 0.2.1 AutoJev balanced-skill 56.40 and
Jebadiah 54.67 figures are **opponent-selection context only**, not scores on
JevArena or this first-release panel.

`frontier-infra/jebadiah-27b` is a further same-size strong control only if
its exact HF revision, native three-type runtime, calibration, full-file
attestation and two-process smoke qualify **before** candidate v3 inference.
At this policy revision its model page/runtime could not be independently
verified; the private roster must therefore record
`hold_unqualified` with the precise reason. That status does not disqualify
AutoJev, Lux or JevK5, and it cannot be reversed after the candidate score
is known. A later qualified Jebadiah comparison receives a separately
versioned panel/report, never a silent roster rewrite.

The official `Qwen/Qwen3.8-27B` start is a desirable equal-size external
reference, but it has no qualified native Choice/Noul/Score decision adapter
in the current catalog. It cannot silently inherit the candidate's trained
decision head or become an after-the-fact control. A separate, prospectively
frozen generative-output projection may be added as an explicitly labeled
diagnostic after its decoding, invalid-output and confidence rules pass tests;
it cannot replace a member of the locked first-release roster. Neither an
arbitrary generative prompt nor an unqualified historical leaderboard score
is an interchangeable comparator. The first product card needs same-panel
rank and model-by-task charts. Any parameter-efficiency claim requires actual
measured size and exact same-panel results, and is outside its launch charts.

The 27B candidate lock binds its model/package/base/CAL identities, native
parity receipt, comparator attestations and repeatability, protocol and scorer
hashes, and the exact **gold-free** typed FINAL1,600, CSS15 6,547 and separate
JevBench public231 prompts. Typed FINAL has 2,000 scored answers within 1,600
items. Run the frozen candidate and controls on the same prompt bytes and
eligibility rules. Missing, invalid and over-budget answers are failures.
Save prediction and native-manifest digests before any scorer reads labels;
then bind the full roster in the standard v3 pre-key freeze. A candidate-only
seal is not a substitute for that full-roster freeze. No authored v3.1 items
or public231 items enter the v3 capability score.

## Aggregate decision and truthful exposure

The same-panel v3 scalar remains `100 × sqrt(T × H)`, where `T` is typed
FINAL family-macro accuracy and `H` is the median macro-F1 over 15 human
transfer tasks. Use 5,000 paired bootstrap replicates with seed `20260927`,
sampling typed independent groups within family and human tasks/records in
their recorded hierarchy. This 27B prospective **aggregate-priority** gate
requires candidate-minus-Lux v3 score at least **+3.0 points** and a paired
95% lower bound above zero, together with complete panel coverage, native
package parity, comparator qualifications and no broken probability contract.
Retain the coverage-adjusted typed Brier and invalid-answer guardrails from
the historical v3 policy as **explicitly adopted 27B technical checks**:
invalid-or-missing fraction on each sealed axis ≤
`max(0.02, paired Lux fraction + 0.01)`; typed coverage-adjusted
`B*=(m*B+(2000-m))/2000` ≤ paired Lux `B*+0.03`, with probability coverage
`m>0` and all 2,000 typed answers counted. These checks do not reinstate the
historical mandatory T/H or per-slice improvement rules. Individual T, H and
Choice/Noul/Score changes are published with intervals and error examples;
an individual regression does not by itself veto a substantial aggregate
gain. Any change to these numerical rules after 27B prediction sealing must
be versioned and disclosed, never called a pretest pass.

The v3 typed FINAL and CSS15 labels were previously opened during the
excluded 4B research evaluation. A later 27B run on those panels remains a
valuable **same-panel comparison**, but it is **not virgin blind** or a new
independent validation, even if 27B predictions are sealed before local
scoring. The model card must state this plainly. A claim of independent
confirmation requires a genuinely untouched source-disjoint panel or a
qualified external same-task review; otherwise release wording is limited to
the disclosed v3 comparison and separately labeled public231 result. A
JevBench public-subset rank is not the upstream sealed official rank.

## Readiness and publication boundary

This policy does not authorize GPU inference or HF upload. A release gate
remains HOLD until the uppercase package and new parity receipt, qualified
Lux/open roster, gold-free prompt audit, complete candidate and comparator
prediction seals, full v3 pre-key receipt, scored paired report, independent
overlap review, final downloaded-byte/native-output parity and product model
card have each passed. Existing 27B DEV and CSS pilot scores are diagnostic
only. The public README should present use cases and runnable native examples
first, then concise results, direct official-Qwen weight lineage, material
regressions and limits; keep private gate details out of the product card.
