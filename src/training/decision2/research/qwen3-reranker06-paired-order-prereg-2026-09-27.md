# Decision 2.0 0.6B paired-order reranker screen: prospective protocol

**Status: preregistered design; zero optimizer steps under this protocol.**
This is a bounded development experiment, not a release checkpoint. The
protected typed FINAL, CSS 15-task final and authored release labels remain
sealed. The public 231-item subset, if reached, is an exposed independent
replication diagnostic rather than an official closed ranking.

## Why this is a new arm

The previously completed Qwen3-Reranker-0.6B sampled-eight LoRA arm improved
source-disjoint SELECT from 302 to 351/700 but improved structured typed DEV
only 422 to 431/1,600. Its source trains a yes/no score for each nominated
option while every alternative appears in the query. The option serialization
can therefore make the same decision sensitive to a harmless reordering.
Two other Qwen3 0.6B decision heads had weak option-order and counterfactual
consistency on the native development battery. This arm tests **order
equivariance within the existing long-context reranker**, rather than running
another generic continuation or changing its decision adapter.

The [Qwen3-Reranker-0.6B release](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
at revision `e61197ed45024b0ed8a2d74b80b4d909f1255473` is the immutable
source: 595,776,512 measured parameters, pinned model SHA-256
`27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`,
Apache-2.0, and a locally verified 4,096-token native call budget. The old
fixed-step 64 control adapter has tensor SHA-256
`d6fb8ac11f9e9cb95fc082b759c5ec5c716ad200c6efb735fe0bf41cc5e3a05a`.
Its prior prediction SHA-256 values on SELECT700 and typed DEV1600 were
`8ecbe12707f048f3793463d3fe6ad68c9449f9ed7aace8ee48ed562bb53cc20c`
and `572bce9055461de3d9b2d519ad649cde82294967ba7d4c36c59b22a4965e9e35`.
Do not retrain or overwrite the control. Verify the private bytes and native
inference contract before reuse.

## Fixed input and treatment

Use only the existing rights-audited noncommercial TRAIN7,455 and disjoint
SELECT700, source SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`
and `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`.
Reuse the existing 512-row deterministic slice SHA-256
`32a1226931967fd7a0f53cb2a4fd51189d5b643c56eeb1b88cea94ebf4312398`:
192 Choice, 192 Noul, 128 Score, including 157 long inputs above 512 tokens.
Verify source group, canonical-input and exact/near-context disjointness with
SELECT, CAL and the gold-free development panels before step one. No benchmark
answers, API-generated decision labels or new upstream corpus enter TRAIN.

The control is the existing fixed 64-step, sampled-eight cross-entropy LoRA.
The treatment starts afresh from the **same immutable source**, not from the
control adapter. It uses the same row order, selected answer keys, negative
key ranking, 64 updates, accumulation eight, rank eight, alpha 16, Q/K/V/O
and MLP projection targets, BF16, AdamW LR `2e-5`, weight decay `.01`,
gradient clipping one, seed `20260927`, and 4,096-token complete-input
admission. An attempted view above the bound is a hard preflight failure;
neither view may be truncated or silently skipped.

On Choice rows only, render a second complete query with the option entries
in reverse order but their keys, descriptions, state, instruction and correct
semantic key unchanged. Compute nominated-option margins for identical
selected keys in both views. The fixed loss is `0.5 CE(original) + 0.5
CE(reversed) + 0.2 symmetric-KL(original, reversed)`, aligning probability
vectors by **semantic option key**. Noul and Score retain the control's
single-view CE loss. The second Choice view increases FLOPs but not source
rows or optimizer updates; report this compute difference. No inference-time
ensemble or altered answer projection is permitted.

## Pre-optimizer gates and fixed stop rule

Before any optimizer update: (1) verify exact source, control and private
TRAIN/SELECT bytes; (2) verify group/near-duplicate isolation and data rights;
(3) prove reverse-view key/target alignment on Choice examples with
nonalphabetic, uneven key order, and test that the default inference text is
byte-identical to the prior native adapter; (4) run 32 zero-update native
answers and require categorical parity with the immutable source; (5) verify
one visible BF16-capable GPU, finite loss/nonzero LoRA gradients, no base
gradients, a clean task-specific output directory and no collision with other
experiments. Failure at any point is a **no-run** with a receipt; do not
weaken a requirement after observing an output.

Train exactly 64 updates and take only the final adapter. If OOM, nonfinite
gradient, source drift, save/reload failure or incomplete updates occur, stop
and preserve the negative receipt. No adaptive hyperparameters, retries on a
different seed, or development-based checkpoint choice. Seal gold-free SELECT
predictions before opening its labels, and verify 32/32 save-reload categorical
parity. Compare the same-row treatment to the immutable source and completed
control. SELECT advancement requires treatment at least **365/700** (control
351 +14), family macro at least control +`.02`, no Choice/Noul/Score accuracy
decline over `.025` against control, and no extra invalids. Also report
calibration and the six source-family buckets; never select on CSS or public.

If SELECT passes, collect typed DEV1600 gold-free predictions and a fixed
32-Choice-group order-reversal packet before scoring either. The packet is
selected by a seeded ID hash independent of labels and is shared with source
and control. Typed advancement requires **at least 463/1600** (control 431
+32), positive four-family macro change, no family regression over `.02`,
paired simultaneous correctness and order-invariant semantic choice better
than control, and no extra native invalids. Compare to the already measured
Kai 1.0 425/1600 and GLiNER source 652/1600 under the exact same prompt
hashes. Even a typed pass below GLiNER is a diagnostic, not a 0.6B backbone
advance.

Only if both SELECT and typed gates pass, collect treatment and control on
the same CSS three-task pilot1430 and exposed public231 under the unchanged
native inference adapter, then compare to Kai 1.0 and GLiNER source using
their pinned same-panel receipts. For architecture advance require CSS pilot
correct at least GLiNER source 561/1430, median task macro-F1 at least
`.31035`, public correct above GLiNER source 116/231, and no loss versus
source/control in either of the three CSS tasks. Report missing/overflow as
wrong, per-type/task/tier results, calibration, per-group paired confidence
intervals, and parameter counts. The CSS pilot may share TRAIN source families
and cannot prove unseen-task transfer. A new independently annotated transfer
panel would still be required for a release decision.

The predeclared **stop** applies even if the model improves an isolated
metric: failure of SELECT blocks DEV/CSS/public; failure of typed blocks
CSS/public. Preserve a negative arm, do not use exposed development results
to tune this same experiment, and do not publish `dev-2.0-0.6b` from it.

## Evidence boundary

Keep row text, training labels, target joins, predictions and weights in the
private experiment store. The public note may include input/model/code hashes,
aggregate results, failed gates, and limitations. Public scores reuse historical
baseline values only after verifying exact prompt and scorer hashes; otherwise
recompute them natively. Historical scores from different protocols are never
mixed into the rank or Pareto chart.
