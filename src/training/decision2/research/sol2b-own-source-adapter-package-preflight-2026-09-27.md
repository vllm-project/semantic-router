# Own-Sol 2B unmerged package preflight

Status: prospective package experiment. This does not resume training, select
a new checkpoint, or inherit a historical development or benchmark score.

## Question and immutable inputs

The existing targeted3024 BEST160 adapter directly initialized from our
Decision-1.0-Sol-2B at HF revision
`0665a41108e8f0b33a9515c98311c45947b99399`. Its previous BF16 native
DEV/CSS pilot was 946/1,600 and median macro-F1 0.34763; public231 was
161/231. Those are development diagnostics from the original scored runtime,
not scores of a new package. Its later full-weight FP32 merge changed native
probabilities and is held as a separate model. Test whether the **unmerged**
adapter and exact own-family source can be distributed with a reproducible
Decision 2.0 native loader.

Pin the existing `checkpoint-0000160`, original hard900 calibration, our
immutable Decision 1.0 repository revision and a fixed 32-item, gold-free
Choice/Noul/Score roster. No weights, CAL temperatures, data or prompts may be
changed to repair a failed comparison. Inventory hashes and the original
scored manifest must be recorded privately before any GPU run.

The current package builder requires the exact six inference-source hashes
from the scored prediction manifest. The historical BEST160 manifest was
produced by an older source revision. A new, versioned **current-runtime**
32-item native prediction manifest must first bind the current inference
code, unchanged checkpoint/source and original CAL. Compare two fresh
current-runtime processes: 32/32 valid, zero categorical changes and maximum
option probability drift at most `1e-4`. Historical predictions are a
diagnostic, not a gate that can import their scores.

## Package and advancement gates

Build the existing source-plus-adapter candidate with the current package
builder only after exact source ownership/revision, file inventory, CAL and
dependency lock pass. Do not modify the scored manifest or suppress a mismatch.
Compare the original unmerged model and the **copied package loader** on the
same 32-item gold-free roster, same physical GPU and pinned runtime. Require
zero invalid/missing answers, zero changed categories, maximum calibrated
option/Score drift at most `1e-4`, and an unchanged package inventory after
inference. A failure stops before any scored panel. Budget this preflight at
most 0.15 GPU-hour on one otherwise idle GPU.

If package parity passes, freeze that package and run it once on the complete
typed DEV1,600, CSS pilot1,430 and public JevBench231 panels; repeat our
Decision 1.0 Sol on the identical current-runtime prompts/adapters for the
matched control. Invalid and over-budget answers fail. A prospective formal
v3 roster is justified only if the package's development proxy
`100*sqrt(T_dev*H_pilot)` exceeds matched own-Sol1 by at least two points,
public231 is no worse than the matched own-Sol1 count, full DEV/CSS source to
package parity passes, and no dataset-overlap or package audit fails. Report
every regressing axis. If the package does not pass, retain the prior
development candidate as research only. No formal FINAL/CSS15 labels are
used in this preflight or to alter the gates.

The protected v3 key has been used previously on an unrelated arm. Any later
2B formal run is a post-key prospective same-panel comparison, not a virgin
blind test. Peer selection from Decision Index is external context only;
near-size peers must be rerun on the identical v3/public231 panels before
appearing in those ranks.
