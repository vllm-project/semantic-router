# Kev 4B TRAIN-only teacher signal screen

**Status: prospective, no new teacher inference or student training.** This
96-row diagnostic asks whether a stronger public native comparator supplies
useful soft targets on our existing rights-clean TRAIN. Kev is an external
teacher and evaluation peer only, never a Decision 2.0 direct weight start.
It cannot by itself establish student quality or a release result.

The exact release is `jaredpalmer/kev-4b` revision
`139fdd94f1b6a6ad80cc15e08fcb99cac885a101`, with publisher source
revision `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`. The existing
[native Kev adapter](../inference/kev.py) verifies model provenance and source
hashes and calls its published typed checkpoint path. It retains Choice,
Noul and ordered Score semantics, rather than a chat completion. The same
published teacher scored JevArena v3 **59.226** and public JevBench **175/231**
on a separate same-panel comparison. Those external scores motivate the
screen; they are not TRAIN targets or student results.

## Frozen pilot

| Component | Commitment |
| --- | --- |
| TRAIN | Rights-clean v2, 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| Roster | Exactly the earlier Eikos teacher's deterministic 96 independent TRAIN groups, 32 per type; identity SHA-256 `18dce35a5ca58864f1b92399344fab679ec98fb7ff4ddd05ee71cfeecebb1722` |
| Native path | Pinned Kev release code and checkpoint, its fitted temperature, exact typed state/instructions/options, no gold label in request |
| Report | By type: full-denominator valid/invalid/tie and hard-label agreement, mean gold probability, normalized Brier; aggregate only, no raw rows or per-item predictions |
| Resource | At most 15 minutes on one exclusive GPU; abort on unexpected exception; no changed roster, model or prompt retry |

Before allocating a GPU, independently verify the full model and source
revisions, all cached release files, TRAIN hash and roster dry-run identity,
script/source hashes, qualified runtime and an idle reservation. A local signed
commit must be mirrored exactly. Overflow is counted invalid; malformed or
missing probabilities stop the pilot. The output file is private, mode 0600,
and written once. No protected panel or public benchmark labels are accessed.

The descriptive decision rule for a *future separately preregistered*
three-type distillation arm is all 96 valid, at least 22/32 Choice, 26/32
Noul and 20/32 Score TRAIN hard-label agreement, and Score normalized Brier
at most 0.25. These are source-signal thresholds, not model release gates.
If they fail, do not launch a three-type student. A type-restricted follow-up
would require its own lock, protection check, identical student/control budget
and independent development promotion rule; this pilot alone authorizes no
optimizer or JevArena scoring.
