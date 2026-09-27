# Qwen3.8-27B group-atomic Score≥400: locked CPU admission, revision r1

**Status: prospectively locked candidate validation; GPU HOLD.** This is a
new data-schedule version. The original exact-quota Score≥460 arm failed,
its second exact-subset attempt failed, and the later pooled candidate failed
whole-source-group and near-state checks. None is repaired or reinterpreted
by this lock. No candidate model, selector score, protected answer key or
optimizer result was used to choose this version.

## Honest selection history and immutable candidate

The preceding CPU capacity study searched **128 deterministic source-group
hash orders** after inspecting the pinned TRAIN input, teacher mask and the
gold-free SELECT/CAL input projections. It selected seed **96** for token
exposure proximity, not for any model outcome. This is an exploratory search,
so this document prospectively locks *validation of that already found
candidate*; it does not claim the seed was registered before the search.

The selected-row/input/native-token digest is
`0743da3f1ed18cc8eb824c41c736d92b23fb3d8a738e1b3401ff830229424be5`.
The mode-0600 private candidate-and-bounded-pair receipt has SHA-256
`809912b45c75aa7b630fd8fb40cafe898dc55a7201cc47cb9084484b0abb134d`.
The selected 2,560 IDs and text remain private. The receipt fixes every
selected ID, input hash and native token count; no row, seed, order, source
group or threshold can be swapped after this lock. Any failure ends **r1**.

The only source TRAIN is rights-clean v2 at SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
Pin its rights manifest
`61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`,
Choice/Noul and Score teacher artifacts respectively
`8cf211e5af88920de556dc84aa0fcb8b16677fdfd5209abb04db0ffe8e8b195c`
and `072cd519657caaa883eea1f5077789e5bacbf85f8ee20ab44cd562acc317701b`,
and eight-role gold-free inventory manifest
`26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`.
The official Qwen3.8-27B source revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; its tokenizer JSON
SHA-256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.

## Fixed budget, mask and controlled contrast

The witness contains exactly **2,560 rows from whole `(source, group_id)` groups**
across **160 nominal updates** at microbatch 1 and accumulation 16. Its fixed
type mix is Choice 1,201, Noul 940, Score 419. The source/type mix follows
from the locked selected IDs, not the older near-even quota. The selected
order in the private receipt is the sole allowed one for both future arms.
Maximum native segmented input length is 4,096 tokens without truncation.
Raw native exposure is exactly **1,244,573** tokens and eight-token-rounded
exposure exactly **1,253,880**; these are within **1%** of the earlier fixed
2,560-row control totals (1,244,036 and 1,253,048). The actual microbatch
dynamic padded exposure must be measured against the same order and runtime
before training and remain within **5%** between future A/B arms. Rounded
exposure is not a substitute for that check.

Every selected row gets native hard-label CE. If a separately admitted A/B
training contrast proceeds, A adds `.02 × CE(teacher distribution, student)`
on the fixed mask and B adds `.02 × CE(one-hot TRAIN gold, student)` on the
**same** mask and rows. Both begin from the official Qwen revision plus our
own `BEST368` LoRA/head fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`
with a fresh optimizer; neither starts from third-party Decision weights.
Teacher probabilities are row-local targets only. The earlier
`qwen38-27b-train-teacher-residual-screen` note defines the unchanged native
head, rank-8 direct LoRA, optimizer, one-epoch cap, zero/one-step parity and
single-checkpoint selection logic. Those technical gates must receive their
own versioned receipt; this CPU lock authorizes none of them.

The pinned teacher vectors must pass all 7,455 TRAIN ordered row/input/option
identities and probability checks. For this candidate, the mask must reproduce
971 eligible rows (Choice 429, Noul 368, Score 174), including 63 three-level
Score rows, with mask digest
`778eb599aa5f442a62d241394c65fd3b86e5a4aa4aad9c200bd360da86c0ab82`.
The original minima remain: masked Score≥160, three-level Score≥40, and
genuine-source Choice/Noul≥400. **Score≥400 refers to selected TRAIN rows,
not teacher-masked rows.**

## Fail-closed CPU admission before any GPU preflight

1. Recompute selected ID, full native-input hash and token identity from the
   pinned TRAIN and tokenizer. Require exactly 2,560 unique rows, full
   `(source, group_id)` inclusion across all task types, fixed type/exposure
   counts and every input≤4,096. The 83 TRAIN rows in saved bounded near-state
   SELECT/CAL pair IDs exclude their **entire** source groups. No post hoc
   replacement or backfill is allowed.
2. Recheck rights-ledger bytes, schema, all raw-source-to-ledger mappings,
   declared use terms and selected-source coverage. A declared manifest is
   not an independent legal conclusion or permission to redistribute source
   text; unresolved source use or attribution is a HOLD.
3. Recheck all eight pinned gold-free roles and their file hashes. The TRAIN
   role must contain the same projected selected inputs. For each of the
   seven disjoint roles, require zero same-row ID and zero exact raw or
   normalized **complete native-input** matches, and zero exact or bounded
   near **state-evidence** matches. Also enumerate bounded near complete-input
   pairs privately and review their distinctive evidence, options and source
   provenance. Any unresolved semantic or source overlap is a HOLD. Generic
   shared instruction/option scaffolds alone are not a duplicate claim.
4. Save exact private pair IDs and reason codes, but publish only aggregate
   counts and cryptographic digests. Neither protected labels nor model
   predictions may be opened for this admission. A missing role, stale hash,
   ambiguous source, malformed teacher vector, failed mask or overlap blocks
   r1. A clean bounded lexical screen cannot prove absence of paraphrase or
   third-party pretraining exposure; state its limits explicitly.

Only after a signed CPU PASS and separate source/semantic clearance may a
versioned zero-step GPU preflight be considered. No SELECT/CAL scoring, DEV,
JevArena, JevBench, model download, training or HF mutation is part of this
admission. The Score≥460 failures and the pooled candidate HOLD remain in the
record.
