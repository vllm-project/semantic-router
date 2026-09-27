# ANLI TRAIN native Score source screen

**Decision: HOLD; admitted rows: 0.** This is a CPU-only data-source audit,
not a trained Decision 2.0 model or a model evaluation. It follows the signed
[prospective protocol](anli-score-train-source-prereg-2026-09-28.md) at
`9231450e5` (spelling-only correction `385ce740d`) and the signed auditor
at `90f8ab98a`. The frozen candidate was not reseeded, refilled or filtered
after any source label or overlap result. GPU-hours: **0**.

## Pinned source and sampling

Official [`facebook/anli`](https://huggingface.co/datasets/facebook/anli/blob/main/README.md)
revision `8e4813d81f46d313dac7892e1c28076917cfcdf9` supplies the three
TRAIN rounds. Source parquet SHA-256 hashes, by round, are
`de2d038ae67f1fb1872073490b9e7685e9114d5f278ddd4631905fe0a4ecbcff`,
`209f4a15bf77224c62ffbde5f150fda928a7e2f5175366f4cacc3c7588aab13d`
and `c1d3f614d673888ac56b9ab62324e21583c98a11c4fef84e938d0f8fc414b29a`.
The audit used the official Qwen3.8-27B tokenizer revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and the strict eight-role,
input-only protected manifest SHA-256
`26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`.
No protected answer key or model prediction was read. The private aggregate
audit receipt has SHA-256
`8a31e42bf11c3b317b2506520a303f6d4277a600375d94bf38a13b2da18ffe62`.

| Round | Source rows | Source premise groups | Selected whole-group rows | Selected groups |
| --- | ---: | ---: | ---: | ---: |
| R1 | 16,946 | 2,053 | 4,000 | within combined total |
| R2 | 45,460 | 2,685 | 4,000 | within combined total |
| R3 | 100,459 | 5,996 | 4,000 | within combined total |
| **All** | **162,865** | **10,721 distinct** | **12,000** | **930 distinct** |

The round group counts sum above the distinct total because 13 premise
groups cross rounds; the signed selector excludes all 13. It also excludes
2,809 exact premise groups shared with public ANLI DEV. The selected complete
native Score requests total **2,599,097 tokens**, under the 3.6M-token
ceiling; the maximum selected request is 324 tokens. All three source
relations exceed the predeclared 15% floor in every selected round. The
source has two normalized premise–hypothesis pairs with conflicting labels;
neither appears in the fixed selected candidate. Nine selected groups contain
repeated normalized pairs and cover 260 group rows; that figure is not a
duplicate-row count and needs later adjudication.

## Overlap, shortcuts and admissible upper bound

The whole-source exact input check and selected bounded-near check found no
matches against the eight pinned protected roles. That is a **lexical
screen**, not proof of source or semantic independence; historical optional
roles are outside this manifest. The fixed selected candidate has **one
exact raw and normalized input-span match** against public ANLI DEV, affecting
one premise group containing 26 selected rows. The signed protocol requires
group-level quarantine and a zero-overlap candidate. A separate signed
quarantine counter (`867db736b`) computed the consequence without changing
the selector or backfilling: at most **929 groups / 11,974 rows** remain.
Its private aggregate-only receipt SHA-256 is
`d88a8277a02531252f98b6fc77fa4ece28101b6cdd34258bfb5a621219147589`.
This is a *technical upper bound*, **not** an admitted training dataset.

The preregistered no-model screens show per-round majority label rates of
about 40–46%, no file-position modulo-3/9 association above the majority
trigger, and no fixed hypothesis-only cue over its stated conditional-lift
trigger. These simple tests do not rule out more complex shortcuts. The
dataset is English and selected prompts are short; it supplies no direct
evidence for multilingual or long-document ability.

## Meaning and source-use gates

The publisher's labels are entailment, neutral and contradiction, not an
arbitrary numeric Score. A native evidence-relation mapping of contradiction
to 0, neutral to 1 and entailment to 2 is plausible, but independent blinded
review must check that the full System One wording makes the answer unambiguous,
especially for neutral. The source card declares [CC BY-NC 4.0](https://huggingface.co/datasets/facebook/anli/blob/main/README.md);
the original [paper](https://aclanthology.org/2020.acl-main.441.pdf) describes
Wikipedia/HotpotQA-derived premises in earlier rounds and additional news,
fiction, procedural and entailment corpora in R3. The distributed row schema
does not identify each original corpus. Thus attribution, source terms,
possible original-corpus reuse and any downstream redistribution scope remain
unresolved at row level. Private storage alone does not clear those checks.

The current state is **HOLD_NO_TRAINING_OR_REDISTRIBUTION**: one candidate
group needs quarantine; source identity and rights need an auditable mapping;
the three-level rubric needs a blinded review; repeated pairs need quality
resolution; and protected lexical zero needs a separate semantic/source
review. Do not use the 11,974-row upper bound to claim a passed gate, run a
GPU arm or release weights. The open ANLI DEV rounds are development diagnostics,
never untouched release tests.

### Native rubric review packet

A signed [blind-packet builder](../training/data/prepare_anli_score_blind_review.py)
at `9f99fb43b` generated a **private 24-item review packet** from the
unchanged selected TRAIN candidate and pinned parent source-screen receipt.
It samples eight premise groups, allocated 3/3/2 across the source rounds,
and three distinct source relations per group. The original sampled groups
contain 4–12 rows; the packet covers three claims per group and is **not** a
complete review of those groups or of the candidate source. The relation
balanced review selection occurred after the aggregate source audit; it is a
rubric diagnostic, not the predeclared training selector or an independent
evaluation set. The reviewer sees native Score inputs but not source labels,
mapped answers or reasons. The answer key and review inputs are separate
private files. Packet SHA-256:
`ffe307e908623bfcc9a8894a71aaa19b2900f9d80543d954bbcb838038ee116a`;
aggregate receipt SHA-256:
`cdd4c89fa2521a25d0841e829f2ba6cce4c2af1873598193f51436282bb90fe5`.
All 24 requests have the exact native Score input keys, and the packet
contains eight groups with three requests each. Source-file, tokenizer and
parent-receipt hashes matched the frozen audit. The packet was generated on
an authorized private CPU environment with **0 GPU-hours**.

### Independent answer-blind rubric result

An independent reviewer froze item-level ordinal meaning, unique support,
evidence necessity, shortcut and language criteria, plus group-level relation
integrity, **before** opening the packet. Its private criterion receipt is
SHA-256 `17c84c524e8130e7b8b509c17dd67b159fde36d70b87401372967e6b10730f04`.
The reviewer checked the packet hash, saw no separate key, source label,
prior receipt, model prediction or benchmark answer, and sealed a reasoned
answer guess for every item.

Only **9/24** individual native Score requests and **2/8** complete three-item
groups passed the conservative rubric. Eleven requests had wording or label
shortcut risk, six lacked a robust unique ordinal answer and one had a major
language defect; categories overlap. Recurrent relation errors included
conflating a named actor's conduct with an institution's, inferring causality
from co-occurrence, and treating absent evidence as contradiction. The
private item-level and public-safe aggregate receipts have SHA-256 values
`127f9304033970814378c39da5e3d4128c9a7a28de3a7ca5d97890f89e740511`
and `6336c1ca235639f408642cf453aff642a404e29384a9dc580c748565b9246510`.
This is a sample-level construct check, not a source accuracy estimate or a
comparison with official ANLI labels.

**Whole-source automatic Score admission remains HOLD, zero rows.** A distinct
future item-level filter or authored evidence-relation source would require a
new prospective protocol, larger independent review, complete source-rights
and protected-overlap checks. This negative result authorizes no GPU training.

## Single conditional causal experiment

If a later version independently clears every gate, the narrowest useful
Score-source contrast is one official-Qwen-start native-model A/B experiment.
Both arms would start from the *same* byte-pinned Decision checkpoint and
fresh optimizer, use the same frozen parent Choice/Noul replay and exactly
the same fixed total rows, native-token exposure, update count, context budget,
seed and final checkpoint. Only the Score-source slice changes: A gets a
prospectively selected whole-group ANLI evidence-relation slice, B gets an
equal-size eligible parent Score slice. This isolates whether external,
human-labeled evidence relations improve native Score beyond replay alone.
The existing 2,288-row, 143-update 27B design is a possible capacity target,
**not** an inherited authorization: before any run, a new signed protocol
must demonstrate an exactly matched 240-row whole-group Score slice, raw and
padded token parity, rights, independent selector, native zero-step parity
and fixed stop rules. If those constraints cannot be met, record HOLD instead
of changing the source hash, selecting a favorable subset or widening the
claim. No A/B model result exists.
