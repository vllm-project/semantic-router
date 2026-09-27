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
