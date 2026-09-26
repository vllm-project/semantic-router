# Hard multilingual DEV v7: frozen author candidate, awaiting qualified review

Status: **BLOCK_FOR_INFERENCE_PENDING_QUALIFIED_BILINGUAL_REVIEW**. This is a
private development editor's candidate, not a model result, training source,
or release benchmark. It follows the prospective v7 method in R154. The
independently rejected r6 packet remains frozen and blocked.

The first r7 author draft was never sealed or exposed to a reviewer or model.
Its manifest SHA-256 was
`32645730c2a22d92f791cec9ef736fdb33941251aec6e6df31ff8f5b2e9f898f`.
A Spanish wording change to resolve the repository spellcheck gate produced
the separate canonical r7-r2 candidate. No r7 answer was used to make that
source-only change, and the first draft is retained as `DO_NOT_REVIEW`.

## Frozen inputs and shape

Signed local source revision:
`8e965125c2955e73bfdaef0d402fbe086ecbc0a1`. Generator SHA-256:
`815b031f0d5031b9ef9b94bf50cd6c262991a703ec0a5db7f5c5c53b15f7c131`.
The separate private casebook commitment is
`1bcbbe3328435094b56b7a92c2d3954ad65af237e1dae54316f321a31992ee4c`.
The frozen manifest, gold-free prompt and private target SHA-256 values are,
respectively,
`811924721951d518572601b040dc38e1ee1c51b7456023214764d37bb08c2739`,
`214e1588081574780a7e7ff1cc80fd042fef36db8e15d34e88877e3c0a439a32`
and
`bbc77fd2bbc18a291bc20a87e6303dd17ef96c712d1104249cb2cd9a61c180f5`.
The separate freeze receipt SHA-256 is
`063154b60cf6228a29e2d9370580e8e70754e3db45fc4acc95b3722c9b36b1ef`;
it records 2026-09-26 21:41:16 UTC. No case fact, answer or private location is
included here.

There are **18 independent base cases × four languages = 72 prompts**,
balanced at six Choice, six Noul and six Score bases. English, Chinese,
Spanish and Japanese each have 18 gold-free prompts. Choice has two genuine
HOLD bases; the six semantic results have maximum frequency two, and native
option positions occur 2/2/1/1. Noul has three true and three false bases.
Score levels 0/1/2/3 occur 2/1/2/1 times. Same-day expiry and next-day
revocation are explicitly resolved by rules in every locale; the Score
mitigation credit is explicitly additive. Equal mark counts yield different
weighted levels, and equal pass counts yield different ordered-stage levels.

## Automated gates

The mechanical oracle passed the prospective balance, temporal-boundary and
counterfactual screens. All 18 Choice candidate records had a direct
single-field decision-changing counterfactual. Conjunctive roster/chain and
ordered-stage checks used repaired positive witnesses; that establishes
conditional necessity, **not** that each entry directly changes an already
failed answer. The operation-only leave-one-out heuristic was correct on
zero of the 18 bases; a fixed native Choice position reaches at most two of
six, never-HOLD at most four of six, and a fixed Score level at most two of
six. These shallow baselines do not prove resistance to learned shortcuts.

Eleven frozen reference files totaling **12,323 rows with state** were screened
using NFKC/casefold/exact match and a ≥0.94 near-match threshold. These
include original TRAIN/SELECT/CAL, visible DEV/public references, r6, the
sealed 144-row Score v3 gold-free packet and its separate 960-row gold-free
TRAIN projection. The Score v3 projection commitment is
`247f49b06c110ba5d15ee9e434a8635ca5923d296d830216e70fb4987dc57c49`.
The 144 rows are also represented in the full projection, so 12,323 is a
screened-file row count, **not** an independent example count. Exact and near
matches were both **zero**. This cannot exclude semantic overlap or unseen
sealed FINAL overlap. No Score training labels were opened for this audit.

The pinned native option parser accepted all 72 Choice/Noul/Score question
shapes. Its source SHA-256 is
`9ff8d754ce99c6539fc7f5bd88c10b196357c70124f59ed3d1bd1a0d3fdbb7d0`;
the preflight receipt SHA-256 is
`2c813bddc3290f5f78b0d957fc8fa6f1fbea96a28eebb505c0df7e034e724dda`.
The local focused tests and repository `make check` passed. No GPU or model
inference was run.

## Blinded handoff and remaining gate

The four separate, gold-free stage-one review packet SHA-256 values are:

| Locale | Frozen packet SHA-256 |
| --- | --- |
| EN | `d753a90ac0175f8082febec68d54fe6bd562e6b66d5513ea439bd331916af01e` |
| ZH | `9ff9b6b1ac412a5f314f5833ce08b7e200acce8b77adf5ca3adf0aff45197b81` |
| ES | `ad1813de395ad461b3adcba82ec55904637bbc4c2c7c4503e56dedda2b76861b` |
| JA | `26c16e74021346bbe06a47b2b963d09c78aab0941a8085d436b1429109adc217` |

Each non-English handoff has the English gold-free packet as a separate
stage-two comparison attachment. A qualified native or bilingual reviewer
must independently solve and seal all local-language rows **before** opening
that attachment, then seal translation-fidelity judgments. No reviewer has
yet passed this gate; language naturalness and exact cross-language meaning
are unverified. The frozen candidate must not be edited after review starts.
Any material ambiguity or localization error blocks this candidate and
requires a new version. Until the independent reviews pass, this panel is
ineligible for model diagnostic inference, training and release claims.
