# Decision Index 0.2.1 protocol and Decision 1.0 audit

Audit date: 2026-09-27 UTC. This is a read-only audit of the community-maintained
[Jev Decision Index Space](https://huggingface.co/spaces/multimodalart/jev-decision-index),
not a Decision 2.0 result or a JevBench result. The Space says it is unaffiliated
with TypeSafe. No Decision 2.0 checkpoint has been submitted or scored here.

## Immutable evidence

| Item | Pinned identity |
| --- | --- |
| Space Git revision | [`99decba1a937df51ec43b4bfbbed82c0707d20e4`](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/99decba1a937df51ec43b4bfbbed82c0707d20e4), last modified 2026-09-27 01:46 UTC |
| [Index bundle](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/99decba1a937df51ec43b4bfbbed82c0707d20e4/data/index-v0.2.1.json) | `data/index.json` and `data/index-v0.2.1.json` identical; SHA-256 `e822c6717eb4007559013a70ddd47a42253fc3fe9bafde248d3e9d53847bfd2e`; generated 2026-09-27 01:44 UTC |
| [Methodology bundle](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/99decba1a937df51ec43b4bfbbed82c0707d20e4/data/methodology-v0.2.1.json) | `data/methodology.json` and `data/methodology-v0.2.1.json` identical; SHA-256 `542b508673a1ed2827fd83c31cc3cf5bc51250aa86b479858973a0b663667980` |
| Corpus identity in both editions | Source corpus SHA-256 `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`; this hash alone does **not** establish identical scoring, row selection, or index versions. |
| Public [reproduction kit](https://github.com/apolinario/decision-index/tree/19ad28ec9485493cc4f7fc07d91c178f948e6434) | Git HEAD `19ad28ec9485493cc4f7fc07d91c178f948e6434`; currently implements editions 0.1 and 0.2 only. |

The Space's README and some explanatory HTML still describe 0.2's 40 benchmarks
and equal area weights. For the current scoreboard, use the **0.2.1 JSON**
`suite.edition`, `suite.areas`, `index.weights`, `index.gold`, `index.formulas`,
and scored rows. `methodology.index.functions.category` also retains an older
equal-weight expression; its explicit 0.2.1 weights and formula take precedence.
I independently recomputed the weighted headline from all 38 per-benchmark
skill entries for Jev plus 64 open models; the maximum deviation from the
bundle's rounded headline was 0.0051 index points.

## Panel and scoring

The edition is `release-v2.1` / **Decision Index 0.2.1**: 120,340 scheduled
requests across 43 static suite benchmarks, 442 common exclusions, 119,898
scoreable rows, and **38 benchmarks in the headline**. Six interactive
environments have not been run for any entrant and are outside the index.
RouterBench, SGD, MMLU, ARC-Easy, and ARC-Challenge remain displayed but outside
the 38-item headline. Original upstream tasks are projected to typed decisions;
a result on the projection is not a result on the upstream task's unrestricted
generation or execution protocol.

| Area (headline weight) | All headline benchmarks, with Space IDs |
| --- | --- |
| Knowledge & Reasoning (25.85%) | GPQA Diamond `25`, GSM8K `30`, ChessBench `31`, MuSR `32`, SATA-Bench `33`, CRUXEval `43`, CLadder `44`, HLE `45`, MMLU-Pro `57`, BBH `58` |
| Language Understanding (25.85%) | ContractNLI `11`, ANLI `12`, WinoGrande `28`, HellaSwag `29`, ACOS `38`, FinEntity `39`, iSarcasmEval `40`, VAST `41`, NLI4CT `42`, RAGTruth `59` |
| Retrieval & Classification (20.02%) | BANKING77 `4`, CLINC150+OOS `5`, BRIGHT `36`, Amazon ESCI `37`, PhishNChips `56`, HoVer `61` |
| Tools & Automation (18.28%) | BFCL `1`, ToolRet `2`, API-Bank `3`, Home appliance simulator `9`, When2Call `62` |
| Arts & Human Taste (10%) | BPoMP `20`, Humicroedit `21`, POP909-CL `22`, cfcolor `23`, ForecastBench `48`, Habermas Machine `50`, New Yorker caption matching `64` |

The headline is `balanced_skill`, not raw accuracy: each benchmark is
chance-corrected, clipped to [0,1], then combined within an area. Thirteen
`gold` benchmarks have weight 1.2 within their area; the rest weight 1.0.
The four non-Arts area weights scale with the square root of their benchmark
counts after Arts is fixed at 10%. The combined index is the weighted sum of
five area skills times 100. ForecastBench instead uses
`clip((0.25 - Brier) / 0.25) × coverage`, so an always-0.5 forecast scores zero.
The Space also reports `balanced_raw`, `breadth_skill`, per-benchmark native
metrics, calibration, latency, and coverage separately; speed does not enter
the headline. Its method specifies 2,000 paired hierarchical bootstrap
replicates for uncertainty.

The 0.2.1 rescoring matters: ToolRet and BRIGHT count only queries whose 32
candidates include a relevant item, with chance recomputed on those queries;
Home appliances drops 24 duplicate test rows and 48 dev-identical test rows;
ACOS uses per-review F1; RAGTruth uses the stronger always-"hallucinated" F1
chance baseline. RouterBench leaves the index because its prompt leaks the
best route; SGD leaves pending a fixed builder. These are useful quality
lessons for JevArena's independent authored panel and benchmark audit. They
must not retroactively change a frozen JevArena release formula.

## Published Decision 1.0 rows in this exact edition

These are **Space-published 0.2.1** results, not new evaluations. The rank
column replays the Space UI rule over Jev plus 64 open models: rows within
0.25 points of the row immediately above share rank. It is an external Index
rank, not a JevBench or JevArena rank.

| Decision 1.0 model | Loaded/served parameters recorded by Space | Estimated count? | 0.2.1 skill /100 | UI rank /65 |
| --- | ---: | --- | ---: | ---: |
| Kai | 307,797,035 | yes | 6.52 | 52 |
| Lex | 307,797,035 | yes | 4.54 | 58 |
| Eos | 873,438,784 | no | 18.41 | 43 |
| Sol | 2,274,069,824 | no | 25.32 | 39 |
| Nox | 4,659,865,088 | no | 34.36 | 29 |
| Lux | 9,653,104,368 | no | 43.49 | 12 |

The Kai/Lex loaded counts are estimates in this bundle, despite `0.6B` in
their Hub names; verify parameter identity from the actual native package
before using them on a strict parameter/score Pareto plot. Jev is 57.89 on
this same edition, and the highest listed open result is Surogate Rune
26B-A4B v3 at 57.44. Eos and Lux lie on a provisional score/served-parameter
frontier **when restricting to the 60 open rows with nonestimated parameter
counts**; this is an exact-edition metadata calculation, not a target claim
for Decision 2.0.

## What can currently be reproduced

The public kit's [README](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/README.md),
[`editions.py`](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/decision_index/editions.py),
and [`index02.py`](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/decision_index/scoring/index02.py)
rebuild and score **0.2**. That edition has 121,057 scheduled / 120,615
scoreable requests, 40 headline benchmarks, five equal-weight areas, and no
0.2.1 `gold` weighting. The same Space archive gives Jev 51.67 and Eos 17.49
on 0.2, versus Jev 57.89 and Eos 18.41 on 0.2.1. Running the old kit and
labeling its score `0.2.1` would be a version error.

The public Space repository contains the static aggregate JSON and pages,
but no complete per-row prompts, gold labels, result records, or executable
0.2.1 scorer. Its methodology names `typesafe-diffusion-lab` generation
scripts and source artifact hashes; a public checkout of those scripts was
not identified in this audit. The public kit can rebuild 0.2 source rows from
pinned upstream datasets under their individual terms (including gated and
nonredistributable sources); its README says approximately 7 GB of downloads
and 17 GB of workspace. The 0.2/0.2.1 source corpus hash matches, but 0.2.1
has 717 fewer scheduled requests and several changed scoring rules. A future
0.2.1 port must reconstruct its exact row exclusions and score all 65
published rows to rounding parity before accepting a new entrant's result.

Minimal verifiable port: retain the kit's hash-checked 0.2 source rows,
added rows, typed Choice/Noul runner, native metric implementations, and
per-row outputs. Introduce an immutable `0.2.1` edition selector with an
exact 717-request filter (ToolRet 315, BRIGHT 330, Home appliances 72),
the 38-ID panel, 13 gold weights, five new area weights, and the documented
ACOS/RAGTruth/retrieval rescoring. The matching source-corpus hash supports
reuse of the old **base** rows, but an exact added-row hash, row-ID filter,
gold linkage, all metric edge cases, and model-native output parity still
require verification. Replay the published open-model per-benchmark raw
metrics, all six 1.0 area/headline rows, and Jev's published summary within
displayed rounding. Then inspect original row-level results and failure
counts where available, and compare an independently run native engine with
the same packaged checkpoint. Aggregate-score parity alone cannot certify
identical per-request predictions or latency. No such 0.2.1 port or
exact-output validation has been completed here.

The kit's 0.2 [submission instructions](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/README.md#submitting-a-model-to-the-leaderboard)
ask for complete, untouched run artifacts linked in a GitHub pull request.
They do not establish a self-service 0.2.1 submission or guarantee a new
entrant will appear on the Space. An author-operated 0.2.1 evaluation could
provide a same-edition result after its runner, inputs, calibration, and
package revision are disclosed; otherwise await an updated public kit or
build and parity-test the port. No 2.0 model should be called a 0.2.1
participant from an old-kit score, a partial run, or a projected estimate.

## Native Decision 2.0 feasibility and release order

The kit's [engine contract](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/docs/engines.md)
is `Engine(state, questions) -> (response, raw)`. Its validator accepts only
`choice` and `noul`: every question answered, full option probabilities
finite and summing to one, and a chosen key in the supplied options. Decision
2.0's native Choice/Noul/Score heads can address the first two types; **Score
is not tested by this external edition**. Upstream generative datasets such
as GSM8K and BFCL are evaluated through the board's constrained typed
adaptations. If a native checkpoint cannot faithfully process an original
typed request because of context, options, or capacity, report the original
request as unsupported and wrong on the full denominator. Never truncate,
filter options, insert gold-aware hints, substitute a generative readout, or
call a typed-projection score an unrestricted generative-task score.

The kit's generic `transformers` engine scores option tokens from a stock
causal LM; that would bypass Decision 2.0's trained native heads. Each
released 2.0 size therefore needs a pinned custom in-process engine or
`/v1/systemone` server that calls its **packaged native inference**, preserving
its own prompts, calibration, batching limits, and abstentions. The current
Decision 2.0 inference code is file-oriented, so a bridge and response parity
tests are still required. Run the kit's 86-request compatibility pass first
(two requests per benchmark/subtrack), then the complete fixed edition; log
checkpoint/source/corpus/scorer hashes and every invalid reason. The full run
gate requires all 119,898 scoreable rows to reach a valid answer or certified
model failure, with no evaluator defects. The 1.0 Space scores may be reused
for this **external exact-edition** comparison, while paired JevArena claims
still need 1.0 and 2.0 reruns on the same JevArena panel.

Do not run the approximately 120,000-request external suite for all six
candidate sizes while their native release gates and the exact 0.2.1 scorer
are unresolved. Freeze one qualifying candidate per size first; then run
full-index inference once per accepted package and seek author adjudication
for inclusion. JevArena should borrow the Index's attention to retrieval
candidate recall, tool-set exactness, calibrated forecasting, semantic option
audits, and task-level grouping, while preserving its own sealed typed,
transfer, robustness, and multilingual strata. The public Index panel is
not a dedicated multilingual evaluation and does not replace those strata
or a hidden independent test set.

## Current Space snapshot addendum (2026-09-27 UTC)

The immutable `99decba` snapshot above remains a historical audit. The Space
subsequently advanced to Git revision
[`ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4`](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4).
At this revision, the archived [0.2 index](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4/data/index-v2.json)
is unchanged (SHA-256 `fad0e6b0ee996de2543c2aecb73325464338b8437286ebc8162b652efd8d51bc`).
The current [0.2.1 index](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4/data/index-v0.2.1.json)
has SHA-256 `5444deeacd9bd6ea9e8ccf008f99f1223fe1d43af6e40739259aec804284ec55`;
its [methodology bundle](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4/data/methodology-v0.2.1.json)
has SHA-256 `235384612203690889a4d82ff22dab16fd3c65a6c8f5db0ef28b361f9ed9f665`.
Both differ from the historical bundle hashes above. The old 0.2 methodology
bundle remains SHA-256 `903cee829e68a7cb4592f91ba55558a5c725f8c80692cf0c846efa8b34f67b2f`.

The 0.2.1 **evaluation panel, five area weights, 13 gold weights, scoring
rules, 119,898 scoreable rows, Jev record, and all six Decision 1.0 records
are unchanged** from `99decba`. Board membership changed: JPT 0.8B, JPT 9B,
and Lavoir were added; the earlier reflex 27B record was replaced by the
full-coverage reflex 27B v2 record. The 0.2.1 board therefore now contains
67 open models plus Jev (68 rows), versus 64 open models plus Jev in archived
0.2 (65 rows). The current 0.2.1 release notes add the reflex replacement;
the version-to-version protocol changes described above remain the same.

| Model | Archived 0.2 skill /100, UI rank /65 | Current 0.2.1 skill /100, UI rank /68 |
| --- | ---: | ---: |
| Jev | 51.67, 2 | 57.89, 1 |
| Decision 1.0 Kai | 7.03, 50 | 6.52, 55 |
| Decision 1.0 Lex | 4.31, 58 | 4.54, 61 |
| Decision 1.0 Eos | 17.49, 43 | 18.41, 45 |
| Decision 1.0 Sol | 22.90, 39 | 25.32, 40 |
| Decision 1.0 Nox | 31.06, 28 | 34.36, 30 |
| Decision 1.0 Lux | 38.98, 12 | 43.49, 14 |

The UI shares a rank when adjacent headline scores differ by at most 0.25
points; the denominator includes Jev. Rank movement combines changed panel
scoring and changed entrants, so it is not a controlled measure of model
improvement. The 0.2 public kit still has no native 0.2.1 edition. Future
citations must pin the index bundle revision as well as its edition; the
`99decba` ranks /65 above must not be presented as the current board.
