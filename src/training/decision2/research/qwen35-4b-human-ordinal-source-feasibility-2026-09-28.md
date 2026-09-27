# Official-Qwen 4B Score: human ordinal source feasibility, metadata only

**Status: SOURCE SCREEN, not admitted data.** This is a read-only audit of
publisher cards and task definitions after the fresh synthetic SELECT3 failed
its policy-free shortcut control. No raw records, labels, evaluation answers,
model inference or GPU work were downloaded or used for this note. Therefore
label distribution, exact/near overlap, group isolation, row length and native
4B token exposure remain **unmeasured**. They must not be inferred from the
published row counts.

| Source | Published label semantics | Published scale and rights | Proposed role | Main uncertainty |
| --- | --- | --- | --- | --- |
| [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Five human-rated absolute response attributes: helpfulness, correctness, coherence, complexity, verbosity; each 0–4 | 20,324 train and 1,038 validation rows; English; CC BY 4.0 | Candidate **TRAIN** arm using one prespecified ordinal attribute, with group-level source split | It scores assistant-response quality, not rule/state decisions; class balance, overlap and response-length cues unmeasured |
| [NVIDIA HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | Relative preference of response 1 versus response 2, −3…3, including tie at 0; multilingual subset | 38,459 train and 2,017 validation preference pairs; CC BY 4.0 | Untouched **external diagnostic** of relative decisions and language, kept out of this TRAIN arm | Its score is *not* an absolute five- or three-level Score label; prompt-source overlap with H2 possible |
| [NIST TREC DL 2023](https://trec.nist.gov/data/deep2023.html) | Document relevance 0=not, 1=relevant, 2=highly, 3=perfect; passage 1=related but does not answer | NIST qrels; associated [MS MARCO](https://microsoft.github.io/msmarco/) corpus is noncommercial research use with independent source rights | Source-disjoint **supplemental diagnostic**, not an automatic 3-level TRAIN conversion | Full query/document join, corpus permission, duplicate groups, label distribution and overlap unmeasured |

HelpSteer2 explicitly says consecutive responses can share a prompt and about
29% of prompts are multi-turn. Its `preference` add-on has train/validation
rows from the original absolute-rating split; that add-on must not be treated
as an independent holdout. The publisher says prompts largely originate in
user-contributed ShareGPT and responses from a mix of ten in-house models.
For an absolute Score arm, preserve the **native 0–4 rubric** for one
preregistered attribute (first candidate: correctness, because it is closer
to evidence quality than verbosity/complexity). Do not average five ratings
into a self-defined target, turn paired rows into independent groups, or
collapse 0–4 to 0/1/2 before seeing its distribution and rubric. A semantic
mapping into Decision Score remains a separate hypothesis; H2 alone cannot
repair the 4B's observed three-level rule Score collapse.

HelpSteer3 preference examples contain `context`, `response1`, `response2`,
`overall_preference`, domain, language and annotator preferences. The source
card says the multilingual domain uses native-language prompts and response
annotations, while general/STEM/code differ. The −3…3 sign encodes **which
response is preferred**, so a polarity flip or option order change must be
handled explicitly if it becomes a Choice or ordinal-relative diagnostic.
Keep the publisher validation split untouched and group by normalized
conversation context; check potential shared ShareGPT prompt origins with
HelpSteer2 before claiming cross-source independence. Its different task
semantics make it useful as a supplemental pressure test, not a replacement
for an absolute Score selector.

The NIST 2023 page says documents and passages have different meanings at
level 1. Its qrels cover judged items; **unjudged is missing, not label 0**.
Passage qrels propagate judgments to near-duplicate equivalence classes,
which must be grouped rather than counted as independent examples. Query,
document and duplicate-class identities must be sealed across any split.
The qrels alone do not include all required document text; the MS MARCO terms
limit the corpus to noncommercial research and do not grant underlying
document rights for redistribution. A private query/document join and
rights check would precede even a supplemental evaluation. Native four-level
relevance should be reported as such; combining levels just to fit a
three-level gate would discard the published semantics.

## Concrete next CPU gate, before a model run

1. Pin each dataset revision and fetch **TRAIN-only** HelpSteer2 into an
   authorized private scratch area. Record actual attribute histograms,
   prompt-group counts, response length by class, EN quality and token lengths
   under the pinned Qwen3.5-4B tokenizer. Keep publisher validation and
   HelpSteer3 diagnostic labels sealed from transformation design.
2. Register the attribute choice and prompt/response-to-native-Score mapping
   from publisher rubric. Split at normalized prompt/conversation ID, never
   response row. Audit exact, near and semantic overlap against existing
   TRAIN/SELECT/CAL and gold-free DEV/formal/public rosters; ShareGPT reuse is
   a specific risk. Quarantine matches and record IDs/reasons privately.
3. Compute whether a capped H2 subset fits a **new** exact exposure-matched
   official-Base arm. Predeclare a parent-only control, native and padded
   token budgets, ordinal and retention gates, then independently review a
   source-disjoint gate. No GPU authorization follows from metadata alone.
4. If a separate TREC diagnostic is built, use only judged qrels and
   authorized corpus access; group by query plus duplicate class; report its
   native four-level semantics separately. Audit its source overlap and
   sampling before scoring any model.

Primary sources: [HelpSteer2 card](https://huggingface.co/datasets/nvidia/HelpSteer2),
[HelpSteer3 card](https://huggingface.co/datasets/nvidia/HelpSteer3),
[NIST TREC 2023 judgments](https://trec.nist.gov/data/deep2023.html),
[MS MARCO terms](https://microsoft.github.io/msmarco/),
[TREC FAQ on collection access](https://trec.nist.gov/faq.html).
