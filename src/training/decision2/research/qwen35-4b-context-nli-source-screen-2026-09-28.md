# 4B contextual NLI source screen: candidate, not an admitted training arm

**Decision: CPU source candidate; GPU training HOLD.** This screen spent zero
GPU-hours and did not generate a student TRAIN partition, use any model for
inference, fit a checkpoint, or read publisher development/test labels. It
targets the completed official Qwen3.5-4B Base control's actual v3 gaps:
JevArena v3 **53.218** versus own Nox-4B **56.470**, with exception-stack
Noul `−16.00` percentage points and three-level Score `−4.25` points. That
official-source candidate had a much stronger development proxy before
transfer failed. The own-Nox clean-v2 continuation likewise reached SELECT
611/700 but only a 53.94 development composite and remains HOLD. Neither
control should be retrained just to test another source.

The previously audited SNLI/OCNLI source had a strong hypothesis-only
shortcut; the initial synthetic three-level SELECT3 was answerable without
the intended rule. This screen asks whether a *different* real-label source
is worth the next construction effort. It does not claim a new recipe works.

## Publisher sources and exact CPU receipts

| Source | Original source and terms | Role in this screen |
| --- | --- | --- |
| [ConTRoL](https://github.com/csitfun/ConTRoL-dataset) and its [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17580) | Expert-designed contextual NLI; publisher commit `d7acc335bef6c716f2830e1413d0d90c133ad6e9`, TRAIN SHA-256 `e51b63fa1da381a27fb5244e6f3c8f317eed51921e20fc05334023db3e2e834f`, CC BY-NC-SA 4.0. | Possible human-label **TRAIN** source after admission. Publisher dev/test unopened. |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) and its [paper](https://aclanthology.org/2021.findings-emnlp.164/) | Original publisher archive SHA-256 `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b`; bundled terms and dataset are CC BY 4.0. | Possible **external long-contract diagnostic**, not default student TRAIN. Publisher dev/test unopened. |

Both archives and all source text remain in private experiment storage. Exact
local audit code, remote mirror, and raw receipt fingerprints were checked.
The ConTRoL source audit used `training/data/audit_control_nli_source.py`;
the ContractNLI audit used `training/data/audit_contractnli_probe.py`. The
official Qwen3.5-4B tokenizer files used for the raw token screens have
SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`
and `353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2`.
These counts are *before* the native System One wrapper and therefore cannot
authorize a full-input 8,192-token run. The private aggregate ConTRoL and
ContractNLI receipts have SHA-256
`7cae4644594e7e412ae436e407ed4aa754082387928e943e777788e4dc891854`
and `2daa1c5b7e7a9f718041c8decf59cc23c8cc508a5c3e98ccba858d472e5f76f5`.
The ConTRoL receipt is byte-identical under Python hash seeds 0, 1 and 42.
The scanner now breaks equal-frequency rare-gram ties lexically; before that
fix, the heuristic overlap count varied between otherwise identical runs.

## ConTRoL: more plausible than caption NLI, but not yet clean TRAIN

| Publisher TRAIN property | Measured result |
| --- | ---: |
| Source rows / normalized independent premise groups | 6,719 / 1,527 |
| Contradiction / neutral / entailment | 2,293 / 1,946 / 2,480 |
| Exact duplicate normalized premise-hypothesis pairs | 26 rows |
| Raw pair tokens under pinned 4B tokenizer | median 351, p90 1,179, p99 1,558, max 2,018 |
| Group-disjoint hypothesis-only char-gram negative control | 44.23% accuracy, 44.17% balanced accuracy |
| Group-disjoint majority / original-position controls | 36.81% / at most 36.81% |

The shallow hypothesis-only control is **+7.42 points** above majority. It is
less extreme than the preceding SNLI/OCNLI screen but still means the raw
corpus must not be called artifact-free. It also does not prove any native
Decision model will use or ignore the premise. The protected-role inventory
included rights-clean TRAIN/SELECT/CAL, typed DEV and gold-free FINAL,
CSS pilot and gold-free CSS15, public JevBench/Decision Bench and previous
development packets. It found **five source premise groups with heuristic
near matches in public Decision Bench**, zero exact matches and no detected
matches in the other roles. Those five whole groups must be quarantined.
The lexical near stage did not inspect 9,444 longer protected text leaves,
and semantic paraphrases remain possible. The screen is not an overlap
clearance. The audited private role inventory SHA-256 is
`cc1988b9c7589b8011b2bb8ac02257bdd4423712a9c34b65bfb30980f4f0079b`.

An eventual native projection can ask whether a shown passage **refutes,
does not determine or supports** a hypothesis, mapping publisher
contradiction/neutral/entailment to Score levels 0/1/2. Separate Noul
projections could ask entailment and contradiction as binary questions, but
multiple projections of a source pair share one premise group. Neither
projection, the exact group sample, nor its rubric has passed blind semantic
review. ConTRoL's real labels are not a programmatic oracle for our native
question. Keep all derivatives in the same group, and do not count them as
independent task scenarios.

## ContractNLI: useful long-context challenge, weak as a sole gate

The publisher TRAIN has 423 annotated contracts, 17 *fixed* hypotheses and
7,191 labels (contradiction 841, not-mentioned 2,820, entailment 3,530).
Publisher span indices and bounds had zero integrity errors. Document-only
native token length has median 1,947, p90 3,933, p99 5,729, and two documents
exceed 8,192 **before** the native wrapper. Evidence positions reach p99 93%
of document length, so a front-only crop can lose the decisive span.

On a deterministic document-disjoint split of TRAIN, a classifier that sees
only the fixed hypothesis identity, never the contract, gets 1,112/1,598
correct (**69.59%**, macro-F1 `.661969`). A global majority gets 836/1,598
(`52.32%`, macro-F1 `.228978`). Twelve hypotheses have all three label
classes; five have only two classes in TRAIN. A raw score on the publisher
split could therefore reward clause priors without document reasoning.
Reserve ContractNLI as a *secondary* source-disjoint long-contract stress
view, report per-hypothesis/class metrics and this negative control, and do
not use it as the sole admission or release gate. Do not open its dev/test
labels before freezing the candidate, native prompt and scoring rule.

## Next discriminating 4B experiment and admission gates

1. **CPU first:** quarantine ConTRoL duplicate pairs and the five protected
   near-match premise groups; adjudicate suspected semantic neighbors,
   inspect source-label-to-native-rubric agreement blind, and build complete
   premise-group-disjoint native Choice/Noul/Score projections. The original
   source and license/attribution remain in the private ledger; neither a
   private dataset nor an Apache model license erases source obligations.
2. Preserve all already admitted human-labeled rows and the old verified
   rule/exception groups in rights-clean v2. Construct **one data-only
   substitution** from the same official Base revision as its completed
   control, replacing a prospectively chosen redundant synthetic/replay
   subset with reviewed contextual human-label projections. Fix row IDs,
   source/group distribution, option and class balance, oracle/rubric review,
   native prompt tokens, padded exposure, 7,455-row/466-update schedule,
   seed and unchanged optimizer/LoRA/head/checkpoint selection before any
   training. The exact replacement pool and budget have **not** been frozen;
   do not imply they fit merely because raw ConTRoL pairs are short.
3. Build a separately sourced, never-used three-level evidence diagnostic
   and an independently rendered **rule-exception/state** diagnostic. Source
   dev/test or prior Score r1/r2/v6, typed DEV, CSS pilot, formal and public
   labels are not fresh hidden selectors. Freeze negative controls: omit
   premise, omit exception clause, and swap/rename options. A candidate must
   beat shortcuts and retain old capabilities, not merely fit ConTRoL.
4. Use the original SELECT700 to choose the checkpoint, compare completed
   official Base control and one treatment under the same native runtime,
   then run one fixed diagnostic pass. The own-Nox start is a separate
   initialization factor and cannot be pooled with this data-only contrast.
   Existing 4B formal labels have been accessed, so a later v3 comparison is
   post-key same-panel evidence and needs separate untouched corroboration.

**HOLD until all four gates are concrete.** No 4B GPU reservation, formal
readout, model package or SOTA/parameter-efficiency claim follows from this
source screen. The most informative next action is the private native-row
projection and group/overlap/oracle audit, not another broad optimizer run.

## Native Choice/Noul projection preflight, still HOLD

The subsequent CPU-only `audit_control_native_projection.py` screen used the
same pinned source and protected inventory, the actual official 4B tokenizer,
and the `training.model.decision_model.encode` segmented-option prompt aligned
to the [typed System One request contract](https://docs.typesafe.ai/api). It
created no student TRAIN file or optimizer job. The aggregate receipt SHA-256
is `42f924c07f4c2982510f4954aa68012ba29b9cb4dece16006180193eda243fe2`;
it is byte-identical under Python hash seeds 0 and 42. The canonical native
input-payload stream SHA-256 is
`61e0100d2c5f8ddd820facbb34d6a05cdbfde68efcb61026026d82d292b6183a`.
Choice maps publisher entailment, contradiction and neutral to distinct
`supports`, `contradicts` and `undetermined` options. The two Noul projections
ask separately whether support or contradiction is established. These are
semantic hypotheses pending independent rubric review, not verified oracles.
**ConTRoL provides no ordinal Score supervision.**

| Native projection gate | CPU result |
| --- | ---: |
| Protected near-match premise groups / source rows quarantined | 5 / 25 |
| Repeated premise-hypothesis rows removed | 26 |
| Conflicting-label source groups removed | 13 |
| Remaining source pairs / independent premise groups | 6,618 / 1,509 |
| Potential Choice / Noul / Score rows, never admitted to TRAIN | 6,618 / 13,236 / 0 |
| Full native Choice tokens, median / p99 / max | 476 / 1,667 / 2,137 |
| Full native support Noul tokens, median / p99 / max | 446 / 1,637 / 2,107 |
| Full native contradiction Noul tokens, median / p99 / max | 449 / 1,640 / 2,110 |
| Prompts above frozen 8,192-token limit | 0 |

Choice gold keys are supports 2,448, contradicts 2,258 and undetermined
1,912. Rotated Choice gold positions are 2,236 / 2,179 / 2,203; support Noul
positions are 3,292 / 3,326 and contradiction Noul positions are 3,322 /
3,296. The projection passes syntax, position-balance and native length
preflight. No exact or short-leaf heuristic near overlap was found in the
protected SELECT/CAL roles, but the scanner did not near-scan 9,444 long
protected leaves or assess semantic paraphrases. The 13 conflicting-label
groups are an additional quality warning, not a reason to relabel them.

**TRAIN admission remains HOLD** until a separate reviewer checks source
label versus native rubric on a blind group sample, long-leaf and semantic
neighbors are adjudicated, and a same-initialization data-only substitution
with matched token/padded exposure and explicit Score retention is frozen.
This Choice/Noul source gate cannot by itself justify a 4B release recipe.
