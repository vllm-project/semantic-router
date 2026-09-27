# Decision 2.0 0.8B and 2B: next-arm admission audit

**Status: data-first HOLD, no GPU-ready optimizer arm.** This is a prospective
choice of one next intervention per size, not a model score or a revision of
earlier negative experiments. The source branch at this audit was
`81b59bc2f14f9bbd38554cea0276135065da9c81`. Do not rerun completed
controls or inspect JevArena v3/public231 labels to choose these mixtures.
The common data admission below must be signed with exact row, tokenizer,
rights and overlap hashes before a training command is frozen. Its failure
stops both arms; no replacement template or seed is selected after a failed
review.

## Evidence and decision

| Size | Eligible direct source for the next arm | Already measured | Next question |
| --- | --- | --- | --- |
| 0.8B | Our `Decision-1.0-Eos-0.8B@3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd` | Matched hard/soft 498-step replay changed the open DEV/CSS proxy by only **+0.1627**, below its predeclared +1.0 gate. Both retained **85/400** typed DEV Score correct and predicted level 0 on all 400; CAL temperature changed no categorical answers. No formal result exists for these arms. | Can independent, source-dependent ordinal evidence break the level-0 collapse while preserving real-task transfer? |
| 2B | Official `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` general posttrained weights | Its completed 466-step clean-v2 development arm reached proxy **41.9113**, below own Sol1 **43.1450**; Score **95/400**, predicted levels 0/1/2 on 390/0/10. Human CSS pilot median macro-F1 **.3580** exceeded Sol1 **.3152**. Separately, own-Sol BEST160 formal v3 **43.9596** was below Sol1 **45.5804**, and the official Base arm also failed development promotion. | Can a Score-targeted data substitution retain the posttrained source's human-transfer gain and recover the lost ordinal axis? |

The 0.8B replay intervention, Qwen 0.8B Base/posttrained controls, own-Sol
continuation, 2B official Base/posttrained controls, 270-row small structured
replay, and stopped Sol soft-replay zero-step test remain immutable. The
present proposal is not a rescue by retuning a failed checkpoint. A third-party
Decision model is neither source nor initialization in either arm. Historical
Decision Index scores may select peers only; later JevArena and public231
numbers must come from new same-panel native runs.

## Shared data admission, before any optimizer

The completed Score v8.4 blind review quarantined all 30 groups for shortcut,
padding and copy defects. The v8.5 90-row pilot failed its frozen long-input
gate (only 3 rather than at least 6 rows above 1,500 tokens); its complete
overlap scan was not run. Neither corpus can be silently admitted. Follow the
already recorded v8.5 failure analysis with a **new, hand-composed 12-group /
36-row quality probe**, two groups for each of six mechanisms, pairing a
concise and a genuinely long document. Mechanisms must include exception
precedence, state transition, multi-source evidence, numeric boundary,
temporal precedence and evidence insufficiency. Each three-row group changes
only the decisive fact and has one oracle-grounded level 0, 1 and 2. Vary
answer order, decisive position, document genre, language and surface wording.
At least one case per mechanism must require a distant source span; deletion
of that span must make the answer unresolved. No inert length padding.

Freeze source facts and oracle before rendering. Two reviewers independently
answer the 36 rendered questions without the key, cite the operative spans,
and flag ambiguity, implausible state, language defects and shortcuts. Run
source ablation, single-field and option-position shortcut probes, exact and
near-overlap checks against TRAIN/SELECT/CAL, typed DEV/FINAL prompt rosters,
CSS pilot/15-task prompt rosters, public231 and earlier Score packets. Record
any unavailable protected roster as an explicit admission failure. A wrong or
ambiguous answer, systematic surface shortcut, missing source necessity,
unresolved overlap or failed native-language review keeps this pilot HOLD.
The 12 groups test construct quality only; they do not become 36 independent
training scenarios or a model-transfer score.

Only after that pass, prospectively build one immutable **128 independent
TRAIN groups / 384 Score rows**, **32 SELECT groups / 96 rows**, **32 CAL groups /
96 rows**, plus **32 untouched diagnostic groups / 96 rows**, all whole-group
and template-family disjoint. Use four mechanisms or more, including the six
above, with both English and native Chinese cases and all three levels within
every group. Count groups, examples and native tokens separately by language,
mechanism and level. Blind-review every admitted group and retain the
quarantine ledger; translation variants of one case stay in the same group.
The diagnostic group labels remain inaccessible until both size-specific
candidate packages and SELECT choices are frozen. Construct two further
human-labeled, original-source-disjoint transfer tasks of at least 200 items
each for development corroboration; neither task may share a source, task
identity or original record with the training mixture or CSS15. Fix their
task mapping, source revisions, rights and score before candidate inference.
Synthetic Score success alone is not cross-source transfer.

All new rows need explicit upstream rights and re-distribution decisions.
The actual native token counts, row IDs, generator/author identities, exact
source bytes and SHA-256 manifests are frozen **after** quality admission and
**before** a GPU preregistration. Do not assert that a data arm exists until
these gates pass. Keep raw text, labels and private paths outside Git/gist.

## Conditional 0.8B arm: own-Eos ordinal substitution

Use the pinned own-Eos source and original deterministic native runtime from
the completed 498-update hard-label arm. Retain rights-clean v2 TRAIN 7,455
rows, SELECT 700 and CAL 700 byte-identically. Replace 384 of that hard arm's
512 prospectively repeated TRAIN occurrences with the 384 new admitted Score
rows; retain the other 128 occurrences. Select replacements by a fixed
gold-free row-ID hash and choose length-matched rows so the total Eos-native
input-token budget differs from the completed hard arm by at most 1%.
Otherwise HOLD: do not pad with irrelevant prose or change the optimizer
horizon. Freeze exact 7,967-row order, per-step tokens, 498-update schedule,
seed, optimizer, head and LoRA settings, image/code digest and old hard-arm
reference before launch. No teacher KL or new temperature trick enters this
arm. The old hard arm is a descriptive same-source control; differing data
are the intended treatment, not a claim of strict token identity.

Preflight all row/group/near-overlap and native-length checks, a 32-item
same-batch zero-step/reload test, finite one-update gradients, and source
checksum. Stop on any mismatch, invalid output, non-finite step or resource
cap. At the fixed final update, freeze the single candidate; SELECT may
report it but must not pick a different checkpoint. Use CAL only after the
package is fixed. A development advance requires the same-panel typed
DEV/CSS pilot proxy to exceed own Eos1 by **at least 2.0 points**, Score
three-level macro recall **at least .45** on typed DEV and zero missing or
over-budget answers beyond the native source's declared limit. Report every
Choice/Noul/Score and human-task tradeoff, calibration and paired uncertainty.
Then score the separately frozen new Score diagnostic and two human transfer
tasks once. Do not call a within-template gain real transfer. Only a candidate
that survives this sequence may get a new post-key v3/public231 prediction
lock; the formal release decision is separate.

## Conditional 2B arm: official-posttrained ordinal substitution

Use the pinned official Qwen3.5-2B posttrained source, original dynamic-option
head initialization, 466-update trainer, SELECT700/CAL700 and native adapter
from the completed 2B posttrained control. Keep every original human-label
row and every original Score row in rights-clean v2. Replace exactly 384
gold-free-hash-selected **nonhuman, non-Score** TRAIN rows with the shared
admitted 384 new Score rows, preserving 7,455 row slots, 466 optimizer updates
and native TRAIN input tokens within 1% of the control's **4,194,465**. If
the source-specific token matcher cannot meet the bound without dropping a
human or Score row, HOLD; do not silently append rows or increase steps.
Freeze the exact replacement IDs, original/new token counts, deterministic
order, source package hash, optimizer and selection rule before an update.
No new real-label source is added in this single arm, so it isolates the
ordinal-data substitution rather than conflating it with transfer data.

Repeat official-source zero-step native identity and one finite optimizer
step on throwaway TRAIN examples, with no reuse of the smoke checkpoint.
Use the same complete 466-step horizon and SELECT family-macro/Brier/earliest
BEST selector as the original posttrained arm, then reload the frozen BEST
and fit temperatures only on CAL. Cap the complete training plus selection
at 2.5 GPU-hours; stop on source/data/rights/length mismatch, numerical
failure, invalid native response or failed reload parity. A development
advance requires the frozen typed DEV/CSS proxy to exceed own Sol1 by **at
least 2.0 points** (old own-Sol reference **43.1450**), with 100% native-valid
typed/CSS responses. Score level balance, Score macro recall, CSS task scores
and Brier/ECE are mandatory disclosures, not separately optimized gates;
individual tradeoffs are allowed. The same independent Score diagnostic and
two human transfer tasks must then be reported, with their predictions
sealed before scoring. No formal/public comparison or HF upload follows a
failed development gate.

## Release interpretation

The current formal v3 key was accessed earlier in the project. A later same-
panel comparison remains useful but is **post-key**, not untouched blind
evidence. Freeze each qualifying package and all predictions before its v3
and public231 score; run own 1.0 and an Index-selected peer freshly under the
same protocol. The new source-disjoint diagnostic supplies independent
corroboration. A first release needs a material combined gain over own 1.0
and proximity to the selected peer; per-task regressions may be disclosed
without vetoing an otherwise supported overall advance. Neither arm here has
yet met the data gate or produced a new model score.
