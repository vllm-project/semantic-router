# Sol 2B structured replay recovery experiment

This experiment asks whether bounded rehearsal of the
published Decision 1.0 structured TRAIN lineage can recover synthetic rule,
set, and transition
abilities lost or held flat when the 2B model receives human-label transfer
training. The synthetic family-disjoint final and CSS15 labels remain sealed.

**Rights scope:** the 5,824-row base contains all 3,600 TweetEval-derived
TRAIN rows. A later source-rights audit found explicit noncommercial or
research-only terms in at least 2,800 of those source rows. The user has
authorized noncommercial research with these sources, so training and
evaluation resumed after a temporary pause. A public release from this
lineage needs explicit source-rights review and attribution, with license
and use terms consistent with the underlying restrictions; it cannot be
described as unrestricted Apache-2.0-only training. This finding does not
change the source-file and isolation hashes below.

## Data and isolation

The fixed balanced-human 5,824 TRAIN (SHA-256
`e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2`)
is retained byte-for-byte as rows, including all 3,600 TweetEval human rows.
The [audited builder](../training/data/build_structured_replay.py) at signed
commit `b85a1ed` selects complete Stage4 groups from the pinned legacy 6k
TRAIN (SHA-256
`603600a6d1aeabe9179e5d3a85f275817f79a614e372bd64cb16584f50b83e17`).
It includes only `legacy:stage4-general-composition-v2` and
`legacy:stage3_replay`, excluding MultiNLI because its derivative rights
review is unresolved. These are legacy rehearsal examples, not independent
new supervision. The added source licenses and attribution are embedded with
source hashes, complete selected IDs, and input hashes in the private manifest.
Stage3 replay includes CLINC/Banking-attributed text, so downstream model
publication needs source review.

From 2,580 eligible rows after exact/group filtering, 44 near-context rows
were quarantined. The final mix adds 2,536 rows (1,769 Choice, 540 Noul,
227 Score; 1,833 internal Stage4 and 703 Stage3 replay) for 8,360 total
TRAIN rows, SHA-256
`399dc5322a8daf98316c3f9255c257140047331806190955a57bf632fea726dd`.
The private selection manifest SHA-256 is
`f2e2e35ac552731495679a00a9674fd4347ee9f07ff5cf7a5cf66e42e810c2da`.
The added rows total 3,451,144 Sol-tokenized input tokens, maximum 6,596
under the 8,192 cap. The full TRAIN has 5,297,993 tokens. The mixing ratio
is 30.33% added rows and 65.14% added tokens; training samples rows, so the
gradient mixture is row weighted while computation is dominated by long
structured examples.

The builder checks exact file hashes and whole source groups. Its added rows
have zero ID, group, canonical input, raw context, normalized context, and
detected near-context overlap against SELECT600, hard CAL900, synthetic
DEV1600, CSS pilot1430, and the 6,547 **gold-free** CSS15 prompts. The
balanced base's same zero-overlap audit is bound by its pinned manifest.
Near detection uses SimHash candidates and a 0.94 similarity threshold; it
is approximate, so it does not prove absence of paraphrases. SELECT and CAL
files are copied byte-for-byte with SHA-256
`d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38`
and `bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf`.

## Frozen training and evaluation recipe

Warm start from published Decision 1.0 Sol 2B revision
`0665a41108e8f0b33a9515c98311c45947b99399`, with dynamic candidate
head and rank-16 LoRA (alpha 32, dropout .05). Run one epoch of mixed hard
labels with CE + 0.5 Brier, microbatch 1, accumulation 16, maximum 8,192
tokens, LoRA LR `1.5e-5`, head LR `7.5e-6`, seed `20260926`, and checkpoint
every 32 steps. This uses no teacher-probability KL replay. BEST is fixed by
SELECT family-macro accuracy, normalized Brier, then earliest step. Hard
CAL900 is never used for training or checkpoint choice. The planned
post-training procedure was to fit native Choice/Noul/Score temperatures
and score synthetic DEV1600 and CSS three-task pilot1430. A temporary rights
pause interrupted training before that procedure; exact checkpoint resume
began after the noncommercial research scope was clarified. All inference
and scoring run on the experiment GPU host. The accelerated run
uses the pinned training container image SHA-256
`f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
and the bundled FLA gated-delta chunk source SHA-256
`fd4e01dc22a8c139c2a6eb61e47ae472a50322e4b4fff006cc5039a4602b310e`.
A short earlier fallback-operator launch was stopped before any checkpoint
and excluded from selection; no optimizer state was reused.

## Prior development baselines

All rows use the same DEV1600 and CSS pilot1430 items. CSS entries are micro
accuracy and median task macro-F1. Each arm was independently fit on the same
hard CAL900 and uses its native calibrated adapter; historical Sol 1.0 uses
its published calibration. Synthetic families each have 400 questions.

| Model | DEV acc | Attribute | Rule | Set | Transition | CSS acc | CSS F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision 1.0 Sol 2B | 58.94% | 65.25% | 54.25% | 78.25% | 38.00% | 38.46% | .31554 |
| Targeted3024 LoRA BEST160 | 59.13% | 67.00% | 55.25% | 73.25% | 41.00% | 42.52% | .34763 |
| Balanced human5824 LoRA BEST160 | 57.81% | 60.75% | 56.00% | 77.25% | 37.25% | 40.35% | .31009 |
| Structured replay8360 LoRA BEST288 | 58.63% | 62.75% | 54.75% | 77.50% | 39.50% | 40.42% | .32781 |

## Temporary pause and limited SELECT observation

The task-owned accelerated run was temporarily stopped immediately after the
atomic `checkpoint-0000256` was written. At the pause `LATEST.json` pointed
to step 256; its
SHA-256 is `c714d9925dfee83cad5602f63bc7ad354f7742b12404f32c0bb7ea838356ff31`.
At the pause `COMPLETE.json` was absent. The stopped container did not corrupt
the checkpoint; exact resume uses its saved optimizer and RNG state. The frozen training
provenance SHA-256 is
`20d485d5b2c61be6718daf7fa790c32f39337acdad2a8f77f39d32e90b72e731`;
step-256 checkpoint metadata SHA-256 is
`cca4e80d8d3f13eceafa9fd547769211365643c1930fbb22dd4be63ef4279aab`.

At interruption, `BEST.json` selected step 192, SHA-256
`e030acd750869c460a8e72a2864e0575b4f8238ebdb7c3c78597bebc22236ca5`.
Its SELECT600 metric receipt SHA-256 is
`7654199c12039c031931869d742b7e45f569af3eef2fa9cf0b71b9dcb4211026`:
236/600, **39.33%** family-macro/micro accuracy and 0.36030 family-macro
Brier. SELECT task accuracy is discourse 32.00%, implicit hate 30.00%,
stance 56.00%. The previously measured Sol 1.0 SELECT baseline was 38.17%.
The earlier completed targeted and balanced 2B arms reached 41.67% and
39.67%, respectively, on this same SELECT. These values are checkpoint
selection diagnostics from a partial arm, not a DEV/CSS comparison or a
release claim. They do not show recovery beyond the prior targeted arm.

## Completed run and native development evaluation

Exact resume completed all 523/523 planned updates. The final `BEST.json`
selected `checkpoint-0000288`, 239/600 (**39.83%**) on SELECT600 with
family-macro Brier 0.36295. Later checkpoints fell to 39.00% by step 448;
the full run did not surpass the earlier targeted3024 SELECT 41.67%.
Final `BEST.json` SHA-256 is
`7ceda36fff6ee42652b22a8daaad3f40c1740372146c8b1bbc54eb4717af0205`;
`COMPLETE.json` SHA-256 is
`5de308baf42eb50c02e895d30a7505212facefec74f2873a755a638bc4a206b6`.
The selected model's inference fingerprint is
`6325dc8857eb0b36fc6dace4ccb5f0dcd734b59c1a13322195e787f40ad4813e`.

The audited hard CAL900 fit native temperatures Choice 1.41807, Noul
1.52551, Score 2.09331. Its receipt SHA-256 is
`4d11fd70427b68dca0a15f92cf48e4dff90b5e8465226eb95825f6a7de4a85c8`.
No CAL labels entered training or SELECT. Native calibrated inference gave
1,600/1,600 valid synthetic DEV questions and 1,430/1,430 valid CSS pilot
questions, with zero truncation or over-budget questions. DEV accuracy was
938/1,600 (**58.625%**): attribute 251/400, rule 219/400, set 310/400,
transition 158/400. CSS pilot micro accuracy was 578/1,430 (**40.420%**),
three-task median macro-F1 **0.32781**; discourse 172/497, implicit hate
158/498, stance 248/435. These CSS tasks are development checks and do not
substitute for the sealed CSS15 evaluation.

Raw DEV predictions SHA-256
`b6a383b4dfaf3faeec43efb5f3cbc9dc0f888e229cef5937361941bdbf391d24`;
their manifest `177f7d45ecb365f6b25921e3bbb4db060338327f48388f92863aa46777ab99e3`;
DEV score report `011859a61f6833edc9e8bf0b3950defe57b093629b11ac9b8b1fb210d4226336`.
Raw CSS pilot predictions SHA-256
`a61ba53ddd3ede7ab98c885b2ccfffe85d73b52cc32fd3176329a61648091523`;
their manifest `4cb4f38aa12e4004d55f91c6aec4b77f83fc6e3edc94f70c9645135965fd6845`;
CSS score report `d4ca690d01d36c601d03b79a9b305049ff1d86dd27a9539077a34e24e89d799a`.

The extra structured replay improves CSS pilot micro accuracy by 1.96 points
and median macro-F1 by 0.0123 relative to Sol 1.0, but synthetic DEV falls
0.31 points. It trails the targeted3024 2B adapter on both DEV and CSS pilot.
The replay mixture is 30.33% of rows yet 65.14% of input tokens, and its
selected SELECT discourse accuracy (34.0%) trails targeted3024 (37.0%).
This is consistent with short human-label transfer being diluted by the long
structured rows, although the experiment does not isolate sample weighting,
length, source, or update count. This arm is **not selected** as the 2B
Decision 2.0 release candidate.

The Sol 1.0 warm start's [pinned public attribution](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B/blob/0665a41108e8f0b33a9515c98311c45947b99399/ATTRIBUTIONS.md)
at revision `0665a41108e8f0b33a9515c98311c45947b99399` names internal generators,
CLINC/Banking, non-fiction MultiNLI, Cosmos QA, SQuAD2 and SNLI as its
custom-training sources in `ATTRIBUTIONS.md`; it does not disclose TweetEval,
Twitter or SemEval training rows. The original 47,842-row TRAIN pool was not
available for a row-by-row independent audit here, and upstream Qwen
pretraining data cannot be excluded. Therefore the evidence supports only
"not disclosed in Sol's custom-training record," not a claim that the
weights contain no such material. The new 3,600-row TweetEval exposure in
this experiment is independently established by the exact TRAIN manifest.

## Clean 2B successor plan

The [rights-clean 4,655-row control](sol2b-rights-clean-control-2026-09-26.md)
removes directly identified TweetEval and MultiNLI rows and uses new
independent SELECT300/CAL300. A separate 7,455-row successor adds 2,800
GoEmotions human TRAIN rows and 400 human rows to each SELECT/CAL, retaining
synthetic anchors. These have distinct frozen holdouts, so their SELECT
percentages cannot be compared directly to the 8,360-row arm; DEV1600 and
CSS pilot1430 remain common development outcome panels. Preserve the Sol 1.0
source attribution and its row-level historical limitation above. A model
from any lineage requires substantive synthetic and transfer improvement
over Sol 1.0 before a release claim.

If the Sol 1.0 custom-training lineage fails rights review, initialize the
same candidate head from pinned [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
Apache-2.0 weights instead. The trainer already supports `--init-kind base`;
that path starts with a random head and needs a longer, higher-LR clean-data
schedule plus a matched SELECT/CAL/DEV/CSS audit. With a roughly 8,000-row
clean TRAIN, accumulation 16 means about 500 updates per epoch. The
accelerated 2B LoRA pilot measured roughly 3–4 seconds per update at this
sequence mix, so two epochs plus periodic SELECT, calibration and evaluation
are a practical single-GPU run on the assigned 256 GiB card. This is a
resource estimate, not a measured base-init result or performance promise.
