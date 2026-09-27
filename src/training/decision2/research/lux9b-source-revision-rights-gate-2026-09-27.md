# Decision 2.0 nominal 9B: source revision and release gate

Status: **research checkpoint retained; public release HOLD**. This is a
gold-free provenance and rights preflight for the existing structured replay
BEST224. It does not create another checkpoint, qualify a JevArena v3 result,
or assess the legality of a public weight release.

## What is verified

The completed `lux9b-human-structured8360-low-fla-r1` selected
`checkpoint-0000224`. Its run provenance SHA-256 is
`be9a2ed31b2e9dd1410123c64f63fa6a0c50402242f8886be94c87f94c5b0b3e`.
The source-file map in that provenance matched **16/16** files, including all
backbone weight shards, tokenizer and decision head, in the cached Hugging Face
snapshot of
[`llm-semantic-router/Decision-1.0-Lux-9B`](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B)
at revision `bd45a30aee8c84032791c245c70f86dee5389cc8`. The Hugging Face
CLI independently confirmed that revision exists. The private, mode-0600
file-byte verification receipt has SHA-256
`650f88a05db7545ec263f4dea6cdc7a42247d526647253604861d05e8ed1b127`.
This resolves the source commit identity missing from the earlier
[rights/package preflight](lux9b-structured8360-best224-rights-package-preflight-2026-09-27.md).
It does not establish rights for the source model's training lineage.

The measured native candidate loads **7,984,174,080** parameters including
its unmerged LoRA and head. A full merge is expected to contain **7,940,895,744**
active parameters and still needs direct verification. The model is nominally
9B, not exactly 9 billion active parameters.

The recorded, open development scores are typed DEV **1,398/1,600** (87.375%),
CSS three-task pilot median macro-F1 **0.573976**, and public JevBench
**183/231**. The corresponding 1.0 Lux observations are 86.750%, 0.57011,
and 183/231 on the recorded development panels. Another completed 9B
human-only BEST192 reached pilot F1 0.580324, so BEST224 is not a uniform
improvement. None of these numbers is a sealed JevArena v3 score or an
official JevBench rank. A separate `human-no-mnli8255` directory has only
provenance and an empty metrics file, with no BEST receipt; it is not a
trained control or release candidate.

## Remaining release gates

The structured replay TRAIN has **8,360 rows** and manifest SHA-256
`f2e2e35ac552731495679a00a9674fd4347ee9f07ff5cf7a5cf66e42e810c2da`.
Its `merged_counts.source` lists 19 source keys. The added-row manifest's
`source_license_evidence` lists two aggregate keys and does not attest all
inherited rows. In particular, the inherited set includes 67 MultiNLI rows;
the merged set includes 3,600 TweetEval rows and 897 mixed Stage3 replay rows.
The upstream [TweetEval license section](https://github.com/cardiffnlp/tweeteval#license)
defers to task-specific and platform restrictions;
[SemEval irony](https://github.com/Cyvhee/SemEval2018-Task3) describes
participant/academic-use and distribution-by-request conditions;
[NRC labeled-data terms](https://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html)
allow research use but prohibit data redistribution. A private dataset and
noncommercial intent alone do not document a source-specific determination
for generally accessible model weights. The exact-source public-weight rights
ledger, including inherited Lux lineage and attribution, remains open.

The current publication code also fails closed for this lineage:
`publication.rights_gate` recognizes clean-v2 and two other research schemas,
but not `decision2-human-structured-replay/1`; `publication.training_record`
expects `counts`, while this manifest uses `merged_counts`. A schema-aware
verifier must preserve the exact run, source, partition and reviewed-rights
bindings before it accepts this candidate. There is no completed merged 9B
package, package-native full-panel parity or v3 prediction seal. No sealed
FINAL or CSS15 labels were accessed for this audit.

## Next discriminating action

First finish a source-by-source, noncommercial **public-weight** determination
for this exact manifest and Lux initializer. If it passes, add tests and the
schema-aware publication binding, materialize **only BEST224**, verify the
merged parameter count, then run same-runtime gold-free adapter/package
Choice/Noul/Score parity before any v3 prediction. A failed rights or parity
gate keeps this candidate on HOLD; do not choose another checkpoint after the
fact.

If that source determination cannot pass, start a separate prospective 9B
lineage from a rights-audited source and the existing rights-clean v2 data,
without reusing this checkpoint's scores. The Hugging Face CLI confirmed
Apache-2.0 tagged [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
at `68c46c4b3498877f3ef123c856ecfde50c39f404` and
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` as possible starts. Freeze
the native head, source revision, data/token budget, SELECT/CAL, controls and
stop rules before a training run; perform a zero-step length/numeric check
first. This is a proposed experiment, not an observed model gain.
