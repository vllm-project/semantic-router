# Lux 9B structured8360 BEST224: rights and package preflight

Status: **prospective, release on hold**. This note inventories the existing
research checkpoint and the work needed for a noncommercial public **weights
and model-card only** release. It neither selects a new checkpoint nor uses
sealed FINAL or CSS15 labels. No raw TRAIN, SELECT, CAL, or individual text
predictions belong in a public package. A dataset license, a repository license,
and a weight-release permission are distinct claims.

## Frozen candidate and output identity

The completed `lux9b-human-structured8360-low-fla-r1` research run has 523/523
updates; `BEST.json` selects `checkpoint-0000224` by SELECT family macro,
normalized Brier, then earliest step. The scored native identity is
`decision2-lux-structured8360-r1` / `checkpoint-0000224`, model fingerprint
`0279db1b42131db737c03041c7e9d142d2efa0ac233970a3ddd8dfd672e3b7a7`.
These are private-run receipts, not a proposed public HF revision.

| Frozen input or receipt | SHA-256 or count |
| --- | --- |
| TRAIN `human_structured_replay.train.jsonl` | 8,360 rows; `399dc5322a8daf98316c3f9255c257140047331806190955a57bf632fea726dd` |
| Source builder manifest | `f2e2e35ac552731495679a00a9674fd4347ee9f07ff5cf7a5cf66e42e810c2da` |
| SELECT | 600 rows; `d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38` |
| Independent hard CAL input | 900 rows; `bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf` |
| Run provenance / selected checkpoint file inventory | `be9a2ed31b2e9dd1410123c64f63fa6a0c50402242f8886be94c87f94c5b0b3e` / `1148dbd73f21dc5b771acf604231b2cebc2c13e9d91770e0f5fe7c5cbd6cde70` |
| Selected `checkpoint.json` | `a1d26181ef670bb7944b39d4defc7984f5b013bc686620b4c2bd9d60581e297b` |
| Selected adapter / FP32 head safetensors | `7e0ebb0b0412806c805dd1a407236124308ed6b4839ec57fb861cc80ec0cd6b8` / `0b89fa73e39a0236c276091dc7960bfc605cf140d19f25910e500b21f21cabe7` |
| Fitted `hard-calibration-v2.json` | `8da35b1ba2095086bbb17aea702845cb6841922fb63b82a9015d2633a4cb326d`; Choice 1.500636, Noul 1.199237, Score 2.068282 |

| Existing output | Prediction SHA-256 | Native manifest SHA-256 | Score report SHA-256 |
| --- | --- | --- | --- |
| DEV1,600 | `dfe380cbee874aead39d9a0d8a7182a7c99332fddeab350d880a988900f56e77` | `4651afb299d35447d9aabbd34ddb8d6a32758d57c02f447d3fd355b0a85a0ef2` | `fd321a43dd71710cf0236fa6a9163d7265e521b3dc49e9473ce9fc3430eaf333` |
| CSS pilot1,430 | `9bd1848ec62766bf76435c7bbf1a38847c38e2b25db96244657dcc77ee14bf5d` | `3cc60eed0ac188e9e1f5017c342bff3b7498776b1da8794fd56c9c6d4ac82885` | `fe12a134b47f909599ff5556d8e3b42bf6affd19edf0a87b0c0692f447f36f33` |
| Public231 | `8f2e20195d258a11782567c49b0a90302d50d71dae735a86760a53dfff56df06` | `d8a2fa51fbfd3208e6c9726ae76f14031cd6321ad54d47c644da70255d53cf6b` | `81330996ce8bd7c5986f1b7130ab5baf90ceee5d7cef3103ac0620ad9b6b6264` |

Those native manifests record 1,600/1,430/231 valid, zero truncated or
over-budget questions, and the same model/CAL identity. These development
outputs do not qualify a release score.

The initializer is the published
[`Decision-1.0-Lux-9B`](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B)
text backbone and candidate head. The run pins its source-file hashes; its
`bundle-manifest.json` SHA-256 is
`985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`.
That source names upstream Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Before packaging, prove
which immutable **Lux** repository commit supplied the source bytes. The
published model's Apache-2.0 tag and [source attributions](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/main/ATTRIBUTIONS.md)
do not replace its upstream dataset conditions.

Safetensors header counts are 7,936,684,544 loaded text-backbone parameters,
43,278,336 LoRA parameters, and 4,211,200 decision-head parameters. The
unmerged scored model loads **7,984,174,080** parameters; 47,489,536 were
trainable. A full merged model should contain **7,940,895,744** parameters
(text backbone plus head); verify that number against its materialized files.
The `9B` name is nominal and must not be reported as the exact count.

## Source-rights decision for this TRAIN

The private structured manifest gives all 8,360 `merged_counts.source` rows.
It carries detailed license evidence for the **2,536 added** replay rows, but
not a complete public release attestation for the inherited 5,824-row base.
The replay builder excluded new MultiNLI rows; the final TRAIN still contains
**67 inherited MultiNLI rows**. Each assessment below concerns a public
noncommercial weight release, not redistribution of dataset text.

| Used TRAIN source | Rows | Preflight conclusion |
| --- | ---: | --- |
| Internal `decision2_programmatic_original_v1`, `decision2_targeted_programmatic_v1`, `legacy:stage4-general-composition-v2` | 70 + 600 + 2,600 | Project-generated/oracle lineage is documented. Confirm authorship and the project release declaration; no external corpus is asserted for these keys. |
| `legacy:stage3_replay` | 897 | Mixed internal examples and [CLINC150 CC BY 3.0](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE) / [BANKING77 CC BY 4.0](https://huggingface.co/datasets/PolyAI/banking77) text. The cited grants require attribution. Confirm the exact inherited-row mapping, applicable notices, and weight-release assessment; the aggregate key is insufficient. |
| `legacy:cosmos_qa` | 148 | [Author-reported CC BY 4.0](https://huggingface.co/datasets/allenai/cosmos_qa). Conditional on attribution and source/revision mapping. |
| `legacy:snli`, `legacy:squad2_answerability` | 72 + 186 | [SNLI](https://nlp.stanford.edu/projects/snli/) and [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) state CC BY-SA 4.0. Conditional on attribution and review of any adaptation/ShareAlike obligations for the proposed artifact. |
| `css_flute_official_train` | 120 | The [official FLUTE card](https://huggingface.co/datasets/ColumbiaNLP/FLUTE) marks AFL-3.0. Conditional on the corresponding notices and source receipt. This makes CSS FLUTE same-task supervised. |
| `legacy:nyu-mll/multi_nli` | 67 | **Unresolved.** The [MultiNLI card](https://huggingface.co/datasets/nyu-mll/multi_nli) describes mixed source-specific terms. The source manifest says non-fiction genres were selected, but the exact 67-row origin/derivative assessment has not been attested for this release. |
| TweetEval stance: `stance/abortion`, `atheism`, `climate`, `feminist`, `hillary`; emotion | 1,500 + 400 | **Unresolved for public weights.** [NRC's source terms](https://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html) make resources free for research, require citation, prohibit data redistribution, and direct commercial users to contact NRC. Bind the exact task files to those terms and obtain a reviewed determination or permission covering a publicly accessible noncommercial model. |
| TweetEval hate | 500 | The [organizer's HatEval card](https://huggingface.co/datasets/valeriobasile/HatEval) identifies CC BY-NC 4.0 for the task data and gated access conditions. Confirm attribution, access and NC conditions, platform terms, and whether they govern the proposed model weights. No weight-release clearance is attested. |
| TweetEval irony | 400 | **Unresolved.** The [SemEval irony authors](https://github.com/Cyvhee/SemEval2018-Task3) state CC BY-NC-SA 4.0 **and** participant/academic-use and distribution-by-request wording. Obtain author confirmation for a generally accessible public research model before release. |
| TweetEval offensive; sentiment | 400 + 400 | **Unresolved.** The [TweetEval umbrella](https://github.com/cardiffnlp/tweeteval#license) defers to each original task and Twitter. The retained manifest does not establish a task-specific public weight grant for these two sources. |

TweetEval's umbrella statement also leaves Twitter/platform terms applicable
to **all 3,600** converted posts. Do not infer that the umbrella permits public
weights. The inherited Lux 1.0 training lineage includes CLINC, BANKING77,
MultiNLI, CosmosQA, SNLI and SQuAD; its existing publication is evidence of
source disclosure, not a new license for the 2.0 checkpoint. The assessment
above therefore does **not** clear this candidate for public distribution.

## Required attestations and package sequence

1. Complete a reviewed, exact-source rights ledger covering every
   `merged_counts.source` key and the underlying mixed Stage3 and TweetEval
   tasks. Bind it to TRAIN/SELECT/CAL hashes, source-builder manifest, run
   provenance, Lux source-file inventory, and the selected checkpoint. Record
   the noncommercial weight-only/no-raw-rows scope, attributions, Twitter
   constraints, inherited Lux conditions, and decisions or permission records
   for irony, NRC, offensive, sentiment and MultiNLI. No such **Lux
   structured8360** attestation is present among the retained receipts; an
   attestation for another model or TRAIN is not transferable.
2. Extend and test the publication verifier for
   `decision2-human-structured-replay/1`: its source/type inventory is under
   `merged_counts`, whereas `publication.training_record` currently expects
   `counts`; `publication.rights_gate` currently recognizes balanced-human
   and Nox-structured schemas only. Require the exact rights attestation and
   license metadata `other` with reviewed noncommercial terms. A source hash
   or formal JSON match alone does not establish permission.
3. Once rights are cleared, verify the immutable Lux source revision and
   checkpoint/CAL inventories. Materialize **only** `checkpoint-0000224`
   through `training.model.materialize`; preserve its portable merge receipt
   and check the 7,940,895,744 full-model parameter count. The current
   external-base PEFT prototype supports Qwen sources, not this Decision 1.0
   initializer, so the full-materialization path is the supported starting
   point unless a separate portable dependency contract is implemented.
4. On a later authorized model runtime, compare selected adapter and merged
   native Choice/Noul/Score answers on a private gold-free roster under the
   same calibration, precision and context limit. Then run the package's
   gold-free DEV1,600/CSS-pilot1,430 source-versus-package parity gate:
   complete IDs, no invalid/missing or categorical changes, p99 probability
   drift at most 0.005 and maximum drift at most 0.02. Preserve a private
   hash-bound receipt; stop on a failed gate. This note ran no GPU parity.
5. Assemble a **local, unpublished** weights/model-card candidate only after
   the rights and parity gates. Bind public source counts, actual parameters,
   attribution, noncommercial conditions, CAL, model and artifact hashes;
   scan the staged files for raw rows, individual text predictions, private
   locations and secrets. Verify the copied native loader from that staged
   artifact. Later score/release decisions require their own frozen evidence;
   this note provides no score or publication authorization.
