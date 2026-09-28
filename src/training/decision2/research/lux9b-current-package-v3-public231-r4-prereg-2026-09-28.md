# Lux 1.0 current-package comparator: prospective r4 lock

**Freeze this document before r4 formal or public inference.** This is one
same-panel Decision 1.0 Lux control for future Decision 2.0 comparisons. The
project has accessed v3 keys elsewhere, so any complete outcome is expressly
**post-key same-panel**, not blind evidence. Prior r3 ended incomplete on a
native no-truncation input-limit error (private STOP SHA-256
`accb56ca630c6dc8f5868bfef7a3f3d43a19b6317e426071a0112db4451c52d9`);
none of its predictions may enter r4.

## Locked identities

| Component | Frozen identity |
| --- | --- |
| Model | Own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`, deployed parameters 7,940,895,744; 29-file bundle SHA-256 `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`; release manifest SHA-256 `f786ce80a159c716f22b7d9c652af765b06f9db083bf43a8b08e67c3be3e69c5`. Require exact revision metadata and every bundle file hash. |
| Native inference | `DecisionModel.decide` receives each original state/questions, without truncation, option removal or chat conversion. Collector SHA-256 `9925fa0d486de6ee2117553ba8f72413df32528b3b66ecc90427f974a1f9f26e`, adapter `native-published-v2-overbudget-invalid-v1`. Its only new behavior is the row-level failure rule below. |
| Runtime | Offline ROCm image digest `sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb`; only `ROCR_VISIBLE_DEVICES=7`, exposed device `cuda:0`. Require one device, correct native runtime and fresh GPU ownership check. |
| Typed FINAL | 1,600 originals, 2,000 answers; gold-free prompt SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`; protected target SHA-256 `707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e`. |
| CSS15 | 6,547 originals and answers, 15 human-label tasks; gold-free prompt SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`; protected target SHA-256 `1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4`. |
| Public JevBench subset | Separately score 231 originals and answers (easy48, standard72, hard111). Gold-free prompt SHA-256 `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`; panel manifest SHA-256 `e0e7c67701cf05f996d3b4eac09abf1bb3e6bb363cfe007089acd3644a38ed35`; public target SHA-256 `abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f`. This set never enters v3. |
| Sealing and scoring | New three-panel gold-free sealer SHA-256 `361631b95cc2b7162ca74391b67d983322461605f0bde4037455bccc901c3220`. Typed scorer `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`, CSS scorer `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`, public scorer `aec840b6497beeae8b63e9270670c22506265c086a84e889bc82f694146518cf`. V3 `100 × sqrt(T × H)`: typed four-family macro accuracy T and CSS15 task-median macro-F1 H. |

## Frozen row-level failure rule

The known r3 failure occurred at CSS ordinal 3,515 (one-based); SHA-256 of
its original row ID is
`2ce492bf65ab3de5cd155918a153fdb7cca9d6aad1e01bdc3998c12aa1cf5ee6`.
Its native exception reported `label: 18198 tokens exceeds max_length=16384;
no truncation allowed`. R4 first performs one bounded technical dry-run on
that exact gold-free original to verify the adapter. This diagnostic is not
scored or reused in the formal prediction files.

For **any** original, only an exact native `ValueError` of the form
`<offered question ID>: <integer> tokens exceeds max_length=<integer>; no
truncation allowed`, with tokens strictly above the positive maximum, is
recoverable. Store the unchanged original input digest and **all** its
question IDs with `null` answers, plus a structured error category and token
counts. Every answer slot in that original is invalid and wrong under the
unchanged scorers. Continue to the next original. Do not shorten the input,
remove options, convert outputs or retry the question. Any other exception,
invalid native response, missing item, runtime mismatch or stale input stops
the run. Record over-budget counts by panel, as original rows and answer
slots; do not drop these rows from denominators. The native adapter version
differs from r3 and its scores must be labeled separately.

## Execution and stop conditions

1. Before GPU use, verify the signed local source, exact remote mirror, image,
   model revision/file hashes, three prompt hashes, r3 STOP, source and scorer
   hashes, and current GPU ownership. Technical dry-run uses only the single
   preidentified gold-free CSS row. If it fails or its output is anything
   other than exactly one invalid answer with the frozen error category, stop
   with no formal r4 inference.
2. Create a private immutable r4 run lock containing this preregistration
   commit and SHA, all above hashes, the dry-run receipt, GPU identity and
   **0.8 GPU-hour / 48-minute** cap. Gold/targets are not mounted into the
   inference container. Freshly collect typed FINAL, CSS15 and public231 once
   each. Existing partial r3 files are not inputs. If cap or another error
   interrupts any panel, preserve a STOP receipt and do not score.
3. Validate all 8,378 original IDs and 8,778 answer slots, exact input/model
   metadata, runtime and over-budget markers. The frozen r4 sealer writes one
   fsynced joint three-panel prediction seal before any protected or public
   target is read. Only after the seal run all three unchanged scorers once.
   Report T, H, v3, typed families/types, 15 transfer tasks, public
   overall/easy/standard/hard, calibration and invalidity. Record all GPU wall
   time including dry-run and model loads. No output is called an official
   closed JevBench rank.
4. An existing JPT-9B same-panel v3 60.994/public231 197 may be shown only as
   an external peer under its own pinned protocol. Do not merge older Lux
   historical DEV/public-card numbers or use this control as a Decision 2.0
   result. No HF publication or repository upload occurs in r4.

Raw prompts, predictions, targets and logs stay private. Public research
notes may include only aggregate scores, invalid counts and safe digests.
