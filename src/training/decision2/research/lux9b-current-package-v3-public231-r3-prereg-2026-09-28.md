# Lux 1.0 current package: prospective v3 and public-231 control r3

**Locked before r3 formal/public prediction or scoring.** This is one pinned
own Decision 1.0 Lux control for same-panel Decision 2.0 comparisons. No Lux
current-package complete v3 prediction pair exists yet. The r1 attempt
produced zero items due to ROCm device discovery; r2 separately proved that
the same image with only `ROCR_VISIBLE_DEVICES=7` exposes one `gfx942` device
and reproduces all five gold-free current-package answers with zero drift.
The r2 stop receipt SHA-256 is
`1fea5d5243882dd21005a1a842a923b43693de670ba96bf871b1c00aa3436324`.
The project has already accessed v3 keys elsewhere: this is a disclosed
**post-key same-panel** control, not a new blind test or a 2.0 result.

## Fixed package, inference and panels

| Component | Locked identity |
| --- | --- |
| Weights | Own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`; deployed parameters 7,940,895,744; current 29-file bundle manifest SHA-256 `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`; release manifest SHA-256 `f786ce80a159c716f22b7d9c652af765b06f9db083bf43a8b08e67c3be3e69c5`. Require exact HF revision metadata and full file hashes. |
| Native inference | Unmodified `inference/run.py` SHA-256 `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`, published `DecisionModel.decide` with original state/questions, backend `lux`; no chat wrapper, truncation, option deletion or runtime override. |
| Runtime | Exact offline ROCm image `sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb`; only `ROCR_VISIBLE_DEVICES=7`, no `HIP_VISIBLE_DEVICES`; exposed device `cuda:0`. Current-package r2 native parity is the prerequisite, not a substitute for full collection. |
| Typed FINAL | 1,600 original items / 2,000 answer slots; gold-free prompt SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`; protected target SHA-256 `707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e`. |
| CSS15 | 6,547 original items / answer slots over 15 human-label tasks; gold-free prompt SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`; protected target SHA-256 `1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4`. |
| JevBench public subset | Separate **231 public items**: easy48, standard72, hard111. Prompt SHA-256 `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`; public panel manifest `e0e7c67701cf05f996d3b4eac09abf1bb3e6bb363cfe007089acd3644a38ed35`; target SHA-256 `abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f`. Public scorer SHA-256 `aec840b6497beeae8b63e9270670c22506265c086a84e889bc82f694146518cf`. The public set does not enter v3. |
| Seals/scorers | Lux native v3 sealer SHA-256 `ddc526425b6377f15ea37117c95a9315c7bc7cb5b27a80eedeed5c8571ef2753`. Typed scorer SHA-256 `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`; CSS scorer SHA-256 `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`. v3 score is `100 × sqrt(T × H)`, with typed four-family macro accuracy `T` and CSS15 task-median macro-F1 `H`. |
| Budget | One live-confirmed idle GPU, at most **0.8 GPU-hour / 48 minutes elapsed** for source/model load and three collectors; no retry, alternate image/mask/model/adapter/prompt/calibration or checkpoint. Technical r1/r2 GPU-hours are reported separately. |

## Execution, freeze and failure

1. Before GPU use, revalidate every exact package file, model revision,
   source/image/prompt/manifest hashes, r2 parity receipt and GPU occupancy.
   Stage only gold-free prompts and exact source; model, source and panels are
   read-only inside each inference container. If anything differs, stop with
   zero r3 GPU use. Do not mount formal gold or public targets for inference.
2. Collect complete typed FINAL, CSS15 and then public231 once each with the
   unchanged native collector and fixed runtime. Missing, invalid and
   over-budget answers are failures, never dropped. A crash, incomplete panel
   or total cap exhaustion leaves an incomplete run; preserve partial outputs
   and stop without alternate scoring.
3. Verify all **8,378 original IDs** and **8,778 answer slots** across the
   three panels, with exact input hashes and model/runtime metadata. Seal typed
   and CSS with the frozen Lux sealer, validate public231 native rows, then
   hash and fsync a **joint three-panel prediction freeze before reading any
   target**. Only after that freeze run the unchanged typed, CSS and public
   scorers once. Report T, H, v3 score, Choice/Noul/Score, four typed
   families, 15 transfer tasks, invalidity and probability quality. Report
   public231 overall and easy/standard/hard separately.
4. A paired interval against JPT-9B or a later eligible Decision 2.0 model is
   computed only when that model has full predictions for the same panel and
   scorer hashes. JPT's already published same-panel v3 60.994 and public231
   197/231 may be shown as an external peer, not as a Decision 2.0 result.
   Older Lux historical DEV/public card numbers are not merged into this
   run's rank. The public result is an independent **public-subset
   reproduction**, never an official closed JevBench ranking.

No HF upload or model publication occurs in r3. Keep raw prompts,
predictions, keys, logs, internal paths and host details in private receipts;
the repository and unified gist receive only aggregate outcomes and
non-sensitive digests.
