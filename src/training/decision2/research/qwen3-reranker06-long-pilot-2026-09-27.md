# Decision 2.0: preregistered 0.6B long-context reranker pilot

Status: **completed, negative architecture-advance verdict**. The original plan and v2 memory amendment were preregistered before their respective optimizer steps. This is one bounded, noncommercial research arm. A source-versus-trained SELECT gain alone cannot qualify a Decision 2.0 checkpoint. No FINAL, CSS15 or other sealed gold was used for training, selection or this pilot's score gate. Raw rows, individual predictions and checkpoint bytes remain private.

## Candidate comparison and reason for the arm

| Candidate | Measured parameters | Native context | Evidence and limitation |
| --- | ---: | ---: | --- |
| Decision-1.0-Kai-0.6B | 571,909,635 | 1,024 | Published typed scorer; previous continuation has answer-class collapse and 22 CSS pilot long-input overflows. |
| Laya typed decisions | 421,293,830 | ModernBERT 8,192 positions; released call budget 1,024 | Same-weight 4,096-call ablation improves coverage but its trained arm loses typed DEV accuracy. |
| GLiNER2.5-Decide English | 486,444,053 | 512 | Prior fixed 64-step continuation gained SELECT but fell 652→584 on typed DEV. |
| GLiNER2.5-multi-Decide | 287,355,159 | 4,096 | Better coverage and multilingual reach, but smaller than the requested 0.6B slot and weaker on typed DEV than English GLiNER. |
| [Jina embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3) | HF tensor count 572,310,396 | 8,192 claimed | Multilingual embedding encoder with custom remote code and CC BY-NC 4.0; its similarity objective needs a new typed decision readout. Not selected for this one arm. |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | HF tensor count 595,776,512 | 32K claimed | Apache-2.0 multilingual embedding; retrieval pooling would require a new decision head. Not selected for this one arm. |
| **[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)** | **595,776,512 loaded** | 32K model-card claim; 40,960 configured positions; **4,096 actually forward-tested** | Apache-2.0, multilingual, instruction-aware yes/no relevance scorer. The published terminal yes/no logits can rank every Choice, Noul or Score candidate while keeping the full state and all alternatives in view. This is the sole training candidate. |

The reranker is pinned to Hugging Face revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, from Qwen3-0.6B-Base. Its local model/config/tokenizer SHA-256 values are `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`, `d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0`, and `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. On the authorized AMD environment, BF16 loading measured 595,776,512 parameters; a 4,096-token forward with only terminal logits returned finite scores. This validates 4,096-token *loadability*, not a 32K quality or throughput claim. [Qwen's published native reranker card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) supplies the prompt wrapper and yes/no token scoring method. Our task-specific `<Instruct>/<Query>/<Document>` content and full-slate option normalization are explicitly a Decision adapter, so future comparisons must use the same adapter version.

## Frozen data, mapping and compute

Use the existing rights-clean v2 TRAIN7,455 and disjoint SELECT700 only; byte hashes are `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` and `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`. The builder rechecks source bytes and ID/group/input isolation. Gold enters only the private TRAIN loss. For each candidate, the gold-free adapter presents the entire option slate and state under the official reranker wrapper, scores the terminal `yes` and `no` logits, and softmaxes the yes-minus-no margins over options. Choice chooses the highest candidate, Noul returns the yes probability, and Score returns the top ordinal level plus expected level. All invalid or overflowing answers count as failures. No truncation is permitted.

Set the adapter maximum to **4,096 tokens** even though the model advertises 32K, so long-sequence cost remains bounded and empirically load-tested. A deterministic ID-hash draw selects 512 TRAIN rows: Choice 96 structured + 96 other; Noul 96 + 96; Score 96 + 32. Structured means the `stage4_` source families. Within structured quotas, force at least 32 Choice, 32 Noul and 8 Score rows above 512 native tokens, without looking at SELECT or DEV labels. The preflight admitted 512 rows (192 Choice, 192 Noul, 128 Score), of which **157 exceed 512 native tokens**; the longest is 4,042. Whole-row admission excludes 66 Choice, 40 Noul and 9 Score TRAIN rows above 4,096. The exact private TRAIN slice hash is `32a1226931967fd7a0f53cb2a4fd51189d5b643c56eeb1b88cea94ebf4312398`.

Run **one** LoRA arm from the immutable published reranker: rank 8, alpha 16, zero dropout, Q/K/V/O and gate/up/down projections, BF16 backbone compute, AdamW LR `2e-5`, weight decay `.01`, gradient norm cap 1.0, single row per microbatch with all its candidates, accumulation 8, exactly **64 updates / one pass over 512 rows**, seed 20260927. Full-slate cross-entropy on candidate yes-minus-no margins is the sole loss. No checkpoint selection, threshold tuning, calibration fitting or data change is allowed after updates begin; evaluate the fixed final adapter. If source load, backward, saving or native reload fails, record it and stop. Any recipe change needs another preregistration.

## Source-matched panels and advance gate

Evaluate the frozen source and fixed final adapter through `qwen3-rerank06-full-slate-v1` on the **same** SELECT700 and typed DEV1,600. Typed DEV is an independent synthetic development panel with 400 groups and four structure families; it is not the sealed release set. Bind source/model/adapter/prompt/scorer/data digests, report Choice/Noul/Score and the four DEV families, all invalids, and 400-group paired uncertainty. Prediction files are generated gold-free before the DEV scorer reads development gold. No CSS15, FINAL or public-ranking panel is included in this arm.

**Architecture-advance gate**, predeclared without model output: final adapter must gain at least **21/700** SELECT correct and **.03** SELECT family-macro accuracy without a type drop over .05 or increased invalids **and** gain at least **32/1,600** typed DEV correct over source, exceed the existing English GLiNER **652/1,600**, show positive family-macro DEV change, and have no DEV family drop exceeding **.02**. Report SELECT and DEV regardless of which gate fails, so a source-matched SELECT gain cannot hide a cross-distribution loss. Any failure means research-only negative arm, no release or sealed evaluation. Passing only permits a subsequent multilingual, CSS pilot, calibration and robustness audit; it does not itself justify publishing `dev-2.0-0.6b`.

This short pilot has no Chinese or other non-English training examples and cannot establish multilingual transfer. It also cannot prove 32K decision quality from a 4,096-token operational budget.

## Pre-optimizer panel audit and source scores

Before starting any optimizer update, the source reranker produced **302/700** on clean SELECT, family-macro **.37798**, with 700 valid; Choice 158/320, Noul 139/290, Score 5/90. Source typed DEV scored **422/1,600**, all valid, with family correct counts attribute gate149, rule precedence181, set reconciliation80, transition table12 (each 400). Source SELECT prediction SHA-256 is `88d3e50cf11cd7dd2f0adc30f04c7e3b84075762bb93c1f10868ea070fb19024`; source DEV prediction SHA-256 is `d85be1e054e7912bc30709b0282d37fbbe8979e5c60e98cae349680664f28b0d`. The native adapter answered one gold-free DEV smoke prompt before the full source run.

An additional possible development split was audited for lineage and native token lengths, **without opening its gold**. It turned out to consist entirely of CSS pilot Choice tasks, so it is excluded from this arm under the no-CSS-heldout rule. Crucially, all 700 clean SELECT and all 1,600 typed DEV prompts fit within 512 reranker tokens; neither panel isolates long-context generalization. This does not change the frozen architecture-advance gate above. The pilot will report long TRAIN admission and 4,096-token loadability, while treating real held-out long-input decision quality as **unmeasured**. A later independently reviewed, non-CSS long panel is required before any 0.6B release claim.

## v1 memory failure and v2 compute amendment, before v2 optimizer

The original **full-slate training recipe failed with GPU OOM before its eighth logged update**. No checkpoint was saved; because logging was every eight updates, zero to seven updates may have occurred in that abandoned process. The frozen first-64 draw contains a 128-option Choice row at position 57, with 2,213 native tokens per option, or about 283K batched tokens. The GPU reported 252.57 GiB PyTorch allocation when another 1.62 GiB was requested. This high-cardinality batch is a plausible trigger, but the exact failing row was not logged and is not asserted. The failed process is **discarded** and no partial optimizer state is reused. The source SELECT/DEV predictions above remain valid and the data/prompt/decision gate do not change.

Before any optimizer update of the replacement, freeze one **v2** recipe: restart from the same immutable source, same 512-row order, same 64-update/accumulation-8 budget and all original hyperparameters; enable gradient checkpointing and disable model KV caching during training. On every training row, render the full state and **all alternatives** in the Query, but optimize a deterministic set of at most eight terminal yes/no candidate margins: the gold candidate plus up to seven ID/key-hash-ranked negatives. For rows with at most eight options, retain the original full-slate loss exactly. The final native inference adapter still scores **every** offered option with no cap. Thus high-cardinality training uses sampled-negative cross-entropy and cannot be described as full-slate training; parameter count, model source, evaluation adapter and score gate remain unchanged. The v2 trainer source SHA-256 is `eafdaa1de3e04188b34023060f69c319515b0e2ccbd42b384278b7bc35bd7142`; candidate rendering source SHA-256 is `2892c0b577c5867d03c66ea72871077e26cf1144e2a605d9d1225b882971b9a1`.

If v2 encounters OOM, nonfinite gradients, save/reload failure or cannot finish exactly 64 updates, stop this candidate as a no-run. Only the fixed v2 final adapter, if completed, is compared with the already measured source on SELECT700 and typed DEV1,600. The preregistered dual-panel advance gate above applies unchanged. No long-context or multilingual quality claim follows even if it passes.

## Fixed-final v2 result and verdict

The v2 run completed **64/64 updates** from the immutable source in 120.6 seconds with maximum admitted training length 4,042 native tokens. The adapter saved, reloaded through the pinned native yes/no scorer, and returned 700/700 valid SELECT and 1,600/1,600 valid typed DEV answers. Signed local commits `cbb988486` and `ab1673076` hold the pilot and v2/paired-analysis code; `make check` passed after the second commit's source edits. The fixed adapter tensor SHA-256 is `d6fb8ac11f9e9cb95fc082b759c5ec5c716ad200c6efb735fe0bf41cc5e3a05a`. The private training receipt SHA-256 is `cad1ac32a3479b5b01532611ea2124167acdf0db84cc8e115176f537d87a3bf5`. No intermediate checkpoint was selected.

| Same frozen panel | Source | Fixed v2 final | Delta |
| --- | ---: | ---: | ---: |
| Clean SELECT correct / 700 | 302 | 351 | +49 |
| Clean SELECT family-macro accuracy | .37798 | .43355 | +.05557 |
| SELECT Choice / 320 | 158 | 167 | +9 |
| SELECT Noul / 290 | 139 | 160 | +21 |
| SELECT Score / 90 | 5 | 24 | +19 |
| Typed DEV correct / 1,600 | 422 | 431 | +9 |
| Typed DEV Choice / 800 | 161 | 164 | +3 |
| Typed DEV Noul / 400 | 181 | 185 | +4 |
| Typed DEV Score / 400 | 80 | 82 | +2 |
| DEV attribute gate / 400 | 149 | 139 | **−10** |
| DEV rule precedence / 400 | 181 | 185 | +4 |
| DEV set reconciliation / 400 | 80 | 82 | +2 |
| DEV transition table / 400 | 12 | 25 | +13 |

The 400-independent-group, family-stratified paired bootstrap gives DEV family-macro delta **+.005625**, 95% CI **[−.019375, +.030000]** (10,000 seeded draws); there were 200 candidate row wins and 191 losses. Attribute gate falls **.025**, beyond the frozen .02 maximum family decline. DEV gain is only +9 rather than the required +32, and 431 is far below the independently measured English GLiNER 652. Thus the clean SELECT half of the gate passes, but **the architecture-advance gate fails on three DEV conditions**. This remains a research-only negative arm; do not release or promote it to CSS/FINAL evaluation. The largest structural deficiency remains transition-table reasoning despite its +13 gain.

Calibration did not clearly improve: overall typed DEV Brier .345960→.346277 and NLL 1.129456→1.132348 worsened, while ECE10 .192239→.177123 improved; Score MAE .743041→.753536 worsened. SELECT source/final prediction SHA-256 values are `88d3e50cf11cd7dd2f0adc30f04c7e3b84075762bb93c1f10868ea070fb19024` / `8ecbe12707f048f3793463d3fe6ad68c9449f9ed7aace8ee48ed562bb53cc20c`. Typed DEV source/final prediction SHA-256 values are `d85be1e054e7912bc30709b0282d37fbbe8979e5c60e98cae349680664f28b0d` / `572bce9055461de3d9b2d519ad649cde82294967ba7d4c36c59b22a4965e9e35`. Private final SELECT score, typed DEV score and paired contrast SHA-256 values are `72caf24695eff764baaaaaa8f80df802366f61986b361eee114b67ee04641104`, `575cce92873fb86d0ec41595ba72ba9aca134a82b67add13dc4715abb4f260b6`, and `71b779b2fcdbc7b5206baebe29e074bd7c22f49fbb3fe3386c327058bdaff2c2`.

**Interpretation:** sampled-negative continuation clearly adapts the short clean SELECT mixture, but the gain does not transport to structured decision tasks and degrades attribute-gate generalization. This supports shifting the 0.6B architecture search toward native typed supervision and held-out long-input tasks rather than repeating more SELECT-driven reranker updates. All scored SELECT and DEV prompts were under 512 tokens, so this arm supplies no held-out long-context accuracy evidence. The multilingual claim likewise remains untested. A new arm requires a new preregistration and truly independent development evidence.
