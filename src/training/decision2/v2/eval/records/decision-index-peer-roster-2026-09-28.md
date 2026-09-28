# Decision Index 0.2.1 same-size peer roster, refreshed 2026-09-28

This is a comparator **selection** record for Decision 2.0, not a result.
**Decision Index numbers below are used only to choose peers. They must never
enter our JevArena or JevBench rank charts and must never be compared
numerically with our results.** Every ranked number we publish must come from
the peer rerun on our frozen panels with our scorer. The machine-readable twin
is [`decision-index-peer-roster-2026-09-28.json`](decision-index-peer-roster-2026-09-28.json),
which also maps all 70 entrants to tiers.

## Sources

- Space [`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index)
  at `7cdcea3dd14615192ff2e1f6fd13936a547b55d8`. The public Space API returned
  that `sha`, `lastModified` 2026-09-28T00:40:34Z, public, not gated, not
  disabled, both at the start and at the end of this work.
- Selection file `data/index.json`, SHA-256
  `a5a4aa0a2cce152056964a5c732c919f55fd8e0d1115f9cb7f849c35322eb9df`
  (952,794 bytes; byte-identical to `data/index-v0.2.1.json`). Methodology
  `data/methodology.json`, SHA-256
  `2ff75bb862498ddba4c4e39698bbe57118ad26252d21fa4de6e6ba4641ee1612`
  (= `methodology-v0.2.1.json`). Not used: `data/index-v2.json`
  `fad0e6b0ee996de2543c2aecb73325464338b8437286ebc8162b652efd8d51bc`
  (edition 0.2, 64 entrants) and `data/methodology-v2.json`
  `903cee829e68a7cb4592f91ba55558a5c725f8c80692cf0c846efa8b34f67b2f`.
  All eight data files match the Space tree blob ids at this revision.
- Edition inside the files: `suite.edition` `release-v2.1`, `suite.label`
  "Decision Index 0.2.1", `panel_id` `decision-index-0.2.1`; methodology
  edition `v0.2.1` dated 2026-09-28. Generated 2026-09-28T00:39:36Z (index)
  and 00:39:48Z (methodology). Position metric: `suite.headline` =
  `balanced_skill` (chance-corrected, 38 benchmarks, five areas). "Index #" is
  the rank among the 70 entrants by that value.
- 70 open entrants: 55 publish their own weights; 15 are inference techniques
  that serve a stock checkpoint through their own code. The hosted Jev 1.13.0
  API (separate `jev` block, 57.91) is closed and excluded.
- Index `served_params` is the backbone size (methodology
  `parameter_count_precedence`), so every count below was measured by us.
  We read safetensors headers (8-byte length plus JSON) and GGUF headers via
  HTTP range requests, and parsed the zip directory and pickle opcodes of
  `head.pt` without executing them. No weights were downloaded. All access was
  public and unauthenticated: HF API/tree/resolve, `git ls-remote`, blobless
  GitHub clones and raw files.

## Tiers and selection rule

Size is the number of parameters the model's native text-decision path
instantiates. Tied weights count once and a LoRA merged at load adds nothing.
Vision, audio and MTP tensors that are stored but not used for text readout
are listed separately. Each model goes to the nearest tier center in log space
(0.6, 0.8, 2, 4, 9, 27 B dense; MoE checkpoints only to the 26 B MoE tier).
A model is **same-size** if `max(P/c, c/P) <= 1.25`, the project's frozen
same-size ratio, and **adjacent** (optional stress test only) up to 1.5.
Same-size bands: 0.6B 0.48-0.69, 0.8B 0.69-1.00, 2B 1.60-2.50, 4B 3.20-5.00,
9B 7.20-11.25, 27B dense 21.6-33.75, 26B MoE 20.8-32.5 B total.

Within a tier, candidates are taken in Index order and we pick the top two or
three. We skip our own `llm-semantic-router/Decision-1.0-*` models (shown for
context only), hosted systems, inference techniques without their own weights,
and models with licence or weight blockers. Every skip is listed with its reason.

C/N/S means native support for Choice/Noul/Score: Y (native), P (projection
through another head or generic options) or N (none). Our eval hosts are AMD ROCm gfx942
([`inference/README.md`](../../../inference/README.md)), so CUDA-only runtimes
are listed as blockers.

### 0.6B

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `Hanno-Labs/bosun-v3.1-0.6b@1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b` (09-23) | 49 / 14.32 | 606,131,200 = `Qwen/Qwen3-0.6B@c1899de2` tied body 596,049,920 − 11,264 (vocab resized to 151,925) + LoRA 10,092,544; 524,288 decision rows overwrite existing rows | Apache-2.0 (LICENSE, NOTICE) | remote-code `BosunForDecision.predict()`, 256 decision-token slots, ≤255 options, no truncation; Y/Y/Y | `bosun06.py` at same rev; [v3 result](../../../research/kai06-v3-postkey-result-2026-09-27.md) at same rev | none |
| Peer | `fastino/GLiNER2.5-Decide@5a7adf72a23b4d311abae6ce050d7f0012bb3416` (09-28) | 53 / 11.21 | 486,444,053 (F32, ratio 1.233) | Apache-2.0 (card only); DeBERTa-v3-large MIT | `gliner2@55656fbf` classifier over caller labels, 512-token encoder window; Y/P/P | `gliner25.py` pins `7ee5da4c`: README-only diff, still valid; public231 pilot only, no v3 | low |
| Own | Decision 1.0 Kai / Lex | 59 / 6.52; 63 / 4.54 | Kai 571,909,635 (our record) | — | context only | — | — |
| Skip | jeff | 55 / 8.04 | stock `knowledgator/gliformer-large-v1` | — | inference technique, no own weights | — | — |
| Skip | Lumma-Fev-0.6B | 67 / 2.97 | 649,282,476 | Apache-2.0 | in band but not first tier (near chance) | — | — |
| Skip | Lavoir; Laya | 54 / 8.69; 60 / 6.04 | ~0.42B each | CC BY-NC 4.0; Apache-2.0 | adjacent (ratio 1.42) | `laya.py` covers the separate `laya-typed-decisions` repo | — |

### 0.8B

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `kirp/jpt-0.8b@1431c0509bbc10772cd59964cc7af5835c8720d6` (09-27) | 46 / 19.22 | 852,985,920 = text 752,393,024 + vision 100,592,896 (unused by text) | **CC BY-NC 4.0** (card); Qwen3.5 Apache-2.0 | `llm2jev@2b252d50` label log-probs on chat template, T=1.140, 2-255 options; Y/Y/Y | `jpt.py` is 9B-only (model id, rev, T constants); no v3 | low |
| Peer | `jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e` (09-24) | 48 / 14.60 | 752,917,824 = `Qwen3.5-0.8B-Base@dc7cdfe2` text + FP32 pointer head 524,800 (LoRA 10,822,656 merged; base vision/MTP not loaded) | Apache-2.0 (card) | `kev@45923b7a` `Checkpoint.load`/`DecisionModel.probs`, strict 8,192-token limits; Y/Y/Y | `kev.py` pins Kev-4B and source `6d02f5d0`; no v3 | low-medium |
| Peer | `internlm/Intern-Decision-0.8B@85a0cc5a99d67ea8d56dfe98115689212867171d` (09-26) | 51 / 11.94 | 852,985,920 = text + vision 100,592,896 (its VLM loader builds both) | Apache-2.0 (LICENSE, LICENSE-QWEN) | bundled `inference.py` `DecisionEngine`, one forward, ≤62 options, 8,192 tokens rejected, T=2.748; Y/Y/Y | none; no v3 | low-medium |
| Own | Decision 1.0 Eos | 47 / 18.41 | — | — | context only | — | — |
| Skip | Tev1-0.8B-experimental | 50 / 12.85 | 873,438,784 | **none granted** | card: weights licence "being finalized"; letter generation, Choice only, 2-24 options | — | — |
| Skip | MoJev | 52 / 11.69 | 854,036,544 | MIT | next after the three picks | — | — |

### 2B

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `Mapika/decider-2b@533964dae8be954c5b5e19fa4948e48408094c1e` (09-24) | 37 / 28.97 (Index served FP8) | 1,881,825,088 BF16 text | Apache-2.0 (card) | bundled `decider/` `system_one`, option-letter logits, per-type T, 255 options, 32,768 state tokens; Y/Y/Y | `run.py --backend decider`; [v3 result](../../../research/sol2b-own-source-formal-v3-result-2026-09-27.md) at same rev | none |
| Peer | `flock-io/this-that-model-1.2@c4d1c30b8d512d278726de439b8cc81fccc70c8f` (09-23) | 39 / 28.14 | 1,881,825,088 BF16 | MIT (card: adapted from Apache-2.0 Decider 2B) | `thisthat.TypedDecider.decide`, source `f57c9f0` (1.2 release); 1,536-token state cut unchanged since 1.0; Y/P/P | `this_that.py` pins 1.0 (`3d927195`, source `4efe782c`); no v3 | low |
| Peer | `Hanno-Labs/bosun-v3.1-1.7b@1d8dc82a20e4a32ed60927a47272d6efff48eed2` (09-23) | 44 / 20.10 | 1,737,985,024 = `Qwen3-1.7B@70d244cc` tied body 1,720,574,976 − 22,528 + LoRA 17,432,576 (same loader as 0.6B; not run-verified) | Apache-2.0 (LICENSE, NOTICE) | same Bosun contract; Y/Y/Y | `bosun06.py` is 0.6B-only; no v3 | low |
| Own | Decision 1.0 Sol | 42 / 25.32 | — | — | context only | — | — |
| Skip | Intern-Decision-2B; LFM2.5-2.6B-RLCD | 45 / 19.38; 58 / 6.76 | 2.21B; 2.70B | Apache-2.0; other | next pick; adjacent (1.35) | — | — |

### 4B

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `kirp/jpt-4b@78312f855b8bebf83ae7e9e8f05b5a0f73f519a9` (09-27) | 15 / 43.04 | 4,539,265,536 = text 4,205,751,296 + vision 333,514,240 | **CC BY-NC 4.0** | `llm2jev@2b252d50`, T=1.036; Y/Y/Y | `jpt.py` needs parameters; no v3 | low |
| Peer | `michaljach/jet@fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a` (09-26; tree = tag v6.2.0) | 16 / 42.60 | 4,205,751,296 BF16 text | Apache-2.0 (LICENSE) | bundled `jet.py` `Jet().decide`, label-token readout, 16,384 tokens without truncation; Choice 2-255, Score 2-10 levels; Y/Y/Y | none; no v3 | low-medium (card: CUDA + FLA 0.5.2) |
| Peer | `HopitAI/hopper-g@71d991f4df78c445a50a4bc952748cea89c2ac21` (09-27) | 18 / 40.77 | 4,205,751,296 text (LoRA 32,464,896 merged into `Qwen3.5-4B@851bf6e8`; base also stores vision 333,514,240 + MTP 120,599,552) | **research-and-demo, no commercial use** (card); code Apache-2.0 | `hopit-ai/hopper` tag `g-1.2.0` → `0204f929`: letter readout, per-kind calibration, ≤26 options/levels, larger Choice menus via disclosed shortlist; Y/Y/Y | none; no v3 | medium |
| Keep | `Mapika/decider-4b@eb5fbdfc9448473ec25e399882912863afbdb70e` (09-24) | 19 / 40.70 | 4,205,751,296 BF16 text | Apache-2.0 (card) | `system_one`, 255 options, 32,768 state tokens; Y/Y/Y | `run.py --backend decider`; [v3 result](../../../research/decider4b-v3-peer-result-2026-09-27.md) at same rev | none |
| Own | Decision 1.0 Nox | 32 / 34.36 | 4,208,383,488 (our record) | — | context only | — | — |
| Ctx | JevK5; Kev 4B | 24 / 38.81; 31 / 34.64 | 4.21B each | Apache-2.0 | below the picks | `jevk5.py` at `c4f7fdb3` (= main), no v3; Kev 4B [v3 result](../../../research/decision2-4b-jevarena-v3-first-release-hold-2026-09-27.md) at `139fdd94` (= main) | — |

### 9B

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `kirp/jpt-9b@b447cc7ee105c0a76a22f8fde8ecf05074dc8be0` (09-27) | 13 / 46.89 | 9,409,813,744 = text 8,953,803,264 + vision 456,010,480 | **CC BY-NC 4.0** | `llm2jev@2b252d50`, T=1.087; Y/Y/Y | `jpt.py` at `7114b0c3`; [v3 result](../../../research/jpt9b-native-v3-peer-result-2026-09-28.md) at `7114b0c3` (README/assets-only diff to main) | none |
| Peer | `EldanRing/Winnow-E4B@734302fe5fbfeb3f21a7ece62653c9539be4aaf3` (09-24) | 22 / 39.89 (Q8_0) | 7,518,069,290 in the Q8_0 GGUF LM (same count in BF16), incl. 2,818,572,288 per-layer embeddings, so "E4B" is the effective size; the 478,088,384-parameter projector is not loaded text-only | Apache-2.0 (LICENSE, NOTICE) | `winnow-inference@77d14580` (patched llama.cpp) `/v1/systemone`, T=1.2574; Y/Y/Y | none; no v3 | **high**: CUDA/Metal backends only |
| Peer | `bespokelabs/Bespoke-Nimble-9B-v2@4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe` (09-23) | 23 / 39.57 | 9,453,092,080 = `Qwen3.5-9B@c2022362` text + vision (VLM class) + unmerged LoRA 43,278,336; MTP 243,290,624 not loaded | Apache-2.0 (LICENSE) | bundled `ParallelScorer.score`, answer-token logits, T=2.179 (transferred, not refit), ≤255 choices, 8,192 tokens rejected; Y/Y/Y (Score as ordinal enum) | none (CPU preflight only); no v3 | low-medium |
| Own | Decision 1.0 Lux | 14 / 43.49 | — | — | context only | — | — |
| Skip | Winnow-12B; Jev-Omni | 9 / 50.02; 20 / 40.53 | 11,907,350,576 (GGUF LM); 11.96B | Apache-2.0 | adjacent (1.32; 1.33); Winnow-12B = optional larger stress test | — | — |
| Skip | Kev 9B | 26 / 38.48 | ~9.7B | Apache-2.0 | next after the three picks | — | — |

### ~27B dense

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015` (09-22) | 3 / 56.40 | 26,086,635,760 = text incl. readout 25,625,905,664 + vision 460,730,096 | Apache-2.0 weights, MIT code | bundled `source/` `DecisionModel`, `autojev@ee63c151`; Y/Y/Y | `autojev27.py` at same rev; [v3 result](../../../research/autojev27-v3-public-peer-result-2026-09-28.md) at same rev | none |
| Peer | `frontier-infra/jebadiah-27b@c68db2b5570f3c2bdab511088885a88bd8b3a814` (09-26) | 5 / 54.67 | 26,895,998,464 via the text-only `Qwen3_5ForCausalLM` loader (stored 27,781,427,952 incl. vision 460,730,096 + MTP 424,699,392) | Apache-2.0 (LICENSE) | `scripts/decide_standalone.py`, label-token logits at last position, per-type T; `/v1/systemone` Choice ≤20 options (68-label alphabet); Y/Y/Y | none (27B plan: `hold_unqualified`); no v3 | medium |
| Peer | `caiovicentino1/Eikos-27B-FP8@bbdb02333652ad91cfa0dc2163efe8429138eb66` (09-27) | 6 / 53.13 (FP8) | 27,356,728,560 under vLLM = text 26,895,998,464 (24,350,556,160 FP8 E4M3) + vision 460,730,096; MTP 424,699,392 not loaded; plus 3,904,000 scales | MIT + Qwen Apache-2.0 (LICENSE, LICENSE-Qwen, NOTICE) | letter-logit readout via vLLM ≥0.30 `serve_vllm.sh`, ≤160 options per pass, 16,384 max length; Y/Y/Y | `eikos.py` covers Eikos-4B only; no v3 | medium-high |
| Skip | Decider chat · Gemma-4-31B | 2 / 57.33 | stock `google/gemma-4-31B-it@842da379` (31,273,088,876) | Apache-2.0 | inference technique with `Mapika/decider` code, no own weights; optional technique comparator | `decider` backend is for trained packages | — |
| Skip | simple-jev; reflex 27B v2; Decider chat · Qwen3.6-27B; Jevfire | 4; 7; 8; 11 | stock Qwen3.8/3.6-27B | Apache-2.0 | inference techniques, no own weights | — | — |

No own Decision 1.0 model exists at this size.

### ~26B MoE (Gemma-4-26B-A4B class)

| Role | Model @ pinned `main` (lastModified) | Index # / skill | Loaded / active params | Licence | Native path; C/N/S | Our adapter; prior v3 | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Peer | `surogate/rune-26b-a4b-GGUF@c6b360d47895bb77bdf3805a13ee5a5557ab1921` (09-26; BF16 safetensors despite the name) | 1 / 57.44 | 25,233,141,790 text (+ vision 572,794,416 only with `--vision`; stored 25,805,936,206); **active 3,822,530,590/token** | Apache-2.0 (card) | `surogate@9071a52a` `/api/alpha/decisions` (C++/CUDA 13); card documents a transformers letter-softmax fallback; Y/Y/Y; 262k context, thinking off by default | none; no v3 | **high** |
| Adj | Decider 35B-A3B: Index ran `Mapika/decider-35b-a3b-nvfp4@798555c06e419c4638c9ebd06c78ed8b5e92c868`; BF16 sibling `Mapika/decider-35b-a3b@91470e65478656eb68fde8c0ccdfe575d69d0309` | 12 / 47.11 | 34,660,610,688 (ratio 1.333); active 3,454,988,928 | Apache-2.0 | `system_one`; NVFP4 build targets Blackwell | `run.py --backend decider` with the BF16 sibling (not the exact Index artifact) | low-medium |
| Adj | `juspay/xor@8d2d70892c0d5f18e197793fffbfb3374a2c5fc8` | 17 / 41.48 | 35,107,181,936 = text 34,660,610,688 + vision 446,571,248 (ratio 1.350); active 3,454,988,928 | Apache-2.0 | patched-SGLang serving bundle, forward+reverse option order; 12.1% unanswered on the Index | none | high |
| Skip | JoshuaSP diffusiongemma; djev; razorback; vLLM PR 57250 | 10; 21; 28; 34 | stock `diffusiongemma-26B-A4B-it` (25.8B) | Apache-2.0 | inference techniques (diffusion), no own weights | — | — |

How the MoE active counts were computed from header tensor shapes:

- **Rune:** 30 layers × 128 experts × (gate_up 1408×2816 + down 2816×704 = 5,947,392) gives 22,837,985,280 routed parameters. The other 2,395,156,510 text parameters include the tied 738,197,504 embedding. Active per token = 2,395,156,510 + 30×8×5,947,392 = 3,822,530,590 (3,084,333,086 without the embedding).
- **Decider 35B-A3B and Xor (text):** 40 layers × 256 experts × 3,145,728 gives 32,212,254,720 routed parameters. The other 2,448,355,968 include shared experts 125,911,040 and the untied embedding 508,559,360. Active per token = 2,448,355,968 + 40×8×3,145,728 = 3,454,988,928.
- **FP8 and NVFP4 counts:** NVFP4 stores 16,808,673,280 packed U8 elements (two values each) + 1,043,264,128 BF16 = 34,660,610,688, the same as the BF16 sibling, plus 2,101,146,100 scale elements.

## Identity checks since earlier runs

We compared tree `expand=true` LFS SHA-256 and blob ids at each earlier
revision with `main`. The only differences were cards, assets, checksums or
metadata; weights, configs, tokenizers and templates were unchanged:

- JPT-9B `7114b0c3→b447cc7e` (README, 3 PNGs).
- JPT-4B `277ae37b→78312f85` (README, PNGs, `.gitattributes`).
- JPT-0.8B `5921ae03→1431c050`.
- GLiNER2.5-Decide `7ee5da4c→5a7adf72` (README).
- Rune `bd4a7cbb→c6b360d4` and Jebadiah `1c0d794f→c68db2b5` (README only).
- Hopper-G `d60a1d6c→71d991f4`: `adapter_config.json` `task_type` null→`CAUSAL_LM`, checksums and card; adapter weights and `hopper.json` unchanged.

Runtime HEADs equal our pins for `llm2jev` `2b252d50`, `GLiNER2` `55656fbf` and
`autojev` `ee63c151`.

Qualification caveat: Kev-0.8B's `provenance.json` `head_sha256` (`bc77488a…`)
does not equal the LFS SHA-256 of `head.pt` (`f400bd12…`). Confirm the
publisher's hashing scheme at qualification.

## New vs prior roster

The prior record [`decision-index-021-peer-roster-2026-09-27.md`](../../../research/decision-index-021-peer-roster-2026-09-27.md)
used `index.json` SHA-256 `0e48d92d…f73c`. We matched that file exactly to
Space commit `34426d6a979c066e397aff63cf1385b4cd990f6c` (67 entrants,
generated 2026-09-27T03:42:13Z).

Board changes since then (commits `c3e02501`, `f1d9308d`, `df7f948e`, `7cdcea3d`):

- Hopper (G) 1.2 replaced Hopper 1.1.1 (39.67).
- Eikos-27B-FP8 (53.13), JPT-4B (43.04) and Jet v6.2 (42.60) joined.
- The other 66 entrants have identical scores and metadata.

Pick changes:

- **0.8B:** adds Kev-0.8B and Intern-Decision-0.8B; Tev1 is skipped on licence.
- **2B:** adds Bosun 1.7B.
- **4B:** the top three are now JPT-4B, Jet v6.2 and Hopper (G) 1.2, and Decider 4B stays as a near-tied fourth.
- **9B:** adds Winnow-E4B (7.52B by actual count) and Nimble v2.
- **~27B:** the prior row is split. The dense tier is AutoJev, Jebadiah and Eikos-27B-FP8; Decider chat · Gemma-4-31B is now a technique comparator. The MoE tier is Rune v3, with the 35B-A3B models only adjacent.

## Actions for the eval track

1. **Reuse by identity, no rerun:** Bosun 0.6B `1d8b6f96`, Decider 2B
   `533964da`, Decider 4B `eb5fbdfc` and AutoJev 27B `6f5b557e` are unchanged
   on `main`. Keep JPT-9B at `7114b0c3`: `main` differs only in card and
   assets, and `jpt.py` enforces the pin. Kev-4B's result at `139fdd94` is
   also current.
2. **Existing adapter family, re-pin or parameterize, then run:**
   - GLiNER2.5-Decide (pin still valid; needs its first v3 run).
   - JPT-0.8B and JPT-4B (model id, revision, T 1.140/1.036).
   - Bosun 1.7B (verify 1,737,985,024 at load).
   - this-that 1.2 (weights `c4d1c30b`, source `f57c9f0`).
   - Kev-0.8B (source `45923b7a` and its hashes).
3. **New adapters over bundled public runtimes:** Intern-Decision-0.8B, Jet
   v6.2, Nimble v2 and Jebadiah 27B (fix the 20- vs 68-option behaviour
   first) are low-medium; Hopper (G) 1.2 is medium.
4. **High effort or blocked on ROCm:**
   - Winnow-E4B: llama.cpp with CUDA/Metal backends only.
   - Rune v3: surogate is CUDA-only, so a transformers fallback needs a parity check.
   - Eikos-27B-FP8: FP8 vLLM on ROCm; a BF16 sibling `caiovicentino1/Eikos-27B@103a5647` exists but is not the board artifact.
   - Xor: CUDA SGLang image.
5. **Licence review before publishing peer rows:** JPT-0.8B/4B/9B (CC BY-NC
   4.0) and Hopper (G) (research and demo only). Tev1 stays excluded until its
   weights are licensed.
6. Put only same-panel reruns in charts. Index values stay in this selection
   record.
