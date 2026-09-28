# 0.6B encoder track: M2a diagnostics and Milestone 2 proposal

Development readouts only; no formal v3 or public 231 run was made.

## M2a results (preregistered in `m2a-prereg-2026-09-28.md`)

| Arm | SELECT BEST (step) | Typed DEV T | Choice / Noul / Score | CSS pilot H | Proxy P | Note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| L8H, seed 1 (`m1-l8h`) | 460 (102) | .3881 | 222 / 191 / 208 | .2016 | 27.97 | Milestone 1 |
| L8H, seed 2 (`m2a-l8h-seed2`) | 498 (117) | .3944 | 231 / 192 / 208 | .2134 | 29.01 | order seed only |
| EB610, LR 1e-5 + 10% warmup (`m2a-eb610-lr1e5`) | 247 (175) | .3313 | 232 / 192 / 106 | .0849 | 16.77 | collapsed again |

- **Seed-order noise is large.** Changing only the data order moved SELECT by
  38/700 and the proxy by 1.04. SELECT differences under ~40 answers between
  single-seed arms are not interpretable; any finalist needs at least two seeds.
- L8H beats unchanged Lex1@8K (26.01) on both seeds (+1.96, +3.00), mainly on
  CSS pilot discourse, but stays below the 31.02 promotion bar and still
  predicts one constant typed-DEV Score level.
- **EuroBERT-610m is blocked.** Halving the backbone LR and doubling warmup did
  not stop the collapse (gradient norm ~20 → ~0.01, loss at chance). mmBERT-base
  trains under the identical code, head, data and recipe, so the cause is
  specific to EuroBERT in this setup (candidates: BF16 autocast with its custom
  4D float mask, its checkpointed layer call, or marker-state degeneration), not
  the learning rate. No further EuroBERT arm without a root-cause diagnostic.
- Receipts: L8H seed 2 export `f9deb7403b94e8282a4fe144b84b0089f00992cee688702266345cbad2b4af9d`,
  DEV predictions `da9800ae…68e477`, CSS pilot `9be83a13…d04a8`; EB610 LR repair
  export `a27ec68a520d92d16693e6d8fa4c4f62ed7f1a53ffa0dde8e63a85b9c81d4e91`,
  DEV `8c003f61…23e34`, CSS pilot `cd5f9520…7eb1f`. M2a GPU-hours 0.27.

## What the same-panel baselines add

The eval track's post-key v3 baselines (`01-decision-2-eval-peers.md`) show
**Lex1 at v3 31.022 (T .3619, H .2659) below Kai1 at 35.938 (T .3619, H
.3569)**, while the development proxy ranks Lex above Kai (26.01 versus 21.55).
Typed DEV uses different families from typed FINAL, and the three CSS pilot
tasks do not track the 15-task median. The proxy therefore cannot by itself
choose between lineages. L8H inherits Lex's weak transfer, so a Lex-lineage
gain of ~2–3 proxy points is not evidence of a v3 gain over Kai.

At 0.6B the strongest same-panel transfer remains the official-Qwen causal
control (H .4801, v3 38.520) and the strongest typed score Bosun v3.1 (T .4338,
v3 38.524, the only 0.6B model with a paired interval above Kai1:
+2.586 [+0.374, +9.089]).

## Milestone 2 proposal

1. **Bidirectional official-Qwen marker encoder (backup-weights arm).** Start
   from `Qwen/Qwen3-0.6B-Base@da87bfb6`, run attention bidirectionally over
   question, candidates and state with Kai's marker layout and the shared
   candidate head, same data and recipe as the causal control (466 updates).
   Hypothesis: keep the control's transfer (H .48) while the joint candidate
   encoding repairs typed Choice (109/800 causal versus 277/800 for Kai's joint
   encoder). Two seeds. About 0.8 GPU-hour including readouts. Needs the user's
   agreement, since the brief treats official Qwen 0.6B as a backup.
2. **Kai1@8K formal post-key run (runtime-only control, ~0.1 GPU-hour).** Kai
   answers none of the 402 long CSS items it currently fails at 1,024 tokens;
   this isolates the cap effect before crediting any 8K-trained model with it.
3. **Own-Lux1 soft-teacher screen, then distillation.** Lux1 is our own 1.0
   model (v3 65.8–66.3). Screen its native distributions on 96 TRAIN rows under
   the eval track's frozen Lux1 runtime (the cross-node repeat currently fails
   by 43/8,778 categories), with the same thresholds as the own-Kai screen; only
   if it passes, distil into the best 0.6B student on the TRAIN inputs. About
   0.6 GPU-hour.
4. **Data arms on the best student** once the data track freezes them, Score
   levels first (A6): every 0.6B model here predicts one constant typed-DEV Score
   level, and TRAIN has only 102 three-level Score rows. Then hard distractors
   (A4) and option-order copies (A0p). About 1.0 GPU-hour.
5. **EuroBERT root-cause diagnostic** (~0.1 GPU-hour) before any new EuroBERT
   arm: FP32 versus BF16 one-update gradients, checkpointing on/off, and marker
   versus candidate-span pooling on fixed TRAIN rows.
6. **Selection discipline.** Two seeds for any finalist; at most two finalists
   per milestone go to the eval track's frozen same-panel runner
   (`v2/eval/run_same_panel.sh`, with this track's native 8K collector as the
   adapter spec) and are compared with Kai1, Lex1, Bosun and the causal control.

GPU need: node A GPU0–1 suffice for this plan, about 3–4 GPU-hours.
