# Reasoning wave 1 (4B) — preregistration

Track: Decision 2.0 Reasoning (`v2/reasoning/`). Written 2026-10-04 before any training run of this wave. Data hashes
are filled in by the data-lock amendment once the teacher graphs finish; nothing else changes after this file is
pushed except through numbered amendments.

## Question

Does supervising the intermediate judgments of a reasoning graph (each node asked with its parents' true
conclusions in the state; inference asks only the final question) raise the Jev Decision Index 0.2.1 score of the
released Decision 2.0 4B model, beyond what the same problems with final answers only give?

## Start point

- `vllm-sr/Decision-2.0-Nox-4B`, `main` `25e8f67d1b486c647222df3aac640d2d5d736bbe` (weights of the `4b-LRHxALL`
  release; Index 0.2.1 43.77). Training-format FP32 checkpoint on node F: `runs/af/4b/soup/4b-LRHxALL/build/4b-LRHxALL`,
  model SHA-256 `2fdee0bc8055115c6fa6f96b641cbaa086f60a15774d0637f5574d8da3f58be2`.
- Full continuation (`train_dec --init decision2 --train-mode full`); same architecture, head, prompt and runtime.

## Data (build `v2.reasoning.build_v1`)

- Program-verified graph problems (new instances, node truth from the program): arithmetic quantity graphs 4,500,
  code-execution traces 4,000, causal-graph queries 4,000, deductive logic (rule chaining, truth chains, ordering,
  boolean evaluation, object swaps, nested arithmetic) 5,000.
- Teacher graph problems: reasoning-heavy rows of the released model's own TRAIN (GSM8K, MathQA, programmatic rule
  decisions, HoVer, WinoGrande, MultiNLI, SNLI, SciTail, CommonsenseQA, QuaRTz, ROPES, MuSiQue, 2WikiMultihopQA,
  HotpotQA, COPA; English; licences as audited for the release). Teacher `openai/gpt-oss-120b`
  (`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, Apache-2.0) served locally with vLLM: two solutions, keep the problem
  only if one reaches the gold key; graph extraction; each node re-asked without the reasoning and kept only if the
  fresh answer agrees.
- Replay: 30,000 rows of the released TRAIN union (fixed-seed sample, teacher-problem rows excluded), labelled with
  the released model itself (T = 1); KL weight 1.0 on replay rows and on teacher-problem final rows.
- Node views per problem: at most 6 (3 when the state is longer than 6,000 characters), total weight 0.5 per
  problem; view forms native (Choice / Noul / Score 0–9), statement verification (polarity balanced) and
  multi-statement Choice with "None of these".
- Isolation: SELECT / CAL = the release's `sel700-cal698`; program train and dev use disjoint seed namespaces (dev
  finals equal to a train final are dropped); teacher problems split train / dev by row-id hash (4% dev).
- Decontamination: 13-gram scan of every non-replay row against the Index 0.2.1 suite (selected and added rows), the
  JevArena v3 typed and human-transfer prompts, JevBench public 231, HT-DEV v2, typed DEV, CSS pilot and mlx-diag. A
  flagged final drops its problem; a flagged node view drops the view. 13-grams in more than 25 evaluation records
  count as boilerplate.

## Arms (two seeds each: 20261004, 20261005; node F GPU2–7, one seed per GPU)

| Arm | Node views | Purpose |
| --- | --- | --- |
| `R4-TF` | true parent conclusions, weight 0.5 / problem | the method |
| `R4-TFM` | rewired conclusions (same count, non-parent non-ancestor non-descendant nodes, depth-matched) | control: graph structure |
| `R4-F0` | the `R4-TF` views at weight 1e-6 | control: same problems and batches, final answers only |

Shared trainer settings: `--backbone-lr 5e-6 --head-lr 5e-5 --warmup-ratio 0.05 --epochs 1 --weight-decay 0.01
--brier-weight 0.5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --max-length 8192
--checkpoint-schedule even8 --selection matrix-v1 --teacher-kl-weight 1.0 --teacher-partial` with the arm's
`--example-weights`. Every seed runs zero-step, one-step and the `preflight_dec` gate first; a failure stops that
seed and is recorded; nothing is rerun.

## Candidates and selection (dev only; no Index or JevArena item is used for any choice)

1. Per arm, the uniform FP32 soup of its two seeds' final checkpoints (the trainer's SELECT-based checkpoint choice
   is recorded but not used: in a continuation it favours the least-trained checkpoint; the interpolation below,
   with its SELECT floor, is the registered guard against forgetting).
2. Interpolation with the release: points α ∈ {0.5, 0.75, 1.0} (soup multiplicities [soup, release], [soup × 3,
   release], soup).
3. Dev readouts (`v2.reasoning.devread`): RP-DEV finals (program families, held-out seeds), RP-DEV node views
   (mechanism check, program and teacher dev problems) and SELECT 700.
4. The `R4-TF` candidate is the α point with the highest RP-DEV final macro accuracy among points whose SELECT
   family-macro accuracy is at least the release's minus 0.01; ties go to the larger α. The `R4-F0` and `R4-TFM`
   points at the same α are its controls.
5. Formal reads (at most three in this wave): the `R4-TF` candidate, and its `R4-F0` and `R4-TFM` controls, each as a
   BF16 release copy restaged on the released 4B Index package, run with the program's IX1 harness (same image,
   kit `87d4650b`, panel) and paired-bootstrapped (2,000 replicates) against the release's reference run
   `AF-4b-LRHxALL-bf16`.

## Release gate (Nox-4B-Reasoning)

- Index 0.2.1 overall paired 95% CI lower bound > 0 against the release run.
- No Index area down by more than 2 points; Index row contamination audit clean; JevArena v3 post-key and the
  reasoning panel reported (not gating).
- Package: BF16 copy answers identical to the scored package; `MODEL_MANIFEST.json` verification, Hub remote-code
  smoke, System One examples, same latency as the release within noise on the same GPU.

## Stop rules

- If the `R4-TF` soup does not beat the `R4-F0` soup on RP-DEV finals (paired, lower bound > 0), the tree signal is
  not shown at 4B: record it, read only the better of the two formally, and do not transfer the recipe unchanged to
  9B / 2B.
- No candidate passes the release gate: no release; record and plan wave 2 from the dev evidence.
