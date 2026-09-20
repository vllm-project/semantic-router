# Decision Record: First Router-Native Model Candidate Experiment

Tracking issue: [vllm-project/semantic-router#3198](https://github.com/vllm-project/semantic-router/issues/3198)
("[Research] Define routing-native task contracts, baselines, and the first
candidate experiment"). Parent epic:
[#2974](https://github.com/vllm-project/semantic-router/issues/2974).

This record covers piece 3 of the 3-way split agreed in the #3198 thread:

- Piece 1, [#3856](https://github.com/vllm-project/semantic-router/issues/3856):
  same-run latency, CPU and memory harness. Owned by someone else, still open.
- Piece 2, [#3857](https://github.com/vllm-project/semantic-router/issues/3857):
  controls to compare against. Owned by someone else, still open.
- Piece 3, this record: train one candidate model and write down what we found.

Results from #3856 and #3857 get added here once they exist. This record does
not build them.

## Summary

- **Task:** route a prompt to text only (`AR`), image only (`DIFFUSION`), or
  text plus image (`BOTH`).
- **Candidate:** DistilBERT (66M parameters), trained by distillation from the
  production mmBERT-32K model.
- **Headline result:** on the original labels the candidate scores 96.31%
  against 97.10% for a clean mmBERT baseline. The gap is small and not
  statistically significant (p = 0.31).
- **The catch:** a blind re-judgement of the labels (§11) disagrees with 7.4%
  of them, almost all in the `BOTH` class. Whether the candidate is "as good
  as" or "worse than" the baseline depends on what `BOTH` is supposed to mean.
- **Decision needed:** should `BOTH` mean "the user asks for visuals" or
  "visuals would clearly help"? Raised with adaamko in the #3198 thread.
- **Still open:** latency (#3856), controls (#3857), the accuracy tolerance
  (§9), and a human check of the re-judged labels (§11.5).

## 1. Task contract

- **Task type:** classification with three classes: `AR`, `DIFFUSION`, `BOTH`.
- **Input:** one free-text prompt.
- **Output:** the kind of reply the prompt needs: text only, an image only,
  or both.
- **Why this task:** as adaamko noted in the thread, it is one of the few
  router tasks with a label that exists for every request. Model-choice
  routing does not have that, because "which model would have succeeded" is
  never observed.
- **Caveat found later (§11):** the label exists, but for `BOTH` it is a
  policy choice, not a fact about the prompt. Does "How do I tile a floor?"
  need images? About 7% of labels change under a blind re-judgement, almost
  all of them `BOTH` becoming `AR`. Read every accuracy number here with that
  in mind.

## 2. Baselines

Two baselines are reported. The reason is in §4.

### Published baseline

- **Model:** `llm-semantic-router/mmbert32k-modality-router-merged`
  (mmBERT-32K plus LoRA, 307M parameters).
- **Where it is used:** it is the checkpoint wired into `registry.go` and
  served today.
- **What its model card says:** 10 epochs, batch size 32, learning rate 2e-5,
  Focal Loss (gamma 2.0).
- **What it does not say:** the `--max-samples` value, whether vLLM synthesis
  was used, or the dataset revisions.
- **What the repo says:** git history and the original PR (#1310) list only the
  default parameters of the training script, never the run that produced this
  checkpoint.
- **Result on `test.jsonl` (759 rows):** 96.84% accuracy, 0.9684 weighted F1.

### Clean baseline

- **What it is:** the same architecture and hyperparameters (mmBERT-32K plus
  LoRA), retrained from scratch on this record's own `train.jsonl` and
  `validation.jsonl` (§4). Command: `modality_routing_fixed_split_trainer.py
  --model mmbert-32k`, with no teacher.
- **Why:** it is the leakage-free comparison point for the graduation gate.
- **Result on `test.jsonl`:** 97.10% accuracy, 0.9711 weighted F1 at 10 epochs.
  That is 0.26 points above the published baseline.
- **Training time:** about 310 seconds on an RTX 3090.

### Why 10 epochs

- The first pass used 8 epochs, a round number that did not match the
  published model.
- Validation curves showed the clean baseline had nearly stopped improving
  (+0.004 accuracy from epoch 5 to 8). The candidate was still improving, and
  its weakest metric (`BOTH` precision) was still rising.
- The published model card says 10 epochs, so both models were retrained at 10.
- Effect on the clean baseline: 95.78% to 97.10% (+1.32 points, a real jump).
- Effect on the candidate: 95.39% to 96.31% (+0.92 points).
- Every number in this record is from the 10-epoch run. The 8-epoch logs are
  kept locally and superseded.

### Did leakage help the published baseline?

- The two baselines agree on 96.84% of test rows (24 of 759 differ). See §7.
- The clean baseline scores slightly higher than the published one.
- So there is no sign that leakage gave the published model an edge. This
  particular run may simply have generalized a little better on this split.

### Why two baselines and not one

- We cannot rebuild the published model's training data from this repo:
  - `random.seed(42)` in `ModalityRoutingDataset.load_datasets()` fixes the
    shuffle order, not which rows are in the pool.
  - Every Hugging Face source is loaded without a pinned `revision=`.
  - Several fallback generators and the `--synthesize-both` vLLM path are not
    seeded.
  - `--max-samples` feeds into every per-source quota, so a different value
    changes the whole pool.
- So whether that model saw our test rows cannot be checked, even in
  principle. We searched git history and read the full loader to be sure.
- Retraining on our own split is the only reliable fix.
- The published model stays in the tables because it is what runs in
  production and what #3856 will probably benchmark.

## 3. Candidate and architecture rationale

**Candidate: DistilBERT-base-uncased** (66M parameters, 6 transformer blocks).

- **Method:** knowledge distillation. The frozen published model is the
  teacher.
- **Loss:** temperature-scaled soft-label KL (Hinton et al., 2015) combined
  with the same hard-label Focal Loss used elsewhere in this pipeline.
- **Command:** `modality_routing_fixed_split_trainer.py --model
  distilbert-base-uncased --teacher-model-path
  llm-semantic-router/mmbert32k-modality-router-merged`.

### Options considered

| Candidate | Verdict | Reason |
|---|---|---|
| **DistilBERT-base-uncased** | **Chosen** | Already listed in `common_lora_utils.get_model_mapping()`. Six blocks against mmBERT-32K's 22 gives a real latency win. It is the standard "distilled BERT student", so the recipe is well known. |
| ALBERT-base | Rejected | Shares parameters but keeps 12 blocks, so no real latency gain. |
| TinyBERT / MiniLM | Rejected | Not in this pipeline's model mapping. More integration work for a similar gain. |
| Same-family lower-rank LoRA | Rejected | Same 22-layer trunk. It would shrink the adapter, not train a student. |
| LFM2.5-Encoder / SCX Router v0.1 | Deferred, later tried informally (§12) | See below. |

### LFM2.5-Encoder and SCX Router

- **What I said in the #3198 thread:** "Not picking up LFM2.5-Encoder or SCX
  Router for this pass. I'll just note both in the decision record as
  candidates for next time, along with the traps @adaamko flagged. I don't
  know either model well enough yet to trust myself not to mix up 'is the
  model good' with 'did I load it right', so keeping this pass to what I know
  well feels like the safer move."
- **Traps adaamko flagged for LFM2.5:**
  - `AutoModel.from_pretrained` returns a randomly initialised trunk, because
    the weights sit under the MaskedLM prefix.
  - An Opir-style learning rate of 1e-6 is wrong for a raw MLM trunk. Closer
    is about 3e-5 for the encoder and 1e-4 for the head.
- **SCX Router v0.1** (Knowledgator, Qwen3-0.6B backbone, GLiClass-style head,
  Apache 2.0) was also flagged as a candidate for a future pass.

### Known limitation

- DistilBERT is English-only. mmBERT-32K is multilingual (1800+ languages).
- The dataset (§4) is mostly English, so this experiment cannot show a
  multilingual regression. That is a limit of this pass, not a hidden defect.

### Measured result (10 epochs, `test.jsonl`, 759 rows)

- **Accuracy:** 96.31%. **Weighted F1:** 0.9634.
- **Against the baselines:** 0.79 points below the clean baseline and 0.53
  points below the published baseline.
- **Is the gap real?** Not on the original labels.
  - The two models differ in correctness on 24 rows.
  - The clean baseline is right on 15 of them and the candidate on 9.
  - Exact McNemar test, two-sided: p = 0.31.
  - Under the stricter re-judged labels the gap becomes significant (§11.3).
  - **Training seed noise is not measured.** Each model was trained once,
    before the trainer had a `--seed` option. A gap of 0.79 points is about 6
    rows, and the McNemar test only covers which test rows were sampled, not
    how much a retrained model would move. A seed sweep (several seeds per
    model) is still to do before reading a small gap as real.
- **Speed and size:**
  - Trained in about 120 seconds on an RTX 3090, roughly 2.5x faster than the
    clean baseline.
  - The merged checkpoint is 4.6x smaller (268 MB against 1.2 GB).

### The weak spot: too many `BOTH` predictions

The candidate is not just slightly worse everywhere. It has one specific habit.

- **`BOTH` precision:** 0.8966. **`BOTH` recall:** 0.9811.
- **In plain terms:** it finds nearly every real `BOTH` prompt, but it also
  labels some `AR` and `DIFFUSION` prompts as `BOTH`.
- **How many:** 10 `AR` prompts and 8 `DIFFUSION` prompts, out of 300 each.
- **Same pattern at 8 epochs:** `BOTH` precision was 0.8667. More training
  helped but did not remove it.
- **The other two classes:** `AR` and `DIFFUSION` precision and recall are
  close to or above the clean baseline's.

Confusion matrix (rows are the true class, columns are the prediction):

| True \ Predicted | AR | DIFFUSION | BOTH |
|---|---|---|---|
| AR | 286 | 4 | 10 |
| DIFFUSION | 3 | 289 | 8 |
| BOTH | 3 | 0 | 156 |

**Why it matters for the tolerance (§9):** a single accuracy tolerance could
let this candidate through and hide the `BOTH` bias. A per-class floor would
catch it.

## 4. Dataset and objective

### Where the data comes from

- **Produced by:** `export_modality_dataset.py --max-samples 6000`.
- **Stored in:** `exported_modality_routing_dataset/` in this directory.
- **Reproducibility:** the export is deterministic for the same code, flags and
  upstream dataset revisions. That is not the same as reproducible across
  different runs or dates (see §2).

| Split | Total | AR | DIFFUSION | BOTH |
|---|---|---|---|---|
| train | 3,538 | 1,400 | 1,400 | 738 |
| validation | 758 | 300 | 300 | 158 |
| test | 759 | 300 | 300 | 159 |

- **Total:** 5,055 rows. The target was 6,000. Several sources, mainly WildChat
  mining, came in under quota, which the pipeline accepts by design.
- **Sources this run:** `WildChat_AR=500`, `WildChat_DIFF=136`,
  `WildChat_BOTH=18`, `Diffusion_SD=1864`, `OASST2=500`, `Alpaca=500`,
  `Dolly=500`, `InterleavedBench=447`, `BOTH_seeds=75`, `BOTH_fallback=515`.
- **Synthesis:** no vLLM synthesis (`synthesize_both=0`).
- **Licenses:** every row keeps the license of its source. The Alpaca rows are
  CC BY-NC 4.0 (non-commercial), so this export is not cleanly reusable under
  the repository's license. The full table is in the dataset README, and how to
  handle it is a decision for the maintainers.

### Training objective

- **Loss:** `kd_alpha * soft_KL(student/T, teacher/T) * T^2 + (1 - kd_alpha) *
  FocalLoss(student, hard_labels)`.
- **Settings:** `T=3.0`, `kd_alpha=0.5`. These are standard starting values and
  have **not** been tuned on this dataset. A small sweep is worth doing before
  treating the numbers as final.

### Data-quality check before training

Two kinds of leakage were checked.

1. **Across runs** (the published model's unknown training data against our
   export). This cannot be verified. See §2.
2. **Within one run** (the same or nearly the same text landing in both train
   and test, because the split never removes duplicates first).

For the second kind, we pulled real rows from four of the named source
datasets and measured duplicates:

| Source | Rows pulled | Exact duplicates | Overlap between two disjoint windows |
|---|---|---|---|
| `tatsu-lab/alpaca` (AR) | 3,000 | 0.0% | 0.0% |
| `databricks/databricks-dolly-15k` (AR) | 3,000 | 1.23% | 0.6% |
| `Gustavosta/Stable-Diffusion-Prompts` (DIFFUSION) | 73,718 (all) | **32.44%** | **16.47%** |
| `nateraw/parti-prompts` (DIFFUSION) | 1,632 | 0.0% | 0.0% |

That showed the risk is real at the source level. So we then measured it on
our actual split. `evaluate_modality_candidate.py` checks whether each test
row appears (exactly, or after normalising case and whitespace) in train or
validation:

| Class | Contaminated test rows | Rate |
|---|---|---|
| Overall | 1 of 759 | **0.13%** |
| AR | 0 of 300 | 0.0% |
| DIFFUSION | 1 of 300 | 0.33% |
| BOTH | 0 of 159 | 0.0% |

- **Conclusion:** the high duplicate rate at the source did not turn into a
  real train/test contamination problem on this split.
- **Keep the check:** it is cheap, and another run or `--max-samples` value
  could be worse.
- Both raw and contamination-filtered numbers are reported in §7.

## 5. Same-run latency

**Status: `PENDING`.** Blocked on
[#3856](https://github.com/vllm-project/semantic-router/issues/3856), which is
not built yet.

Once it exists, give the harness the merged clean-baseline and candidate
checkpoints (§6) and fill in this table:

| Model | p50 | p99 | CPU | Mem HWM |
|---|---|---|---|---|
| clean_baseline (mmBERT-32K) | TBD | TBD | TBD | TBD |
| candidate (DistilBERT) | TBD | TBD | TBD | TBD |

- **Expectation, not a result:** 6 blocks against 22 should give a clear p99
  win. Nothing is claimed until #3856 measures it.

### What the #3856 spec requires (as agreed in the thread)

- Single stream, batch size 1, with a 20-iteration warmup before timing.
- Both models measured in the same run on the same rows.
- A host fingerprint recorded with every run, so numbers from different
  machines are not compared by accident.
- Re-check all of this against the real harness once it lands.

### Problem to settle first: the candle binding

- Production serves models through the candle binding. Its sequence-classifier
  loaders only support the ModernBERT family.
- A merged DistilBERT directory loads with Hugging Face `transformers`, but as
  far as I can tell from the loader code, **not** with candle.
- So #3856 has two options:
  - Run both models through Hugging Face. That measures the models, not the
    production path.
  - Write a candle DistilBERT loader first.
- That choice belongs to #3856 and should be made on purpose.
- It also limits what this candidate could deliver in production without new
  Rust work.

## 6. Reproducing this experiment

- **Testing before the real run:** both trainer modes, the merge step and the
  evaluation script were smoke-tested on a tiny synthetic fixture.
- **CPU was too slow:** the first real attempt measured 71 seconds per step for
  the clean baseline (448 steps, about 8.8 hours), so it was abandoned.
- **Real runs:** all numbers here come from an RTX 3090, 10 epochs. The clean
  baseline took about 310 s and the candidate about 120 s.
- **What is not checked in:** merged checkpoints (clean baseline about 1.2 GB,
  candidate about 268 MB) are too large for a research artifact. Use the
  commands below with the exported split in this directory.

```bash
# Clean baseline (mmBERT-32K, no teacher, a few minutes on an RTX 3090)
python modality_routing_fixed_split_trainer.py \
  --model mmbert-32k \
  --train-file exported_modality_routing_dataset/train.jsonl \
  --val-file exported_modality_routing_dataset/validation.jsonl \
  --epochs 10 --batch-size 32 \
  --output-dir lora_modality_router_mmbert32k_clean_baseline \
  --merge-output-dir models/mmbert32k-modality-router-clean-merged

# Candidate (DistilBERT student, distilled from the published production baseline)
python modality_routing_fixed_split_trainer.py \
  --model distilbert-base-uncased \
  --teacher-model-path llm-semantic-router/mmbert32k-modality-router-merged \
  --train-file exported_modality_routing_dataset/train.jsonl \
  --val-file exported_modality_routing_dataset/validation.jsonl \
  --epochs 10 --batch-size 32 --temperature 3.0 --kd-alpha 0.5 \
  --output-dir lora_modality_router_distilbert_candidate \
  --merge-output-dir models/distilbert-modality-router-candidate-merged

# Three-way evaluation (accuracy, agreement, contamination check)
python evaluate_modality_candidate.py \
  --test-file exported_modality_routing_dataset/test.jsonl \
  --train-file exported_modality_routing_dataset/train.jsonl \
  --val-file exported_modality_routing_dataset/validation.jsonl \
  --published-baseline-model-path llm-semantic-router/mmbert32k-modality-router-merged \
  --clean-baseline-model-path models/mmbert32k-modality-router-clean-merged \
  --candidate-model-path models/distilbert-modality-router-candidate-merged \
  --output-report modality_candidate_eval_report.json
```

- **Pinned environment:** `requirements-lock.txt` in this directory records the
  exact package versions used (torch 2.14.0, transformers 5.17.0, peft 0.21.0).
- **Seeds:** the trainer takes `--seed` (default 42) and, with the same seed,
  gives the same result twice. The reported numbers come from runs made before
  that option existed, so the commands above reproduce the setup and not the
  exact numbers. See the seed note in §3.
- **Tests:** run `pip install pytest` and then `pytest` in this directory. No
  GPU, checkpoint or network is needed. The suite covers the label mapping, the
  data helpers, the losses, the trainer, the evaluation, and the audit tool
  (`label_audit/`, including judging through a fake API client).

### Code layout

Each file has one job, and the heavy parts (torch, checkpoints) sit at the edges.

- `modality_routing_fixed_split_trainer.py`: the training entry point. A
  `TrainConfig` goes in, and one `ModalityTrainer` handles both plain
  fine-tuning and distillation.
- `modality_data.py`: class weights and oversampling. No torch.
- `modality_losses.py`: the distillation loss.
- `modality_label_mapping.py`: the canonical labels and the checks that a
  checkpoint's labels map onto them. No torch.
- `evaluate_modality_candidate.py`: loads each checkpoint and predicts.
- `modality_eval_metrics.py`: contamination check, metrics and the report.
  No torch, so it can be tested with plain arrays.
- `label_audit/judge_labels.py`: the audit command line. The work is in
  `label_audit/audit_lib/`, one module per job (dataset, judgments, checkpoint,
  statistics, human review, API judging, report).

## 7. Routing agreement

The full report is `modality_candidate_eval_report.json` in this directory.
It has per-example predictions, all three pairwise agreements, and metrics on
the full and the contamination-filtered test set. This is the 10-epoch run.

Agreement on the 759-row `test.jsonl`. The filtered set (758 rows) moves every
number by 0.4 points or less.

| Pair | Agreement | Rows that differ |
|---|---|---|
| **candidate vs. clean_baseline** (main gate) | **96.44%** (96.57% filtered) | 27 |
| candidate vs. published_baseline | 96.05% (96.17% filtered) | 30 |
| clean_baseline vs. published_baseline | 96.84% (96.83% filtered) | 24 |

Candidate against clean baseline, the 27 rows that differ:

- **Clean baseline right, candidate wrong:** 15 rows (55.6%).
- **Candidate right, clean baseline wrong:** 9 rows (33.3%).
- **Both wrong, with different labels:** 3 rows (11.1%).

What that says:

- The clean baseline wins more of the disagreements.
- The candidate still wins a third of them, so it is not uniformly worse.
- It makes a different and smaller set of mistakes, mostly the `BOTH`
  over-prediction described in §3.

## 8. Controls

**Status: `PENDING`.** Blocked on
[#3857](https://github.com/vllm-project/semantic-router/issues/3857), which
itself waits on #3856's run format.

### Scope changed in the #3857 thread

- adaamko dropped the cost-based controls (always-cheapest, always-strongest,
  best fixed split at matched cost). This task has no per-model cost to match.
- The controls are now:

| Control | Notes |
|---|---|
| Lexical / regex baseline | Report per-class recall. **Circularity warning:** many labels were made by regexes (WildChat mining) and templates (`BOTH_fallback`). A regex baseline will score high on exactly those rows. Report regex-labelled and dataset-labelled rows separately, or it will overstate how easy the task is. |
| Majority class | 300 of 759 = **39.5%** on this `test.jsonl` (`AR` and `DIFFUSION` tie). |
| Prior-matched random | Expected accuracy **35.6%** on this split's class mix. |

- Per-class recall is reported for every control, not only accuracy.
- The two chance baselines follow directly from the split table in §4.
- The lexical baseline numbers are `PENDING`.

## 9. Tolerance and graduation gate

**Status: `TODO`.** adaamko left the tolerance number open for reviewers
("happy to have the tolerance itself be part of the decision record review").
It is not invented here.

What reviewers have to work with:

- **Accuracy gap:** the candidate is 0.79 points below the clean baseline
  (96.31% against 97.10%). Not significant on the original labels (p = 0.31,
  §3).
- **Agreement:** 96.44% with the clean baseline.
- **Direction of the errors:** the candidate over-predicts `BOTH` (§3, §7). It
  is not simply weaker everywhere.
- **Label policy first:** the label audit (§11) shows the significance of the
  gap depends on what `BOTH` means. Settle that before setting any tolerance.

Choices for whoever sets the tolerance:

- An aggregate accuracy band.
- A per-class floor. The `BOTH` precision number would fail this more visibly.
- An agreement-rate floor.
- The required p99 improvement from §5, still pending #3856.

## 10. Stop criteria

The candidate does not move forward if any of these hold:

- Its accuracy against the clean baseline falls outside the tolerance from §9.
- No meaningful p99 win is measured once §5 lands.
- Once §8 lands, it does not clearly beat the controls (lexical, majority
  class, prior-matched random), judged per class and not only on accuracy.
- The `BOTH` policy (§11.4) has been decided and the comparison rerun, and the
  candidate only passes under one of the two policies.

## 11. Label audit

### Why we did it

- Every model scores 96% to 99%, which is suspiciously high for a task about
  what a user wants.
- The labels come from where a prompt was found, plus regexes and templates.
  Nobody read each prompt:
  - `AR`: Alpaca, Dolly, OASST2.
  - `DIFFUSION`: Stable-Diffusion-Prompts.
  - `BOTH`: InterleavedBench plus a template generator.
  - WildChat rows: regexes.
- So every row of `validation.jsonl` and `test.jsonl` was re-judged blind,
  without seeing the original label, against a written rubric.

### Where the files are

All in `label_audit/`:

- `RUBRIC.md` (rubric hash `75a64835`).
- `judge_labels.py` with subcommands `next`, `save`, `status`, `report`,
  `sheet`, `api`.
- `checkpoint.jsonl`: append-only, one record per judged row.
- `dataset_sha256.json`: pins each split by hash, so judgments cannot be
  joined to a re-exported dataset. This applies to every `--data-dir`: a custom
  checkpoint gets its own manifest beside it, and judgments that exist without a
  recorded hash are refused.
- `predictions/`: the SCX and LFM2.5 test predictions used in §11.2, each with
  the sha256 of every prompt. The baseline and candidate predictions are in
  `modality_candidate_eval_report.json`. The report tool aligns predictions by
  row and rejects any that were made for different prompts, instead of joining
  them by position.
- `audit_report_test.txt` and `audit_report_validation.txt`.
- `human_review.tsv`: blinded sheet for a human spot-check (§11.5).

### 11.1 How often the judge agrees with the original labels

| Split | Rows | Strict agreement | Cohen's kappa | Inclusive agreement |
|---|---|---|---|---|
| test | 759 | 703 (92.6%) | 0.884 | 721 (95.0%) |
| validation | 758 | 702 (92.6%) | 0.883 | 725 (95.7%) |

- **Strict:** the judge's label exactly as given.
- **Inclusive:** a judge `AR` counts as `BOTH` when the judge flagged that
  visuals would clearly help.
- **Both splits match almost exactly.** This is a property of the dataset, not
  of one split.

Confusion (rows are the original label, columns are the judge's label, in
order `AR`, `DIFFUSION`, `BOTH`):

| Original | Test | Validation |
|---|---|---|
| AR | 299 / 1 / 0 | 299 / 1 / 0 |
| DIFFUSION | 15 / 285 / 0 | 12 / 288 / 0 |
| BOTH | **40** / 0 / 119 | **43** / 0 / 115 |

What stands out:

- **`BOTH` is the unstable class.** 25% of test and 27% of validation `BOTH`
  rows are judged `AR`. `AR` and `DIFFUSION` are 95% to 99% stable.
- **The noise goes one way.** No original `BOTH` row is judged `DIFFUSION`, and
  the judge never promotes an `AR` or `DIFFUSION` row to `BOTH`.
- **The 40 test `BOTH` rows that flipped to `AR`:**
  - 21 are implicit-visual how-tos, such as "How do I install a dimmer
    switch?" and "Guide me through concrete curing".
  - 19 are InterleavedBench wikihow rows ("In this task, you are given a
    high-level goal...") whose text never asks for an image.
  - The how-tos are **deliberate**. `_generate_fallback_both_prompts` has an
    "IMPLICIT" template branch that labels topics that "inherently need
    visuals" as `BOTH`. So this is a policy baked into the data, not a
    labelling mistake.
  - The wikihow rows are the closer call. The judge marked all 19 as medium
    confidence.
  - The validation flips have not been broken down by source.
- **`DIFFUSION` rows judged `AR` (15 in test, 12 in validation):** mostly
  prompts *about* image generation, such as "Can you create images?" or "Write
  a Stable Diffusion prompt for...". The original pipeline labelled them
  `DIFFUSION` from their source. Under the rubric they need a text answer.

### 11.2 The models scored against the judged labels (test)

| Model | Original labels | Judge, strict | Judge, inclusive |
|---|---|---|---|
| published_baseline | 96.84% | 91.70% | 93.54% |
| clean_baseline | 97.10% | 93.41% | 94.73% |
| candidate (DistilBERT) | 96.31% | 91.57% | 93.68% |
| scx_finetuned (§12) | 99.60% | 93.02% | 95.13% |
| lfm25_lora (§12) | 98.16% | 93.02% | 94.86% |

- Every model loses 3 to 7 points against the stricter labels.
- The gap between models shrinks. All five land near 92% to 93% strict.
- **SCX's 99.6% mostly shows it learned the dataset's labelling rules
  tightly.** It does not show it is the model most often right about intent.
- Treat the original-label numbers as "how well a model reproduces this
  dataset's rules".
- SCX against the clean baseline shows it. On the original labels SCX is right
  on 19 rows the baseline misses and never the other way round (p < 0.001). On
  the strict labels it is 10 against 7 (p = 0.63), which is noise.

### 11.3 Does the candidate comparison change?

Exact McNemar test on paired correctness, candidate against clean baseline:

| Labels used | Clean right, candidate wrong | Candidate right, clean wrong | p (two-sided) |
|---|---|---|---|
| Original | 15 | 9 | 0.31 |
| Judged, strict | 19 | 5 | **0.007** |
| Judged, inclusive | 16 | 8 | 0.15 |

- Original labels: the gap is noise.
- Strict labels: the gap is significant.
- Inclusive labels: the gap is noise again.
- **The verdict depends on the `BOTH` policy**, and the data does not settle
  that. I would not claim "the student matches the teacher" or "the student is
  worse" until it is decided.

To reproduce the tables in §11.2 and §11.3:

```bash
cd label_audit
python judge_labels.py report --split test \
  --eval-report ../modality_candidate_eval_report.json \
  --preds scx_finetuned=predictions/scx_finetuned.json \
  --preds lfm25_lora=predictions/lfm25_lora.json \
  --compare clean_baseline,candidate --compare clean_baseline,scx_finetuned
```

### 11.4 Decision needed: what should `BOTH` mean?

- **Option A, strict:** `BOTH` only when the user asks for visuals.
- **Option B, inclusive:** `BOTH` also when visuals would clearly help.
- The answer probably depends on what the downstream text and image pipeline
  can act on. That is a question for whoever owns routing behaviour.
- It is raised with adaamko in the #3198 thread.
- Until it is decided:
  - Report both the strict and the inclusive numbers.
  - Do not set the §9 tolerance from original-label accuracy alone.

### 11.5 Limits of this audit (read before citing it)

- **The judge is not independent.**
  - All 1,517 judgments came from one model in one session (`claude-sonnet-5`),
    using a rubric I wrote after looking at the data.
  - It gave the same answer on a 76-row re-judge (100% self-consistency). That
    is inflated, because it was the same session and context. It shows the
    judge is steady, not that it is right.
- **A small label leak.** I had seen about 20 original labels earlier in the
  session, before the judged rows were sealed. The effect on 1,517 rows should
  be small but is not zero.
- **Clipped text.** The judge saw rows cut at 400 characters (106 test rows and
  87 validation rows are tagged `truncated`). Those judgments reflect what was
  visible.
- **No human check yet.**
  - `label_audit/human_review.tsv` is a blinded sheet of 100 test rows: 56
    where the judge disagreed with the original label, and 44 where it agreed.
  - Fill it in, then run `judge_labels.py report --human <file>` to estimate
    the judge's own error rate.
- **The API mode is untested.** `judge_labels.py api` should reproduce the audit
  with an independent model, but it has not been run against the real API (no
  key was available).
- **Training labels not audited.** Only validation and test were judged. The
  3,538 `train` rows probably carry the same kind and rate of noise, unmeasured.

## 12. Exploratory: LFM2.5-Encoder and SCX Router

- **Status:** not part of the planned experiment. §3 deferred both.
- **What was done:** tried afterwards, out of curiosity, using the same fixed
  split. The scripts are in a separate draft PR, #3934, not in this one.
- **How to read the table:** it is not a leaderboard.
  - Each model used a different recipe and budget.
  - One seed, no tuning.
  - §11.2 shows the original-label ranking is dominated by label noise.

| Model | Setup | Test accuracy (original labels) |
|---|---|---|
| SCX Router v0.1, zero-shot | As released | 43.35% |
| SCX Router v0.1, fine-tuned | GLiClass native `multi_label_classification`, bf16 | 99.60% (756 of 759) |
| LFM2.5-Encoder-350M plus LoRA (r=16) | Masked mean pool plus a linear head | 98.16% (745 of 759) |

Problems hit, for whoever picks this up:

- **LFM2.5 `AutoModel` gives random weights.**
  - Load `AutoModelForMaskedLM(...)` and use its `.lfm2` trunk.
  - A wrapper that skips this looks like a bad model, but it is a bad load.
- **GLiClass training can show loss 0 while doing nothing.**
  - The single-label path is broken for the decoder-kv variant.
  - The collator drops scalar (0-dim) labels.
  - The Trainer swallows the error.
  - Use the native multi-label mode with multi-hot labels.
- **Big checkpoints crashed WSL.** Writing 3.4 GB intermediate checkpoints
  lined up with crashes on this machine. The SCX script uses
  `save_strategy="no"`, no optimizer state, and a bf16 final save.
- **Zero-shot SCX is weak.** 43.35% is barely above the 39.5% majority class,
  so the label descriptions matter and the released head is not a drop-in for
  this taxonomy.
