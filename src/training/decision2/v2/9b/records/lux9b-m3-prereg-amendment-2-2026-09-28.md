# 9B Milestone 3 preregistration, amendment 2: arm D (E8F-style full-data fine-tuning)

Frozen 2026-09-28 before any arm D step and before any arm A or B readout. Trigger: the
coordinator's 21:00 note — A7's Stage1–4 curricula trained only Sol and Nox, so A7 is mostly
**new** data for Lux (which saw only the natural 24k pool); an E8F-style run (full-data
fine-tuning, three seeds, uniform soup, own-Lux soft targets as a trust region) is the
highest-expected-value 9B arm; and candidates compare against the **stricter** of the adopted
1.0 run and a same-renderer, same-limit 1.0 control. Everything else is as in the
[preregistration](lux9b-m3-prereg-2026-09-28.md) and [amendment 1](lux9b-m3-prereg-amendment-1-2026-09-28.md).

## Arm D

| Field | Value |
| --- | --- |
| Start | own Lux 1.0 `bd45a30a…` |
| Training | **full fine-tuning** (the 0.8B E8F recipe): backbone lr 1e-5, head lr 1e-4, weight decay .01, warmup .05, one epoch; token-budget micro-batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per update; max length 8,192; gradient checkpointing |
| Objective | CE + 0.5·Brier on every row, + 0.5·KL(own Lux ‖ student) on the full-M recipe rows (trust region; A7 and v1 rows carry gold labels only, as no own-Lux targets exist for them) |
| Seeds | 20260926, 1, 2; SELECT700 even8 per seed |
| Artifact | uniform FP32 soup of the three SELECT-chosen checkpoints (`v2.dec.soup`) if its development proxy is ≥ the seed mean, otherwise the median seed (the coordinator's seed rule, used here to cut seed variance); CAL698 temperatures with `v2.dec.calibrate_ckpt`; packaged `qwen-full` at 16,384 tokens |

## Arm D data (spec `lux9b/specs/m3-d-full-M-a7-v1.json`, `lux9b.m3_data`)

In order, each deduplicated against pk1 A0, the recipe and every earlier component by current
and pre-renumbering input hash:

1. **mx-v2-full-M** with own-Lux targets exactly as arm A, **minus the A0s rows of the two
   shortcut families** (`natural_cosmos_qa`, `natural_squad2_answerability`), per the coordinator
   (no strict A0s file is published yet; arms A/B keep the frozen recipe and disclose them).
2. **A7 v3 core** (`3a4efc77…`): A7h, A7i, A7m, A7o, A7p, whole; the two families excluded.
3. **A7g** (Stage4 v2 generated, 93.95M tokens): whole groups stratified by source × type ×
   language to **45M tokens** (budget-driven; a full pass would add ≈ 3 h per seed).
4. **v1 arms** (`5c0255ed…`): A1, A2, A3, A4v2h, A5, A6g, A6h (rows already in the recipe drop).
5. **New A7 human Score / multilingual sub-arms** (`0b47239e…`, adopted per the 21:05 note):
   A7k, A7s, A7r whole; **A7q** whole groups to 5M tokens. A7x (MASSIVE) is never used.

Guards as before (sealed C1 candidates, MASSIVE / PAWS-X / XNLI, isolation vs SELECT700 and
CAL698). The build prints and records the final rows and tokens before any GPU step.

## Comparison and promotion (all arms)

- Formal comparator = the **stricter** (higher) of the native Lux1 16K run (65.808) and the
  same-renderer Lux1 16K control collected under amendment 1. The release gate is a paired 95%
  lower bound > 0 against that stricter run.
- Arm D goes to the formal runner under the preregistered finalist rule applied to its
  artifact (soup or median seed); A and B keep their rule. C (v2-L) is deprioritized: it runs
  only on GPUs left idle after the A/B/D decisions.

## Budget

Arm D ≈ 3 × 6.5 GPU-h (estimated ≈ 4,800 tokens/s for full fine-tuning; measured in the
preflight and first updates, recorded with the result). **Milestone 3 cap raised to 34
GPU-hours.** A reproduced ROCm fault, OOM or a failed preflight stops arm D without retry.
