# Official Qwen3.5 4B source ablation: Posttrained versus completed Base

**Prospective development-only protocol. Status: HOLD before optimizer.** This
record freezes one possible official-source Posttrained arm against the
already-completed [Base BEST466 control](qwen35-4b-official-base-full-development-result-2026-09-27.md).
It does not authorize GPU training, score protected labels, replace the Base
checkpoint, or make a release claim. The prior
[Base v3 HOLD](qwen35-4b-official-best466-v3-formal-hold-2026-09-27.md)
remains fixed. The independent variable is the official general Posttrained
initialization. A second Base training run would add no matched information.

## Source identity and CPU admission

| Source | Immutable Hugging Face revision | Weight shard SHA-256, in index order |
| --- | --- | --- |
| [Official Qwen3.5-4B Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | `1001bb4d826a52d1f399e183466143f4da7b741b` | `df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712`; `590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef` |
| [Official Qwen3.5-4B Posttrained](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`; `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |

The official Posttrained model is a **general Qwen model** whose card names
the official Base as its parent; it is not a third-party Decision checkpoint.
Its metadata and full shards were fetched at the fixed revision through the
HF CLI. Both configurations declare `Qwen3_5ForConditionalGeneration` and
have the same config SHA-256
`ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`,
the same 738 indexed tensor names and 9,319,737,856 indexed weight bytes.
The weight indices differ, as expected for separate releases: Base
`eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb`,
Posttrained
`cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d`.
The two tokenizer JSON SHA-256 values are Base
`fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`
and Posttrained
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
The BPE vocabulary/merges and existing special-token IDs match; the latter
adds chat/reasoning special tokens and changes EOS. The native decision
renderer uses raw segmented input with `add_special_tokens=False`, not the
chat template, so those differences require an **actual row-wise** audit.

The CPU-only
[`preflight_qwen35_4b_posttrained_cpu.py`](../scripts/preflight_qwen35_4b_posttrained_cpu.py)
checks indexed weight headers without loading tensor data onto a device,
configuration, tokenizer, exact raw native token IDs and row counts. Its
code SHA-256 is `1eecdd3dc07112e7a16411b8fcd2e5bf76e4b1629fdca9b3e84dff99ba38c0d6`.
The completed CPU report SHA-256 is
`0c45417d4cc8f3e4c32edc55aec3c230e0922476500acc1fabc3b5c2151ac82b`.
It verified all 738 indexed tensors and **4,659,865,088 full safetensors
parameters** per source, including vision weights; actual text-native loaded
parameters plus the new head/LoRA still require the zero-step load gate. The
audit across all 8,855 fixed TRAIN/SELECT/CAL rows found **zero changed
token-ID sequences** and identical pad token ID `248044`.
TRAIN is 7,455 rows, 4,194,465 unpadded input tokens and maximum 6,596
tokens; SELECT and CAL are 700 each. This establishes native **input** and
token-budget parity for these rows, not zero-step output or optimization
parity. Source EOS IDs differ (`248044` Base, `248046` Posttrained), but the
raw native renderer inserts neither EOS. Any missing/mismatched tensor keeps
HOLD.

The private fixed-data SHA-256 values are TRAIN
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
SELECT `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
CAL `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
The script reads no JevArena v3, CSS15 or JevBench labels. The private report
must retain the ordered row-ID/token digest and exact source shard hashes;
it must not publish the training records.

## Why this experiment is narrow and uncertain

The Base BEST466 passed open DEV/pilot strongly, then failed the separately
locked post-key same-panel formal comparison: v3 **53.218 versus own Nox 1.0
56.470**, Noul **−9.00 pp**, Score **−4.25 pp**, exception stack **−16.00 pp**,
and CSS task-median macro-F1 **−.022723**. Increasing Choice gradient mass
does not directly explain those misses; a separate audited 2,120-row Choice
reweighting proposal is on HOLD for that reason. A general Posttrained source
could improve language/transfer, but can also make ordinal Score worse. At
2B, the same official-source, same-data Posttrained arm previously yielded
only **95/400 Score** on typed DEV while Base yielded 241/400. That is a
cross-size warning, not a prediction of 4B behavior.
Changing initialization alone is also not designed to cure the over-budget
long CSS examples under the unchanged 8,192-token cap.

The [Decider 4B author's model card](https://huggingface.co/Mapika/decider-4b)
reports about **742M** first-stage training tokens and a 29,325-row later
hard-example/replay stage with soft-distribution replay. Our fixed Base arm
consumed **4.194M native input tokens**, a nominal **~177×** smaller token
count. Tokenizers, objectives, data, training stages and compute are not
matched. This gap is a hypothesis for weak transfer and an important
limitation of the small ablation; it is **not evidence that scale caused** the
measured difference or that a Posttrained initialization closes it.

## Locked arm and control comparison

Only if all admission gates below pass, initialize a **new** 256-dimensional
dynamic-option decision head with seed `20260926` and rank-16 LoRA,
alpha 32, dropout `.05`, directly on the pinned Posttrained weights. Match
the completed Base control exactly: the same rights-clean v2 row order and
one full epoch (7,455 rows, 4,194,465 input tokens), microbatch 1,
accumulation 16 with partial final window, **466 optimizer updates**, max
input 8,192 with no truncation, CE plus `.5` Brier, AdamW weight decay `.01`,
gradient clip `1.0`, BF16 backbone compute and FP32 head/loss, LoRA peak LR
`1e-4`, head peak LR `2e-4`, `.05` warmup and cosine decay. Use the same
pinned runtime image ID
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
and the same source code and exact optimizer schedule as Base; if these
cannot be reproduced, label the result **unmatched**, not a source ablation.
No teacher, replay, added row, changed loss, chat template, or calibration
is admitted. The source tokenizer is used only after the exact native input
parity gate. The already-completed Base BEST466 is the sole Base control.

Save and score SELECT after updates **64, 128, 192, 256, 320, 384, 448 and
466**. Choose one BEST by SELECT family-macro accuracy descending,
normalized Brier ascending, then earliest step; do not use DEV, CSS pilot,
public231 or formal labels to select a checkpoint. CAL is untouched until
after selection. Cap the arm at **3.0 one-GPU hours** including SELECT checks
and saves. A nonfinite loss/gradient, OOM, missing row/type, source mismatch,
failed save or cap stops the arm with an incomplete receipt. No other step,
seed or learning rate is searched after a failed screen.

## Admission and development gates

1. **CPU source/input:** the full pinned Posttrained shards must match the
   index roster and be counted without loading tensors to GPU. The same
   training code/runtime must accept both sources. All 8,855 native encoded
   inputs, pad IDs, 7,455 TRAIN row identities, data digests and token budget
   must remain identical. A mismatch stops before optimizer.
2. **Native zero-step and one-update:** before full training, load the new
   Posttrained source, new seeded head and adapter through the same native
   path as Base. Run the fixed 32 SELECT prompts in **two fresh processes**;
   require 32/32 valid, zero category changes, p99 probability drift at most
   `.005` and maximum drift at most `.02`. This is within-source
   repeatability, **not** equality of Base and Posttrained predictions. One
   optimizer update must have finite loss/gradients and a reload parity check
   at the same limits. If any gate fails, stop. These are GPU gates pending
   after this CPU-only prereg, not completed evidence.
3. **Complete fixed budget and selector:** exactly 466 updates, a complete
   source/data/optimizer receipt, SELECT-only BEST and exact 32-row native
   BEST reload at the same drift limits are required. An interrupted run may
   resume only from its exact durable optimizer/data cursor without changing
   schedule; retain failures and spent GPU-hours.
4. **One open-panel decision:** run the selected package once on the existing
   typed DEV 1,600 and CSS pilot 1,430 with the same native collector as the
   archived Base and own Nox controls. With `T_dev` the four-family macro
   accuracy and `H_pilot` the three-task median macro-F1, promote only if
   `100*sqrt(T_dev*H_pilot)` is at least **Base 65.33730 + 2.0** points,
   valid/within-budget coverage is at least 99%, Noul is at least Base
   **230/400**, Score at least Base **355/400** (allowing eight misses), and
   the DEV exception-family count is at least the archived same-panel Base
   count; CSS pilot `H_pilot` must be at least Base **.520209**. Report four typed
   families, all three types, the three human tasks, Score level histogram,
   Brier/ECE and over-budget cases. These prospective thresholds make the
   source hypothesis falsifiable; the screens have been exposed before and
   cannot establish independent transfer.
5. **After a pass only:** prepare a new candidate/prediction lock before any
   formal readout, with a predeclared same-panel v3 and public231 analysis.
   The existing formal labels have already been accessed by the project, so
   these panels are **post-key comparisons**, not untouched blind tests.
   Obtain fresh untouched or independent external corroboration before
   calling the arm independently validated. No candidate formal score may
   be transferred from Base, changed after seeing it, or used to retune CAL
   or select a different update. A development pass is not a release gate.

The main counterfactual remains data-composition rather than initialization:
if this narrow arm fails, construct a separately frozen, source-disjoint
real-label/Score/long-evidence mixture and compare it under matched budgets.
Do not reinterpret repeated exposure to typed DEV or CSS pilot as additional
independent evidence.
