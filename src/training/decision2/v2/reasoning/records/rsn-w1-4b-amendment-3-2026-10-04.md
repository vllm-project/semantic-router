# Reasoning wave 1 (4B) — amendment 3: one interpolation read (wave 1b), written before it runs

Disclosed: written after the first formal read (`RS-R4-TFM-bf16`) and before the `RS-R4-F0-bf16` / `RS-R4-TF-bf16`
reads finish.

## The first read

`RS-R4-TFM-bf16` (α 1.0): Index 0.2.1 42.99 against the release's 43.77. Knowledge & Reasoning rises (27.6 → 28.5;
GSM8K +17.7, BBH +8.0, MuSR +4.9) but Language Understanding falls (48.6 → 44.5; RAGTruth −24.3, FinEntity −12.3),
with CRUXEval −15.9 and MMLU-Pro −6.0. The full continuation moves the 4B release (a soup of LoRA-merged models) too
far: the gains are where the training families point, the losses are interference elsewhere.

## Wave 1b (one more formal read; the wave-1 reads `RS-R4-F0-bf16` and `RS-R4-TF-bf16` continue unchanged)

- Candidate: `RS-R4-TFM-a50-bf16` = release + 0.5 × (`R4-TFM` soup − release), built with
  `v2.reasoning.interpolate`; the registered interpolation family of the prereg, at the point that halves the move.
- Eligibility before the read: its SELECT family-macro accuracy must be at least the release's minus 0.01 (.886),
  read with `v2.reasoning.devread` on node C.
- The read: the same IX1 harness on node C GPU5–7 (image, kit, panel, reference run `AF-4b-LRHxALL-bf16`), paired
  bootstrap 2,000 replicates.
- No further 4B point is read in this wave. If it fails the release gate, the 4B result of wave 1 is negative and the
  next 4B attempt needs a new preregistration (lower learning rate and a larger replay share are the obvious levers).
