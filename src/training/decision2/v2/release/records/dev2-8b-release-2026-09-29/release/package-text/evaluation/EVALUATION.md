# Evaluation

**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, where T is the four-family macro accuracy of typed decisions and H is the median task macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every denominator. The answers of this panel were available during development, so results are same-panel comparisons rather than an untouched blind test.

JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. It is an independent rerun of the public questions, not the upstream four-axis score or the official sealed JevBench rank.

Every model ran through its own native inference path on the same frozen prompts and was scored by the same scorers; each row reports the parameters its native loader instantiates. Typed Brier and ECE (10 bins) measure calibration on the typed decisions. The paired interval is a joint bootstrap over typed groups (within family) and transfer tasks then items, 5,000 replicates. The candidate-minus-Decision 1.0 Lux v3 difference is +1.929 with paired 95% interval [+0.607, +4.144]. Against the other models shown: Nimble v2 +5.681 [+3.190, +10.090].

mlx-diag (development diagnostic, 2,275 prompts, seven languages, English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split (CC BY 4.0) and Noul from the PAWS-X test split; the columns show mean non-English accuracy. Its Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in backbone pretraining data, so this is not a sealed test.

Comparators under non-commercial, research-only or unknown licences are not shown on this card. Decision 1.0 Lux is shown from its native run at this model's 16,384-token limit (JevArena v3 65.81), the comparator of this size's release bar; run through this model's own renderer at the same limit it scores 65.23. Nimble v2 (a LoRA over Qwen3.5-9B; Apache-2.0 with a LICENSE file) ran with its bundled scorer on ROCm, a platform its card does not list, at its native 8,192-token limit (21 human-transfer answers over that limit count as failures); its parameter count includes its vision tower and the unmerged LoRA.

| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Typed Brier / ECE | mlx-diag non-English Choice / Noul | Invalid typed / transfer / public |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DEV2.0-8B | 7,940,895,744 | 67.737 | 0.8106 | 0.5660 | 736/800 / 715/800 / 246/400 | 178 (48/66/64) | 0.100 / 0.016 | 79.4 / 80.8 | 0/4/0 |
| Decision 1.0 Lux | 7,940,895,744 | 65.808 | 0.7762 | 0.5579 | 711/800 / 704/800 / 227/400 | 183 (48/67/68) | 0.119 / 0.017 | 78.3 / 83.3 | 0/4/0 |
| Nimble v2 | 9,453,092,080 | 62.056 | 0.7188 | 0.5358 | 784/800 / 654/800 / 107/400 | 185 (48/68/69) | 0.138 / 0.061 | 75.0 / 73.5 | 0/21/0 |

Report, panel and figure digests: [manifest.json](manifest.json).
