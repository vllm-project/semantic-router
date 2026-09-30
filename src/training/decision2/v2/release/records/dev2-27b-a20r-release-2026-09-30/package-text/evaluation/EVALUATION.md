# Evaluation

**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, where T is the four-family macro accuracy of typed decisions and H is the median task macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every denominator. The answers of this panel were available during development, so results are same-panel comparisons rather than an untouched blind test.

JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. It is an independent rerun of the public questions, not the upstream four-axis score or the official sealed JevBench rank.

Every model ran through its own native inference path on the same frozen prompts and was scored by the same scorers; each row reports the parameters its native loader instantiates. Typed Brier and ECE (10 bins) measure calibration on the typed decisions. The paired interval is a joint bootstrap over typed groups (within family) and transfer tasks then items, 5,000 replicates. There is no Decision 1.0 model at this size. The candidate-minus-AutoJev-27B v3 difference is +0.227 with paired 95% interval [-1.598, +4.739]. Against the other models shown: Eikos-27B +3.070 [+0.197, +7.751]; Jebadiah-27B +6.888 [+4.330, +10.540].

mlx-diag (development diagnostic, 2,275 prompts, seven languages, English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split (CC BY 4.0) and Noul from the PAWS-X test split; the columns show mean non-English accuracy. Its Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in backbone pretraining data, so this is not a sealed test.

Comparators under non-commercial, research-only or unknown licences are not shown on this card. The three 27B peers were rerun on the same kernel image as this model. Eikos-27B is shown from its BF16 weights (`caiovicentino1/Eikos-27B`), the sibling of the FP8 build listed on the Decision Index; its card declares MIT for the authors' contributions on the Apache-2.0 base. AutoJev-27B declares Apache-2.0 weights and MIT code, Jebadiah-27B Apache-2.0. Invalid answers count as failures: none for this model; 13, 4 and 153 transfer answers for AutoJev-27B, Eikos-27B and Jebadiah-27B, and 36 public-231 answers for Jebadiah-27B.

| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Typed Brier / ECE | mlx-diag non-English Choice / Noul | Invalid typed / transfer / public |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DEV2.0-27B | 26,096,775,168 | 72.360 | 0.8962 | 0.5842 | 753/800 / 737/800 / 344/400 | 203 (48/72/83) | 0.059 / 0.013 | 81.0 / 86.0 | 0/0/0 |
| AutoJev-27B | 26,086,635,760 | 72.133 | 0.8869 | 0.5867 | 800/800 / 737/800 / 282/400 | 201 (48/72/81) | 0.064 / 0.082 | 80.6 / 88.7 | 0/13/0 |
| Eikos-27B | 27,781,427,952 | 69.290 | 0.8175 | 0.5873 | 783/800 / 589/800 / 336/400 | 212 (48/72/92) | 0.107 / 0.015 | 78.4 / 85.0 | 0/4/0 |
| Jebadiah-27B | 26,895,998,464 | 65.472 | 0.7419 | 0.5778 | 617/800 / 720/800 / 250/400 | 176 (48/70/58) | 0.143 / 0.062 | 77.6 / 85.3 | 0/153/36 |

Report, panel and figure digests: [manifest.json](manifest.json).
