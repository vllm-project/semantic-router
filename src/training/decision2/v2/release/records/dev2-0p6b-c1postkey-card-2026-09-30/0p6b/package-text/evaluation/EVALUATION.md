# Evaluation

**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, where T is the four-family macro accuracy of typed decisions and H is the median task macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every denominator. The answers of this panel were available during development, so results are same-panel comparisons rather than an untouched blind test.

JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. It is an independent rerun of the public questions, not the upstream four-axis score or the official sealed JevBench rank.

Every model ran through its own native inference path on the same frozen prompts and was scored by the same scorers; each row reports the parameters its native loader instantiates. Typed Brier and ECE (10 bins) measure calibration on the typed decisions. The paired interval is a joint bootstrap over typed groups (within family) and transfer tasks then items, 5,000 replicates. The candidate-minus-Decision 1.0 Kai v3 difference is +12.698 with paired 95% interval [+10.002, +17.500].

mlx-diag (development diagnostic, 2,275 prompts, seven languages, English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split (CC BY 4.0) and Noul from the PAWS-X test split; the columns show mean non-English accuracy. Its Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in backbone pretraining data, so this is not a sealed test.

Comparators under non-commercial, research-only or unknown licences are not shown on this card. GLiNER2.5-Decide declares Apache-2.0 in its model card metadata, but its repository has no LICENSE file.

| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Typed Brier / ECE | mlx-diag non-English Choice / Noul | Invalid typed / transfer / public |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 597,103,104 | 48.636 | 0.5153 | 0.4590 | 389/800 / 517/800 / 137/400 | 152 (47/61/44) | 0.321 / 0.155 | 70.2 / 52.8 | 0/15/0 |
| GLiNER2.5-Decide | 486,444,053 | 42.524 | 0.4100 | 0.4410 | 313/800 / 502/800 / 88/400 | 116 (48/46/22) | 0.336 / 0.128 | 57.7 / 52.5 | 0/1072/56 |
| Bosun v3.1 0.6B | 606,131,200 | 38.524 | 0.4338 | 0.3422 | 365/800 / 541/800 / 83/400 | 133 (47/50/36) | 0.302 / 0.090 | 63.2 / 61.2 | 0/0/0 |
| Decision 1.0 Kai | 571,909,635 | 35.938 | 0.3619 | 0.3569 | 277/800 / 404/800 / 98/400 | 114 (45/39/30) | 0.390 / 0.209 | 49.1 / 47.7 | 0/404/44 |
| Decision 1.0 Lex | 571,909,635 | 31.022 | 0.3619 | 0.2659 | 235/800 / 402/800 / 107/400 | 113 (42/42/29) | 0.336 / 0.156 | 26.5 / 51.0 | 0/404/44 |

Report, panel and figure digests: [manifest.json](manifest.json).
