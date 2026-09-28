# Evaluation

**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, where T is the four-family macro accuracy of typed decisions and H is the median task macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every denominator. The answers of this panel were available during development, so results are same-panel comparisons rather than an untouched blind test.

JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. It is an independent rerun of the public questions, not the upstream four-axis score or the official sealed JevBench rank.

Every model ran through its own native inference path on the same frozen prompts and was scored by the same scorers; each row reports the parameters its native loader instantiates. Typed Brier and ECE (10 bins) measure calibration on the typed decisions. The paired interval is a joint bootstrap over typed groups (within family) and transfer tasks then items, 5,000 replicates. The candidate-minus-Decision 1.0 Eos v3 difference is +7.689 with paired 95% interval [+3.646, +13.316].

mlx-diag (development diagnostic, 2,275 prompts, seven languages, English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split (CC BY 4.0) and Noul from the PAWS-X test split; the columns show mean non-English accuracy. Its Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in backbone pretraining data, so this is not a sealed test.

Comparators under non-commercial, research-only or unknown licences are not shown on this card.

| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Typed Brier / ECE | mlx-diag non-English Choice / Noul | Invalid typed / transfer / public |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DEV2.0-0.8B | 753,446,208 | 50.236 | 0.5734 | 0.4401 | 529/800 / 611/800 / 107/400 | 156 (48/63/45) | 0.254 / 0.103 | 68.7 / 54.2 | 0/4/0 |
| Intern-Decision-0.8B | 852,985,920 | 43.535 | 0.4956 | 0.3824 | 495/800 / 431/800 / 88/400 | 164 (47/57/60) | 0.297 / 0.093 | 64.2 / 62.7 | 0/115/0 |
| Kev-0.8B | 752,917,824 | 43.217 | 0.4794 | 0.3896 | 489/800 / 407/800 / 82/400 | 147 (48/58/41) | 0.303 / 0.066 | 61.5 / 58.7 | 0/13/0 |
| Decision 1.0 Eos | 753,446,208 | 42.547 | 0.3925 | 0.4612 | 315/800 / 410/800 / 120/400 | 142 (48/55/39) | 0.333 / 0.165 | 68.5 / 59.2 | 0/4/0 |

Report, panel and figure digests: [manifest.json](manifest.json).
