# Evaluation

**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, where T is the four-family macro accuracy of typed decisions and H is the median task macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every denominator. The answers of this panel were available during development, so results are same-panel comparisons rather than an untouched blind test.

JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. It is an independent rerun of the public questions, not the upstream four-axis score or the official sealed JevBench rank.

Every model ran through its own native inference path on the same frozen prompts and was scored by the same scorers; each row reports the parameters its native loader instantiates. Typed Brier and ECE (10 bins) measure calibration on the typed decisions. The paired interval is a joint bootstrap over typed groups (within family) and transfer tasks then items, 5,000 replicates. The candidate-minus-Decision 1.0 Sol v3 difference is +7.656 with paired 95% interval [+3.265, +10.805].

mlx-diag (development diagnostic, 2,275 prompts, seven languages, English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split (CC BY 4.0) and Noul from the PAWS-X test split; the columns show mean non-English accuracy. Its Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in backbone pretraining data, so this is not a sealed test.

Comparators under non-commercial, research-only or unknown licences are not shown on this card. Decision 1.0 Sol is shown from its stricter same-renderer run at this model's 16,384-token limit (JevArena v3 45.78; its adopted run scores 45.58). Decider 2B (Apache-2.0) and This-That 1.2 (MIT) declare their licences in model card metadata only; their repositories have no LICENSE file.

| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Typed Brier / ECE | mlx-diag non-English Choice / Noul | Invalid typed / transfer / public |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DEV2.0-2B | 1,883,930,944 | 53.437 | 0.5434 | 0.5255 | 445/800 / 567/800 / 175/400 | 171 (48/66/57) | 0.271 / 0.103 | 66.5 / 65.8 | 0/4/0 |
| Decider 2B | 1,881,825,088 | 49.499 | 0.5831 | 0.4202 | 545/800 / 497/800 / 133/400 | 175 (48/64/63) | 0.259 / 0.063 | 75.4 / 66.0 | 0/0/0 |
| This-That 1.2 | 1,881,825,088 | 46.112 | 0.5250 | 0.4050 | 529/800 / 431/800 / 79/400 | 147 (48/64/35) | 0.344 / 0.260 | 76.2 / 66.5 | 0/169/37 |
| Decision 1.0 Sol | 1,883,930,944 | 45.781 | 0.4253 | 0.4928 | 374/800 / 438/800 / 155/400 | 160 (48/66/46) | 0.345 / 0.249 | 67.1 / 68.0 | 0/4/0 |
| Bosun v3.1 1.7B | 1,737,985,024 | 42.117 | 0.4669 | 0.3799 | 415/800 / 493/800 / 99/400 | 151 (46/62/43) | 0.312 / 0.127 | 70.0 / 70.2 | 0/0/0 |

Report, panel and figure digests: [manifest.json](manifest.json).
