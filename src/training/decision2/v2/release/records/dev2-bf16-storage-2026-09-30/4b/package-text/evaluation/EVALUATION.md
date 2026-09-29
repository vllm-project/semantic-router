# Evaluation

**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, where T is the four-family macro accuracy of typed decisions and H is the median task macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every denominator. The answers of this panel were available during development, so results are same-panel comparisons rather than an untouched blind test.

JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. It is an independent rerun of the public questions, not the upstream four-axis score or the official sealed JevBench rank.

Every model ran through its own native inference path on the same frozen prompts and was scored by the same scorers; each row reports the parameters its native loader instantiates. Typed Brier and ECE (10 bins) measure calibration on the typed decisions. The paired interval is a joint bootstrap over typed groups (within family) and transfer tasks then items, 5,000 replicates. The candidate-minus-Decision 1.0 Nox v3 difference is +6.681 with paired 95% interval [+0.992, +9.636].

mlx-diag (development diagnostic, 2,275 prompts, seven languages, English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split (CC BY 4.0) and Noul from the PAWS-X test split; the columns show mean non-English accuracy. Its Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in backbone pretraining data, so this is not a sealed test.

Comparators under non-commercial, research-only or unknown licences are not shown on this card. Decision 1.0 Nox is shown from its adopted run (JevArena v3 56.47), the comparator of this size's release bar; its same-renderer run at this model's 16,384-token limit scores 55.69. Decider 4B (Apache-2.0) declares its licence in model card metadata only; its repository has no LICENSE file.

| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Typed Brier / ECE | mlx-diag non-English Choice / Noul | Invalid typed / transfer / public |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DEV2.0-4B | 4,208,383,488 | 63.151 | 0.6881 | 0.5796 | 582/800 / 734/800 / 185/400 | 171 (48/67/56) | 0.175 / 0.116 | 75.1 / 72.7 | 0/4/0 |
| Decider 4B | 4,205,751,296 | 61.882 | 0.6894 | 0.5555 | 750/800 / 623/800 / 114/400 | 192 (48/71/73) | 0.154 / 0.056 | 79.4 / 75.3 | 0/0/0 |
| Jet v6.2 | 4,205,751,296 | 60.375 | 0.6781 | 0.5375 | 720/800 / 640/800 / 125/400 | 174 (48/69/57) | 0.171 / 0.077 | 77.5 / 77.8 | 0/3/0 |
| Decision 1.0 Nox | 4,208,383,488 | 56.470 | 0.6144 | 0.5190 | 552/800 / 653/800 / 178/400 | 173 (48/66/59) | 0.205 / 0.092 | 75.9 / 80.0 | 0/4/0 |

Report, panel and figure digests: [manifest.json](manifest.json).
