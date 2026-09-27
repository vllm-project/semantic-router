# F1000RD human recommendation source: CPU admission screen

**Decision: TRAIN HOLD; possible independent long-document diagnostic only.**
This is a source metadata and text-length audit, not a new model result. No
Decision weights, optimizer, native inference, GPU, protected evaluation labels
or HF dataset upload were used. The source was not admitted to TRAIN, SELECT,
CAL or JevArena.

## Pinned primary source and actual contents

The [UKP Lab F1000RD repository](https://github.com/UKPLab/f1000rd) at Git
revision `77555d5c138bd770ccd9048b94cc3b2b0637b0be` contains the
*study sample*, not its on-request full crawl. Its README describes manuscript
versions, reports on the first version, and a submission-disjoint
`data/simple/splits.csv`; the split CSV SHA-256 is
`ac7284e0746df8767422784ccfa1dda6206520b82673f6a9ce7c1aa785461985`.
The [paper](https://doi.org/10.1162/coli_a_00455) describes the open-review
collection and its cross-document annotations. The publisher's
[review guidance](https://f1000research.com/for-authors/peer-review) defines
the three recommendation categories. This source is journal peer review, a
different task and document domain from the current Decision training mix;
source-name absence alone does not prove record-level disjointness.

| Property from pinned source | CPU observation |
| --- | ---: |
| Distinct submission IDs / initial-version files | 172 / 172 |
| Human review reports with a recommendation | 224 |
| Article groups by published train/dev/test split | 120 / 17 / 35 |
| Review reports by train/dev/test | 160 / 22 / 42 |
| Recommendation reports: reject / approve with reservations / approve | 26 / 118 / 80 |
| Train recommendation reports in that order | 14 / 89 / 57 |
| Article groups whose reviewers disagree | 21 (19 train, 1 dev, 1 test) |
| Unanimous train article groups in that order | 7 / 60 / 34 |
| Version-1 article core text, whitespace words | median 2,929; p90 6,672; max 19,436 |
| Review report text, whitespace words | median 388; p90 690; max 1,464 |

Counts come from `data/itg/<submission-id>/v1.json` and
`data/itg/<submission-id>/reviews/*.json`, grouped on the submission directory
ID and joined to the publisher's split CSV. Review metadata supplies
`recommendation` and a review ID. Article and review texts have separate
versioned records. Lengths are **whitespace words**, not native Qwen tokens;
an 8,192-token input limit has not been checked. The source has no explicit
language field in the checked article metadata. Do not claim multilingual
coverage without a language audit.

## Native decision interpretation and limits

A possible `Score` request would provide the *first manuscript version* as
`state` and ask for the degree of reviewer approval on the ordered rubric
`not approved < approved with reservations < approved`. This is an ordinal
recommendation about a real artifact, but each report is a different human
judgment. Twenty-one submission groups have discordant labels, so one article
does not have a single deterministic answer. Treating all reviews of that
article as independent questions or making a random row split would leak
article content and overstate precision. A soft empirical vote distribution
could be explored later, but 123/172 articles have only one report, and no
consistent probability target is thereby established. Later manuscript
versions and review text must not be supplied to an article-only prediction;
they may reveal the outcome or subsequent revision.

The three labels could also be projected to a binary `Noul` or three-option
`Choice` question, but those would be *the same underlying observations*, not
additional task coverage. The source does not provide a verified rule,
evidence-chain or state-update oracle. It cannot by itself train or certify
general Choice/Noul/Score transfer. The official source's dev/test groups
measure within-source generalization, not genuine cross-source transfer after
training on its train split.

## Rights and admission disposition

The repository README states **CC BY-SA 4.0 for the dataset**. Its root
`LICENSE.txt` says Apache-2.0 for repository software; it must not be read as
relicensing the data. The checked review records embed CC BY 4.0 metadata;
the article versions embed CC BY 4.0 (288 files), CC BY 3.0 (19), or CC BY 3.0
IGO (4). The publisher's [policies](https://f1000research.com/about/policies)
state that article and report rights are item-specific. Any private derived
corpus needs per-item provenance, attribution and applicable share-alike
handling; a private HF repository does not resolve those conditions. No raw
source text or reports were copied into this code repository or the gist.

**Why TRAIN is on HOLD:** only seven unanimous TRAIN articles carry the low
recommendation, the report-level target has reviewer disagreement, native
token exposure and source-overlap gates are undone, and this task alone cannot
repair the observed typed Score collapse or prove three-type transfer. Do not
launch a matched 2B/4B training arm from this source.

**Smallest useful next CPU gate:** if a long-document external diagnostic is
desired, pin an article-only `v1` extraction and reviewer-group scoring rule;
exclude/handle disagreement without reading protected Decision labels; verify
language, per-item rights, native 2B/4B token lengths, and exact/near matches
against TRAIN/SELECT/CAL and gold-free typed/CSS/public prompts. Reserve the
F1000RD publisher test groups before any model scoring. This screen already
inspected their aggregate grade counts, so they cannot be called untouched
blind labels for a design chosen afterward. Then run a
simple title/abstract-only and document-length shortcut check before any
model evaluation. Without those gates, keep it as a documented source screen.
