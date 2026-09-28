# PeerRead ordinal Score source triage v1

**Status: prospective source inventory only.** This screen asks whether the
publisher's human review ratings could provide a *distinct* ordinal Score
source for future 0.8B/27B training. It does not admit rows, launch a model,
or alter any existing training or release gate. The repository HEAD to inspect
is `9bb37751781a900cee9e74ec3105997732c8e8e5`; record the exact checked-out
commit and file hashes before any data analysis.

The [PeerRead publisher repository](https://github.com/allenai/PeerRead) contains
venue-specific review sections and says licenses vary by section. The
[source paper](https://aclanthology.org/N18-1149/) reports numeric review
aspects and recommendations. Its authors also found that about half of aspect
ratings could not be inferred from the corresponding review text in a small
feasibility check. Those two facts make rights and evidence sufficiency hard
gates, not details to infer from a GitHub repository's public visibility.

Inspect the publisher's repository tree and section-specific terms first.
Read only publisher TRAIN records for any aggregate counts, never its DEV/TEST
records or protected JevArena labels. Report per-section counts of review
records with nonempty review text and an **explicit human numeric overall
recommendation**, the original scale, unique paper IDs, review-text word
lengths, and score distribution. Keep original text, paper IDs, annotator IDs,
and any score-bearing examples in the private source environment. Do not
combine incompatible rating scales or turn an accept/reject paper decision
into an ordinal review score. Identify literal score leaks in review text
without printing them.

An affirmative triage requires a section-specific use/retention conclusion,
at least 400 distinct review records over at least 100 distinct papers with
human numeric overall ratings, at least three populated ordinal grades, and
no obvious score field copied into the model input. These are only source
feasibility criteria. Before any training admission, separately require
paper-disjoint sampling, native request-length analysis, exact/near/semantic
source overlap against TRAIN/SELECT/CAL and all protected panels, a frozen
answer-blind rubric review, and matched-token controls. If the section terms
or labels cannot be verified, record HOLD with zero admitted rows and zero
GPU-hours.

The potential task is to rate the decision-relevant **review text** using
the venue's stated recommendation rubric. It is not evidence that the model
can judge a paper from its abstract or infer a reviewer's hidden score when
the rating is not supported by the text. Any later Score projection must
explicitly handle uncertain/undiscussed reviews rather than manufacture a
three-grade label.
