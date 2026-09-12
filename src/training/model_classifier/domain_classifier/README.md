# Vela Domain classifier data

Domain is the academic-subject signal historically called `intent`. It does not
predict arbitrary user intents or select a downstream model by itself. The Vela
application name is `Vela-1.0-Encoder-307M-Domain`; runtime label IDs remain:

| ID | Domain | ID | Domain |
| --- | --- | --- | --- |
| 0 | biology | 7 | history |
| 1 | business | 8 | law |
| 2 | chemistry | 9 | math |
| 3 | computer science | 10 | other |
| 4 | economics | 11 | philosophy |
| 5 | engineering | 12 | physics |
| 6 | health | 13 | psychology |

`prepare_data.py` maps Global-MMLU subjects to this existing ontology, keeps the
question and answer choices, and excludes the answer key. English source
questions and their translations stay together. If normalized text connects two
source groups, all their translated siblings are unioned transitively before
splitting. Conflicting-label components are excluded.

All available MMLU-Pro test/validation rows and the category supplement are used
for historical exposure screening. Exact matches or English unigram/bigram TF-IDF
cosine similarity of at least 0.75 force the whole translated group into training.
The screen is deliberately conservative and does not establish absence from old
weights or base pretraining. It also does not turn a benchmark question into a
natural conversational request. Independently reviewed Aya requests are added
with contributor-group splits and reported as a separate source.

Fetch the pinned files and hydrate Aya labels using the
[shared workflow](../sequence_repair/README.md), then prepare the corpus:

```bash
python -m src.training.model_classifier.domain_classifier.prepare_data \
  --sources artifacts/vela/sources \
  --legacy-datasets artifacts/vela/sources \
  --annotations artifacts/vela/annotations/aya-reviewed-labels-v1.jsonl \
  --contract artifacts/vela/reference-domain/config.json \
  --output artifacts/vela/domain
```

The reference contract is the original 14-label classifier configuration. The
initial continuation experiment uses the original fixed Domain adapter on
`mmbert-32k-yarn@72a23a6640489471eb4ff7ad3ec5bc80af8a27de`; every candidate receipt
must identify its actual base and adapter hashes. A later Base migration requires
a development-only short/long comparison before freezing final lineage.

Use source-balanced development selection to keep the small natural-request
source visible. Report per-class and per-language quality, not only the aggregate
academic score. Long context variants mark the requested task within independent
authored background narratives; they are controlled stress tests, not natural
long-document estimates. Independent test evaluation occurs only after the
candidate and base lineage are frozen.
