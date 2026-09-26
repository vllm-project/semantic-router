# Multilingual development audit and paired pilot

This directory audits model-visible text in training and **public development**
panels and builds a small, self-authored multilingual typed-decision pilot. It
does not read sealed final prompts or labels. The pilot is diagnostic; it is
not a language benchmark or evidence for a state-of-the-art claim.

## Language coverage census

`python -m multilingual.audit` records the input SHA256, source buckets,
declared language where available, Unicode script counts, and optional
`langid==1.1.6` guesses. It emits **counts only**, never source text or training
rows. The projection uses `instructions`, `state`, and option descriptions for
training, and `state` and `questions` for benchmark prompts. Labels and audit
metadata are excluded. Model-visible text can contain JSON, code, names,
English instructions and another language in the state, so the classifier's
top label is not an independently verified language label. We retain both the
exact script census and the metadata labels and report detector disagreement
instead of treating its log-score margin as a calibrated confidence.

The frozen clean-v2 TRAIN input contains 7,455 rows: 6,085 declared English
and 1,370 declared Chinese. The 1,370 Chinese rows are internal structured or
programmatic examples; the 2,800 GoEmotions human rows are English. No row is
declared Spanish, French, German, Japanese or Arabic. The full-text `langid`
top label is Chinese for 1,352 rows and English for 5,953; 112 rows have fewer
than 60 visible letters and are left unclassified. Mixed language and code can
mask other languages. These are corpus counts, without sampling error, but
neither the training metadata nor automatic classifier establishes semantic
language ground truth for every row.

The existing public DEV 1,600 and JevBench public 231 panels are detected as
English throughout. The three-task CSS pilot 1,430 has one apparent Icelandic
classifier outlier and no row with significant non-Latin script. The 1,041
text-readable Decision Bench v4 cases have four non-English detector guesses
in document-related material and no significant non-Latin row. Those guesses
must be reviewed as possible code/template false positives. Existing panels
therefore provide little direct evidence for Spanish, French, German, Japanese
or Arabic transfer.

Example audit command, with a local data mount and an installed `langid`:

```bash
python -m multilingual.audit --input <workspace>/data/train.jsonl \
  --output <workspace>/runs/train-language-audit.json \
  --kind training --source-key source --langid
```

## Paired native typed-decision slice

`python -m multilingual.pilot --output-dir <workspace>/bench/multilingual-dev`
builds 210 prompts and separate targets from 18 controlled English base cases
in English, Chinese, Spanish, French, German, Japanese and Arabic. There are
six Choice, six Noul and six Score base cases. Each language has 30 native
prompts: Choice has an option order and key rotation; Noul has a label wording
variant; Score preserves ordinal level order. Option perturbations and
translations remain paired to the same base case. The independence unit is
18 base cases, not 210 prompts.

The author checked the controlled translations. The builder verifies numeric
facts, the protected entity, rubric cut points, target script presence,
complete language coverage and gold-free prompts. A native parser preflight
checks Eikos Choice, Noul and Score option semantics. There has been no
independent native-speaker review, and the small, simple set has wide
uncertainty for real multilingual traffic.

```bash
python -m multilingual.native_preflight \
  --panel <workspace>/bench/multilingual-dev \
  --eikos-source <workspace>/models/Eikos-4B
python -m multilingual.score \
  --panel <workspace>/bench/multilingual-dev \
  --predictions <workspace>/runs/model.predictions.jsonl \
  --output <workspace>/runs/model.multilingual.score.json
```

The scorer verifies input, target and prediction hashes and native answer
identity. It reports accuracy by language and type, English-paired semantic
prediction agreement, option-perturbation flips and invalid counts. Each
language's macro accuracy and gap versus English average over the same 18
base cases. Model selection must continue to use its independent frozen
selection split; this authored slice is an exploratory development diagnostic.

## Human-translated parallel development slice

The controlled slice has a ceiling effect: all four tested models answered
210/210 prompts. `python -m multilingual.parallel` builds a harder,
independent follow-up from pinned validation revisions of
[XNLI](https://arxiv.org/abs/1809.05053) and
[PAWS-X](https://arxiv.org/abs/1908.11828). It selects 60 XNLI source IDs
(20 per NLI label) and 40 PAWS-X source IDs (20 per paraphrase label), then
pairs each with the English and five translated versions. Thus 600 prompts
represent **100 independent source IDs**. XNLI covers English, Arabic,
German, Spanish, French and Chinese; PAWS-X covers English, German, Spanish,
French, Japanese and Chinese. They use native Choice and Noul readouts,
respectively. The panel stays local: source text and translations are not
committed.

The builder validates one-to-one XNLI language coverage, matched PAWS-X IDs,
nonempty translations and equal labels across all six PAWS-X languages. It
quarantines 59/2,000 PAWS-X validation IDs with inconsistent upstream labels
before deterministic label-stratified sampling. These mechanical checks do
not replace bilingual semantic review; translation artifacts and remaining
annotation noise may affect scores. The official validation rows must never
be used to train or select a checkpoint. The local manifest pins upstream
revisions, parquet hashes, script hash, prompt hash, target hash and the
mismatch quarantine hash; local target rows include selected source-row
digests.

```bash
python -m multilingual.parallel \
  --xnli-root <workspace>/datasets/xnli \
  --pawsx-root <workspace>/datasets/paws-x \
  --output-dir <workspace>/bench/multilingual-parallel-dev
python -m multilingual.native_preflight \
  --panel <workspace>/bench/multilingual-parallel-dev \
  --eikos-source <workspace>/models/Eikos-4B
python -m multilingual.parallel_score \
  --panel <workspace>/bench/multilingual-parallel-dev \
  --predictions <workspace>/runs/model.predictions.jsonl \
  --output <workspace>/runs/model.parallel.score.json
python -m multilingual.compare \
  --panel <workspace>/bench/multilingual-parallel-dev \
  --baseline <workspace>/runs/baseline.predictions.jsonl \
  --candidate <workspace>/runs/candidate.predictions.jsonl \
  --output <workspace>/runs/paired-model-compare.json
```

Both scorers check the frozen panel and prediction input hashes. They report
invalid native answers as wrong and compare each translation to the *same*
English source ID. `compare` gives matched model deltas and two-sided exact
McNemar p-values by corpus and language. Each language slice has only 60 or
40 independent IDs, and multiple languages are inspected. Individual small
deltas or nominal p-values are exploratory evidence, not release claims.
