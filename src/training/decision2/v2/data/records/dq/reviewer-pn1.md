# Blind review instructions: sentence-pair items (PN1)

You are one of several independent reviewers checking a sample of training items. Each item is a yes/no
question about two sentences written in one language (Japanese, Chinese, German, Russian, Korean, Arabic,
Spanish or French). You do not see the answer stored in the dataset. Read only the packet file you are given
and write only your answer file. Do not open any other file, repository, dataset or web page, and do not run
programs that read other files.

## What each item contains

The packet has one JSON object per line:

- `rid`: the review id. Copy it into your answer.
- `language`: the language of the two sentences.
- `instructions`: the question, exactly as the training item asks it.
- `state`: the two sentences, one per line, each after a short English label (for example `Sentence A:`).
- `options`: the two answer descriptions, always `["No", "Yes"]`.

## How to answer

Answer the question in `instructions` for the two sentences in `state`, as a careful, fluent speaker of the
language would.

- Judge meaning, not form. Differences in punctuation, spacing, script width or capitalisation never matter.
- Apply the question as written. Some questions ask whether the sentences say *exactly* the same thing, or
  whether the same people and things are in the same roles; take that literally.
- Decide every item yes or no, even when unsure; use `confidence` to say how sure you are.

Also rate each sentence on its own, in state order (first line, then second line):

- `ok`: natural and grammatical;
- `awkward`: grammatical but unnatural;
- `ungrammatical`: a fluent speaker would not produce it (broken grammar, wrong word, garbled text).

## Output

Write one JSON object per line to your answer file, one per item, in any order:

```json
{"rid": "p001", "answer": "yes", "confidence": "high", "fluency": ["ok", "ok"], "note": ""}
```

- `answer`: `"yes"` or `"no"`.
- `confidence`: `"high"`, `"medium"` or `"low"`.
- `fluency`: two values, one per sentence in state order.
- `note`: at most 25 words. Required when the answer is `no`, when confidence is `low`, or when a sentence
  is not `ok`: say what differs or what is wrong. Otherwise it may be empty.

Review every item; do not skip any. Write your answers to the file as you go, so that finished work is kept.
