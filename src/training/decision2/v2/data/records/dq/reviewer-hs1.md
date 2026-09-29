# Blind review instructions: generated decision items (HS1)

You are an independent reviewer checking a sample of generated training items. You do not see the answer
stored in the dataset. Read only the packet file you are given and write only your answer file. Do not open
any other file, repository, dataset or web page, and do not run programs that read other files.

## What each item contains

The packet has one JSON object per line:

- `rid`: the review id. Copy it into your answer.
- `task_type`: `choice`, `noul` or `score`.
- `instructions`: the question.
- `state`: the text the question is about. It is fictional and self-contained, for example a quoted person's
  conclusion with the evidence it is about, a long policy with amendments and exceptions, or a requirement
  and a case.
- `options`: a list of `{"key", "description"}`.
  - `choice`: pick the key of the single best option.
  - `noul`: the keys are `"false"` (No) and `"true"` (Yes).
  - `score`: the keys `"0"`, `"1"`, ... are levels; pick the level whose description fits.

## How to answer

Answer from the state only, as a careful expert. Read the whole state. Policies may contain amendments,
withdrawals, annexes, temporary measures, exceptions and stated precedence rules, and any of them can decide
the answer. A quoted conclusion can be wrong. A condition can be unmet because of one detail.

The items are meant to be hard. Hardness is not a defect. Report a defect only in the item itself:

- `garbled`: broken rendering, such as template placeholders, repeated or truncated text, wrong names or
  numbers substituted, or formatting debris;
- `contradiction`: the state contradicts itself in a way the item does not intend (a quoted claim that
  disagrees with the evidence is intended);
- `ambiguous`: more than one option is defensible, or the state does not determine an answer;
- `options_malformed`: duplicate, missing or mis-worded options, or options that do not fit the question;
- `gold_unclear`: you doubt that the item has a single intended answer for another reason;
- `other`: anything else; explain it in the note.

## Output

Write one JSON object per line to your answer file, one per item, in any order:

```json
{"rid": "h001", "answer": "o2", "confidence": "high", "flags": [], "note": ""}
```

- `answer`: one option key, copied exactly.
- `confidence`: `"high"`, `"medium"` or `"low"`.
- `flags`: a list of the defect names above; empty if there is no defect.
- `note`: at most 40 words. Required when `flags` is not empty or confidence is `low`: quote or point to the
  sentence(s) that decide the answer or show the defect.

Review every item; do not skip any. Write your answers to the file as you go, so that finished work is kept.
