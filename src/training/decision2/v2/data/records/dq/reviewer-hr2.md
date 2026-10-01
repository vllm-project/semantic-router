# Blind review instructions: HR2 items (Choice, Noul, Score)

You are one of several independent reviewers checking a sample of training items. Each item is a decision
question: pick one option (Choice), answer yes or no (Noul), or pick a level on an ordered scale (Score). Texts
may be in English, Chinese, Korean, Japanese, Indonesian, Polish, French, Spanish, German, Russian, Portuguese,
Italian, Dutch, Vietnamese, or contain program code or mathematics. You do not see the answer stored in the
dataset. Read only the packet file you are given and write only your answer file. Do not open any other file,
repository, dataset or web page, and do not run programs that read other files.

## What each item contains

The packet has one JSON object per line:

- `rid`: the review id. Copy it into your answer.
- `task_type`: `choice`, `noul` or `score`.
- `instructions`: the question, exactly as the training item asks it.
- `state`: the material the question is about (a conversation and candidate responses, a scenario, a problem
  with solution steps, a claim with evidence, a passage, a review, ...).
- `options`: the answer options as `[{"key": ..., "description": ...}, ...]`. For `noul` they are always
  `false` "No" and `true` "Yes"; for `score` the keys are levels `0`, `1`, ... in increasing order.

## How to answer

Answer the question in `instructions` about `state` as a careful expert would, using your own judgment.

- **Choice:** give the key of the option you judge correct or better.
- **Noul:** give `true` for yes or `false` for no.
- **Score:** give the key of the level you judge right.
- Work it out yourself (check a mathematical step, compare the two responses on helpfulness, correctness and
  following the user's instructions, read the passage carefully, etc.). Do not guess from surface features.
- Decide every item even when unsure; use `confidence` to say how sure you are.
- Set `defect` to `true` only when the item itself is broken so that no careful reader could answer it (for
  example missing or garbled material, or a question that does not fit the state), and say why in `note`.

## Output

Write one JSON object per line to your answer file, one per item, in any order:

```json
{"rid": "h001", "answer": "a", "confidence": "high", "defect": false, "note": ""}
```

- `answer`: an option key from the item (`"a"`, `"b"`, `"true"`, `"false"`, `"0"` … `"4"`).
- `confidence`: `"high"`, `"medium"` or `"low"`.
- `defect`: `true` or `false`.
- `note`: at most 25 words. Required when confidence is `low` or `defect` is `true`; otherwise it may be empty.

Review every item; do not skip any. Write your answers to the file as you go, so that finished work is kept.
