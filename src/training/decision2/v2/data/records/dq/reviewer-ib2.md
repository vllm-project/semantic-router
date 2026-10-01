# Blind review instructions: IB2 items (Choice, Noul)

You are one of several independent reviewers checking a sample of training items. Each item is a decision
question: pick one option (Choice) or answer yes or no (Noul). Items cover many tasks: whether one of an assistant's
available functions can carry out a user's request, which function to call, which arguments to pass, whether a request
already contains every required argument, whether a YouTube comment is spam, whether an argument supports or opposes a
statement, whether Wikipedia passages support a claim, science multiple-choice questions, whether a stated answer to a
math word problem is correct, and whether a contract entails a statement. Texts are English. You do not see the answer
stored in the dataset. Read only the packet file you are given and write only your answer file. Do not open any other
file, repository, dataset or web page, and do not run programs that read other files.

## What each item contains

The packet has one JSON object per line:

- `rid`: the review id. Copy it into your answer.
- `task_type`: `choice` or `noul`.
- `instructions`: the question, exactly as the training item asks it.
- `state`: the material the question is about (function definitions and a user request, a comment, a statement and an
  argument, a claim with Wikipedia passages, a question, a math problem with a stated answer, a contract, ...).
- `options`: the answer options as `[{"key": ..., "description": ...}, ...]`. For `noul` they are always `false`
  "No" and `true` "Yes".

## How to answer

Answer the question in `instructions` about `state` as a careful expert would, using your own judgment.

- **Choice:** give the key of the option you judge correct or best.
- **Noul:** give `true` for yes or `false` for no.
- Work it out yourself (solve the math problem, check each part of the claim against the passages, compare the
  request with each function and argument value, read the contract carefully, etc.). Do not guess from surface
  features such as length or position.
- Decide every item even when unsure; use `confidence` to say how sure you are.
- Set `defect` to `true` only when the item itself is broken so that no careful reader could answer it (for example
  missing or garbled material, or a question that does not fit the state), and say why in `note`.

## Output

Write one JSON object per line to your answer file, one per item, in any order:

```json
{"rid": "u001", "answer": "o2", "confidence": "high", "defect": false, "note": ""}
```

- `answer`: an option key from the item (for example `"o2"`, `"support"`, `"true"`, `"false"`).
- `confidence`: `"high"`, `"medium"` or `"low"`.
- `defect`: `true` or `false`.
- `note`: at most 25 words. Required when confidence is `low` or `defect` is `true`; otherwise it may be empty.

Review every item; do not skip any. Write your answers to the file as you go, so that finished work is kept.
