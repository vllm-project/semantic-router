# Blind review instructions: IB1 items (Choice, Noul)

You are one of several independent reviewers checking a sample of training items. Each item is a decision
question: pick one option (Choice) or answer yes or no (Noul). Items cover many tasks: whether a summary or a claim
is supported by a document or evidence, whether a text message is spam, which reply an assistant should give or which
function it should call, the stance of an argument, which of two texts is sarcastic, whether a math solution is
correct, the sentiment of a news headline toward a company, how well a product matches a search query, deal points
in a merger agreement, commonsense and exam questions, the relation between a premise and a hypothesis, and which of
two poem stanzas is the original. Texts are mostly English; some are Arabic. You do not see the answer stored in the
dataset. Read only the packet file you are given and write only your answer file. Do not open any other file,
repository, dataset or web page, and do not run programs that read other files.

## What each item contains

The packet has one JSON object per line:

- `rid`: the review id. Copy it into your answer.
- `task_type`: `choice` or `noul`.
- `instructions`: the question, exactly as the training item asks it.
- `state`: the material the question is about (a document and a summary, a claim with evidence, a message, tool
  definitions and candidate replies, a statement and an argument, two texts, a problem with solution steps, a
  headline and an entity, a query and a product, a contract excerpt, a question, a premise and a hypothesis, two
  stanzas, ...).
- `options`: the answer options as `[{"key": ..., "description": ...}, ...]`. For `noul` they are always `false`
  "No" and `true` "Yes".

## How to answer

Answer the question in `instructions` about `state` as a careful expert would, using your own judgment.

- **Choice:** give the key of the option you judge correct or best.
- **Noul:** give `true` for yes or `false` for no.
- Work it out yourself (check each mathematical step, compare the summary or claim with the source sentence by
  sentence, read the contract excerpt carefully, compare the two candidates, etc.). Do not guess from surface
  features such as length or position.
- Decide every item even when unsure; use `confidence` to say how sure you are.
- Set `defect` to `true` only when the item itself is broken so that no careful reader could answer it (for example
  missing or garbled material, or a question that does not fit the state), and say why in `note`.

## Output

Write one JSON object per line to your answer file, one per item, in any order:

```json
{"rid": "q001", "answer": "a", "confidence": "high", "defect": false, "note": ""}
```

- `answer`: an option key from the item (for example `"a"`, `"o2"`, `"true"`, `"false"`).
- `confidence`: `"high"`, `"medium"` or `"low"`.
- `defect`: `true` or `false`.
- `note`: at most 25 words. Required when confidence is `low` or `defect` is `true`; otherwise it may be empty.

Review every item; do not skip any. Write your answers to the file as you go, so that finished work is kept.
