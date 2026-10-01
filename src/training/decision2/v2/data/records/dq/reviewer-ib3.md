# Blind review instructions: IB3 items (yes / no)

You are one of several independent reviewers checking a sample of training items. Each item is a yes / no decision
question. Items cover several tasks: whether a link or a web page is a phishing link or page, whether every statement in
a dialogue response is supported by a knowledge passage, whether an answer to a question is supported by passages,
whether a product exactly matches a shopper's search query, whether a proposed answer to a math multiple-choice problem
is the correct option, and whether a merger-agreement excerpt supports a proposed answer to a deal-point question.
Texts are English, Spanish or Japanese. You do not see the answer stored in the dataset. Read only the packet file you
are given and write only your answer file. Do not open any other file, repository, dataset or web page, and do not run
programs that read other files.

## What each item contains

The packet has one JSON object per line:

- `rid`: the review id. Copy it into your answer.
- `task_type`: always `noul` (yes / no).
- `instructions`: the question, exactly as the training item asks it, including any definition it gives.
- `state`: the material the question is about (a URL, a URL with a page title, a knowledge passage with a dialogue
  turn and a response, passages with a question and an answer, a search query with a product, a math problem with its
  options and a proposed answer, a contract excerpt with a deal-point question and a proposed answer).
- `options`: always `false` "No" and `true` "Yes".

## How to answer

Answer the question in `instructions` about `state` as a careful expert would, using your own judgment and the
definitions the question gives.

- Work it out yourself: look at the whole web address (the registered domain, subdomains, path and any brand names in
  it) and the page title; check each statement of a response or answer against the passages; compare the product with
  every attribute in the query; solve the math problem; read the contract excerpt carefully. Do not guess from surface
  features such as length.
- Answer `true` for yes or `false` for no. Decide every item even when unsure; use `confidence` to say how sure you
  are.
- Set `defect` to `true` only when the item itself is broken so that no careful reader could answer it (for example
  missing or garbled material, or a question that does not fit the state), and say why in `note`.

## Output

Write one JSON object per line to your answer file, one per item, in any order:

```json
{"rid": "w001", "answer": "true", "confidence": "high", "defect": false, "note": ""}
```

- `answer`: `"true"` or `"false"`.
- `confidence`: `"high"`, `"medium"` or `"low"`.
- `defect`: `true` or `false`.
- `note`: at most 25 words. Required when confidence is `low` or `defect` is `true`; otherwise it may be empty.

Review every item; do not skip any. Write your answers to the file as you go, so that finished work is kept.
