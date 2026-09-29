# Adjudication instructions (PN1 third review, HS1 adjudication)

You are an independent reviewer. You do not see the answers stored in the datasets, and you do not know why
an item was selected. Read only the packet files you are given and write only your answer files. Do not open
any other file, repository, dataset or web page, and do not run programs that read other files.

## Part 1: sentence-pair items (PN1)

Follow `reviewer-pn1.md` exactly (the text is repeated in your task). The packet and the answer format are
the same as there.

## Part 2: generated decision items (HS1)

The items have the fields of `reviewer-hs1.md` (`task_type`, `instructions`, `state`, `options`), an `aid`
to copy into your answer, and a `type`:

- `type: "disagree"`: `candidates` is `{"A": key, "B": key}`, two different answers that two different
  sources gave for this item. Read the whole state and decide:
  - `"A"` or `"B"`: that answer is correct and the other is not;
  - `"both"`: both are defensible, because the item does not determine a single answer;
  - `"neither"`: neither is correct.
- `type: "flag"`: `reported_issue` holds the defect names and the note of a reviewer who suspected a defect in
  the item. Decide:
  - `"defect"`: the item really is broken: garbled rendering, an unintended contradiction, an ambiguity that
    leaves the answer undetermined, or malformed options;
  - `"no_defect"`: the item is sound (hard, unusual or wordy is not a defect).

Write one JSON object per line, one per item:

```json
{"aid": "a001", "verdict": "A", "reason": ""}
```

`reason`: at most 60 words; quote or point to the sentence(s) that decide. Review every item; write your
answers to the file as you go.
