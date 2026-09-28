# Sol 2B: human rule-state source feasibility, prospective CPU screen

**Status: source screen only.** The policy-conflict v1 and v2 generators remain
HOLD after their frozen state-removed shortcut tests. This is a distinct
human-annotated source hypothesis; it does not admit training rows, change a
release threshold, or authorize a formal evaluation.

The [ShARC publisher's data description](https://sharc-data.github.io/data.html)
defines `tree_id` as a common rule snippet and user question with varying
scenario and follow-up history. Its input is exactly snippet, question,
scenario and history; `evidence` and `answer` are targets, not input. The
publisher marks the dataset CC BY-SA 3.0. The [original paper](https://aclanthology.org/D18-1233/)
describes rule-based conversational machine reading, a plausible source of
real policy-state contrast for System One. This is a hypothesis, not evidence
that it improves Choice or transfer.

## Fixed cheap screen

1. Use only the publisher's `sharc1-official.zip` TRAIN JSON, pin its byte
   SHA-256 and enumerate exact archive members. Never read DEV/TEST answers
   for source selection. Reject malformed rows; report their count rather
   than silently interpreting them.
2. Count original `tree_id`, `source_url`, exact snippet hashes, Yes/No and
   other answer classes. For a possible native Choice pair, require two
   distinct TRAIN utterances in one tree with exact `Yes` and `No` targets,
   different visible scenario/history, identical snippet and question, and
   nonempty visible rule text. Pick at most one pair per tree by a fixed hash
   of the publisher utterance IDs, without looking at the text or model score.
   Report pair and source counts and visible-input length distribution. This
   first screen creates no model inputs or training package.
3. Continue only if there are at least 100 such independent trees, 50 source
   URLs and 100 distinct exact snippets. Otherwise retain the negative count
   receipt and stop. These are feasibility floors, not training admission.
4. Before any training arm, separately freeze native rendering, pair-order
   counterbalancing, a 24-pair answer-blind semantic review with at least
   18/24 correct and evidence-necessary judgments, exact/near/semantic
   overlap against TRAIN/SELECT/CAL and protected panels at the *source URL,
   tree and snippet* level, rights/attribution treatment, a state-removed
   shortcut check, a matched token roster, and zero-step source parity.
   Any failure is HOLD, with no GPU training or release score.

The intended target would be a Choice between two fully visible policy
situations under one human-written rule, with reversed option order inside
the same group. It must never use an upstream follow-up question as a binary
label, include `evidence` in the input, or treat the two option orders as
independent examples. Native Noul/Score mappings require separate rubrics and
are outside this source screen. Even a passing source audit would not establish
an improvement; it would only justify a prospectively matched Sol1 training
contrast and later independent transfer evaluation.
