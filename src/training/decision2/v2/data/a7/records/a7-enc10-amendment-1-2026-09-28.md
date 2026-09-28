# A7 encoder corpora, amendment 1: A7x distractors from the same scenario only (2026-09-28)

Committed before any `a7-enc10-v2` row is built. Version `a7-enc10-v2` = the
rules of `a7-enc10-prereg-2026-09-28.md` with the A7x option rule below; A7q,
A7k and A7s are built by unchanged code and must equal `a7-enc10-v1` row for row
apart from the version string in `audit_metadata.a7` (checked). `a7-enc10-v1` was built once (node B,
commit `7db3b8feb`) and is not published.

## Finding in the v1 build (gate failure)

The single A7x cell (`massive_intent` Choice, 47,987 rows before admission)
failed rule 7c: option-only 0.520 and state-removed 0.524 against a 0.252
majority, so the whole sub-arm was dropped as preregistered. Cause: for the
many MASSIVE scenarios with fewer than four intents, v1 filled the remaining
distractors with intents of other scenarios, so the gold was always one of the
options that share a scenario — readable without the request.

## Rule change (A7x only)

Distractors come only from the gold intent's own scenario. A row offers
K = min(4, number of intents in its scenario) options: the gold and K − 1
distractors in `sha256("massive-distractor:" + id + intent)` order; the gold
position is `sha256("massive-position:" + id) % K`. Scenarios with one intent
stay excluded. Everything else (train partition, 12 locales, per-(locale,
scenario) intent balance at 1.2 × the rarest intent, 4,000 rows per locale in
utterance-id hash order, ablation-only status) is unchanged, and the gates of
the preregistration apply as written.
