# JevArena authored v13: frozen DEV editorial pilot

**Status: FROZEN_DEV_EDITORIAL_AWAITING_INDEPENDENT_BLIND_REVIEWS.** This is a
small source-quality pilot under the [prospective v13 method](jev-arena-authored-v13-prereg-2026-09-27.md).
It is neither a release benchmark nor a training source. No model inference,
protected FINAL access or publication is part of this candidate.
The later blind and post-key outcome is recorded separately in
[the v13 post-key audit](jev-arena-authored-v13-postkey-2026-09-27.md); this
freeze-time status is retained for chronology.

The signed builder source commit is `3fc789943284fdd82863a28a9911719744bba9f2`.
The private candidate froze at **2026-09-26 22:43:29.923926 UTC**. The
prompt-only overlap and tokenizer audit completed before the original packet
handoff; its separate preflight seal was written at **2026-09-26
22:44:42.127250 UTC**. At that time, the substitution packet had not been
handed out. The
preflight seal has SHA-256
`6279a7fc3cf578be814f172c902644a972ced7eeaaa8f6c39c07d1539abf4185`.
The blinded packets contain only an opaque ID, state, native question and
criteria. Their IDs are independently salted and disjoint; neither packet
contains a target, proof, source-intervention label or join.

| Frozen component | SHA-256 |
| --- | --- |
| Prospective method | `08ee7e97491081bd05dff89b3492f9ff3791aececfc2e955d97938008d346149` |
| Builder/oracle source | `fb0d051849e6c338e5abebd5919dea985837f74091abf2f350846a5858c9c734` |
| Native scorer source | `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc` |
| Private casebook commitment | `70f0c70b9866861cc36db5a64404f5fcd0827a3760aeebd04d4e02c6b6189d78` |
| Original reviewer packet / manifest | `6b76e22cdbf5fbe8252c0acecbfa9bb961ce403197254a91da5eb105011dbe91` / `e3d0151e4e2c02175aa381033479d20f20f6632c45630058bebcbf27fafd9f6d` |
| Substitution reviewer packet / manifest | `777ece895641d38f05c2e0dd3b305ce5aa463491789fc70f391aa980c7072c63` / `0c3f351c5f2f9df54c453135156f144e3241fbf9f3a9f1e1833dcc88b93197e0` |
| Prompt-only overlap and token audit | `bc1b916f9f668ba533d0f9bcbd257d870281db05196bf5232ebedb65a542bab3` |
| Frozen tokenizer JSON | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |

There are **12 independent originals**, balanced at four each for Choice,
Noul and Score, plus **12 paired complete-source substitutions**. The paired
views are robustness checks and do not increase the independent question
count. Twelve distinct semantic operations span ten evidence presentation
forms. Original Choice answers occupy positions 1–4 once each. The generic
oracle derives typed answers from private structured facts; hand-entered
targets are not used. It verifies an answer change under each full-source
substitution and two alternative-completion witnesses for each source in every
original. It also checks both sources remain decision-bearing in each
substituted view. Noul answers are Boolean, and Score answers are within the
declared ordered levels 0–2. The prompt, rule and native criteria remain
invariant across each pair.

The prompt-only screen covered **110 rosters and 159,030 rows**, including
available TRAIN, calibration, DEV, multilingual, authored and exposed public
prompt rosters. It found no normalized exact match, no three-gram Jaccard
match at or above 0.7, and no shared eight-word span with the candidate.
The maximum roster three-gram Jaccard was 0.011834; the maximum among
different originals was 0.01938. The complete native prompts, including
criteria, consumed **202–279 tokens** under the pinned Qwen3.5 0.8B Base
tokenizer JSON, leaving a 64-token reserve inside the 1,024-token envelope.
The screen accessed prompt fields only; it did not inspect reference labels.

The v12 failures motivated full, native-valid source substitutions here.
No Noul or Score answer depends on a deleted fact or an unstated abstention
label, and no later correction supersedes an earlier full source. These are
mechanical checks, not editorial approval. Text similarity does not establish
semantic independence. Independent blind reviewers must still judge direct
answers, source necessity, plausibility, ambiguity and shortcuts, then seal
their row-level judgments before the private key is opened. A single material
failure holds the entire frozen pilot. A successful review would permit only
planning a separately frozen release pool; it would not make these 12 cases
release evidence.
