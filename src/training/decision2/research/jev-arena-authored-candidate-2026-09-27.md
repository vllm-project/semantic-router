# JevArena authored candidate: sealed-prompt quality audit

This note records a **development candidate**, not a sixth ranked axis or an
official JevBench result. The authored panel is English-only. Existing
synthetic FINAL and CSS15 target labels were not opened for this work.

## Construction and receipts

The signed builder/scorer source is commit `ad10ec0`; its exact module
SHA-256 in both new panel manifests is
`9b602988999395fb778c326048342f40a8e6559c177d2d7f518a9c970665f97b`.
The initial 144-item DEV candidate was rejected after audit found only
Choice branches 0/1, Score levels 0/1, one Noul question polarity, and
implausible actions for complete evidence. It remains as a negative design
artifact. A corrected DEV r2 was built from its separate private seed:

| Artifact | SHA-256 |
| --- | --- |
| DEV r2 prompts | `f654f13c59aaff077c5285ec4026e2e9c35905be5f64e2bc09083e0b2c5c14a3` |
| DEV r2 targets | `d91d2e6c8bb8cb1c2c9ff915fa0491fb3d9d43f55c4f9db538858a7d6a6ac16e` |
| Release candidate prompts | `cc8d0b8f01b79adafeef08dc2da966f70fda8b2ef25fcb18bb8ca23bc5fd5875` |
| Release candidate targets | `c9cd6e2753af74265d4865d5e8a43b8e0a9de0a39273e4d6b27db6ba301509f7` |
| Release candidate manifest | `da341dd5f2fb46a01918ff0ecde44c23720823f2931de803a518a2b7d116afe9` |
| Release automated audit | `eed15ef6880aa2598db6c500b8339f79e50e32504cc2e7d6ed267b2572f5c018` |
| Release template-diversity audit | `77fb84b1979e092f384e7c2733ba190de55293548289db82605c9e75bd7c04ee` |
| Private 20-item QA packet | `788d3930b38dfdb073ccad005935f694d82ea4abec79b775a22b065c80f453fc` |

The release candidate has 1,296 unique prompt/target pairs, one per record
group. It has 432 of each type, 324 of each challenge, and 216 of each
domain. Direct and rendered oracles agree on all 1,296. Exact input/state
overlap and candidate-screened token-5-gram near overlap were zero against
128,716 protected TRAIN/SELECT/CAL and benchmark prompt rows, including
both authored DEV versions. This audit is not a proof against unknown source
data or semantic paraphrases. Its protected-list SHA-256 is
`91f2438cee10330a72ae24441330998e4bc81819f8a866ae0d4bec445244ff56`.
The newly authored target/spec/review/audit files are stored privately with
mode `0600`; only gold-free prompts are eligible for model inference.

The 324 release long-context prompts have 2,958 or more words. Under the
Sol 2B tokenizer, their full encoded requests span 6,840–7,637 tokens
(median 7,150); no item exceeded the 8,192-token adapter limit. The other
challenges had maximum encoded lengths 526 or less. No truncation is allowed.

## Template diversity and ambiguity check

The supplemental audit verified all 1,296 challenge constructions. Long
context target positions range from the beginning to the end of the 193-row
ledger (median normalized position `0.5078`). However, the panel contains
only **three semantic operations**, **12 type/challenge mechanisms**, and
**16 logical-question templates** after Noul polarity. Its six domains are
lexical substitutions of the same policy/question structures; four heading
styles are cosmetic. Only four normalized policy/question signatures remain
after removing IDs and domain vocabulary. Accordingly,
`independent_groups: 1296` means 1,296 distinct record instantiations; it
does **not** mean 1,296 independent scenario designs. Statistical uncertainty
must account for template cells, and even a cell bootstrap cannot remove
synthetic-template bias.

A SHA-selected 20-item private sample covered all 12 type/challenge families
and all six domains. An agent audit found 20/20 mechanically unambiguous
operative rows and answers, matching the dual oracle. It also found a
systematic grammar defect: all eight sampled Score prompts say
`1 verified checks` in the criterion list. Two of five Noul questions use
the awkward negative wording `Is required verified evidence lacking`.
Three sampled long-context items are long repetitive ledgers rather than
natural documents. These are editorial quality failures; an AI agent's
review is **not** a human signoff. The private stratified 144-item review
packet has not been approved by an independent human reviewer.

## DEV-only model diagnostic

The calibrated targeted-to-clean-v2 Sol 2B selected adapter was run natively
on DEV r2. All 144/144 answers were valid, with zero over-budget or
truncated questions. It scored 74/144 (`51.39%` family macro and overall):
Choice 23/48, Noul 32/48, Score 19/48. Insufficient evidence was 13/36;
long context was 19/36. A deterministic 2,000-replicate domain-cell
bootstrap interval for the family macro is `[43.75%, 58.33%]`.
Prediction, companion manifest, and score SHA-256 values are respectively
`1dde71ae3e32aa2d7cb6065eaed32c55a625f4bd1d1789c5ade08e0a9d78f9e5`,
`b051022294caa55fd54418c79eaa5bd4f0cd9631858dee87fcefa2855115b01b`,
and `4fe34e4a74ca4a24fc512d0cb6bb7b8cba0ff6d21d6276e3bb452e2db660f25f`.
This score is a diagnostic of one development model, not a release rank.

## Decision

The authored candidate remains **BLOCKED** as a JevArena v2 axis. The
automated oracle, overlap, and native token checks pass, but template
diversity is narrow, sampled wording needs revision, and genuine editorial
approval is absent. No model has been scored on the release targets. The
current six-axis rank must not silently treat this panel as passed. A future
protocol needs genuinely different source narratives and policy forms,
corrected language, an independently reviewed stratified sample, and a new
sealed release build before this axis becomes eligible.
