# MASSIVE v5 stable-intent Choice pilot: frozen QA and blinded result

**Decision: HOLD_FOR_TRAINING.** The pilot is a narrow four-intent multilingual
Choice data-quality test. It does not provide Noul or Score targets, prove
cross-task transfer, or approve a 1,155-group corpus. The source candidate,
blind packets and prior v1–v4 failures remain immutable. No model inference,
optimizer step, held-out FINAL access or HF upload was performed.

## Prospective sequence and mechanical freeze

The method was signed at `e8eb75951`, blind row/group join separation at
`ef8a64bde`, and a strictly execution-equivalent rank-prefix search at
`3dd7acae2`. Builder commits are `d9a962dc2` and `e1feb4195`; the
independent freeze verifier is `4e712afcc`. The original all-pool audit was
stopped after about six CPU minutes with **no emitted corpus or packet**.
The rank-prefix run used a new private path and did not change the frozen
selection order, overlap rule or four intent classes.

The official MASSIVE 1.1 archive SHA-256 is
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
MASSIVE/SLURP attribution, CC BY 4.0 LICENSE SHA-256
`c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435`
and NOTICE SHA-256
`b90534ccd20c6f0e1e5239567af0d150496339542b75a15bfbc3e1e737593ddb`
were retained in the private candidate. All selected original utterances
come from official TRAIN and have the predeclared six-locale localization-vote
filter; none shares a source ID with the frozen 600-group v1 candidate.

The protected gold-free inventory SHA-256 is
`21f7249fac3e31518221772ab10a66991549b1f05d95c7881cc476e0a00df2ce`.
The four intents had 1,155 qualifying groups after old-source exclusion.
The builder screened a hash-ranked prefix of 24 per intent, 96 source groups.
It found 18 exact and 169 approximate near-matching utterance rows against
the protected inventory and quarantined **71 entire source groups**. These
row-hit counts can overlap. Approximate SimHash retrieval is not a proof that
all semantic paraphrases were removed. Twelve complete, non-overlapping
seven-locale groups were frozen: three groups per intent, **84 rows**, exactly
12 per locale. The private candidate SHA-256 is
`94678169a9c5bc00c2e50785975ab35de365a9e76c7b125e8b71e7e784b3dd23`;
manifest SHA-256 is
`a73d7a2a5d6f735af451801726bf8b47b67c352de38a83e461157632691c2a39`.
The frozen gold-free local packet SHA-256 is
`abd893f5dbde658d8a8f4602439b8ed34ec19adfcf650f18afca1ece093dee85`;
the separate parallel-comparison packet SHA-256 is
`7fc9f976039d65ca58812233cdc2229dc68c0032d2f20308cf582d4fe43e3d2b`.
The candidate remains `training_approved=false`.

An independently signed verifier reloaded the official source and rechecked
every candidate utterance, source partition, intent, option key, gold-free
packet join and rights receipt. It verified 12 seven-locale source groups,
84 unique opaque review IDs, private key separation, 0700 directory and 0600
file permissions. Longest utterance is 98 characters. The v5 source-ID set
was separately compared to the pinned exposed public multilingual DEV
diagnostic's 179 Chinese and 290 Russian MASSIVE source IDs: **zero overlap**
in each language. The private source-ID audit receipt SHA-256 is
`b1d2a744595541692074992dafbe1eb863af16d236cf7aab26c2b6229eeb7d72`.
That check is source-ID isolation only, not proof against paraphrase or
pretraining contamination.

## Blind editorial gate

The independent reviewer received only a reviewer-specific local packet,
without the parallel join, source IDs or key. Its manifest SHA-256 is
`e0aefd6cb4e1ae0cbfde9c1e7f9d2943a6a2dd88afda24e65549f4e2fa4e433c`.
Stage 1 judgments were sealed before Stage 2 was made available: 84/84 row
judgments SHA-256
`1bc0159c85b28de6efc46df4c57d2913aec17da20aa94c771662f40d613ba770`,
receipt SHA-256
`1dd13ad436d4f8ec2609384ca3d0740f0f628411e92cb34577d4a8d7a08e00f3`,
both timestamped 2026-09-26 23:13:14 UTC. Only **41/84** rows have a unique,
precise four-option fit; **43/84** have no exact option. Naturalness is clear
in 47, awkward in 34 and severely distorted in three. Key agreement cannot
rescue a source utterance that fails the exact-option or semantic criterion.
The independent AI reviewer explicitly marked all locale qualification as
HOLD pending qualified native/bilingual human judgment. The predeclared
84/84 exact-fit admission gate therefore fails before any key opening.

Stage 2 was opened only after the Stage 1 seal. The independent reviewer
examined 72 localized source pairs and all 12 source groups without gold. Its
locale judgments SHA-256 is
`5aec6d0779a934c2e80b4f8939f7677d6e0b4fabe62a6cc14cc7c55987d165a3`,
group judgments SHA-256 is
`782e28ccb97b6cfce3a3c1f561cb5a1386d280c20f185b5a4d86a8d6ffc23918`,
and receipt SHA-256 is
`9a29e46489b1d1d0380a8133a64219c446ee713b459309a0bb0f015291706f38`.
The seal completed at 2026-09-26 23:16:50 UTC, before the private key was
opened. Under the predeclared whole-group semantic and naturalness gate, just
**1/12** groups passed and **11/12** were quarantined. Across 72 localized
utterances, the reviewer found 26 exact, 26 minor-drift and **20 material-drift**
translations. Its failure notes cited entity or artist substitutions, changed
temperature numbers, a mistranslated weather phenomenon, a time anomaly and
unnatural wording. These are blinded editorial findings, not model scores.

The signed seal-first post-key script at `9dcb925b2` checked every frozen
candidate, packet, review and receipt SHA-256, receipt references, and the
Stage 1 seal → Stage 2 packet → Stage 2 seal chronology before opening the
private join. Its aggregate receipt SHA-256 is
`7833a3f9f9bb74b60bf149a8252230e72dce5506041db2e553e9915dc19f8955`.
The independent answer agreed with the official intent key for all **41/41**
exact-option-fit rows, but the closest key also agreed for **35/43** rows with
*no* exact option. Thus apparent nearest-key agreement would substantially
overstate valid supervision. Five source groups had seven exact option fits;
only **one** also passed the parallel semantic and naturalness gate. Exact
fit by intent was 14/21 alarm, 13/21 weather, 7/21 music and 7/21 joke;
by locale it was 5/12 Arabic and 6/12 for each other locale.

The data stay **HOLD_FOR_TRAINING** across every locale. An AI editorial
review does not meet the separately preregistered native/bilingual reviewer
qualification. No failed group will be repaired or refilled under v5. The
failure is substantive: official local utterances plus a stable short intent
codebook did not make the precise Choice label or cross-locale equivalence
reliable for this pilot. Source-ID isolation and rights verification remain
valid but cannot override those label-quality failures.
