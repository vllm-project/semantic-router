# Data arms v2 — amendment 1: source pins and projection details (2026-09-28)

Committed before any H1/H3/H5/H6/E11 row is built (the G2/G4/G6 generator files built at
`bdc26e48a` are unaffected; they use unchanged generator code and the preregistered seeds).
Rules of `data-arms-v2-prereg-2026-09-28.md` apply except where stated.

## Source pins

- **GermanQuAD:** the upstream archive (`germanquad.s3.amazonaws.com/GermanQuAD.zip`) no
  longer serves a zip (both nodes received different non-zip responses). Pinned instead:
  the Hub's parquet conversion `deepset/germanquad@a2f3a59f0be843fc305d0417d7292ef0b1a66884`
  (`refs/convert/parquet`), `plain_text/train/0000.parquet` SHA-256
  `319334092605ed9ade8a1e6535584debb2e99def474697417a06ad4a455bf0ec` (identical on both nodes).
- **MTOP:** the official zip (SHA-256 `0c086500…ebaec0b`) bundles `LICENSE.txt` = CC BY-SA 4.0: admitted.
- **QuAC** `train_v0.2.json` SHA-256 `ff5cca5a…c05c56a`.
- **Taskmaster:** the first extraction ran twice concurrently; the directory was re-extracted
  from the verified tarball (`c7e47747…`) on both nodes before any build.

## Projection details fixed by the implementation (before build)

1. **TyDi QA C1 twins use the primary task**, not GoldP (GoldP passages are often a single
   sentence): gold passage plus alternating neighbouring candidate passages up to ≥ 6 units or
   3,000 extra characters; answer = the minimal answer (byte offsets decoded from UTF-8).
   Questions with a minimal answer go to removal or relevance by
   `sha256("tydi-alloc:" + question) % 2`; passage-only answers go to relevance.
2. **C1 extra guard:** a pair is also dropped when the removed twin's title contains the answer or
   the answer survives by plain containment for zh/zh-hant/ja/ko/th/ar/bn/te (word-boundary
   matching misses answers glued to CJK/Hangul text). GermanQuAD title/section lines, JSQuAD
   `title [SEP]` prefixes and SQAC file-name titles are moved into (or dropped from) the title
   slot so they cannot be removed as units; DRCD rows carry no title.
3. **KLUE-MRC and A6h2 exclude v1 items** by v1 local id, by v1 group key and by identical text.
4. **2Wiki twins** use 2–4 gold + 4 distractor paragraphs (de-duplicated), not all 10.
5. **Taskmaster-2 slot Noul:** false slots are chosen to keep each slot's true−false count within
   ±1 (uniform picks let the slot name predict the label: state-removed 78.5% on a replicated
   sample); conversations that cannot satisfy this are dropped.
6. **SciTail:** entails/neutral balanced within each hypothesis; groups join rows sharing a
   premise or a hypothesis.
7. **C10 families cap inside the builder** (balance → whole-group cap → balance) so the 1.2×
   level balance survives the cap; the orchestrator cap is then a no-op. Ties at scale ends empty
   some high-L cells; they are dropped as preregistered (G6 supplies every L = 2..10).
8. **OneStopEnglish:** all three levels are folded to ASCII because the intermediate files lost
   every non-ASCII character upstream (a label cue otherwise).
9. **MTOP** balance over intents present per (language, domain) at 1.2× the rarest; single-intent
   domains dropped.
10. **Generator arms** G2/G4/G6: SHO is moved out of the generator TRAIN file by
    `v2.data.m2.reslice` (rule unchanged).
