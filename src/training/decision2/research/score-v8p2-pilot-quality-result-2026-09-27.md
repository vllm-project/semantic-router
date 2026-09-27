# Score v8.2: CPU-only pilot result and blind-review handoff

**Status: HOLD_OVERLAP, pending independent adjudication.** The signed
[v8.2 preregistration](score-v8p2-pilot-prereg-2026-09-27.md) preceded the
new candidate generation. Signed generator and audit code are at `7fd296571`.
The exact source archive SHA-256 was
`1e5f213df820a1390e1367af453cfc910129b93eb6a8e246482cbcbf6b7cc56e`.
No GPU, optimizer, model output, CAL, or formal benchmark label was used.

The new private 32-byte seed has SHA-256
`ecc19201140d0bd115a68fa8ea4c0d4306c43df2fa7483e58f99b602cc57f1bb`.
The immutable candidate has 15 TRAIN groups/45 rows and 10 SELECT groups/30
rows, with five mechanisms and complete 0/1/2 triplets. All structured and
rendered oracle checks passed. Its manifest SHA-256 is
`cfe6545298a7cb9b011d18c95faed9f6910b1e3db740cb4ef037bae57728abb7`.
Gold-free blind packet SHA-256 values are
`21dbab7ad3b73423c5973acfb3cd92e3ca9134598b138d4a2b96ac51eb824260`
(TRAIN) and
`e9016fafe6fa57339e51e202db33810e2fd6251463dbc2b88b246f87c428bf5b`
(SELECT). Keys and labeled source rows remain separate in the private
candidate directory. **The author has not solved the v8.2 blind packet or
opened its sealed keys.** Another reviewer must seal answers and quality
findings before any key comparison.

The frozen CPU audit receipt SHA-256 is
`e7f0055735a96d8549cf561dd9b2fdc8a47d41f6376ea305c4931cb33f4b23a6`.
Its five evidence-sufficiency triplets now make both the dispatch and meter
single-source projections ambiguous across levels. The five distinct long
dossiers are at least 1,500 characters; normalized cross-group first-variant
similarity ranges from 0.177 to 0.386, below the frozen 0.85 ceiling. The
existing global term-count audit flagged no single-level cue.

The overlap audit checked parent partitions, v7p, v8, v8.1, and 28 gold-free
protected prompt inventories. It found **zero exact input hashes** in the
candidate TRAIN versus its SELECT and the v8/v8.1 TRAIN references, but eight
bounded near-text row hits in four comparisons: one SELECT↔v8 SELECT, one
TRAIN↔v8.2 SELECT, three TRAIN↔v8.1 TRAIN, and three TRAIN↔v8 TRAIN. These
mostly involve inherited numeric-limit and scoped-exception formats; reported
similarities range from 0.942 to 0.995. The preregistered rule treats every
flagged near match as a group-level HOLD pending review. The source and long
document fixes cannot erase that result.

The next reviewer should receive only the two blind packets for item answers,
plus a **separate gold-free near-pair packet** for overlap adjudication. The
eight opaque pairs were produced from the unchanged frozen audit by signed
source `e2b4037d2`, archive SHA-256
`36b16fe80faf209a1663641194bcb74c4100b1d320cd4fca12f3d852019047e2`.
The blind near-pair packet SHA-256 is
`b91645299b8afe4cd221af365e729cf017d76bfd1b8ba40a6472af0241c12003`;
its source-ID mapping is separate and private. The packet includes only the
two rendered inputs per pair in opaque order, with no source IDs or labels.
The author has not adjudicated those pairs. The independent reviewer should
judge all 25 triplets for answerability, source necessity, realism,
and shortcuts without seeing the keys or our v8.1 blind answers. If the near
pairs are genuine same-template situations, a newly versioned pilot should
replace the inherited numeric-limit and scoped-exception renderers before a
matched-budget training experiment. Do not remove only the unfavorable groups
or relabel this v8.2 candidate as clean. BEST368 and 27B first release remain
HOLD pending a validated data intervention and independent same-panel score.
