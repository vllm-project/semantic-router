# Score TRAIN v6: mechanical QA and blind handoff

**Status: HOLD for independent gold-blind editorial review. No training,
model evaluation, protected-label opening or HF upload.** This record
follows the signed prospective v6 method, SHA-256
`e82e92826cf3ae106579b48612b9b435a581662a5b06be78119e73089138ac34`.
The blocked v1–v3 candidates and negative v4/v5 abstract receipts remain
unchanged. The v6 abstract-only screen passed before any text was emitted.

## Source and corpus binding

- The exact text builder SHA-256 is
  `3b16924a3eb7f956ba2c68eb19c4ac1d8874caec07ca68b6c67a6dbd7fc91d85`.
  The independent abstract source is
  `e4e7f758c7fe5a49155b5e91ad1d66f8a2f64ae66abc797bfc7a72e84c31077f`.
- The frozen parent TRAIN, SELECT, CAL and rights-manifest hashes are
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`,
  and `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
  Parent rights are carried forward in the private manifest; v6 adds only
  internally generated TRAIN text. It is not approved for redistribution.
- Protected inventory: 22 gold-free roles, SHA-256
  `21f7249fac3e31518221772ab10a66991549b1f05d95c7881cc476e0a00df2ce`.
  It includes typed DEV, CSS, public benchmark prompts, pressure diagnostics,
  authored v7–v12 rosters, multilingual r6/r7, and Score v1–v3 gold-free
  rosters. The v12 original and variant prompt files were screened by code
  without reading targets, proofs, private joins or reviewer decisions.
- Private candidate manifest SHA-256
  `1217eb9a9003615562715ba11f9164c0e3cf09277433169d8b92ef3d741fe736`;
  candidate TRAIN JSONL SHA-256
  `6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54`.
  The preliminary context-only receipt is preserved separately; the final
  receipt includes complete-prompt near-overlap and uses a new output path.

## Frozen mechanical results

| Check | Result |
| --- | ---: |
| Rows and complete 0/1/2 groups before quarantine | 972 / 324 |
| Same-level cross-group near clones quarantined | 1 whole evidence group |
| Candidate TRAIN rows and groups | 969 / 323 |
| Family groups: evidence / obligation / route / streak | 80 / 81 / 81 / 81 |
| Label rows 0 / 1 / 2 | 323 / 323 / 323 |
| English / Chinese rows | 729 / 240 |
| Protected context and full-prompt exact/near overlaps | 0 across 22 roles |
| Maximum native token length / cap | 348 / 1,024 |
| Evidence shallow best group-held-out | 98 / 240, cap 160 / 240 |
| Evidence shallow perfect triplets | 0 |
| Control count-only correct per family | 81 / 243 |
| Independent oracle / source necessity failures | 0 |

Four evidence formats each remain in 40 admitted groups. Both sources have
two distinct current active attestations and are necessary to determine the
intersection. Complete triplets hold the universe, rule, source identities,
format and answer vocabulary fixed. The code tests independently count the
intersection, enumerate source alternatives, enumerate simple routes,
resolve latest core events and compute calendar runs. Seven focused tests
passed in a pinned CPU container. No GPU was used.

## Blind packet and limits

The new HMAC-opaque reviewer packet contains 48 complete groups and 144
rows, stratified at 9 English and 3 Chinese groups per family. It contains
only aliases, family, language, state, instructions and options; the salt
and source-ID join reside in a separate private directory. Packet SHA-256:
`7c0d65862598f0e90e4f7e24ff296a4c756967af578772756e0c2694fd61a21a`;
manifest SHA-256:
`957e79084c31e17883c6589f245fc811c73f2fd003f60660487e62ef73c019fd`;
packet builder SHA-256:
`0887a2a81eaf50a76d04452980b707a8373f9bad46c482a5f3b3b2d5fd2281ee`.
The private join remains sealed until an independent review is sealed.

This is mechanical admission only. A previous blocked DEV variant has
abstract evidence-join similarity, so that item cannot provide independent
evidence for this curriculum even though exact and bounded near-overlap
find zero matches. Approximate text matching cannot prove semantic
independence. The blind reviewer must assess naturalness, ambiguity,
source necessity and shortcuts without access to source labels or joins.
Any material finding blocks v6; a repair requires a new prospective
version. Even a clean review does not authorize training without the
separate Score SELECT and matched-control gate.
