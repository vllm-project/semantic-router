# Score TRAIN v5: abstract feasibility HOLD

**Disposition: HOLD_NO_CORPUS.** The v5 method was registered in the unified
research ledger before this abstract screen; method note SHA-256 is
`5619c03d6ac19d8e8a9492d41f383409eee4ee0ede14e651363b000f6973d54a`.
Only integer weight/mark pairings were enumerated. No natural-language v5 row,
gold-free review packet, parent dataset access, tokenizer run, protected-set
overlap, GPU step or HF upload occurred. V1–v3 remain blocked and v4 retains
its negative feasibility receipt.

## Ordered source and receipts

The signed abstract-screen source commit `bf0a52d234df85009ee63265188eb9a5c95a00aa`
has script SHA-256
`b3fd268dac0fe7951c25d430aa743027188bfef94bcde4f63551f8dfde930ba1`.
Its first isolated CPU run emitted aggregate-only receipt SHA-256
`2e31c5ed2b5d9ce6ce2dbce76fbbd07b0fc6da318c42eb2c4359a66d3dedb404`.
That run correctly exposed selected-signal excess but incorrectly assessed
the exactly balanced target-position feature with leave-one-group-out (LOO)
majority voting. Excluding the held-out complete triplet depletes its own
label count and makes an otherwise balanced feature look anti-predictive.
The receipt is retained as a failed audit implementation; it is not used as
the final chance result.

Signed commit `4864522cd143995870145b6202cd2e7f825ff013` changed **only** the
chance computation for fixed table/target-position features to empirical
feature-label majority counts. It did not change the pairing selector,
pool schedule, labels, one-signal LOO classifier, 40% gate or preregistered
method. Corrected script SHA-256 is
`e39450caf155c2b1c0eba49be1d2b7c73a6c30c2809cbbcea0fa6b102ddb3377`;
second aggregate-only receipt SHA-256 is
`3505f979e7f64b9129dd9d345667e96f6ab81a48eb7b8e6ac38a89b293732b90`.
The pinned CPU container digest was
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`;
focused arithmetic and balance checks passed 2/2. Both receipts and exact
source mirrors remain immutable.

## Corrected aggregate finding

All **81/81** planned weighted source groups had an eligible three-tier
pairing from the fixed pools. Minimum integer score spans were 2 in 60
groups, 3 in 16, and 4 in five. Each group had 30–54 eligible triples before
the fixed seeded tie-break. The abstract schedule had 60 English and 21
Chinese groups. Target display position, unweighted selected-mark sum, and
maximum selected individual product each scored exactly **81/243**, and
within-language **60/180** and **21/63**, on their empirical one-field
majority baselines.

The material gate still failed. A selected signal's `(weight, mark)` at one
fixed position scored **115/243 = 47.33%**, while the preregistered ≤40%
threshold permits at most **97/243**. English peaked at **79/180** versus
maximum **72/180**; Chinese peaked at **29/63** versus maximum **25/63**.
Several other single-mark/product positions also exceeded their respective
language gate. A correct table- and target-position invariant therefore
does not neutralize a target-bound additive signal proxy. This is a
specific negative result of the frozen v5 selector, not proof that 40% is
mathematically impossible for all additive-score corpora. Do not retune v5
after observing it. A new family/gate needs a prospective v6 method.

## Separate protected-roster handoff

A signed, source-hash-pinned exporter (`70d38113c0bf89e6b0d9a3b8244edfdfad132de7`,
script SHA-256
`69779196e91365b7733936700fdb45df46c51a28da7fbd286d9acef1eb2fc954`)
projected the previously frozen Score **v3** TRAIN candidate to 960 gold-free
prompt/state records in 320 complete groups. The private output prompt and
manifest SHA-256 are
`247f49b06c110ba5d15ee9e434a8635ca5923d296d830216e70fb4987dc57c49`
and `6ac15052d1a3b862b4f610d0d1bf4dfc71a91c08bb77ef182b4dbe69af577e2d`.
Only HMAC-opaque row/group IDs, family, language, state, instructions and
options were exported. Source IDs, labels, private salt and join were
excluded. A focused privacy test passed 1/1, and the artifact was handed to
the multilingual DEV author solely for conservative TRAIN/DEV overlap checks.
This does not approve v3 for training or imply v5 data exists.
