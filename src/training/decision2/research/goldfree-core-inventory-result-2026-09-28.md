# Eight-role input-only projection for prospective source admission

**Result: core input projection constructed; training-source admission remains HOLD.** This CPU-only step used the existing hash-pinned protected-prompt candidate and rights-clean v2 TRAIN/SELECT/CAL. It opened no protected answer key, ran no model and used **0 GPU-hours**. The versioned private projected manifest has SHA-256 `26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`; its source inventory SHA-256 is `c6d3f497b4385ff48817e9a6f98f63baf529b99b27a132a190164d5d729c18c2`. The new materializer is source commit `3f7d6c9a6`.

| Required input-only role | Rows |
| --- | ---: |
| typed DEV | 1,600 |
| CSS pilot | 1,430 |
| typed FINAL | 1,600 |
| CSS 15-task transfer | 6,547 |
| public 231-item supplement | 231 |
| rights-clean TRAIN | 7,455 |
| rights-clean SELECT | 700 |
| rights-clean CAL | 700 |
| **Total role records** | **20,263** |

Each native prompt role was verified against its pinned source SHA-256 and
loaded through the actual gold-free inference-input loader. Its full `state`
and named questions were preserved as text; native input digests are bound in
the private manifest. The three partition sources were also SHA-256 pinned,
and every projected input was checked against its source `input_sha256`.
Their targets were never selected into projected files. The projected role
counts, uniqueness, strict input-only schema and file hashes pass. The private
directory and files have `0700` and `0600` modes. The original 27 optional
roles were excluded rather than implicitly treated as source-attested.

This closes the **core file/schema availability** gap in the earlier 27B
prospective audit; it does not retroactively change either failed 27B quota
receipt. No new 27B schedule or teacher mask was admitted. The inventory has
not yet been used to screen a 27B selected schedule or the QuALITY source.
Exact and bounded near checks must be run on each actual candidate, and
source-level semantics, original-corpus reuse and data rights still require
separate review. Excluded optional roles may not be claimed covered.

Reproduction uses `training.data.materialize_goldfree_inventory` and its
synthetic tests. It refuses changed source hashes or incomplete roles and
never overwrites an existing private output. `make check` passed on the
changed paths before the materializer was mirrored and run on the authorized
CPU node. The private file paths, prompt text and partition targets are not
included here.
