# DEV2.0-0.6B private product-card layout refresh

**Scope:** README copy and section order only. This is not a new training,
calibration, inference-method or evaluation result. The model repository and
Decision 2.0 collection remain private.

The product renderer at local source commit
`08eeff52b7377c5d0b4c1497c6e1d1bd0cbc7ea0` regenerated the card from
the same nine SHA-pinned JevArena v3, human-transfer and public JevBench score
reports. The revised first screen introduces Choice/Noul/Score and the native
System One example before the measured-results table and charts. The card
keeps the owl/model-name banner, matched rank and model-by-task charts,
official Qwen direct-source statement, public-231 scope, paired interval and
Choice/Score regressions. It adds no Pareto or SOTA claim.

The existing private revision was
`f9cbfb192e89c4cf48d3e1a61c959190ba684df0`. The signed-off refreshed
private revision is **`7ac568e6ce99cdeb7cf423b4984a4f87dba8a204`**.
Authenticated HF CLI metadata returned `private=true` and the same 30-file
inventory. The private Decision 2.0 collection also remained private and
contains exactly this eligible 0.6B model.

| File or identity | Before | After |
| --- | --- | --- |
| `README.md` SHA-256 | `3c1cd53b0ef1f4dc38a561e174b48823b980ef7d227650778fffe5aa50a52a8c` | `97923e166649cc4cb6b8e9486e817700a71ca7bb6fa3017133eb7687e2bb5909` |
| `MODEL_MANIFEST.json` SHA-256 | `137eff7391f568e1a25191b5a6ccd6e470ef246a73b610f22bce26b131cdfa81` | `cc2b5a9707b0adfa8732b24e503db73518e202fd0c0f25272cc65a00d278f73f` |
| Root `config.json` SHA-256 | `74bc2bb167d94243f306e91e69ccdee07ee06c998c06bcb0d46d4cbf4ac157a8` | same |
| Model fingerprint | `5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab` | same |

The card-only refresh checker validated the existing full package and
replacement content whitelist, then changed only `README.md` and its manifest
entry. The model-file hashes, calibration hash, sealed prediction identities,
source revision and loaded parameter count were unchanged. The upload stage
contained exactly the two changed files, and the HF CLI reported a two-file
commit. The root `config.json` continues to satisfy the Hub's default model
download-count query-file layout; the private repository's current count does
not establish future public count volume.

An authenticated HF CLI downloaded the **entire exact new revision**. The
downloaded `README.md`, root `config.json` and `MODEL_MANIFEST.json` matched
the hashes above; native `verify_bundle` passed its complete file inventory
and measured **597,103,104 parameters**. All four README image references
resolved, and the new section order was checked. The Python code block is
byte-identical to the previously exercised example, with SHA-256
`e568ea27f0541c0ee1d1d33f62efcf9097fd09a68f894b91971b25c4d848da6f`.
It was executed again directly from the newly downloaded package on one
otherwise idle GPU: Choice, Noul and Score returned valid typed answers with
no error fields (248 input tokens). The isolated validation container exited
and left no GPU process. The example run occupied under 0.005 GPU-hours.

No formal panel was rerun: identical model, runtime, calibration and prompt
bytes preserve the earlier package-native panel-parity evidence. The v3
comparison remains post-key same-panel and its paired interval crosses zero;
this card change is not an independent ability improvement.
