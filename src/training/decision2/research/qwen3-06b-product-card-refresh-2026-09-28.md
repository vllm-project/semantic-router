# 0.6B private product card refresh

**Scope:** model-card copy and layout only. The official-Qwen-origin 466-update
weights, calibration, native runtime, scored predictions, evaluation appendix,
owl banner and three scored figures did not change. This is not a new model
quality result. The model and Decision 2.0 collection remain private.

The [product card renderer](../publication/product_card.py) now leads with a
short description, measured decisions and an understandable same-panel table,
then explains the three typed decision uses and a runnable local System One
request. The exact direct weight source, paired uncertainty and Choice/Score
regressions remain visible. Detailed hashes and statistical methods stay in
the evaluation appendix. The [frozen report renderer](../publication/full_product.py)
regenerated the content from the same nine pinned scored aggregates; comparing
its old and new 11-file content trees found only `README.md` changed. The
Python example code block is byte-identical to the one already exercised on
the previous exact Hub download.

The [card-only refresh tool](../publication/refresh_full_card.py) checked the
old package's full inventory, the replacement content whitelist, and the
byte identity of every non-card product file. It updated only `README.md` and
`MODEL_MANIFEST.json`; the selected model fingerprint remains
`5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab`.
The old manifest SHA-256 was
`08015edfc308572863bc50326833aef4edb2f93cf4851cba837e91f169dd4290`;
the new manifest SHA-256 is
`bace1528a742acb33b6ed167d1a96027a4b6b9415b414f811fcec38f750da3b9`.
The new README SHA-256 is
`4ad399b16ef8036a45234b8f4eedc4dc1e8d99219c20be3709f2ad21eb51e95c`.

The SSH-hosted HF CLI uploaded exactly those two files to the existing
**private** repository at revision
`ec53a8457c12c97183f3ec7f57ed72426e29cda1`. Authenticated metadata
confirmed `private=true`, 30 repository files and a root `config.json`.
The CLI downloaded that exact revision. Its README and manifest match the
staged hashes, and native `verify_bundle` passed with 597,103,104 parameters
and the unchanged model fingerprint. The model card does not include a Pareto
chart or an unverified SOTA claim. Its displayed JevArena result remains a
post-key same-panel comparison with a paired interval crossing zero.

Validation: focused product-card tests passed; the original three-type code
example was byte-identical to the previously GPU-tested example; the full
training contract check passed after the renderer change. The only new
package check was inventory and readback, since the model and runtime files
were unchanged.
