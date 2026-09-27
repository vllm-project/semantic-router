# DEV2.0-0.6B private product-card copy refresh

**Scope:** product copy only. The model remains private. This is not a new
training, inference, calibration or benchmark result. The card is rendered by
`publication.full_product` and `publication.product_card` from the same pinned
scored aggregates and owl banner.

The revised card leads with the native Choice/Noul/Score decision use, keeps
the 8,147-item JevArena and separate 231-item public JevBench same-panel
comparison, and uses the matched rank and model-by-task figures. It names the
official Qwen3-0.6B-Base direct starting weights and retains the paired
interval, Choice/Score regressions, input-length limit and post-key evaluation
qualification. The three-type runnable Python example did not change: its
code-block SHA-256 is
`e568ea27f0541c0ee1d1d33f62efcf9097fd09a68f894b91971b25c4d848da6f`.
No Pareto or SOTA claim was added.

The card-only refresh verified the preceding bundle's complete inventory and
the replacement product-content whitelist, then changed exactly `README.md`
and `MODEL_MANIFEST.json`. The new card and manifest SHA-256 values are
`3c1cd53b0ef1f4dc38a561e174b48823b980ef7d227650778fffe5aa50a52a8c`
and `137eff7391f568e1a25191b5a6ccd6e470ef246a73b610f22bce26b131cdfa81`.
The root `config.json` remains
`74bc2bb167d94243f306e91e69ccdee07ee06c998c06bcb0d46d4cbf4ac157a8`;
the selected model fingerprint remains
`5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab`.

Authenticated HF CLI checked that the repository was private before upload,
uploaded only the two changed files, and downloaded the exact new revision
`f9cbfb192e89c4cf48d3e1a61c959190ba684df0`. The three downloaded
README/manifest/config hashes match the validated staging package. Authenticated
repo listing still reports private. The root `config.json` is present for the
Hub's default download-count query file. Four README image links point to
the unchanged package assets; the runnable code block remains byte-identical
to the previously executed example. Existing full-package native readback and
parity evidence remains bound to the unchanged model/runtime bytes.
