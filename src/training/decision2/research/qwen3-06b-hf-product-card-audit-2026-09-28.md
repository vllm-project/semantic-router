# 0.6B private Hub repository and product-card audit

**Scope:** read-only authenticated Hub CLI audit and local copy proposal. The
model repository, weights, figures, runtime and collection were not changed.
The earlier packaged model and scores retain their original meaning.

The authenticated CLI returned `private=true` at exact revision
`ec53a8457c12c97183f3ec7f57ed72426e29cda1` with 30 files. It includes
root `config.json` (SHA-256
`74bc2bb167d94243f306e91e69ccdee07ee06c998c06bcb0d46d4cbf4ac157a8`),
which points to files actually present under `model/`, plus the native
`decision2/` loader, calibration, three figures, model-name owl banner,
license/attributions, and a concise evaluation appendix. There are no
training rows, raw predictions, private logs or temporary checkpoints. The
root config follows the public Decision 1.0 Kai repository's query-file
layout. The [Hub download-count rule](https://huggingface.co/docs/hub/models-download-stats)
uses root `config.json` by default when no library-specific rule applies.
The Hub currently reports zero downloads for this private repository; that
number is **not** evidence of a missing query file or a promise about future
count volume.

The pinned current README SHA-256 is
`4ad399b16ef8036a45234b8f4eedc4dc1e8d99219c20be3709f2ad21eb51e95c`.
All four banner/figure references resolve against its Hub file tree. The
Choice/Noul/Score Python block parses and has SHA-256
`e568ea27f0541c0ee1d1d33f62efcf9097fd09a68f894b91971b25c4d848da6f`.
This matches the earlier exact-revision full-package readback block whose
execution returned all three answer types; no new GPU run was needed to
establish a change that did not occur.

Compared with Kai 1.0, the current card already has product-first sections,
a model-name owl banner, matched rank/matrix figures and a runnable local
System One example. Some first-screen phrases still read like internal audit
prose, especially the answer-key access wording and repeated uncertainty
explanation. The local renderer proposal changes **copy only**: a clearer
one-state/three-decision introduction, more direct same-panel comparison,
plain-language independent-confirmation note and shorter limits. The exact
direct starting weights, source attribution, 8,147/231 denominators, displayed
scores, 95% paired interval, Choice/Score regressions, 8,192-token limitation,
figures and three-type code block are preserved. No Pareto or SOTA claim is
added. The proposed README SHA-256 is
`3c1cd53b0ef1f4dc38a561e174b48823b980ef7d227650778fffe5aa50a52a8c`.
Only the README would differ from the current private package; any eventual
card-only Hub update must recompute its package manifest and repeat exact
readback and relative-link checks first.
