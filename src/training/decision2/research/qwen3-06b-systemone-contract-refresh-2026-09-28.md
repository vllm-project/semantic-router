# 0.6B private package: System One runtime refresh

**Scope:** package loader only. The selected 466-update model, frozen CAL,
JevArena v3 predictions, JevBench predictions, product card and public figures
were not changed. This is not a new capability score or a re-selection of the
weak 0.6B model. The model repository and Decision 2.0 collection remained
private throughout this update.

The previous package had an older native API loader. It handled string typed
questions, but it did not fully support the [System One request and answer
shapes](https://docs.typesafe.ai/api): structured instructions and criteria,
Choice `null` descriptions, Choice/Score `confidence`, and Score `legend`.
The source implementation and CPU contract tests had already been corrected;
the private package had not yet received those two loader files.

The signed [runtime refresh tool](../publication/refresh_full_contract.py)
verified the previous bundle's complete manifest, proved the benchmark
functions `load_prompts`, `normalized_answer`, `prompt_input_sha256`,
`checkpoint_fingerprint`, and `run_prompts` have identical ASTs in the old and
new inference source, copied the package without rewriting weights, and
replaced only `decision2/infer.py`, `decision2/api.py`, and
`MODEL_MANIFEST.json`. The old and new full-checkpoint fingerprint is
`5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab`.
The old manifest SHA-256 was `9b939b91a6108037141c86ab5c0f27eca78a1493a00f2fb38071463274063fc6`;
the new one is `08015edfc308572863bc50326833aef4edb2f93cf4851cba837e91f169dd4290`.

The old and new adapters converted the same gold-free typed FINAL 1,600 items
and CSS15 6,547 items, **8,547 answer slots**, to identical native rows:
combined canonical row SHA-256
`044297d88e09c8cd5cb05a5eb1d67149cb7dd3a5551dd2b3cd9423bd24dad08e`.
On one fixed 32-item typed native GPU sample, the old bundle, refreshed stage,
and exact downloaded revision all produced the same answer SHA-256
`e907f210d6dd3f4d6be84f7da3ca569203c57411fcfa9208df3eb7e86b94a367`.
The new package also passed a real GPU request with structured state,
structured Choice instructions, `null`/object option descriptions, optional
Noul criteria and structured three-level Score descriptions. Choice/Score
probabilities normalized; Score's value equaled its probability-weighted
zero-based level index. `verify_bundle` passed before and after Hub readback.

The private Hugging Face exact revision after this three-file upload is
`2e83b687fa8d22eb8c6765277682e3c1e798d6a1`; authenticated metadata
reported `private=true` and 30 files. The downloaded loader, root
`config.json` and model head matched the staged hashes. The root
`config.json` is retained because the [Hugging Face default download counter](https://huggingface.co/docs/hub/models-download-stats)
uses it as a query file. The updated loader returns an entropy-based
confidence **defined by this product**, without claiming numerical identity
to TypeSafe's undisclosed confidence calculation.

This package correction leaves the previously measured v3 tradeoff intact:
the 0.6B point estimate above Kai 1.0 has an interval crossing zero and
Choice/Score declined materially. No decision-model SOTA, Pareto or broad
multilingual claim follows from the runtime refresh. A stronger 0.6B
checkpoint still needs new data/architecture evidence and independent
confirmation.
