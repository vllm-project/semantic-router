# Eikos clean-v2 4B PyTorch reference repeatability result

This is the result of the prospective, gold-free runtime arm in
`eikos4b-cleanv2-torch-reference-repeat-prereg-2026-09-27.md`. The single
treatment replaced the installed Qwen3.5 FLA chunk gated-delta function with
its Transformers PyTorch reference before loading the same frozen package.
The installed wrapper's active FLA callable was checked before replacement;
both the prior and selected backend are recorded in each native manifest.

The package `SHA256SUMS` SHA-256 was
`7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`,
calibration SHA-256 was
`6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`,
full CSS pilot prompt SHA-256 was
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`,
and the new collector SHA-256 was
`9fcbdbbefe8767030d0020cc29bf0150d5b4f1cba348523c8b452beee2f08487`.
The runtime retained the preregistered PyTorch, HIP, Transformers and FLA
versions, BF16, deterministic-algorithm setting, and physical GPU across
processes. The private execution receipt records the runtime image digest and
GPU identity.

The fixed 32-item length-stratified preflight finished in 26 seconds, below
the 300-second stop limit. All 32 outputs were valid, with no missing or newly
initialized weight warnings. Its prompt and selected-ID hashes matched the
prospective note. Its prediction SHA-256 was
`60722661c31e61d1d69e9c434064fbaa79ecbb17715955c7b2787f785a8ecd34`.

Exactly two fresh full processes then ran sequentially on that GPU. Each
produced 1,430 valid answers in original input order, with input digests and
token counts matching each other and the earlier gold-free FLA run. Neither
reported missing or newly initialized weights. Their prediction SHA-256 values
were, in run order:

1. `a0fdd38b36868774a4ffc2916caefc8a96cde3b826bb7e6d3c4594daf83be2f9`
2. `8a19b3dacea5f5ba7b7c9765e8272d77f9095bc3ba4ef9620583042d013abc85`

The gold-free auditor found **zero categorical changes and exactly zero maximum
option-probability drift**, passing the preregistered `0` and `1e-6` limits.
The comparator report SHA-256 is
`f0766c4fbc4937c532d604fca394833b30133bd7324d631df6e1705d933a9d74`;
the private execution receipt SHA-256 is
`58751dffcc3af05c83baffffa29034c14a9c48cbf1955a4745c8f1fbf74b6140`.
Across preflight and both full processes, wall GPU time was **0.059722 hours**;
summed synchronized inference time was **0.046831 hours**. Both measures count
one GPU and include no training.

This result establishes repeatability for this one package and input under the
selected PyTorch reference runtime. It does not establish accuracy or parity
with the selected source adapter. The original FLA runtime remains a failed
repeatability arm. No pilot labels, sealed FINAL labels, or public scoring were
used in this experiment.
