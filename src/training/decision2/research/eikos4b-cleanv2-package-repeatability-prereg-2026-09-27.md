# Eikos clean-v2 4B package repeatability before sealed evaluation

**Prospective gold-free diagnostic.** The selected clean-v2 native SemIf
checkpoint 0232 has already passed full DEV 1,600 and CSS pilot 1,430
same-process LoRA-to-merged-package parity with zero category or probability
drift. The predecessor clean-v1 package showed cross-process nondeterminism
under an unattested runtime; its FLA 0.5.2 deterministic configuration later
passed a two-process CSS pilot repeat test. Verify that the current clean-v2
package has the same repeatability before a first-release freeze.

Freeze package `SHA256SUMS` SHA-256
`7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`,
embedded CAL SHA-256
`6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`,
gold-free CSS pilot prompt SHA-256
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`,
and published native collector SHA-256
`607b0fa49dc92be24135caa48517c9a49b9c71a9b9122fb10e40005f88fb5346`.
Use the existing PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`,
Transformers `5.17.0`, FLA `0.5.2` runtime. Bind its image digest and actual
runtime fields in the private receipts. Do not modify the model, CAL, prompts,
collector, option order, or execution mode.

Run **exactly two new independent processes**, sequentially on the same
physical GPU, with `--deterministic-algorithms` and the complete 1,430-item
gold-free pilot. Use distinct absent output paths. The existing
`training.eikos.deterministic_repeat` auditor must check manifest, package,
input and collector hashes and require zero categorical differences and
maximum option-probability drift at most `1e-6`. Preserve both predictions,
manifests, comparator report, runtime and elapsed GPU time. On failure,
report the mismatch and keep the candidate on runtime HOLD; do not change
the threshold or search another runtime after seeing results. No pilot gold,
typed FINAL, CSS evaluation labels or model selection are involved.

## Completed gold-free result

Both independent native processes completed all 1,430 CSS pilot prompts with
zero invalid outputs. Their manifests agree on the frozen package, CAL,
collector, prompt bytes and runtime, including FLA 0.5.2 and
`torch_deterministic_algorithms=true`. The first/second prediction SHA-256s
are `b750c2de4f19b35613c8e397aed9ad1af52c54b875b79c6b8bc38818e9ba141f`
and `806caa5dedf644fd175c1ea7274923f85039d080c3d7f866fb45e9ec7cd3f48b`.
The private comparator report SHA-256 is
`6b90d434988f3db05723c404fc9e2b005e449755840101a0509b12c4218c5fa3`.

The preregistered repeat gate **failed**: 11/1,430 categorical decisions
changed, maximum option-probability drift was 0.065913, p99 drift was 0.035414,
and 278 rows had drift above 0.02. Per-request synchronized latency summed
to 78.68 and 79.24 seconds in the two runs, a combined 0.0439 GPU-hour
lower bound that omits model loading and setup. The initial first launch
failed before writing output because an overridden `PYTHONPATH` hid the pinned
FLA installation; the corrected launch retained the same model, data and
thresholds. Both successful prediction files and manifests are preserved.

**Decision:** 4B clean-v2 stays on runtime HOLD. Same-process LoRA/package
parity does not imply cross-process repeatability. Investigate the inference
numeric path without opening FINAL labels or selecting a favorable repeat.
