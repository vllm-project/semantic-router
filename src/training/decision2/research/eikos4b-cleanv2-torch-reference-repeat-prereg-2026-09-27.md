# Eikos clean-v2 4B PyTorch reference repeatability experiment

Status: prospective, gold-free runtime diagnostic. The frozen clean-v2 package
failed its first independent-process CSS pilot repeatability gate: 11 of 1,430
choices changed and maximum option-probability drift was 0.065913. This arm
tests whether replacing the Qwen3.5 FLA chunk gated-delta operator with the
installed Transformers PyTorch reference makes that same package repeatable.
It does not test accuracy or select a model.

Freeze the standalone package `SHA256SUMS` SHA-256
`7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`,
embedded calibration SHA-256
`6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`,
gold-free CSS pilot prompt SHA-256
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`,
and original published collector SHA-256
`607b0fa49dc92be24135caa48517c9a49b9c71a9b9122fb10e40005f88fb5346`.
Keep PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`, Transformers `5.17.0`,
FLA `0.5.2`, BF16, the package-native one-item-at-a-time SemIf letter readout,
calibration, prompt and option order, and `--deterministic-algorithms` fixed.
Record the exact runtime image digest, GPU identity, new collector source SHA,
package and input hashes, and process start/end times in private receipts.

The sole treatment is an opt-in collector flag. Before model load, inspect the
installed `transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule`
wrapper closure and require that its selected implementation is FLA's
`chunk_gated_delta_rule`. Then bind that module global to its `__wrapped__`
PyTorch reference function. Record both the prior and selected backend in each
manifest. Abort if the installed wrapper is already using PyTorch, the fallback
cannot be verified, or package load reports missing/newly initialized weights.
Leave the collector's default path unchanged.

First run one bounded preflight on a **fixed length-stratified set of 32
prompts**. Rank all 1,430 IDs by the original first process's gold-free
`usage.input_tokens`, then ID to break ties. Select ranks
`floor(i * 1425 / 27)` for `i = 0..27`, plus ranks `1426..1429` (the four
longest). Emit their original prompt JSONL bytes in original input order.
The selected ID-list SHA-256 is
`1865807c6be717d992cab8b83b6c0e53d2e2dc6c8dcd60291ad375d1ee5ecde8`;
the resulting 32-row prompt SHA-256 is
`06b5b13b8a51452ca7f8bae588bb43c7faa6b21d9498d262440041f32c541312`.
These hashes were computed before treatment execution, without labels. The
selected token lengths range from 133 to 5,157. Stop if the preflight has not
finished within **300 seconds of process start**, if any of 32 answers is
invalid, if the requested backend is not attested, or if package loading
reports missing/newly initialized weights. Preserve the preflight receipt and
stop this arm on failure. Do not adjust the limit after seeing it.

Only after that preflight passes, run **exactly two fresh independent full
1,430-item processes**, sequentially on the same physical GPU, with distinct
absent output paths and the same opt-in flag. The gold-free auditor must verify
identical package, calibration, input, new collector, runtime and selected
backend; all 1,430 IDs, input digests and token counts must agree, with 1,430
valid outputs in each process. The numerical repeat gate is **zero categorical
changes and maximum option-probability drift at most `1e-6`**. Compare the two
treatment processes to each other, not to the earlier FLA probabilities.
Preserve both predictions, manifests, comparator report, hashes, wall time and
synchronized inference GPU-hours even on failure. A failed gate keeps clean-v2
on runtime HOLD; do not tune a second runtime using this panel. No training,
sealed FINAL labels, pilot gold, public scoring or publication are involved.
