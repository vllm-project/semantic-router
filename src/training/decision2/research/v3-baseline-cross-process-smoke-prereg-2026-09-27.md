# JevArena v3 baseline native repeatability smoke

Status: prospective, gold-free diagnostic; no model selection, score, or
sealed-answer access. The published Decision 1.0 Nox 4B and independent open
Eikos 4B control each use their pinned Hugging Face revision, their published
native adapter, and released calibration. Exact runtime image IDs, source and
package digests, GPU identity, commands, and output hashes belong in the
owner-private execution receipt and are frozen before GPU execution.

The input is one 32-item gold-free panel: 12 length-stratified Choice rows
drawn from the existing fixed 32-row long-input preflight, plus 10 Noul and
10 Score rows deterministically selected from the already open typed DEV
prompts. `scripts.baseline_repeat_smoke_v3 prepare` verifies both source
digests and writes the selected panel and manifest once in a private 0700
directory. The selected panel SHA-256 and ID digest are recorded before
inference. No target or gold file is opened or copied.

For **each** model, run two fresh independent processes in sequence on the
same physical GPU, with exactly the same model package, native adapter,
calibration, image, prompt bytes, device and inference flags. The output paths
must be distinct and absent. Before each launch, verify that the GPU remains
free of other jobs and that the private reservation is held. Do not use a
different runtime, seed, checkpoint or prompt after seeing the first output.
Separately count parameters from the same native loader object and compare the
integer against the independent safetensors inventory used by v3 baseline
attestation. Preserve that count and its source in the private receipt.

The fixed diagnostic gate is **zero categorical changes** and maximum
per-option probability drift **at most 1e-6**, with all 32 IDs, prompt hashes,
native types, token counts, revision and qualified runtime status matching.
Any invalid answer, input mismatch, unqualified runtime, or changed category
fails. Report maximum and p99 probability drift even on failure. A passing
32-item smoke is only a necessary runtime check; it cannot prove full-panel
repeatability or accuracy. If either model fails, notify the v3 owner before
any 8,147-item inference and keep both raw outputs for diagnosis.

Previous clean-v2 Eikos candidate tests do not qualify the **published open
Eikos 4B** control: the original FLA run changed 11 of 1,430 categories,
while a separate PyTorch-reference treatment was repeatable on its own
package. Neither is reused as this published baseline's result.
