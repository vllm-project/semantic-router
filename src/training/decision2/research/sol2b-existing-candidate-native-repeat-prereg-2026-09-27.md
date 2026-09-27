# Sol 2B existing candidate: new native repeatability check

Recorded before this check reads any new prediction or protected label. This
check concerns the already completed, own-Sol-origin targeted-to-clean-v2
BEST320 continuation. It does not resume training, choose a new checkpoint,
or reclassify the earlier stopped replay arm. The previous four-cell
factorial's failure under its original `1e-4` historical-control threshold
remains a failure.

Run the fixed 32 gold-free prompts with SHA-256
`3376ed4093c7efb591519912b88605960923cf33fd19d8adfa5018eb02270a55`
through `training.model.infer`, twice in separate processes, on one isolated
GPU. Use the same pinned container image, exact source mirror, selected
checkpoint, own-lineage source model, CAL report and maximum input length in
both processes. Refuse overwrite and hash the resulting predictions and
manifests. Verify every prompt ID, native answer validity and model identity
before comparison. Limit the entire check to 0.1 GPU-hour; record failed
attempts and consumed GPU time.

**Prospective outcome:** pass only if all 32 answers are valid, categorical
answers never change, and maximum absolute option-probability drift is at most
0.02 across the two independent processes. This threshold is for qualifying a
new common-runtime evaluation path; it does not imply numerical parity with
the historical control, does not change any earlier gate, and does not by
itself qualify the model for release. On a pass, hold the selected weights and
runtime fixed and perform a separate full package parity check before any
formal v3 evaluation. On a fail, preserve both outputs and keep the candidate
on HOLD until a separately frozen deterministic runtime is available.

Following a pass, materialize BEST320 once into a new portable checkpoint.
Compare the selected unmerged and merged checkpoints on the same 32 gold-free
inputs under one runtime and CAL. This package smoke passes only with zero
changed point decisions, zero malformed answers, p99 absolute probability
drift at most 0.005 and maximum drift at most 0.02. If it fails, retain the
failed materialization for analysis and consider a separately specified
unmerged source-plus-adapter package; do not assign historical development
scores to the new package. Complete DEV and CSS-pilot source/package parity
is required before a formal v3 roster.
