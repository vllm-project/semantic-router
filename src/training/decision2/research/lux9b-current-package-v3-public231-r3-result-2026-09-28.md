# Lux 1.0 current-package comparator r3: incomplete

The prelocked r3 control did **not** produce a JevArena v3 or public-231
score. It was a disclosed post-key, same-panel Decision 1.0 comparator, not
a Decision 2.0 candidate or blind test. The private immutable run lock has
SHA-256 `6c0eeb97b53a57dc7c4bfdc6db4ccba8d8fc1ee40b70208f32ee88d274dfe346`.

The native collector completed typed FINAL (1,600 originals and 2,000 answer
slots), then wrote 3,514 of 6,547 CSS15 originals. On the next unmodified
CSS question the model raised its native input-limit error: `label: 18198
tokens exceeds max_length=16384; no truncation allowed`. Public-231 inference
never started. No protected target or score was read, and no joint prediction
seal was created. The partial typed and CSS files are retained privately as
failure evidence; they are ineligible for reuse in a later run.

The total GPU wall time was 277.536 seconds (0.077093 GPU-hours), including
model loads and partial inference. The task container exited and the GPU was
released. The private STOP receipt SHA-256 is
`accb56ca630c6dc8f5868bfef7a3f3d43a19b6317e426071a0112db4451c52d9`.
This is a collector coverage failure caused by an over-budget native input,
not a measured model-quality loss. A new prospective r4 protocol may count
the complete original row as invalid and continue without shortening or
changing it; it must use fresh predictions for all three panels and a
separate lock before any protected target is read.
