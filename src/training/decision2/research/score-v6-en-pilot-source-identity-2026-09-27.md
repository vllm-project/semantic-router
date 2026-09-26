# Score v6 EN pilot: read-only source identity receipt

This read-only preflight follows the signed English-only matched pilot method
`score-v6-en-matched-pilot-prereg-2026-09-27.md`. No materialization,
optimizer, model inference, arm data output, r2 key access or held-out score
occurred.

The existing Qwen3.8-27B clean-v2 `BEST.json` selects
`checkpoint-0000368`; the completed source run records 458/458 updates. Its
checkpoint metadata declares `peft-lora/1`, LoRA rank 8, alpha 16, dropout
.05, a 256-dimensional Decision head and upstream revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The fresh
`decision2` trainer initializer requires a full checkpoint: passing this
LoRA checkpoint directly omits the immutable base path needed by the LoRA
loader. No existing materialized full copy was found in the task workspace.

An independent CPU-only rehash read **all 28 source files** listed in the
frozen LoRA source fingerprint and **all seven checkpoint inference files**,
including the adapter and head. Every source file matched its frozen hash.
Recomputing the native adapter's combined file-hash identity yielded exactly
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`,
the previously reported BEST368 inference fingerprint. The authorized
experiment environment had adequate observed CPU memory and disk for a
separate merge, but capacity does not establish successful materialization.

**Optimizer status: blocked.** CPU materialization must write a new immutable
full checkpoint and a receipt. A 32-row, gold-free, native original-versus-
merged prediction parity test must then pass before either matched arm loads
the same byte-pinned merged source. Neither arm should silently initialize
from a fresh Qwen backbone or from a nested LoRA checkpoint.
