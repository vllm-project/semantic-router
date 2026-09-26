# Score v6 English matched pilot v1: blocked before optimization

The frozen v1 protocol required a merged full BEST368 start to preserve the
original LoRA's native behavior on 32 gold-free parent English SELECT inputs:
32/32 identical option argmax and maximum absolute option-probability drift
at most `1e-4`. **Native BF16 parity failed. No v1 optimizer step, r2 selector
key access, Chinese-held inference, DEV, transfer, or release inference took
place.** The prepared arm files and all prior model artifacts remain frozen.

The original LoRA inference fingerprint is
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
The existing FP32 materializer produced full-model fingerprint
`59762e359545f3cfa055a38f5f1f63b6c7fb2220a86fd381b05e9e6587e7968b`;
its private source/merged file receipt SHA-256 is
`2675f2cae36c61865622cbadad8f3d2a4fb6af0afde599c64f0e16faac7440aa`.
The 16 Choice plus 16 Noul parity roster SHA-256 is
`193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5`;
its manifest binds every prompt/token hash and is SHA-256
`4070fa2c9f3b421ac84f8eea8bb38ffcee099aedffbec688844cfcc23ec0b861`.

Both source and merged native adapters used the same short inputs, physical
GPU, pinned container image, code, BF16 backbone autocast, FP32 head, SDPA, temperature 1,
one-question batches and no truncation. Both yielded 32/32 valid answers.
Source and merged prediction SHA-256 values are
`8da6756abdc897fbeac696e8ed17dfa7221ca358a0eff82242f059ccdfad5777`
and `f752056eaa6f8fbab35abc4ebb0264f140b2ff2844f1994327f2ed340797bae7`.
The private comparison receipt SHA-256 is
`9cb7b29853349c47f54db91c268034fd1af03e37c86167196f836bc30070ceda`;
it binds model, materializer, adapter, tokenizer, prompt, token, prediction,
software and accelerator identities.

| Native BF16 parity | Choice 16 | Noul 16 | Combined |
| --- | ---: | ---: | ---: |
| Same argmax | 15 | 16 | **31/32** |
| Median maximum option-probability drift | 0.02579 | 0.01283 | — |
| Maximum option-probability drift | 0.15762 | 0.08212 | **0.15762** |
| Rows with drift above `1e-4` | 16 | 16 | **32/32** |
| Rows with drift above `0.01` | 11 | 9 | 20/32 |

The decision head, tokenizer JSON and tokenizer config are byte-identical
between source and merged checkpoints; the head SHA-256 is
`a37b36fc93951b2f401374f8900f03913a19561732d292b366f0c16ed71ce9e3`.
Both loader paths call `model.float().to(device).eval()` and the common native
forward uses `use_cache=False`; LoRA dropout is disabled. The different model
wrappers are the source PEFT adapter versus the merged direct text backbone.
There is no evidence here of an input, tokenizer, head, dropout or cache
identity error.

The mechanism check sampled one first-layer LoRA projection. Its merged FP32
tensor equals the base tensor plus scaled `B @ A` exactly at the stored
precision. The median absolute LoRA delta is `8.27e-6`, while the median
upward BF16 unit in the last place is `6.10e-5`; casting the merged tensor to
BF16 makes 84.06% of its elements equal to the unmodified base tensor. In a
separate read-only FP32 forward on the BF16 flip row and largest-drift row,
the source and merged argmax agree 2/2 and maximum probability drifts are
`9.24e-7` and `2.32e-6`. Its private receipt SHA-256 is
`986daca48caf6ae57fe790f601f5a8845e07fa44d291e43bd49be4b7c2b165ec`.
Together, these observations strongly support loss of small merged adapter
deltas under BF16 matmul weight casts, amplified by the deep backbone. The
FP32 two-row diagnostic does not establish full-panel FP32 equivalence, and
it does not change the failed native BF16 gate.

The current publication builder accepts only a full merged checkpoint. This
merged 27B model therefore cannot be substituted for the selected LoRA in a
release package or inherit its scores. Any future direct-LoRA continuation
needs a new protocol, an adapter-preserving bundle/runtime that pins the
immutable base revision and every source-file hash, native scored-output
parity of the final package, and a parameter count including the full base.
The r2 English selector remains sealed and unused by v1.
