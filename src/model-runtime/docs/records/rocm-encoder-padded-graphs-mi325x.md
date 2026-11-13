# ROCm: padded encoder graphs on MI325X

`EncoderGraphs.padded` takes an encoder family's own padded `[rows, width]`
batches through the encoder graph buckets: the bucket grid on the masked
layout, replayed as a graph from a bucket's second use. The engine sends padded
batches there on NPUs only. On CUDA and ROCm they keep the eager padded layout
at the live grid, so their answers stay byte-identical to the released runtime
by default. This record checks that default on gfx942 and measures the padded
path forced on there, for a later switch that would enable it on GPUs.

Every Vela 3.0 (`vela3-fast`) forward is a padded batch, so it is the case
measured.

- **Date:** 2026-11-11.
- **Device:** one AMD Instinct MI325X (gfx942).
- **Stack:** ROCm 7.2, PyTorch 2.12.0+git6bbd260, Triton 3.7.0, FLA 0.5.2.
- **Code:** main `1f3604871`; the same main with this change (the default); and
  that tree with the engine's NPU gate opened to CUDA and ROCm (forced on).
- **Model:** `vllm-sr/vela3-fast` at the registry pin `7cbefd4b`, `--dtype auto`.

## Method

`tools/cross_process.py answer vllm-sr/vela3-fast --device rocm:0 --panel ...`
once per tree on the same GPU, then `tools/cross_process.py compare` against
main: each prompt is one request alone, through the scheduler, after the
readiness golden check (`unverified` on ROCm, where `vela3-fast` records CPU
references only; its run-to-run check passed on every tree).

The panel is 240 prompts built from 16 fixed sentences, 1 to 40 sentences per
request, so widths run from the first bucket to past the widest (384 tokens);
every third prompt has `source` and `answer` parts besides `request`. Every
prompt asks `vela3-fast`'s golden questions over the request (`topic`,
`urgent`, `domain`, `factcheck`, `jailbreak`, `pii`), and the three-part ones
also `halu` and the `drugs` span question.

## Results

| | main | this change | padded graphs forced on |
| --- | ---: | ---: | ---: |
| Prompts byte-identical to main | - | 240 / 240 | 92 / 240 |
| Encoder graph captures / replays | 0 / 0 | 0 / 0 | 7 / 139 |
| Time to answer the 240 prompts | 19.2 s | 19.1 s | 18.2 s |

- The default answers every prompt byte-identically to main.
- Forced on, 148 prompts differ only in numbers: 2,337 values, the largest
  absolute difference 4.6e-6 and the median 2.0e-8. No category changes: every
  choice, every Noul's side of 0.5 and every score's winner is the same, and no
  other field differs.
- A padded replay equals its eager bucket bit for bit
  (`tests/test_encoder_graphs.py`, `gpu` marker), so with the padded path on,
  an answer would not depend on whether its bucket was captured yet; on the
  test backbone it differs from the eager padded layout by at most 1e-5.
  `tests/test_decision1_gpu.py` checks that the default engine answers padded
  batches bit for bit as an engine without graphs does, and never replays them.

## Implications

- CUDA and ROCm answers stay byte-identical to the released runtime: padded
  encoder batches are replayed as graphs on NPUs only.
- The padded path makes `vela3-fast` about 5% faster on gfx942 here but moves
  its answers in their last digits, so enabling it on GPUs needs its own switch,
  off by default, and its own accuracy record.
