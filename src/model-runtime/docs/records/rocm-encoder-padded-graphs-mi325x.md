# ROCm: padded encoder graphs on MI325X

Padded `[rows, width]` encoder batches take the encoder graph buckets
(`EncoderGraphs.padded`) on every device class that has the graph APIs, not
only on NPUs. On CUDA and ROCm that moves every Vela 3.0 (`vela3-fast`)
forward, which is always a padded batch, from the eager padded layout at the
live grid to the bucket grid on the masked layout, replayed as a graph from a
bucket's second use. This record measures what that changes on gfx942.

- **Date:** 2026-11-11.
- **Device:** one AMD Instinct MI325X (gfx942).
- **Stack:** ROCm 7.2, PyTorch 2.12.0+git6bbd260, Triton 3.7.0, FLA 0.5.2.
- **Code:** main `5aafe5833` against the same main with this change merged.
- **Model:** `vllm-sr/vela3-fast` at the registry pin `7cbefd4b`, `--dtype auto`,
  graphs on (the default).

## Method

`tools/cross_process.py answer vllm-sr/vela3-fast --device rocm:0 --panel ...`
once per side on the same GPU, then `tools/cross_process.py compare`: each
prompt is one request alone, through the scheduler, after the readiness golden
check (`unverified` on ROCm, where `vela3-fast` records CPU references only;
its run-to-run check passed on both sides).

The panel is 240 prompts built from 16 fixed sentences, 1 to 40 sentences per
request, so widths run from the first bucket to past the widest (384 tokens);
every third prompt has `source` and `answer` parts besides `request`. Every
prompt asks `vela3-fast`'s golden questions over the request (`topic`,
`urgent`, `domain`, `factcheck`, `jailbreak`, `pii`), and the three-part ones
also `halu` and the `drugs` span question.

## Results

| | main | this change |
| --- | ---: | ---: |
| Encoder graph captures / replays | 0 / 0 | 7 / 139 |
| Time to answer the 240 prompts | 22.5 s | 19.3 s |

- 92 of 240 prompts answer byte-identically. The other 148 differ only in
  numbers: 2,337 values, the largest absolute difference 4.6e-6 and the median
  2.0e-8.
- No category changes: every choice, every Noul's side of 0.5 and every
  score's winner is the same, and no other field differs.
- A padded replay equals its eager bucket bit for bit
  (`tests/test_encoder_graphs.py`, `gpu` marker, four runs), so an answer does
  not depend on whether its bucket was captured yet. On the test backbone it
  differs from main's eager padded layout by at most 1e-5.

## Implications

- On CUDA and ROCm, padded encoder batches are no longer byte-identical to the
  eager padded layout: answers move in their last digits, well inside the
  golden check's GPU tolerance (0.02), with no decision changed on this panel.
- Each process stays deterministic: the same request gets the same answer
  whether its bucket was replayed or ran eagerly.
