# BF16-resident device memory on CUDA (RTX 4090)

Placement estimates the device memory a model needs before any weights load.
Families with `DtypePolicy.bf16_resident` (Decision 2.0, Vela 2.0 decoders)
hold their BF16-exact Linear weights in BF16 on GPUs and everything else in
FP32. The estimate counts the Linear weights the checkpoint stores in BF16 at
two bytes plus `RESIDENT_HEADROOM`, the rest at four, plus the 2 GiB
activation reserve. This record holds the peaks it is matched to.

## Setup

- **Date:** 2026-10-15.
- **Device:** one NVIDIA GeForce RTX 4090 (sm_89, 24 GB, 22.8 GiB free to
  placement), driver 580.159.04, otherwise idle.
- **Stack:** PyTorch 2.10.0+cu128, Triton 3.6.0, FLA 0.5.2.
- **Command:** `vllm-srun serve vllm-sr/<model> --device cuda:0` with the
  default engine arguments (`max_num_seqs` 32, `max_num_batched_tokens` 4096
  for the under 32 GB device class).
- **Reading:** the device's `memory.used` (`nvidia-smi -lms 20`) minus the idle
  reading taken before `serve`; one model at a time.
- **Requests, each twice, in order:** Lux-9B: seven states of 1 to 31 tokens
  with Choice (3, 60 and 255 options), Score and NoUL questions, then one state
  of 5,846 tokens. Vela-2.0-9B: four states of 6 to 79 tokens with Choice,
  NoUL, Score, Set and PII questions, then one of 114 tokens.

## Results

GiB above the idle reading; weights are the BF16-stored Linear weights in BF16
and the other parameters in FP32. Every request answered HTTP 200.

| Model | Parameters | BF16-stored Linear | Weights | Load | + short requests | + 5,846 tokens | Previous estimate | Estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision-2.0-Lux-9B | 7,940,895,744 | 6,918,504,448 | 16.70 | 17.33 | 19.03 | 19.15 | 31.58 | 19.98 |
| Vela-2.0-9B | 7,949,850,627 | 6,918,504,448 | 16.73 | 18.13 | 18.48 | - | 31.62 | 20.02 |

## Estimate

The estimate is 0.95 GiB (Lux-9B) and 1.54 GiB (Vela-2.0-9B) above the peak
with short requests, so both place on a 24 GB card. The previous estimate
counted every parameter in FP32 and refused them there. As for Decision 3.0
(`decision3-device-memory.md`), much longer inputs can need more than the
estimate.
