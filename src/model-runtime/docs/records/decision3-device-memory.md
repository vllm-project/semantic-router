# Decision 3.0 device memory in Engine mode (MI325X)

Placement estimates the device memory a model needs before any weights load
(design section 10.1). This record holds the peaks the estimate for Decision
3.0 is matched to.

## Setup

- **Command:** `vllm-sr serve -e vllm-sr/<model>` on a clean host with one
  AMD Instinct MI325X (256 GiB), the default engine arguments of the time
  (each request alone), published development
  CLI 0.5.0.dev20261011191319 and `vllm-sr-rocm:latest` (digest
  `sha256:0650ff206bbd92528cc510dd9af095d583d736d1ca7d4f89ba541af22ea89f81`).
- **Reading:** the device's `mem_info_vram_used` (sysfs) about every 10 ms, minus
  the idle baseline (0.28 GiB) measured before `serve`; one model at a time, the
  stack stopped and its containers and volumes removed in between.
- **Requests, in order:** the model card's text (213 input tokens), image
  (chart, 511) and video (5 s clip, 1,039) requests; a 3.5 MB photo (1,311); then
  larger inputs: about 8.5k and 34k text tokens, an image at the
  1,638,400-pixel cap, a video at the 32-frame and per-frame pixel caps, and
  eight questions over 4k tokens. PyTorch keeps the memory it reserved, so each
  column is the peak up to and including that request.

## Results

GiB above the idle baseline; weights are the loaded parameters in BF16.

| Model | Parameters | BF16 weights | Load | Load + card requests | + 3.5 MB photo | + 8.5k tokens | + 34k tokens | Previous estimate | Estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d3 | 26,086,635,760 | 48.59 | 50.03 | 50.28 | 50.46 | 52.71 | 62.11 | 99.18 | 55.45 |
| d3-flash | 8,393,739,504 | 15.63 | 17.05 | 17.21 | 17.36 | 18.98 | 25.56 | 33.27 | 19.20 |
| d3-mini | 4,539,918,336 | 8.46 | 9.93 | 10.09 | 10.38 | 11.79 | 17.58 | 18.91 | 11.30 |
| d3-nano | 2,213,763,904 | 4.12 | 5.53 | 5.74 | 6.03 | 6.68 | 9.72 | 10.25 | 6.54 |
| d3-lite | 853,247,040 | 1.59 | 3.00 | 3.13 | 3.40 | 3.88 | 6.17 | 5.18 | 3.75 |
| d3-edge | 594,582,720 | 1.11 | 2.47 | 2.61 | 2.87 | 3.36 | 5.64 | 4.21 | 3.22 |

Every request answered HTTP 200. The larger image, video and eight-question
requests stayed below the 34k-token peak (d3-lite and d3-edge: 6.43 and 5.91
GiB after the video).

## Estimate

Above its BF16 weights, every model peaked at 1.4 to 1.5 GiB after loading and
at 1.5 to 1.7 GiB with the card's requests, nearly independent of its size.
The previous estimate counted every parameter in FP32 plus 2 GiB. The decision3
family holds its backbone in BF16 on GPUs (`gpu_weights`), so the estimate is
now two bytes a parameter, plus 10% of that (`RESIDENT_HEADROOM`), plus the
2 GiB activation reserve. That is at least 10% and 0.6 GiB above each model's
load and card-request peak, and covers the 3.5 MB photo with at least 0.35 GiB
to spare. Placement compares it with the free memory the worker reads after
initializing the device, so the device context, which the peaks above include,
adds to that margin.

Long inputs need more: about 8.5k tokens exceed the estimate of d3-mini,
d3-nano, d3-lite and d3-edge by 0.1 to 0.5 GiB, and 34k tokens need 3.0 to 11.8
GiB more than the card's requests. A forward that runs out of device memory
fails only its own requests (section 10.2); a GPU shared with other workloads
needs that headroom free for long inputs. Families without `gpu_weights` keep
the FP32 bound.
