# CUDA golden answers on one NVIDIA H200

Golden requests for every built-in model the published CUDA image served on
this card. The `cuda` device class is recorded next to `cpu` and `rocm` in
the registry files. Decision 3.0 and Omni were not run.

Smoke of the same image is [`cuda-router-image.md`](cuda-router-image.md).
An earlier A100 80GB run that did not use the published image and could not
load Vega-27B is the local note `cuda-a100-golden.md`.

- **Date:** 2026-10-12.
- **Hardware:** one NVIDIA H200 (sm_90, 143771 MiB total). Driver 575.57.08.
  Probe free memory 139.29 GiB; BF16 supported. Pod `dq8ck6v4xloqff`.
- **Image:** `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest`, digest
  `sha256:2f05113e5c7399874ef2b1651ecb3745a5dafeff7ef2f9725acd01b643c06730`,
  config created `2026-10-12T07:07:54Z`. Inside the image: Debian 12, Python
  3.12.15, torch 2.10.0+cu128, Triton 3.6.0, fla-core 0.5.2. The image has no
  git-revision label.
- **Head:** `61677a464c2a1b94ef34cd9e22595bab87981b97`, the checkout used for
  the same day's model-runtime measurements on this card.
- **Profile:** `exact`, device `cuda:0`, uid 65532.

The CUDA numbers were merged from that published-image run into:

- `vllm_srun/registry/golden_answers.json` (Decision 2.0, six models including Vega)
- `vllm_srun/registry/golden_answers_decision1.json` (seven models)
- `vllm_srun/registry/golden_answers_vela1.json` (thirteen models)
- `vllm_srun/registry/golden_answers_vela2.json` (four models)

## Against the committed CPU and ROCm keys

No label, choice, or other non-numeric field changed. The largest absolute
difference of any numeric golden value is under the 0.02 GPU readiness
tolerance:

| Model | vs CPU | vs ROCm |
| --- | ---: | ---: |
| Decision 2.0 Kai-0.6B | 0.004510 | 0.005309 |
| Decision 2.0 Eos-0.8B | 0.002926 | 0.004285 |
| Decision 2.0 Sol-2B | 0.002442 | 0.004006 |
| Decision 2.0 Nox-4B | 0.003437 | 0.002531 |
| Decision 2.0 Lux-9B | 0.003473 | 0.002309 |
| Decision 2.0 Vega-27B | 0.001918 | 0.003267 |
| Decision 1.0 Kai-0.6B | 0.000001 | 0.000001 |
| Decision 1.0 Lex-0.6B | 0.000001 | 0.000000 |
| Decision 1.0 Route-0.6B | 0.000001 | 0.000001 |
| Decision 1.0 Eos-0.8B | 0.005322 | 0.004770 |
| Decision 1.0 Sol-2B | 0.005973 | 0.005009 |
| Decision 1.0 Nox-4B | 0.003091 | 0.003068 |
| Decision 1.0 Lux-9B | 0.003428 | 0.004986 |
| Vela 2.0 0.3B | 0.000001 | 0.000001 |
| Vela 2.0 0.8B | 0.003815 | 0.004082 |
| Vela 2.0 4B | 0.001716 | 0.001397 |
| Vela 2.0 9B | 0.001639 | 0.004102 |
| Vela 1.0 encoders (12) | ≤ 0.000002 | ≤ 0.000002 |
| Qwen3-Embedding-0.6B | 0.000000 | 0.000000 |

The largest gap is Decision 1.0 Sol-2B, 0.005973 vs CPU. That is under 0.02.
It is not byte-identical.

## Not in this record

Decision 3.0 (`d3` and the smaller `d3-*` packages) and Vela 1.0 Omni were
not served on this run, so `golden_answers_decision3.json` and
`golden_answers_omni.json` still have only `cpu` and `rocm`.
