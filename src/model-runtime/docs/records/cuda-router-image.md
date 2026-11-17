# CUDA router image smoke on one NVIDIA H200

GPU smoke of `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` as the
charts run it: one runtime process per model inside the published image, uid
65532, model volume at `/app/models`, profile `exact`, device `cuda:0`. Machine
readable summary: [`cuda-router-image-smoke.json`](cuda-router-image-smoke.json).

- **Date:** 2026-10-12.
- **Hardware:** NVIDIA H200 (sm_90, 143771 MiB). Driver 575.57.08.
- **Image:** `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest`, digest
  `sha256:2f05113e5c7399874ef2b1651ecb3745a5dafeff7ef2f9725acd01b643c06730`,
  config created `2026-10-12T07:07:54Z`. Inside the image: Debian 12, Python
  3.12.15, torch 2.10.0+cu128, Triton 3.6.0, fla-core 0.5.2. The image has no
  git-revision label.
- **Head:** `61677a464c2a1b94ef34cd9e22595bab87981b97`, the checkout used for
  the same day's model-runtime measurements on this card.

## Smoke as the charts run it

The chart's seven Vela models: Vela 2.0 4B and 0.3B; Vela 1.0 Domain, PII,
Guard, FactCheck, Feedback. Per-model fields are in `chart_models` in
[`cuda-router-image-smoke.json`](cuda-router-image-smoke.json).

| Model | Ready | Load (s) | Golden checks |
| --- | --- | ---: | ---: |
| Vela 2.0 4B | yes | 64.5 | 7 |
| Vela 2.0 0.3B | yes | 10.0 | 7 |
| Vela 1.0 Domain | yes | 9.3 | 28 |
| Vela 1.0 PII | yes | 12.0 | 2 |
| Vela 1.0 Guard | yes | 10.0 | 4 |
| Vela 1.0 FactCheck | yes | 10.2 | 4 |
| Vela 1.0 Feedback | yes | 9.1 | 10 |

**7 / 7 ready.** The CUDA golden keys for this image, including the 23
models beyond the chart set, are in
[`cuda-golden-answers.md`](cuda-golden-answers.md).
