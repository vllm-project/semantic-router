# CUDA on NVIDIA hardware: measured scope

Issue [#4621](https://github.com/vllm-project/semantic-router/issues/4621).
These records are the measured NVIDIA set (H200 `sm_90`, A100, Tesla T4)
that closes CUDA accelerator validation. `CUDAAccelerator.validated` is
true.

Vela 2.0 0.3B at 32 tokens is a **known regression** on H200: the ten-round
p50 interval is wholly worse. Tesla T4 five-round p50 is Level on that
card; that does not clear the H200 row. Fused kernels without bit-exact
evidence stay off the exact profile. D3/Omni goldens and D3 CUDA FLA pins
were not served by the published CUDA image and are not in this close-out.

## Provenance

| Hardware | Image | Head |
| --- | --- | --- |
| NVIDIA H200, 143771 MiB, driver 575.57.08 | `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` digest `sha256:2f05113e5c7399874ef2b1651ecb3745a5dafeff7ef2f9725acd01b643c06730` (torch 2.10.0+cu128, Triton 3.6.0, fla-core 0.5.2) | `61677a464c2a1b94ef34cd9e22595bab87981b97` |
| NVIDIA H200, 143771 MiB, driver 580.178.04 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus venv (torch 2.10.0+cu128, Triton 3.6.0, fla-core 0.5.2) | `83c1ddced3cbe4b9d82b2f9bd2ad0dad23a34356` |
| NVIDIA H200, 143771 MiB, driver 580.159.04 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8.0+cu128, Triton 3.4.0, FLA 0.5.2) | `ec307ad6ed081dd220b5dbb3571a1f9987d06914` |
| NVIDIA H200 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus venv | `ec307ad6ed081dd220b5dbb3571a1f9987d06914` |
| NVIDIA A100 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus venv | `ec307ad6ed081dd220b5dbb3571a1f9987d06914` |
| NVIDIA Tesla T4 | Google Colab GPU runtime, torch 2.11.0+cu130, CUDA 13.0 | `a1c137bcc9821766f9c0a440cc2912a21c671f3d` |

The published image has no git-revision label. Follow-up benches used the
RunPod image, not the published CUDA image.

## Records

| Gate | Record | Result |
| --- | --- | --- |
| Image smoke (charts) | [`cuda-router-image.md`](cuda-router-image.md) | 7/7 chart models ready |
| Goldens (30 published-image models) | [`cuda-golden-answers.md`](cuda-golden-answers.md) | those 30 under `cuda:sm_90`; D3/Omni captures exist locally and are not incorporated |
| Decision 1.0 parity | [`cuda-decision1-parity.md`](cuda-decision1-parity.md) | 7/7, 400/400 identical on saved MMLU-Pro |
| Decision 2.0 parity | [`cuda-decision2-parity.md`](cuda-decision2-parity.md) | 6/6, 400/400 identical including Vega |
| Vela 2.0 parity | [`cuda-vela2-parity.md`](cuda-vela2-parity.md) | 300 comparison cases passed; 13 oversized 0.3B rows compared after excluding rejected questions |
| Vela 1.0 legacy path | [`cuda-vela1-parity.md`](cuda-vela1-parity.md) | 11/11 jobs pass, agreement 1.0 |
| Fused slots + FLA pins | [`cuda-fused-kernels.md`](cuda-fused-kernels.md) | slots stay off exact; D1/D2/Vela2 `cuda:sm_90` FLA pins recorded; D3 CUDA pins not merged |
| Decision 1.0 performance | [`cuda-decision1-performance.md`](cuda-decision1-performance.md) | all end-to-end `call` intervals Level or Better |
| Decision 2.0 vs released | [`cuda-decision2-performance.md`](cuda-decision2-performance.md) | all six models Better, including Vega |
| Vela 2.0 performance | [`cuda-vela2-performance.md`](cuda-vela2-performance.md) | H200 follow-up 0.8B@512 and 4B@2048 pass; H200 0.3B@32 p50 **known regression**; Tesla T4 five-round 0.3B@32 p50 Level on that card; earlier 4×4 kept separate |
| Vela 1.0 vs CPU legacy | [`cuda-vela1-performance.md`](cuda-vela1-performance.md) | 11/11 Better on A100 |
