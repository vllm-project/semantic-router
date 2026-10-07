# Choosing the decision model: Vela 2.0 0.3B, 0.8B, 4B and 9B

[#4719](https://github.com/vllm-project/semantic-router/issues/4719) lets a
configuration choose the Router's decision model,
`global.model_catalog.system.decision_model` (`vllm-sr serve
--decision-model`). That is the Vela model that answers every built-in signal
it covers, and every `routing.signals.decision` question that names no
`deployment`, in one call per request. This record measures each Vela 2.0
size through the Router, as
[vela2-router-signals.md](vela2-router-signals.md) did for the 0.3B, and gives
the evidence for each size's module thresholds.

- **Accuracy:** the 4B and 9B are ahead of the Vela 1.0 specialists on almost
  every signal and set: domain held-out accuracy +0.122 and +0.138, safety
  held-out AUC +0.101 and +0.112, hallucination held-out AUC +0.135 and
  +0.153. The 0.8B is ahead on domain, prompt guard, safety, modality and
  hallucination (held out), and behind on PII. On every size, user
  feedback's fresh file (CrossWOZ) is the largest regression.
- **Latency:** on a GPU a request takes 6.9 ms at the median on the 0.3B, 40.7 ms on the 0.8B, 56.8 ms on the 4B and 79.0 ms on the 9B, sequentially on one MI325X; at concurrency 16 one GPU serves about 146, 25, 17 and 12 requests per second. On a CPU the 0.8B is a
  decoder: a request takes about 3.0 s at the median on 12 cores (0.32 requests per second), against 116 ms for the 0.3B measured beside it on the same NUMA node, and 79 ms in vela2-router-signals.md, where the 0.3B ran alone.
- **One call per request:** on every size, each request of the A/B reached the
  model as one bundle with one decisions task.
- **Thresholds:** each size has its own module thresholds; a module that sets
  none takes those of the model it runs.

- **Date:** 2026-10-07.
- **Machine:** AMD EPYC 9575F with AMD Instinct MI325X GPUs (gfx942). Each
  process ran in a container pinned to 12 cores of one NUMA node; each GPU arm
  had its own GPU.
- **Commits:** the Router built from this change (on `main` `320d5d49a`) in
  the router image recipe, `tools/docker/Dockerfile.extproc`: the ROCm image
  for the GPU arms (ROCm PyTorch 2.12.0, FLA 0.5.2, `causal-conv1d` 1.7.0)
  and the CPU image (PyTorch 2.10.0, MKL) for the CPU arms. The model runtime
  is `main`'s.
- **Models:** Vela 2.0 at the revisions the runtime pins: 0.3B `a3209a50`,
  0.8B `a778eb2a`, 4B `c1e64d4f`, 9B `bc876163`. The Vela 1.0 arm and the
  0.3B's accuracy arm are vela2-router-signals.md's.

## Accuracy

The rows, metrics, sets and intervals are vela2-router-signals.md's: every
dev, test, source test, held-out and fresh held-out file of the eight signals
whose text [`vllm-sr/router-signal-suite`](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
(revision `fa08b2a6`) publishes; 118,712 request rows and 15,431
hallucination rows per arm; group bootstraps with 2,000 replicates, paired
with the Vela 1.0 arm.

- **Arms:** `tools/router_signal_ab.py config --arm vela2 --decision-model
  <size> --gpu` for the 0.8B, 4B and 9B. Each sets only
  `global.model_catalog.system.decision_model` and puts the modules on the
  GPU, as `vllm-sr serve --platform amd` does, so every built-in signal runs
  on the size's implicit deployment, `@Vela-2.0-<size>/auto`, on `exact`.
  The rows were split over three GPUs per size (two for the 0.8B), each
  shard with its own Router and recording runtime.
- **The 0.3B** is vela2-router-signals.md's arm, on CPU under `max_speed`;
  its scores are within 5e-6 of `exact`. The larger sizes ran on a GPU, where
  their backbones run under bf16 autocast with FP32 heads, as their cards
  evaluate them.
- **Hallucination:** asked as the Router's detector asks it
  (`router_signal_ab.py halu --side vela2`), against runtimes serving each size.

Means per signal (accuracy or AUC):

| Signal (metric) | Set | Files | Vela 1.0 | 0.3B | 0.8B | 4B | 9B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Domain (accuracy) | held out | 5 | 0.638 | 0.600 | 0.701 | 0.759 | 0.775 |
|  | fresh | 3 | 0.561 | 0.473 | 0.550 | 0.658 | 0.671 |
|  | in distribution | 2 | 0.561 | 0.541 | 0.676 | 0.740 | 0.777 |
| Prompt guard (AUC) | held out | 4 | 0.828 | 0.854 | 0.895 | 0.926 | 0.925 |
|  | fresh | 2 | 0.755 | 0.753 | 0.895 | 0.901 | 0.898 |
|  | in distribution | 1 | 0.847 | 0.853 | 0.872 | 0.913 | 0.919 |
| Safety (AUC) | held out | 4 | 0.856 | 0.908 | 0.909 | 0.957 | 0.969 |
|  | fresh | 5 | 0.851 | 0.869 | 0.878 | 0.947 | 0.963 |
|  | in distribution | 4 | 0.880 | 0.914 | 0.876 | 0.925 | 0.930 |
| Fact check (AUC) | held out | 2 | 0.905 | 0.804 | 0.914 | 0.869 | 0.916 |
|  | fresh | 1 | 0.682 | 0.702 | 0.722 | 0.668 | 0.762 |
|  | in distribution | 1 | 0.741 | 0.753 | 0.768 | 0.799 | 0.808 |
| Modality (AUC) | held out | 1 | 0.892 | 0.713 | 0.908 | 0.974 | 0.973 |
|  | in distribution | 1 | 0.817 | 0.708 | 0.841 | 0.900 | 0.933 |
| PII (AUC) | held out | 2 | 0.968 | 0.972 | 0.940 | 0.984 | 0.986 |
|  | fresh | 4 | 0.943 | 0.938 | 0.882 | 0.889 | 0.934 |
|  | in distribution | 4 | 0.954 | 0.924 | 0.851 | 0.903 | 0.905 |
| User feedback (accuracy) | held out | 1 | 0.311 | 0.272 | 0.711 | 0.518 | 0.526 |
|  | fresh | 1 | 0.766 | 0.588 | 0.419 | 0.335 | 0.371 |
|  | in distribution | 2 | 0.609 | 0.595 | 0.644 | 0.578 | 0.717 |
| Hallucination (AUC) | held out | 3 | 0.702 | 0.712 | 0.736 | 0.837 | 0.855 |
|  | fresh | 2 | 0.686 | 0.689 | 0.733 | 0.757 | 0.770 |
|  | in distribution | 4 | 0.782 | 0.757 | 0.721 | 0.834 | 0.855 |

Each size − Vela 1.0 (95% interval):

| Signal (metric) | Set | 0.3B | 0.8B | 4B | 9B |
| --- | --- | ---: | ---: | ---: | ---: |
| Domain (accuracy) | held out | -0.037 [-0.048, -0.028] | +0.063 [+0.054, +0.074] | +0.122 [+0.112, +0.132] | +0.138 [+0.128, +0.147] |
|  | fresh | -0.088 [-0.101, -0.074] | -0.011 [-0.025, +0.003] | +0.098 [+0.084, +0.111] | +0.110 [+0.097, +0.123] |
|  | in distribution | -0.020 [-0.034, -0.006] | +0.115 [+0.099, +0.130] | +0.179 [+0.164, +0.194] | +0.216 [+0.203, +0.231] |
| Prompt guard (AUC) | held out | +0.026 [+0.011, +0.043] | +0.067 [+0.050, +0.089] | +0.098 [+0.083, +0.116] | +0.097 [+0.081, +0.115] |
|  | fresh | -0.002 [-0.025, +0.022] | +0.140 [+0.117, +0.164] | +0.147 [+0.122, +0.173] | +0.144 [+0.121, +0.168] |
|  | in distribution | +0.006 [-0.015, +0.027] | +0.025 [+0.008, +0.046] | +0.065 [+0.047, +0.089] | +0.072 [+0.052, +0.094] |
| Safety (AUC) | held out | +0.052 [+0.038, +0.066] | +0.053 [+0.039, +0.068] | +0.101 [+0.086, +0.117] | +0.112 [+0.097, +0.128] |
|  | fresh | +0.018 [+0.004, +0.033] | +0.027 [+0.013, +0.042] | +0.097 [+0.083, +0.111] | +0.112 [+0.098, +0.127] |
|  | in distribution | +0.034 [+0.025, +0.043] | -0.004 [-0.014, +0.005] | +0.045 [+0.036, +0.054] | +0.050 [+0.041, +0.059] |
| Fact check (AUC) | held out | -0.101 [-0.138, -0.066] | +0.008 [-0.021, +0.036] | -0.037 [-0.069, -0.005] | +0.011 [-0.018, +0.039] |
|  | fresh | +0.020 [-0.075, +0.117] | +0.040 [-0.035, +0.117] | -0.014 [-0.101, +0.070] | +0.080 [+0.006, +0.154] |
|  | in distribution | +0.011 [-0.032, +0.053] | +0.026 [-0.015, +0.065] | +0.058 [+0.018, +0.098] | +0.067 [+0.029, +0.105] |
| Modality (AUC) | held out | -0.180 [-0.198, -0.164] | +0.016 [+0.005, +0.026] | +0.082 [+0.072, +0.091] | +0.081 [+0.072, +0.091] |
|  | in distribution | -0.109 [-0.128, -0.090] | +0.024 [+0.011, +0.037] | +0.083 [+0.070, +0.096] | +0.116 [+0.103, +0.129] |
| PII (AUC) | held out | +0.004 [-0.007, +0.014] | -0.027 [-0.044, -0.012] | +0.017 [+0.011, +0.022] | +0.019 [+0.013, +0.024] |
|  | fresh | -0.005 [-0.013, +0.004] | -0.061 [-0.073, -0.047] | -0.054 [-0.071, -0.038] | -0.009 [-0.022, +0.004] |
|  | in distribution | -0.029 [-0.034, -0.019] | -0.103 [-0.118, -0.094] | -0.051 [-0.059, -0.038] | -0.049 [-0.061, -0.038] |
| User feedback (accuracy) | held out | -0.038 [-0.058, -0.018] | +0.400 [+0.373, +0.430] | +0.207 [+0.183, +0.233] | +0.215 [+0.191, +0.240] |
|  | fresh | -0.178 [-0.206, -0.150] | -0.347 [-0.375, -0.320] | -0.432 [-0.466, -0.399] | -0.396 [-0.423, -0.369] |
|  | in distribution | -0.014 [-0.028, +0.001] | +0.035 [+0.015, +0.055] | -0.031 [-0.051, -0.010] | +0.108 [+0.089, +0.127] |
| Hallucination (AUC) | held out | +0.010 [-0.005, +0.026] | +0.035 [+0.019, +0.051] | +0.135 [+0.119, +0.151] | +0.153 [+0.138, +0.169] |
|  | fresh | +0.003 [-0.032, +0.035] | +0.046 [+0.004, +0.088] | +0.071 [+0.036, +0.108] | +0.083 [+0.050, +0.119] |
|  | in distribution | -0.025 [-0.035, -0.015] | -0.061 [-0.073, -0.046] | +0.052 [+0.041, +0.065] | +0.073 [+0.061, +0.085] |

- **Ahead on every size:** prompt guard and safety (held out and fresh), and,
  from the 0.8B up, domain on held-out files and modality.
- **The 4B and 9B** are ahead on hallucination on every set, where the 0.3B
  and 0.8B are behind in distribution, and on PII held out.
- **Behind on every size:** user feedback on its fresh file (CrossWOZ), by
  0.18 (0.3B) to 0.43 (4B), and PII in distribution. The 0.8B is behind on PII
  on every set.
- **Fact check:** level on most sets. Its two held-out files give their label
  away without the text (vela2-router-signals.md).

Per file, each size against Vela 1.0:

<details>
<summary>0.8B</summary>

| signal | file | rows | metric | Vela 1.0 | Vela 2.0 0.8B | 0.8B − Vela 1.0 [95% CI] |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| domain | dev | 1,498 | accuracy | 0.622 | 0.722 | +0.100 [+0.073, +0.124] |
| domain | fresh-arabicmmlu | 1,999 | accuracy | 0.524 | 0.425 | -0.099 [-0.125, -0.073] |
| domain | fresh-ceval | 1,999 | accuracy | 0.637 | 0.740 | +0.103 [+0.078, +0.127] |
| domain | fresh-indommlu | 1,999 | accuracy | 0.521 | 0.485 | -0.036 [-0.059, -0.015] |
| domain | hold-arena-expert | 987 | accuracy | 0.690 | 0.763 | +0.073 [+0.044, +0.101] |
| domain | hold-mmlu-cf | 1,988 | accuracy | 0.529 | 0.666 | +0.137 [+0.113, +0.161] |
| domain | hold-mmlu-pro | 1,988 | accuracy | 0.725 | 0.689 | -0.036 [-0.054, -0.017] |
| domain | hold-mmlu-prox | 1,988 | accuracy | 0.659 | 0.633 | -0.026 [-0.047, -0.003] |
| domain | hold-supergpqa | 1,882 | accuracy | 0.587 | 0.754 | +0.167 [+0.146, +0.188] |
| domain | test | 2,996 | accuracy | 0.606 | 0.702 | +0.097 [+0.079, +0.115] |
| domain | test-exams | 1,927 | accuracy | 0.516 | 0.650 | +0.133 [+0.108, +0.158] |
| jailbreak | dev | 320 | AUC | 0.912 | 0.962 | +0.050 [+0.019, +0.087] |
| jailbreak | fresh-cyberseceval-indirect | 245 | recall@0.5 | 0.061 | 0.584 | +0.522 [+0.461, +0.580] |
| jailbreak | fresh-guardrail-hn | 244 | AUC | 0.866 | 0.954 | +0.088 [+0.045, +0.133] |
| jailbreak | fresh-ipi-arena | 71 | recall@0.5 | 0.789 | 0.930 | +0.141 [+0.056, +0.225] |
| jailbreak | fresh-sep | 2,000 | AUC | 0.643 | 0.835 | +0.192 [+0.176, +0.208] |
| jailbreak | hold-bipia | 637 | AUC | 0.664 | 0.822 | +0.159 [+0.116, +0.204] |
| jailbreak | hold-jailbreakhub-late | 1,303 | AUC | 0.652 | 0.707 | +0.054 [+0.021, +0.086] |
| jailbreak | hold-llmail | 1,152 | AUC | 0.972 | 1.000 | +0.028 [+0.019, +0.039] |
| jailbreak | hold-notinject | 339 | specificity@0.5 | 0.935 | 0.971 | +0.035 [+0.009, +0.062] |
| jailbreak | hold-promptshield-test | 2,000 | AUC | 0.763 | 0.827 | +0.064 [+0.020, +0.137] |
| jailbreak | hold-toxicchat | 1,152 | AUC | 0.915 | 0.930 | +0.016 [-0.005, +0.037] |
| jailbreak | test | 1,878 | AUC | 0.847 | 0.872 | +0.025 [+0.008, +0.046] |
| safety | dev | 537 | AUC | 0.884 | 0.853 | -0.031 [-0.068, +0.004] |
| safety | fresh-catqa-safety | 1,209 | recall@0.5 | 0.917 | 0.941 | +0.024 [+0.005, +0.043] |
| safety | fresh-cdna | 1,180 | AUC | 0.885 | 0.926 | +0.041 [+0.025, +0.058] |
| safety | fresh-indicsafe | 1,992 | AUC | 0.808 | 0.748 | -0.060 [-0.117, -0.006] |
| safety | fresh-linguasafe-multi | 2,000 | AUC | 0.791 | 0.871 | +0.080 [+0.044, +0.118] |
| safety | fresh-linguasafe-sr | 2,000 | AUC | 0.816 | 0.890 | +0.074 [+0.054, +0.094] |
| safety | fresh-turkish-overrefusal | 480 | AUC | 0.953 | 0.953 | +0.000 [-0.018, +0.017] |
| safety | hold-coconot | 897 | AUC | 0.883 | 0.947 | +0.065 [+0.040, +0.091] |
| safety | hold-coconot-contrast | 379 | specificity@0.5 | 0.757 | 0.807 | +0.050 [+0.013, +0.087] |
| safety | hold-jbb | 200 | AUC | 0.866 | 0.919 | +0.053 [+0.013, +0.098] |
| safety | hold-openai-moderation | 1,473 | AUC | 0.871 | 0.899 | +0.028 [+0.014, +0.044] |
| safety | hold-orbench-hard | 1,319 | specificity@0.5 | 0.774 | 0.278 | -0.496 [-0.525, -0.466] |
| safety | hold-toxicchat | 1,657 | AUC | 0.906 | 0.965 | +0.059 [+0.045, +0.072] |
| safety | hold-xstest | 450 | AUC | 0.782 | 0.854 | +0.072 [+0.040, +0.105] |
| safety | test | 1,120 | AUC | 0.912 | 0.873 | -0.039 [-0.061, -0.017] |
| safety | test-aegis2 | 1,749 | AUC | 0.919 | 0.907 | -0.011 [-0.022, -0.001] |
| safety | test-nemotron-v3 | 2,000 | AUC | 0.890 | 0.846 | -0.043 [-0.061, -0.026] |
| safety | test-polyguardprompts | 2,000 | AUC | 0.799 | 0.876 | +0.077 [+0.053, +0.101] |
| fact_check | dev | 198 | AUC | 0.702 | 0.688 | -0.013 [-0.093, +0.069] |
| fact_check | fresh-halueval-wild | 188 | AUC | 0.682 | 0.722 | +0.040 [-0.035, +0.117] |
| fact_check | fresh-mbpp | 961 | specificity@0.5 | 0.999 | 0.831 | -0.168 [-0.194, -0.145] |
| fact_check | fresh-mgsm | 1,991 | specificity@0.5 | 0.913 | 0.005 | -0.908 [-0.930, -0.885] |
| fact_check | fresh-mintaka | 2,000 | recall@0.5 | 0.951 | 1.000 | +0.050 [+0.041, +0.059] |
| fact_check | fresh-truthfulqa | 726 | recall@0.5 | 0.797 | 0.995 | +0.197 [+0.171, +0.226] |
| fact_check | hold-no_robots | 2,000 | AUC | 0.962 | 0.968 | +0.006 [-0.003, +0.015] |
| fact_check | hold-shipped-factcheck | 2,000 | AUC | 0.724 | 0.894 | +0.170 [+0.149, +0.191] |
| fact_check | hold-simpleqa | 2,000 | recall@0.5 | 0.895 | 1.000 | +0.104 [+0.090, +0.118] |
| fact_check | hold-wildbench | 414 | AUC | 0.849 | 0.860 | +0.011 [-0.047, +0.066] |
| fact_check | test | 800 | AUC | 0.741 | 0.768 | +0.026 [-0.015, +0.065] |
| modality | dev | 1,500 | AUC | 0.812 | 0.840 | +0.028 [+0.008, +0.048] |
| modality | fresh-emu-edit | 2,000 | recall@0.5 | 0.386 | 0.603 | +0.217 [+0.194, +0.238] |
| modality | fresh-oneig-zh | 1,320 | recall@0.5 | 0.230 | 0.687 | +0.457 [+0.429, +0.485] |
| modality | fresh-text-requests | 743 | specificity@0.5 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| modality | hold-alpaca | 2,000 | specificity@0.5 | 0.964 | 0.957 | -0.007 [-0.018, +0.003] |
| modality | hold-arena-t2i-hard | 210 | recall@0.5 | 0.671 | 0.933 | +0.262 [+0.200, +0.324] |
| modality | hold-gedit-bench | 1,189 | recall@0.5 | 0.330 | 0.547 | +0.218 [+0.187, +0.249] |
| modality | hold-parti | 1,559 | recall@0.5 | 0.331 | 0.489 | +0.158 [+0.130, +0.185] |
| modality | hold-realmix | 3,769 | AUC | 0.892 | 0.908 | +0.016 [+0.005, +0.026] |
| modality | hold-search-arena | 2,000 | specificity@0.5 | 0.958 | 0.952 | -0.006 [-0.017, +0.005] |
| modality | test | 3,000 | AUC | 0.817 | 0.841 | +0.024 [+0.011, +0.037] |
| modality | test-dolly | 2,000 | specificity@0.5 | 0.984 | 0.999 | +0.015 [+0.009, +0.021] |
| modality | test-realedit | 2,000 | recall@0.5 | 0.316 | 0.457 | +0.141 [+0.116, +0.167] |
| pii | dev | 1,506 | AUC | 0.940 | 0.885 | -0.055 [-0.074, -0.036] |
| pii | fresh-btc | 1,342 | AUC | 0.914 | 0.833 | -0.082 [-0.105, -0.059] |
| pii | fresh-pii-trace | 1,709 | AUC | 0.998 | 0.997 | -0.002 [-0.005, +0.002] |
| pii | fresh-ru-pii | 1,842 | AUC | 0.939 | 0.938 | -0.001 [-0.015, +0.014] |
| pii | fresh-uner-ewt | 1,780 | AUC | 0.921 | 0.761 | -0.160 [-0.199, -0.114] |
| pii | hold-kaggle-essays | 1,125 | AUC | 0.989 | 0.930 | -0.059 [-0.088, -0.030] |
| pii | hold-pii-prompts | 2,000 | AUC | 0.946 | 0.951 | +0.004 [-0.008, +0.015] |
| pii | hold-wildchat | 2,000 | specificity@0.5 | 0.716 | 0.901 | +0.185 [+0.168, +0.203] |
| pii | test | 3,000 | AUC | 0.947 | 0.888 | -0.059 [-0.073, -0.044] |
| pii | test-abcd | 1,486 | AUC | 0.993 | 0.884 | -0.108 [-0.127, -0.089] |
| pii | test-mapa | 1,679 | AUC | 0.918 | 0.719 | -0.199 [-0.243, -0.192] |
| pii | test-tab | 1,461 | AUC | 0.958 | 0.911 | -0.046 [-0.072, -0.022] |
| feedback | dev | 1,199 | accuracy | 0.574 | 0.553 | -0.021 [-0.069, +0.025] |
| feedback | fresh-crosswoz | 1,995 | accuracy | 0.766 | 0.419 | -0.347 [-0.375, -0.320] |
| feedback | hold-shipped-feedback | 1,715 | accuracy | 0.311 | 0.711 | +0.400 [+0.373, +0.430] |
| feedback | test | 2,842 | accuracy | 0.530 | 0.588 | +0.058 [+0.031, +0.088] |
| feedback | test-sgd | 1,998 | accuracy | 0.688 | 0.700 | +0.011 [-0.014, +0.037] |
| hallucination | dev | 1,210 | AUC | 0.786 | 0.723 | -0.063 [-0.088, -0.040] |
| hallucination | fresh-faithbench | 523 | AUC | 0.646 | 0.677 | +0.030 [-0.042, +0.095] |
| hallucination | fresh-shroom | 749 | AUC | 0.726 | 0.788 | +0.062 [+0.014, +0.112] |
| hallucination | hold-attributionbench-ood | 1,390 | AUC | 0.780 | 0.815 | +0.035 [+0.014, +0.056] |
| hallucination | hold-hallumix | 2,000 | AUC | 0.755 | 0.720 | -0.035 [-0.057, -0.013] |
| hallucination | hold-halubench | 1,877 | AUC | 0.527 | 0.633 | +0.106 [+0.076, +0.136] |
| hallucination | hold-summedits | 578 | AUC | 0.823 | 0.857 | +0.033 [+0.003, +0.064] |
| hallucination | test | 3,000 | AUC | 0.774 | 0.717 | -0.058 [-0.073, -0.042] |
| hallucination | test-psiloqa | 1,265 | AUC | 0.884 | 0.818 | -0.065 [-0.091, -0.041] |
| hallucination | test-ragbench | 1,311 | AUC | 0.605 | 0.577 | -0.028 [-0.060, +0.006] |
| hallucination | test-ragtruth | 1,528 | AUC | 0.864 | 0.773 | -0.091 [-0.116, -0.066] |

</details>

<details>
<summary>4B</summary>

| signal | file | rows | metric | Vela 1.0 | Vela 2.0 4B | 4B − Vela 1.0 [95% CI] |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| domain | dev | 1,498 | accuracy | 0.622 | 0.774 | +0.152 [+0.128, +0.176] |
| domain | fresh-arabicmmlu | 1,999 | accuracy | 0.524 | 0.596 | +0.072 [+0.047, +0.098] |
| domain | fresh-ceval | 1,999 | accuracy | 0.637 | 0.828 | +0.191 [+0.168, +0.214] |
| domain | fresh-indommlu | 1,999 | accuracy | 0.521 | 0.551 | +0.030 [+0.008, +0.051] |
| domain | hold-arena-expert | 987 | accuracy | 0.690 | 0.814 | +0.124 [+0.094, +0.153] |
| domain | hold-mmlu-cf | 1,988 | accuracy | 0.529 | 0.705 | +0.177 [+0.153, +0.200] |
| domain | hold-mmlu-pro | 1,988 | accuracy | 0.725 | 0.764 | +0.039 [+0.019, +0.059] |
| domain | hold-mmlu-prox | 1,988 | accuracy | 0.659 | 0.737 | +0.078 [+0.057, +0.099] |
| domain | hold-supergpqa | 1,882 | accuracy | 0.587 | 0.778 | +0.191 [+0.169, +0.212] |
| domain | test | 2,996 | accuracy | 0.606 | 0.757 | +0.151 [+0.134, +0.169] |
| domain | test-exams | 1,927 | accuracy | 0.516 | 0.723 | +0.207 [+0.182, +0.232] |
| jailbreak | dev | 320 | AUC | 0.912 | 0.966 | +0.054 [+0.023, +0.094] |
| jailbreak | fresh-cyberseceval-indirect | 245 | recall@0.5 | 0.061 | 0.604 | +0.543 [+0.482, +0.608] |
| jailbreak | fresh-guardrail-hn | 244 | AUC | 0.866 | 0.987 | +0.121 [+0.077, +0.167] |
| jailbreak | fresh-ipi-arena | 71 | recall@0.5 | 0.789 | 0.972 | +0.183 [+0.099, +0.268] |
| jailbreak | fresh-sep | 2,000 | AUC | 0.643 | 0.816 | +0.173 [+0.154, +0.192] |
| jailbreak | hold-bipia | 637 | AUC | 0.664 | 0.847 | +0.183 [+0.143, +0.225] |
| jailbreak | hold-jailbreakhub-late | 1,303 | AUC | 0.652 | 0.757 | +0.104 [+0.073, +0.134] |
| jailbreak | hold-llmail | 1,152 | AUC | 0.972 | 1.000 | +0.028 [+0.019, +0.039] |
| jailbreak | hold-notinject | 339 | specificity@0.5 | 0.935 | 0.968 | +0.032 [+0.006, +0.059] |
| jailbreak | hold-promptshield-test | 2,000 | AUC | 0.763 | 0.897 | +0.134 [+0.096, +0.191] |
| jailbreak | hold-toxicchat | 1,152 | AUC | 0.915 | 0.960 | +0.045 [+0.024, +0.069] |
| jailbreak | test | 1,878 | AUC | 0.847 | 0.913 | +0.065 [+0.047, +0.089] |
| safety | dev | 537 | AUC | 0.884 | 0.917 | +0.033 [+0.001, +0.066] |
| safety | fresh-catqa-safety | 1,209 | recall@0.5 | 0.917 | 0.969 | +0.051 [+0.031, +0.071] |
| safety | fresh-cdna | 1,180 | AUC | 0.885 | 0.970 | +0.085 [+0.070, +0.102] |
| safety | fresh-indicsafe | 1,992 | AUC | 0.808 | 0.886 | +0.078 [+0.034, +0.123] |
| safety | fresh-linguasafe-multi | 2,000 | AUC | 0.791 | 0.941 | +0.149 [+0.110, +0.190] |
| safety | fresh-linguasafe-sr | 2,000 | AUC | 0.816 | 0.947 | +0.131 [+0.108, +0.154] |
| safety | fresh-turkish-overrefusal | 480 | AUC | 0.953 | 0.993 | +0.040 [+0.023, +0.059] |
| safety | hold-coconot | 897 | AUC | 0.883 | 0.977 | +0.095 [+0.069, +0.122] |
| safety | hold-coconot-contrast | 379 | specificity@0.5 | 0.757 | 0.968 | +0.211 [+0.172, +0.251] |
| safety | hold-jbb | 200 | AUC | 0.866 | 0.938 | +0.073 [+0.031, +0.117] |
| safety | hold-openai-moderation | 1,473 | AUC | 0.871 | 0.938 | +0.068 [+0.053, +0.083] |
| safety | hold-orbench-hard | 1,319 | specificity@0.5 | 0.774 | 0.309 | -0.466 [-0.497, -0.434] |
| safety | hold-toxicchat | 1,657 | AUC | 0.906 | 0.986 | +0.079 [+0.065, +0.093] |
| safety | hold-xstest | 450 | AUC | 0.782 | 0.968 | +0.185 [+0.146, +0.225] |
| safety | test | 1,120 | AUC | 0.912 | 0.931 | +0.019 [+0.003, +0.034] |
| safety | test-aegis2 | 1,749 | AUC | 0.919 | 0.937 | +0.018 [+0.007, +0.029] |
| safety | test-nemotron-v3 | 2,000 | AUC | 0.890 | 0.899 | +0.010 [-0.004, +0.024] |
| safety | test-polyguardprompts | 2,000 | AUC | 0.799 | 0.933 | +0.134 [+0.109, +0.159] |
| fact_check | dev | 198 | AUC | 0.702 | 0.728 | +0.026 [-0.052, +0.111] |
| fact_check | fresh-halueval-wild | 188 | AUC | 0.682 | 0.668 | -0.014 [-0.101, +0.070] |
| fact_check | fresh-mbpp | 961 | specificity@0.5 | 0.999 | 0.695 | -0.304 [-0.333, -0.275] |
| fact_check | fresh-mgsm | 1,991 | specificity@0.5 | 0.913 | 0.000 | -0.913 [-0.935, -0.890] |
| fact_check | fresh-mintaka | 2,000 | recall@0.5 | 0.951 | 1.000 | +0.050 [+0.041, +0.059] |
| fact_check | fresh-truthfulqa | 726 | recall@0.5 | 0.797 | 1.000 | +0.203 [+0.175, +0.233] |
| fact_check | hold-no_robots | 2,000 | AUC | 0.962 | 0.956 | -0.005 [-0.016, +0.005] |
| fact_check | hold-shipped-factcheck | 2,000 | AUC | 0.724 | 0.918 | +0.194 [+0.173, +0.216] |
| fact_check | hold-simpleqa | 2,000 | recall@0.5 | 0.895 | 1.000 | +0.104 [+0.091, +0.118] |
| fact_check | hold-wildbench | 414 | AUC | 0.849 | 0.781 | -0.068 [-0.134, -0.004] |
| fact_check | test | 800 | AUC | 0.741 | 0.799 | +0.058 [+0.018, +0.098] |
| modality | dev | 1,500 | AUC | 0.812 | 0.902 | +0.090 [+0.070, +0.111] |
| modality | fresh-emu-edit | 2,000 | recall@0.5 | 0.386 | 0.890 | +0.505 [+0.482, +0.527] |
| modality | fresh-oneig-zh | 1,320 | recall@0.5 | 0.230 | 0.820 | +0.589 [+0.561, +0.616] |
| modality | fresh-text-requests | 743 | specificity@0.5 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| modality | hold-alpaca | 2,000 | specificity@0.5 | 0.964 | 0.982 | +0.018 [+0.009, +0.027] |
| modality | hold-arena-t2i-hard | 210 | recall@0.5 | 0.671 | 0.976 | +0.305 [+0.243, +0.367] |
| modality | hold-gedit-bench | 1,189 | recall@0.5 | 0.330 | 0.806 | +0.476 [+0.443, +0.514] |
| modality | hold-parti | 1,559 | recall@0.5 | 0.331 | 0.640 | +0.309 [+0.283, +0.337] |
| modality | hold-realmix | 3,769 | AUC | 0.892 | 0.974 | +0.082 [+0.072, +0.091] |
| modality | hold-search-arena | 2,000 | specificity@0.5 | 0.958 | 0.979 | +0.021 [+0.012, +0.030] |
| modality | test | 3,000 | AUC | 0.817 | 0.900 | +0.083 [+0.070, +0.096] |
| modality | test-dolly | 2,000 | specificity@0.5 | 0.984 | 0.999 | +0.015 [+0.010, +0.021] |
| modality | test-realedit | 2,000 | recall@0.5 | 0.316 | 0.697 | +0.381 [+0.358, +0.405] |
| pii | dev | 1,506 | AUC | 0.940 | 0.943 | +0.003 [-0.012, +0.017] |
| pii | fresh-btc | 1,342 | AUC | 0.914 | 0.837 | -0.077 [-0.103, -0.051] |
| pii | fresh-pii-trace | 1,709 | AUC | 0.998 | 0.994 | -0.004 [-0.008, -0.001] |
| pii | fresh-ru-pii | 1,842 | AUC | 0.939 | 0.912 | -0.027 [-0.042, -0.012] |
| pii | fresh-uner-ewt | 1,780 | AUC | 0.921 | 0.815 | -0.105 [-0.165, -0.052] |
| pii | hold-kaggle-essays | 1,125 | AUC | 0.989 | 0.994 | +0.005 [+0.001, +0.010] |
| pii | hold-pii-prompts | 2,000 | AUC | 0.946 | 0.974 | +0.028 [+0.019, +0.037] |
| pii | hold-wildchat | 2,000 | specificity@0.5 | 0.716 | 0.883 | +0.167 [+0.149, +0.185] |
| pii | test | 3,000 | AUC | 0.947 | 0.942 | -0.005 [-0.015, +0.006] |
| pii | test-abcd | 1,486 | AUC | 0.993 | 0.872 | -0.120 [-0.140, -0.101] |
| pii | test-mapa | 1,679 | AUC | 0.918 | 0.828 | -0.090 [-0.105, -0.059] |
| pii | test-tab | 1,461 | AUC | 0.958 | 0.970 | +0.012 [-0.003, +0.027] |
| feedback | dev | 1,199 | accuracy | 0.574 | 0.531 | -0.043 [-0.093, +0.005] |
| feedback | fresh-crosswoz | 1,995 | accuracy | 0.766 | 0.335 | -0.432 [-0.466, -0.399] |
| feedback | hold-shipped-feedback | 1,715 | accuracy | 0.311 | 0.518 | +0.207 [+0.183, +0.233] |
| feedback | test | 2,842 | accuracy | 0.530 | 0.560 | +0.030 [+0.002, +0.062] |
| feedback | test-sgd | 1,998 | accuracy | 0.688 | 0.597 | -0.092 [-0.119, -0.065] |
| hallucination | dev | 1,210 | AUC | 0.786 | 0.797 | +0.011 [-0.013, +0.033] |
| hallucination | fresh-faithbench | 523 | AUC | 0.646 | 0.657 | +0.011 [-0.046, +0.072] |
| hallucination | fresh-shroom | 749 | AUC | 0.726 | 0.856 | +0.131 [+0.090, +0.173] |
| hallucination | hold-attributionbench-ood | 1,390 | AUC | 0.780 | 0.842 | +0.062 [+0.042, +0.083] |
| hallucination | hold-hallumix | 2,000 | AUC | 0.755 | 0.899 | +0.144 [+0.124, +0.164] |
| hallucination | hold-halubench | 1,877 | AUC | 0.527 | 0.709 | +0.182 [+0.153, +0.211] |
| hallucination | hold-summedits | 578 | AUC | 0.823 | 0.902 | +0.078 [+0.048, +0.110] |
| hallucination | test | 3,000 | AUC | 0.774 | 0.787 | +0.013 [-0.003, +0.028] |
| hallucination | test-psiloqa | 1,265 | AUC | 0.884 | 0.858 | -0.025 [-0.045, -0.006] |
| hallucination | test-ragbench | 1,311 | AUC | 0.605 | 0.799 | +0.194 [+0.160, +0.229] |
| hallucination | test-ragtruth | 1,528 | AUC | 0.864 | 0.893 | +0.029 [+0.010, +0.047] |

</details>

<details>
<summary>9B</summary>

| signal | file | rows | metric | Vela 1.0 | Vela 2.0 9B | 9B − Vela 1.0 [95% CI] |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| domain | dev | 1,498 | accuracy | 0.622 | 0.776 | +0.154 [+0.130, +0.178] |
| domain | fresh-arabicmmlu | 1,999 | accuracy | 0.524 | 0.538 | +0.015 [-0.010, +0.038] |
| domain | fresh-ceval | 1,999 | accuracy | 0.637 | 0.856 | +0.219 [+0.198, +0.241] |
| domain | fresh-indommlu | 1,999 | accuracy | 0.521 | 0.618 | +0.097 [+0.076, +0.118] |
| domain | hold-arena-expert | 987 | accuracy | 0.690 | 0.825 | +0.135 [+0.107, +0.163] |
| domain | hold-mmlu-cf | 1,988 | accuracy | 0.529 | 0.713 | +0.184 [+0.162, +0.206] |
| domain | hold-mmlu-pro | 1,988 | accuracy | 0.725 | 0.787 | +0.062 [+0.043, +0.082] |
| domain | hold-mmlu-prox | 1,988 | accuracy | 0.659 | 0.761 | +0.102 [+0.082, +0.123] |
| domain | hold-supergpqa | 1,882 | accuracy | 0.587 | 0.792 | +0.206 [+0.185, +0.226] |
| domain | test | 2,996 | accuracy | 0.606 | 0.764 | +0.159 [+0.142, +0.177] |
| domain | test-exams | 1,927 | accuracy | 0.516 | 0.790 | +0.274 [+0.250, +0.299] |
| jailbreak | dev | 320 | AUC | 0.912 | 0.955 | +0.043 [+0.011, +0.084] |
| jailbreak | fresh-cyberseceval-indirect | 245 | recall@0.5 | 0.061 | 0.404 | +0.343 [+0.278, +0.404] |
| jailbreak | fresh-guardrail-hn | 244 | AUC | 0.866 | 0.975 | +0.109 [+0.066, +0.155] |
| jailbreak | fresh-ipi-arena | 71 | recall@0.5 | 0.789 | 0.901 | +0.113 [+0.014, +0.211] |
| jailbreak | fresh-sep | 2,000 | AUC | 0.643 | 0.821 | +0.178 [+0.162, +0.195] |
| jailbreak | hold-bipia | 637 | AUC | 0.664 | 0.913 | +0.249 [+0.211, +0.288] |
| jailbreak | hold-jailbreakhub-late | 1,303 | AUC | 0.652 | 0.742 | +0.090 [+0.057, +0.121] |
| jailbreak | hold-llmail | 1,152 | AUC | 0.972 | 1.000 | +0.028 [+0.019, +0.039] |
| jailbreak | hold-notinject | 339 | specificity@0.5 | 0.935 | 0.982 | +0.047 [+0.018, +0.077] |
| jailbreak | hold-promptshield-test | 2,000 | AUC | 0.763 | 0.819 | +0.056 [+0.015, +0.108] |
| jailbreak | hold-toxicchat | 1,152 | AUC | 0.915 | 0.969 | +0.054 [+0.032, +0.078] |
| jailbreak | test | 1,878 | AUC | 0.847 | 0.919 | +0.072 [+0.052, +0.094] |
| safety | dev | 537 | AUC | 0.884 | 0.907 | +0.023 [-0.011, +0.055] |
| safety | fresh-catqa-safety | 1,209 | recall@0.5 | 0.917 | 0.997 | +0.079 [+0.060, +0.099] |
| safety | fresh-cdna | 1,180 | AUC | 0.885 | 0.970 | +0.085 [+0.070, +0.102] |
| safety | fresh-indicsafe | 1,992 | AUC | 0.808 | 0.943 | +0.135 [+0.089, +0.185] |
| safety | fresh-linguasafe-multi | 2,000 | AUC | 0.791 | 0.949 | +0.158 [+0.120, +0.198] |
| safety | fresh-linguasafe-sr | 2,000 | AUC | 0.816 | 0.953 | +0.137 [+0.114, +0.160] |
| safety | fresh-turkish-overrefusal | 480 | AUC | 0.953 | 0.998 | +0.045 [+0.026, +0.065] |
| safety | hold-coconot | 897 | AUC | 0.883 | 0.976 | +0.093 [+0.068, +0.120] |
| safety | hold-coconot-contrast | 379 | specificity@0.5 | 0.757 | 0.921 | +0.164 [+0.124, +0.206] |
| safety | hold-jbb | 200 | AUC | 0.866 | 0.969 | +0.103 [+0.063, +0.149] |
| safety | hold-openai-moderation | 1,473 | AUC | 0.871 | 0.945 | +0.074 [+0.060, +0.089] |
| safety | hold-orbench-hard | 1,319 | specificity@0.5 | 0.774 | 0.230 | -0.544 [-0.575, -0.516] |
| safety | hold-toxicchat | 1,657 | AUC | 0.906 | 0.971 | +0.065 [+0.051, +0.080] |
| safety | hold-xstest | 450 | AUC | 0.782 | 0.990 | +0.207 [+0.166, +0.248] |
| safety | test | 1,120 | AUC | 0.912 | 0.932 | +0.020 [+0.003, +0.036] |
| safety | test-aegis2 | 1,749 | AUC | 0.919 | 0.935 | +0.016 [+0.006, +0.026] |
| safety | test-nemotron-v3 | 2,000 | AUC | 0.890 | 0.913 | +0.023 [+0.009, +0.039] |
| safety | test-polyguardprompts | 2,000 | AUC | 0.799 | 0.941 | +0.141 [+0.117, +0.166] |
| fact_check | dev | 198 | AUC | 0.702 | 0.752 | +0.050 [-0.021, +0.126] |
| fact_check | fresh-halueval-wild | 188 | AUC | 0.682 | 0.762 | +0.080 [+0.006, +0.154] |
| fact_check | fresh-mbpp | 961 | specificity@0.5 | 0.999 | 0.992 | -0.007 [-0.013, -0.002] |
| fact_check | fresh-mgsm | 1,991 | specificity@0.5 | 0.913 | 0.000 | -0.913 [-0.935, -0.890] |
| fact_check | fresh-mintaka | 2,000 | recall@0.5 | 0.951 | 1.000 | +0.050 [+0.041, +0.059] |
| fact_check | fresh-truthfulqa | 726 | recall@0.5 | 0.797 | 0.999 | +0.201 [+0.174, +0.231] |
| fact_check | hold-no_robots | 2,000 | AUC | 0.962 | 0.962 | +0.000 [-0.009, +0.011] |
| fact_check | hold-shipped-factcheck | 2,000 | AUC | 0.724 | 0.906 | +0.183 [+0.161, +0.204] |
| fact_check | hold-simpleqa | 2,000 | recall@0.5 | 0.895 | 1.000 | +0.104 [+0.091, +0.118] |
| fact_check | hold-wildbench | 414 | AUC | 0.849 | 0.871 | +0.021 [-0.034, +0.076] |
| fact_check | test | 800 | AUC | 0.741 | 0.808 | +0.067 [+0.029, +0.105] |
| modality | dev | 1,500 | AUC | 0.812 | 0.935 | +0.123 [+0.104, +0.144] |
| modality | fresh-emu-edit | 2,000 | recall@0.5 | 0.386 | 0.933 | +0.548 [+0.526, +0.571] |
| modality | fresh-oneig-zh | 1,320 | recall@0.5 | 0.230 | 0.561 | +0.330 [+0.301, +0.359] |
| modality | fresh-text-requests | 743 | specificity@0.5 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| modality | hold-alpaca | 2,000 | specificity@0.5 | 0.964 | 0.986 | +0.022 [+0.013, +0.031] |
| modality | hold-arena-t2i-hard | 210 | recall@0.5 | 0.671 | 0.881 | +0.209 [+0.143, +0.281] |
| modality | hold-gedit-bench | 1,189 | recall@0.5 | 0.330 | 0.828 | +0.499 [+0.465, +0.536] |
| modality | hold-parti | 1,559 | recall@0.5 | 0.331 | 0.425 | +0.094 [+0.068, +0.120] |
| modality | hold-realmix | 3,769 | AUC | 0.892 | 0.973 | +0.081 [+0.072, +0.091] |
| modality | hold-search-arena | 2,000 | specificity@0.5 | 0.958 | 0.985 | +0.026 [+0.018, +0.035] |
| modality | test | 3,000 | AUC | 0.817 | 0.933 | +0.116 [+0.103, +0.129] |
| modality | test-dolly | 2,000 | specificity@0.5 | 0.984 | 1.000 | +0.016 [+0.011, +0.021] |
| modality | test-realedit | 2,000 | recall@0.5 | 0.316 | 0.753 | +0.438 [+0.413, +0.463] |
| pii | dev | 1,506 | AUC | 0.940 | 0.963 | +0.024 [+0.011, +0.037] |
| pii | fresh-btc | 1,342 | AUC | 0.914 | 0.895 | -0.020 [-0.041, +0.003] |
| pii | fresh-pii-trace | 1,709 | AUC | 0.998 | 0.998 | -0.000 [-0.003, +0.003] |
| pii | fresh-ru-pii | 1,842 | AUC | 0.939 | 0.949 | +0.011 [-0.003, +0.023] |
| pii | fresh-uner-ewt | 1,780 | AUC | 0.921 | 0.896 | -0.025 [-0.071, +0.022] |
| pii | hold-kaggle-essays | 1,125 | AUC | 0.989 | 0.991 | +0.003 [-0.003, +0.008] |
| pii | hold-pii-prompts | 2,000 | AUC | 0.946 | 0.981 | +0.035 [+0.026, +0.044] |
| pii | hold-wildchat | 2,000 | specificity@0.5 | 0.716 | 0.846 | +0.130 [+0.114, +0.146] |
| pii | test | 3,000 | AUC | 0.947 | 0.960 | +0.013 [+0.003, +0.023] |
| pii | test-abcd | 1,486 | AUC | 0.993 | 0.897 | -0.095 [-0.113, -0.077] |
| pii | test-mapa | 1,679 | AUC | 0.918 | 0.786 | -0.132 [-0.166, -0.102] |
| pii | test-tab | 1,461 | AUC | 0.958 | 0.977 | +0.019 [+0.008, +0.031] |
| feedback | dev | 1,199 | accuracy | 0.574 | 0.623 | +0.049 [+0.000, +0.094] |
| feedback | fresh-crosswoz | 1,995 | accuracy | 0.766 | 0.371 | -0.396 [-0.423, -0.369] |
| feedback | hold-shipped-feedback | 1,715 | accuracy | 0.311 | 0.526 | +0.215 [+0.191, +0.240] |
| feedback | test | 2,842 | accuracy | 0.530 | 0.651 | +0.121 [+0.094, +0.149] |
| feedback | test-sgd | 1,998 | accuracy | 0.688 | 0.783 | +0.095 [+0.069, +0.120] |
| hallucination | dev | 1,210 | AUC | 0.786 | 0.787 | +0.001 [-0.023, +0.024] |
| hallucination | fresh-faithbench | 523 | AUC | 0.646 | 0.682 | +0.036 [-0.019, +0.097] |
| hallucination | fresh-shroom | 749 | AUC | 0.726 | 0.857 | +0.131 [+0.087, +0.175] |
| hallucination | hold-attributionbench-ood | 1,390 | AUC | 0.780 | 0.835 | +0.055 [+0.035, +0.075] |
| hallucination | hold-hallumix | 2,000 | AUC | 0.755 | 0.898 | +0.143 [+0.122, +0.164] |
| hallucination | hold-halubench | 1,877 | AUC | 0.527 | 0.757 | +0.231 [+0.202, +0.260] |
| hallucination | hold-summedits | 578 | AUC | 0.823 | 0.908 | +0.085 [+0.056, +0.115] |
| hallucination | test | 3,000 | AUC | 0.774 | 0.796 | +0.022 [+0.008, +0.037] |
| hallucination | test-psiloqa | 1,265 | AUC | 0.884 | 0.872 | -0.011 [-0.030, +0.007] |
| hallucination | test-ragbench | 1,311 | AUC | 0.605 | 0.843 | +0.238 [+0.203, +0.275] |
| hallucination | test-ragtruth | 1,528 | AUC | 0.864 | 0.906 | +0.042 [+0.022, +0.062] |

</details>

## One call per request

The recording runtime logged every exchange of the A/B. Every request that
reached a size's deployment did so as one `/v1/bundle` with one decisions
task: six questions, or seven after an assistant turn. The few tasks with
fewer questions are the routing previews of rows whose text the Router reads
for fewer signals.

| Size | Requests | One decisions task | Questions per task |
| --- | ---: | ---: | --- |
| 0.8B | 118,717 | 118,717 | six 108,962, seven 9,749, fewer 6 |
| 4B | 118,718 | 118,718 | six 108,963, seven 9,749, fewer 6 |
| 9B | 118,723 | 118,723 | six 108,958, seven 9,749, fewer 16 |

## Thresholds

`router_signal_ab.py calibrate` maps each Vela 1.0 threshold to the size's
threshold that keeps its operating point on the suite's `dev` split, as for
the 0.3B: a binary signal keeps its false-positive rate, a confidence floor its
share of rows below it. The module defaults map Vela 1.0's 0.5 (prompt guard),
0.5 (domain), 0.9 (PII), 0.95 (fact check) and 0.7 (user feedback):

| Size | Prompt guard | Domain | PII | Fact check | User feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vela 1.0 specialists | 0.5 | 0.5 | 0.9 | 0.95 | 0.7 |
| 0.3B | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 |
| 0.8B | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 |
| 4B | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 |
| 9B | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 |

- **A module that sets no threshold** takes those of the model it runs, so
  switching the decision model switches them; a module on any other model keeps
  Vela 1.0's. A threshold the configuration sets stays.
- **Fact check on the larger sizes:** their scores pile up near 1, so two
  decimals rounded the 4B's and 9B's points to 1.00, where the signal flags
  nothing. The tool now keeps as many decimals (up to four) as the operating
  point needs, so these thresholds carry three or four.
- **PII:** as for the 0.3B, every Vela 1.0 threshold maps to one value, since
  the span head keeps only the spans it is confident in and flags fewer dev
  negatives at any threshold than Vela 1.0 does at 0.9.
- **Signal rule thresholds** are the configuration's own, and the maintained
  recipes set the 0.3B's. They stay when the decision model changes: a recipe
  that runs another size should take that size's values from the tables below,
  such as `mom-v1`'s `prompt_attack` 0.75 → 0.71 (0.8B), 0.63 (4B), 0.42 (9B).

Every point of each size (Vela 1.0 threshold → the size's):

**0.8B** (Vela 1.0 threshold → 0.8B; kept rate, then the true-positive rate or balanced accuracy on dev and test, Vela 1.0 / 0.8B):

| signal | Vela 1.0 threshold | Vela 2.0 0.8B threshold | kept rate (Vela 1.0 / Vela 2.0 0.8B) | dev Vela 1.0 / Vela 2.0 0.8B | test Vela 1.0 / Vela 2.0 0.8B |
| --- | ---: | ---: | --- | --- | --- |
| jailbreak | 0.3 | 0.69 | false positive rate 0.062 / 0.062 | 0.688 / 0.828 | 0.606 / 0.669 |
| jailbreak | 0.45 | 0.71 | false positive rate 0.059 / 0.059 | 0.625 / 0.797 | 0.577 / 0.659 |
| jailbreak | 0.5 | 0.71 | false positive rate 0.059 / 0.059 | 0.609 / 0.797 | 0.564 / 0.659 |
| jailbreak | 0.6 | 0.71 | false positive rate 0.059 / 0.059 | 0.609 / 0.797 | 0.545 / 0.659 |
| jailbreak | 0.7 | 0.71 | false positive rate 0.059 / 0.059 | 0.578 / 0.797 | 0.518 / 0.659 |
| jailbreak | 0.8 | 0.72 | false positive rate 0.055 / 0.051 | 0.578 / 0.766 | 0.473 / 0.656 |
| jailbreak | 0.85 | 0.73 | false positive rate 0.051 / 0.051 | 0.547 / 0.750 | 0.463 / 0.656 |
| jailbreak | 0.9 | 0.74 | false positive rate 0.047 / 0.047 | 0.531 / 0.750 | 0.423 / 0.640 |
| pii | 0.4 | 0.07 | false positive rate 0.471 / 0.217 | 0.983 / 0.892 | 0.981 / 0.779 |
| pii | 0.5 | 0.07 | false positive rate 0.451 / 0.217 | 0.983 / 0.892 | 0.978 / 0.779 |
| pii | 0.6 | 0.07 | false positive rate 0.426 / 0.217 | 0.980 / 0.892 | 0.970 / 0.779 |
| pii | 0.7 | 0.07 | false positive rate 0.389 / 0.217 | 0.975 / 0.892 | 0.962 / 0.779 |
| pii | 0.85 | 0.07 | false positive rate 0.300 / 0.217 | 0.955 / 0.892 | 0.939 / 0.779 |
| pii | 0.9 | 0.07 | false positive rate 0.242 / 0.217 | 0.927 / 0.892 | 0.905 / 0.779 |
| safety | 0.5 | 0.51 | false positive rate 0.226 / 0.233 | 0.869 / 0.790 | 0.810 / 0.768 |
| fact_check | 0.65 | 0.95 | false positive rate 0.515 / 0.515 | 0.889 / 0.919 | 0.828 / 0.897 |
| fact_check | 0.85 | 0.99 | false positive rate 0.444 / 0.444 | 0.869 / 0.788 | 0.792 / 0.812 |
| fact_check | 0.95 | 0.99 | false positive rate 0.414 / 0.414 | 0.808 / 0.748 | 0.743 / 0.777 |
| domain | 0.5 | 0.38 | below 0.049 / 0.050 | 0.614 / 0.710 | 0.563 / 0.673 |
| feedback | 0.5 | 0.27 | below 0.001 / 0.002 | 0.563 / 0.624 | 0.558 / 0.680 |
| feedback | 0.7 | 0.34 | below 0.025 / 0.030 | 0.556 / 0.630 | 0.555 / 0.686 |
| modality | 0.5 | 0.36 | below 0.001 / 0.001 | 0.716 / 0.781 | 0.677 / 0.742 |
| modality | 0.6 | 0.48 | below 0.018 / 0.017 | 0.719 / 0.783 | 0.678 / 0.744 |
| modality | 0.7 | 0.51 | below 0.034 / 0.031 | 0.720 / 0.785 | 0.677 / 0.745 |

**4B** (Vela 1.0 threshold → 4B; kept rate, then the true-positive rate or balanced accuracy on dev and test, Vela 1.0 / 4B):

| signal | Vela 1.0 threshold | Vela 2.0 4B threshold | kept rate (Vela 1.0 / Vela 2.0 4B) | dev Vela 1.0 / Vela 2.0 4B | test Vela 1.0 / Vela 2.0 4B |
| --- | ---: | ---: | --- | --- | --- |
| jailbreak | 0.3 | 0.61 | false positive rate 0.062 / 0.062 | 0.688 / 0.844 | 0.606 / 0.765 |
| jailbreak | 0.45 | 0.63 | false positive rate 0.059 / 0.055 | 0.625 / 0.844 | 0.577 / 0.762 |
| jailbreak | 0.5 | 0.63 | false positive rate 0.059 / 0.055 | 0.609 / 0.844 | 0.564 / 0.762 |
| jailbreak | 0.6 | 0.63 | false positive rate 0.059 / 0.055 | 0.609 / 0.844 | 0.545 / 0.762 |
| jailbreak | 0.7 | 0.63 | false positive rate 0.059 / 0.055 | 0.578 / 0.844 | 0.518 / 0.762 |
| jailbreak | 0.8 | 0.67 | false positive rate 0.055 / 0.055 | 0.578 / 0.828 | 0.473 / 0.754 |
| jailbreak | 0.85 | 0.75 | false positive rate 0.051 / 0.051 | 0.547 / 0.781 | 0.463 / 0.725 |
| jailbreak | 0.9 | 0.81 | false positive rate 0.047 / 0.047 | 0.531 / 0.750 | 0.423 / 0.704 |
| pii | 0.4 | 0.05 | false positive rate 0.471 / 0.159 | 0.983 / 0.944 | 0.981 / 0.852 |
| pii | 0.5 | 0.05 | false positive rate 0.451 / 0.159 | 0.983 / 0.944 | 0.978 / 0.852 |
| pii | 0.6 | 0.05 | false positive rate 0.426 / 0.159 | 0.980 / 0.944 | 0.970 / 0.852 |
| pii | 0.7 | 0.05 | false positive rate 0.389 / 0.159 | 0.975 / 0.944 | 0.962 / 0.852 |
| pii | 0.85 | 0.05 | false positive rate 0.300 / 0.159 | 0.955 / 0.944 | 0.939 / 0.852 |
| pii | 0.9 | 0.05 | false positive rate 0.242 / 0.159 | 0.927 / 0.944 | 0.905 / 0.852 |
| safety | 0.5 | 0.47 | false positive rate 0.226 / 0.226 | 0.869 / 0.876 | 0.810 / 0.874 |
| fact_check | 0.65 | 0.99 | false positive rate 0.515 / 0.515 | 0.889 / 0.828 | 0.828 / 0.900 |
| fact_check | 0.85 | 1.00 | false positive rate 0.444 / 0.455 | 0.869 / 0.788 | 0.792 / 0.828 |
| fact_check | 0.95 | 1.00 | false positive rate 0.414 / 0.414 | 0.808 / 0.758 | 0.743 / 0.797 |
| domain | 0.5 | 0.45 | below 0.049 / 0.049 | 0.614 / 0.765 | 0.563 / 0.734 |
| feedback | 0.5 | 0.26 | below 0.001 / 0.001 | 0.563 / 0.597 | 0.558 / 0.617 |
| feedback | 0.7 | 0.33 | below 0.025 / 0.026 | 0.556 / 0.603 | 0.555 / 0.629 |
| modality | 0.5 | 0.45 | below 0.001 / 0.001 | 0.716 / 0.860 | 0.677 / 0.854 |
| modality | 0.6 | 0.55 | below 0.018 / 0.017 | 0.719 / 0.862 | 0.678 / 0.860 |
| modality | 0.7 | 0.61 | below 0.034 / 0.035 | 0.720 / 0.867 | 0.677 / 0.867 |

**9B** (Vela 1.0 threshold → 9B; kept rate, then the true-positive rate or balanced accuracy on dev and test, Vela 1.0 / 9B):

| signal | Vela 1.0 threshold | Vela 2.0 9B threshold | kept rate (Vela 1.0 / Vela 2.0 9B) | dev Vela 1.0 / Vela 2.0 9B | test Vela 1.0 / Vela 2.0 9B |
| --- | ---: | ---: | --- | --- | --- |
| jailbreak | 0.3 | 0.41 | false positive rate 0.062 / 0.062 | 0.688 / 0.750 | 0.606 / 0.788 |
| jailbreak | 0.45 | 0.42 | false positive rate 0.059 / 0.059 | 0.625 / 0.750 | 0.577 / 0.788 |
| jailbreak | 0.5 | 0.42 | false positive rate 0.059 / 0.059 | 0.609 / 0.750 | 0.564 / 0.788 |
| jailbreak | 0.6 | 0.42 | false positive rate 0.059 / 0.059 | 0.609 / 0.750 | 0.545 / 0.788 |
| jailbreak | 0.7 | 0.42 | false positive rate 0.059 / 0.059 | 0.578 / 0.750 | 0.518 / 0.788 |
| jailbreak | 0.8 | 0.46 | false positive rate 0.055 / 0.055 | 0.578 / 0.750 | 0.473 / 0.770 |
| jailbreak | 0.85 | 0.52 | false positive rate 0.051 / 0.051 | 0.547 / 0.719 | 0.463 / 0.741 |
| jailbreak | 0.9 | 0.54 | false positive rate 0.047 / 0.047 | 0.531 / 0.703 | 0.423 / 0.733 |
| pii | 0.4 | 0.14 | false positive rate 0.471 / 0.139 | 0.983 / 0.963 | 0.981 / 0.849 |
| pii | 0.5 | 0.14 | false positive rate 0.451 / 0.139 | 0.983 / 0.963 | 0.978 / 0.849 |
| pii | 0.6 | 0.14 | false positive rate 0.426 / 0.139 | 0.980 / 0.963 | 0.970 / 0.849 |
| pii | 0.7 | 0.14 | false positive rate 0.389 / 0.139 | 0.975 / 0.963 | 0.962 / 0.849 |
| pii | 0.85 | 0.14 | false positive rate 0.300 / 0.139 | 0.955 / 0.963 | 0.939 / 0.849 |
| pii | 0.9 | 0.14 | false positive rate 0.242 / 0.139 | 0.927 / 0.963 | 0.905 / 0.849 |
| safety | 0.5 | 0.56 | false positive rate 0.226 / 0.222 | 0.869 / 0.869 | 0.810 / 0.872 |
| fact_check | 0.65 | 0.98 | false positive rate 0.515 / 0.515 | 0.889 / 0.869 | 0.828 / 0.907 |
| fact_check | 0.85 | 0.99 | false positive rate 0.444 / 0.444 | 0.869 / 0.828 | 0.792 / 0.863 |
| fact_check | 0.95 | 1.00 | false positive rate 0.414 / 0.414 | 0.808 / 0.758 | 0.743 / 0.780 |
| domain | 0.5 | 0.46 | below 0.049 / 0.050 | 0.614 / 0.768 | 0.563 / 0.755 |
| feedback | 0.5 | 0.28 | below 0.001 / 0.001 | 0.563 / 0.672 | 0.558 / 0.725 |
| feedback | 0.7 | 0.35 | below 0.025 / 0.024 | 0.556 / 0.679 | 0.555 / 0.728 |
| modality | 0.5 | 0.49 | below 0.001 / 0.000 | 0.716 / 0.816 | 0.677 / 0.850 |
| modality | 0.6 | 0.55 | below 0.018 / 0.016 | 0.719 / 0.821 | 0.678 / 0.854 |
| modality | 0.7 | 0.59 | below 0.034 / 0.033 | 0.720 / 0.823 | 0.677 / 0.858 |

## Latency

The [router-latency record](router-latency-cpu.md)'s method:
`POST /api/v1/routing/preview` with its configuration's five model-backed
request signals (domain, prompt guard, PII, fact check, feedback), result
caches off on both layers, 20 warm-up requests, then each pass three times
over the inputs. The `vela2` arm of `router_signal_ab.py config --set latency`
with `--decision-model`.

- **GPU:** the 539 inputs of `tools/router_latency.py corpus`, sequentially and
  at concurrency 4 and 16. The four sizes ran at once, each with its own GPU
  and 12 cores, for {"0_3b": 3, "0_8b": 3, "4b": 3, "9b": 3} rounds.
- **CPU:** every 8th input (68), once sequentially and once at concurrency 4
  per round after 5 warm-up requests, since the 0.8B takes seconds per request
  on a CPU. The 0.3B (`max_speed`) and the 0.8B (`exact`) ran at once on 12
  cores each of one NUMA node, for {"0_3b": 2, "0_8b": 2} rounds.
- **Statistics:** the median of the rounds per metric, and the paired
  difference against the 0.3B per round with a 95% t interval.

GPU (`exact`):

| Pass | Metric | 0.3B | 0.8B | 4B | 9B |
| --- | --- | ---: | ---: | ---: | ---: |
| Sequential | p50 (ms) | 6.9 | 40.7 | 56.8 | 79.0 |
| Sequential | p95 (ms) | 8.4 | 46.2 | 68.9 | 97.5 |
| Sequential | requests per second | 139.8 | 24.1 | 16.9 | 12.1 |
| Concurrency 4 | p50 (ms) | 26.3 | 158.2 | 231.2 | 316.4 |
| Concurrency 4 | p95 (ms) | 32.4 | 184.1 | 260.2 | 359.5 |
| Concurrency 4 | requests per second | 148.3 | 24.8 | 17.0 | 12.3 |
| Concurrency 16 | p50 (ms) | 107.3 | 634.0 | 928.1 | 1,316.1 |
| Concurrency 16 | p95 (ms) | 123.4 | 652.5 | 1,024.2 | 1,461.7 |
| Concurrency 16 | requests per second | 146.3 | 25.1 | 17.1 | 12.1 |

Each size − the 0.3B on a GPU:

| Pass | Metric | 0.8B − 0.3B | 4B − 0.3B | 9B − 0.3B |
| --- | --- | ---: | ---: | ---: |
| Sequential | p50 (ms) | +34.0 [+32.1, +35.9] | +49.4 [+46.8, +52.1] | +71.7 [+69.4, +74.1] |
| Sequential | p95 (ms) | +39.1 [+29.6, +48.5] | +61.7 [+51.4, +72.1] | +90.4 [+82.0, +98.8] |
| Sequential | requests per second | -115.1 [-123.3, -106.9] | -122.2 [-129.3, -115.0] | -127.0 [-134.3, -119.7] |
| Concurrency 4 | p50 (ms) | +134.1 [+120.5, +147.7] | +214.9 [+156.5, +273.2] | +291.3 [+279.4, +303.1] |
| Concurrency 4 | p95 (ms) | +164.0 [+103.0, +224.9] | +243.1 [+170.2, +316.0] | +333.0 [+300.9, +365.1] |
| Concurrency 4 | requests per second | -124.4 [-134.7, -114.1] | -132.3 [-141.2, -123.5] | -136.5 [-144.6, -128.3] |
| Concurrency 16 | p50 (ms) | +538.8 [+487.0, +590.5] | +867.4 [+615.9, +1,118.9] | +1,199.2 [+1,139.3, +1,259.0] |
| Concurrency 16 | p95 (ms) | +545.4 [+451.4, +639.4] | +962.2 [+697.6, +1,226.7] | +1,330.8 [+1,291.5, +1,370.1] |
| Concurrency 16 | requests per second | -121.0 [-121.4, -120.7] | -129.4 [-134.7, -124.0] | -133.6 [-136.4, -130.7] |

CPU:

| Pass | Metric | 0.3B | 0.8B |
| --- | --- | ---: | ---: |
| Sequential | p50 (ms) | 116.2 | 2,962.5 |
| Sequential | p95 (ms) | 493.3 | 3,875.8 |
| Sequential | requests per second | 6.2 | 0.3 |
| Concurrency 4 | p50 (ms) | 439.0 | 11,179.8 |
| Concurrency 4 | p95 (ms) | 645.9 | 18,314.8 |
| Concurrency 4 | requests per second | 8.9 | 0.3 |

The 0.8B − the 0.3B on a CPU:

| Pass | Metric | 0.8B − 0.3B |
| --- | --- | ---: |
| Sequential | p50 (ms) | +2,846.2 [-1,945.2, +7,637.7] |
| Sequential | p95 (ms) | +3,382.5 [-8,836.7, +15,601.7] |
| Sequential | requests per second | -5.9 [-19.6, +7.8] |
| Concurrency 4 | p50 (ms) | +10,740.8 [+1,876.5, +19,605.0] |
| Concurrency 4 | p95 (ms) | +17,668.9 [+16,001.8, +19,335.9] |
| Concurrency 4 | requests per second | -8.5 [-8.8, -8.3] |

- **Hardware:** the 0.3B and 0.8B run on a CPU or a GPU; the 4B and 9B on a
  GPU only, with about 17 GB and 32 GB of GPU memory for their FP32 weights.
  The Router refuses a 4B or 9B decision model on a host without a GPU, and
  `vllm-sr serve` refuses it with `--platform cpu`.
- **The ROCm image** runs the 0.8B on a GPU only: its PyTorch has no CPU
  LAPACK, which the 0.8B's CPU kernels need, and the runtime says so. The CPU
  image runs it on a CPU.

## Reproduce

As in vela2-router-signals.md, with the arm configurations of each size:

```bash
python3 tools/router_signal_ab.py config --arm vela2 --decision-model Vela-2.0-4B --gpu > vela2-4b.yaml
# The recording runtime as the Router's runtime command, the Router, then the rows:
python3 tools/router_signal_ab.py run --port 18080 --rows rows.jsonl --out 4b/preview.jsonl
python3 tools/router_signal_ab.py join --rows rows.jsonl --run 4b/preview.jsonl --log-dir 4b/exchanges --arm vela2 --out 4b/preds
python3 tools/router_signal_ab.py halu --side vela2 --suite text --out 4b/preds --ports 8100   # a runtime serving Vela-2.0-4B
python3 tools/router_signal_ab.py score --suite text --a vela1/preds --b 4b/preds --b-name "Vela 2.0 4B" --out score.json --md score.md
python3 tools/router_signal_ab.py calibrate --suite text --a vela1/preds --b 4b/preds --b-name "Vela 2.0 4B" \
  --thresholds thresholds.json --out calibration.json --md calibration.md
# Latency:
python3 tools/router_signal_ab.py config --set latency --arm vela2 --decision-model Vela-2.0-4B --gpu > lat-4b.yaml
python3 tools/router_latency.py run --url http://127.0.0.1:18080 --corpus corpus.json --out r1-4b.json --label 4b --warmup 20 --concurrency 1 4 16
```

The numbers above are in [vela2-decision-model-sizes.json](vela2-decision-model-sizes.json).
