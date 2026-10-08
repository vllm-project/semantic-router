# Vela 2.0 0.3B as the default of the Router's built-in signals

[#4639](https://github.com/vllm-project/semantic-router/issues/4639) made
`vllm-sr/Vela-2.0-0.3B` the default model of every built-in Router signal it
answers. The issue's acceptance asked for accuracy level or better on every
signal and end-to-end router latency level or better, measured on a CPU. **The
maintainers chose to switch although the latency condition fails on CPU**, and
although four signals lose accuracy. This record is the measurement behind that
choice, signal by signal and in latency, and the evidence for the thresholds the
switch recalibrated.
[#4668](https://github.com/vllm-project/semantic-router/issues/4668) follows up
on the CPU latency.

- **Accuracy:** through the Router, against the Vela 1.0 specialists on the
  built-in signals' evaluation rows. The 0.3B is ahead on prompt attacks
  (held-out AUC +0.026) and safety (+0.052), and level on PII and hallucination
  held-out. Modality (held-out AUC −0.180) and feedback (accuracy −0.038, fresh
  −0.178) regress most; domain (−0.037) and fact check (held-out AUC −0.101) are
  behind too.
- **Latency:** with `max_speed`, the 0.3B's default CPU profile, a request takes
  79 ms at the median against 16 ms on the Vela 1.0 models, about 4.9 times as
  long, and the Router serves 11.9 against 38.9 requests per second on 12 cores
  (12.8 against 51.8 at concurrency 16).
- **One call per request:** every built-in signal of a request reaches the 0.3B
  in one `/v1/decisions` task, however long the request. On a heavily loaded
  host, 0.4% of requests sent one question in a second call.
- **Restore:** one block brings back every Vela 1.0 specialist, and one line
  any single signal ([below](#restore-the-vela-10-specialists)).
- **Other sizes:** [vela2-decision-model-sizes.md](vela2-decision-model-sizes.md)
  measures the 0.8B, 4B and 9B the same way, as the decision model
  (`global.model_catalog.system.decision_model`).

- **Date:** 2026-10-07.
- **Machine:** AMD EPYC 9575F, CPU only. The accuracy runs and the latency
  rounds ran on two hosts of this model. Each run's processes ran in `systemd`
  scopes on the cores given below, with memory bound to their NUMA node.
- **Commits:** the Router built from this change in CI's builder image
  (`golang:1.25-bookworm`, the image recipe's flags). The Vela 1.0 arm ran on an
  earlier build of it; no later commit changes a request to a Vela 1.0 model.
  The model runtime is `main`'s (`db35009da`), in a Python 3.12 environment
  with PyTorch 2.10.0 (CPU), as the router image pins it. In the runtime this
  change only marks the Vela 2.0 table's repositories public.
- **Models:** the Vela 1.0 specialists at their pinned revisions (Domain
  `f6354f54`, Guard `087f9e40`, Safety `6e70e725`, FactCheck `99ede1ab`,
  Feedback `47434a7f`, Modality `5384b899`, PII `6d3300c4`, Halu `ca875312`)
  and Vela 2.0 0.3B at `a3209a50`.

## The default

With no model configured, the built-in signals run on one implicit deployment
of the 0.3B, `@Vela-2.0-0.3B`:

- **Questions:** domain, prompt guard, safety rules, fact check, feedback and
  modality ask the question the model card gives for each, with the Vela 1.0
  model's labels. PII and hallucination ask the model's `pii` and `halu`
  presets, which its router span head answers.
- **One call:** a request's questions travel in one `/v1/decisions` task: six,
  or seven after an assistant turn.
  - Every model-backed signal joins the request bundle before any starts, and
    every signal the 0.3B answers reads the request as it came, so prompt
    compression does not split them either.
  - Over the accuracy run below, 480 of the rows' 116,430 distinct texts (0.4%)
    still sent one question in a second call. That host ran at a load of about
    120 on 160 cores, and a signal that started more than the bundle's 2 ms
    window late missed it.
- **Whole text:** a signal on the 0.3B reads the whole text up to the model's
  8,192 tokens. The Router no longer samples or chunks it as it does for the
  sequence classifiers. Beyond 8,192 tokens the model truncates, so a prompt
  attack placed after that point goes unseen; Vela 1.0 Guard scanned up to 32K
  tokens.
- **Profile:** on CPU the deployment runs `max_speed`, which loads the 0.3B's
  consented `float32-packed` copy. [vela2-parity.md](vela2-parity.md) shows it
  keeps every Choice, Score and Set decision and 463 of 464 spans, with
  answers within about 1e-5 of `exact`. On a GPU (`use_cpu: false`) the
  deployment is `@Vela-2.0-0.3B/auto` on `exact`, the faster profile there for
  a single request ([vela2-performance.md](vela2-performance.md)).
- **What stays on Vela 1.0:** Hazard (the card publishes no trained question
  for it), the embedding, Omni and reranker models (Vela 2.0 has no such exit),
  and every module or binding a configuration pins.

**`max_speed` against `exact`.** Replayed on fresh runtimes, 60 recorded
requests of the accuracy run (six questions each) got the same answers under
`max_speed` as under `exact` to within 5e-6, and to within 3e-6 when two ran at
once. The accuracy arm below runs the defaults, so it measures `max_speed`.

## Latency

The [router-latency record](router-latency-cpu.md)'s method and corpus: its
configuration's five model-backed request signals (domain, prompt guard, PII,
fact check, feedback), the 539 inputs of `tools/router_latency.py corpus`,
`POST /api/v1/routing/preview`, result caches off on both layers.

- **Arms:**
  - `vela2` is that configuration with no model catalog, the new defaults;
  - `vela1` adds the one-block restore of the Vela 1.0 specialists
    (`global.model_catalog.system`).
- **Passes:** each sends 20 warm-up requests, then the corpus three times:
  sequentially, then at concurrency 4 and 16.
- **Rounds:** five, alternating the arms with the order rotating each round.
  Each Router and its runtime processes run on cores 32–43 (12 vCPUs), the
  driver on 4 others.
- **Statistics:** per metric, the median of the five rounds and the paired
  difference `vela2 − vela1` per round with a 95% t interval
  (`router_latency.py rounds`). Raw rounds: `vela2-router-signals.json`.

| Pass | Metric | Vela 1.0 | 0.3B | 0.3B − Vela 1.0 [95% CI] |
| --- | --- | ---: | ---: | ---: |
| Sequential | p50 (ms) | 16.2 | 79.5 | +63.17 [+62.58, +63.76] |
| Sequential | p95 (ms) | 57.5 | 100.3 | +42.82 [+42.35, +43.29] |
| Sequential | p99 (ms) | 116.9 | 199.0 | +79.45 [+65.84, +93.05] |
| Sequential | Requests per second | 38.9 | 11.9 | −26.95 [−27.16, −26.74] |
| Concurrency 4 | p50 (ms) | 58.0 | 313.2 | +254.59 [+251.23, +257.95] |
| Concurrency 4 | p95 (ms) | 230.6 | 370.9 | +146.66 [+125.52, +167.80] |
| Concurrency 4 | p99 (ms) | 318.1 | 473.9 | +163.93 [+141.65, +186.20] |
| Concurrency 4 | Requests per second | 49.4 | 12.5 | −36.90 [−37.40, −36.41] |
| Concurrency 16 | p50 (ms) | 283.3 | 1,240.5 | +954.02 [+941.86, +966.19] |
| Concurrency 16 | p95 (ms) | 621.4 | 1,388.9 | +740.71 [+695.49, +785.94] |
| Concurrency 16 | p99 (ms) | 1,120.1 | 1,450.5 | +334.81 [+287.39, +382.23] |
| Concurrency 16 | Requests per second | 51.8 | 12.8 | −38.76 [−39.69, −37.83] |

The Router makes the same routing decision on 269 of 539 inputs
in both arms; the signals' models differ, so their verdicts do.

**Why the 0.3B is slower.** Every request carries the questions, their options
and the 17 PII labels. That is at least 560 tokens of schema ahead of a prompt
whose median is about 15 tokens, through one 307M-parameter forward. Each
Vela 1.0 model is an encoder of the same size, but it reads only the prompt,
and the Router runs the models in separate runtime processes at once. On the
`exact` profile the 0.3B took 132 ms at the median against Vela 1.0's 16 ms;
`max_speed` brings that to 79 ms.

## Accuracy

The evaluation rows are
[`vllm-sr/router-signal-suite`](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
at revision `fa08b2a6`: every dev, test, source test, held-out and fresh
held-out file of the eight signals whose text the suite publishes.

- **Metrics:** each file's headline, as the suite scores it:
  - AUC of the positive class for jailbreak, safety, fact check, PII (any
    sensitive identifier in the request), modality (an image to generate) and
    hallucination;
  - accuracy for domain (14 subjects) and feedback (5 types).
- **Intervals:** group bootstraps with 2,000 replicates. Rows that share a
  source group are resampled together, with the same draws for both arms, so
  the differences are paired. A signal's mean combines one draw of every file
  in its set.

- **Arms:** `tools/router_signal_ab.py config` writes both, with the same
  signals and decisions.
  - `vela1`: the Vela 1.0 specialists, as the restore block names them, with
    their default thresholds. The arm names Vela 1.0 Modality, as
    `config/config.yaml` did.
  - `vela2`: the Router's defaults. No model is configured, and the modality
    classifier has no `model_path`, so every signal runs on the implicit
    `@Vela-2.0-0.3B` deployment under `max_speed`.
- **Request-time signals, through the Router:** each arm's Router answered every
  row through `POST /api/v1/routing/preview`, a feedback row after an assistant
  turn. The Router's managed model runtime was `tools/router_signal_ab.py
  record`, which serves the Router's own socket and logs every exchange. The
  probabilities each signal read are those the runtime returned to the Router.
  Vela 1.0 Guard's score is its riskiest window, as the Router reads it.
- **Hallucination, at response time:** the routing preview never reaches it, so
  each row is asked as the Router's detector asks it (`router_signal_ab.py
  halu`).
  - Vela 1.0 Halu reads the grounded `{context, question, answer}` item, with
    `reject` at 8,192 tokens and threshold 0.5. Its score is the answer's
    highest token probability. Its values equal the ones the suite card
    publishes for this model on every file it lists.
  - The 0.3B reads `{request, context, answer}` with the `halu` preset. Its
    score is the answer's hallucination probability.

Means per signal (`vela2 − vela1`, 95% interval). *Held out* is the suite card's
set of corpora no model trains on; *fresh* is the suite's fresh held-out set,
built after both models' training data; *in distribution* is the suite's own
test and the training corpora's published tests:

| Signal (metric) | Set | Files | Vela 1.0 | 0.3B | 0.3B − Vela 1.0 [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: |
| Domain (accuracy) | held out | 5 | 0.638 | 0.600 | −0.037 [−0.048, −0.028] |
|  | fresh | 3 | 0.561 | 0.473 | −0.088 [−0.101, −0.074] |
|  | in distribution | 2 | 0.561 | 0.541 | −0.020 [−0.034, −0.006] |
| Prompt guard (AUC) | held out | 4 | 0.828 | 0.854 | +0.026 [+0.011, +0.043] |
|  | fresh | 2 | 0.755 | 0.753 | −0.002 [−0.025, +0.022] |
|  | in distribution | 1 | 0.847 | 0.853 | +0.006 [−0.015, +0.027] |
| Safety (AUC) | held out | 4 | 0.856 | 0.908 | +0.052 [+0.038, +0.066] |
|  | fresh | 5 | 0.851 | 0.869 | +0.018 [+0.004, +0.033] |
|  | in distribution | 4 | 0.880 | 0.914 | +0.034 [+0.025, +0.043] |
| Fact check (AUC) | held out | 2 | 0.905 | 0.804 | −0.101 [−0.138, −0.066] |
|  | fresh | 1 | 0.682 | 0.702 | +0.020 [−0.075, +0.117] |
|  | in distribution | 1 | 0.741 | 0.753 | +0.011 [−0.032, +0.053] |
| Modality (AUC) | held out | 1 | 0.892 | 0.713 | −0.180 [−0.198, −0.164] |
|  | in distribution | 1 | 0.817 | 0.708 | −0.109 [−0.128, −0.090] |
| PII (AUC) | held out | 2 | 0.968 | 0.972 | +0.004 [−0.007, +0.014] |
|  | fresh | 4 | 0.943 | 0.938 | −0.005 [−0.013, +0.004] |
|  | in distribution | 4 | 0.954 | 0.924 | −0.029 [−0.034, −0.019] |
| Feedback (accuracy) | held out | 1 | 0.311 | 0.272 | −0.038 [−0.058, −0.018] |
|  | fresh | 1 | 0.766 | 0.588 | −0.178 [−0.206, −0.150] |
|  | in distribution | 2 | 0.609 | 0.595 | −0.014 [−0.028, +0.001] |
| Hallucination (AUC) | held out | 3 | 0.702 | 0.712 | +0.010 [−0.005, +0.026] |
|  | fresh | 2 | 0.686 | 0.689 | +0.003 [−0.032, +0.035] |
|  | in distribution | 4 | 0.782 | 0.757 | −0.025 [−0.035, −0.015] |

- **Ahead:**
  - **Prompt guard:** ahead on held-out attacks, level elsewhere. At its
    matched threshold it catches far more of the attacks the suite's dev and
    test files hold (below).
  - **Safety:** ahead on every set.
- **Level:**
  - **PII:** level on held-out and fresh files, behind in distribution.
  - **Hallucination:** level on held-out and fresh files, behind in
    distribution, where Vela 1.0 Halu trained on RAGTruth's and PsiloQA's
    training splits.
- **Behind:**
  - **Domain:** behind on every set; on fresh-arabicmmlu −0.190.
  - **Feedback:** behind on held-out and fresh files.
  - **Modality:** behind on both sets. It misses most image-generation
    requests at a 0.5 threshold: hold-parti recall 0.056 against 0.331. The
    question is the card's trained wording, asked with the request's other
    questions in one call.
  - **Fact check:** behind on its two held-out files, which the suite card
    notes give the label away without the text. In hold-no_robots, length
    alone separates the classes. At 0.5 the 0.3B also marks every math word
    problem in fresh-mgsm as needing a fact check.
- **Measured here first:** the model card has no published comparison for
  modality, feedback or fact check, so these are their first measurements.

Per file:

| Signal | File | Rows | Metric | Vela 1.0 | 0.3B | 0.3B − Vela 1.0 [95% CI] |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| domain | dev | 1,498 | accuracy | 0.622 | 0.585 | −0.037 [−0.061, −0.013] |
| domain | fresh-arabicmmlu | 1,999 | accuracy | 0.524 | 0.333 | −0.191 [−0.214, −0.168] |
| domain | fresh-ceval | 1,999 | accuracy | 0.637 | 0.650 | +0.013 [−0.009, +0.034] |
| domain | fresh-indommlu | 1,999 | accuracy | 0.521 | 0.435 | −0.086 [−0.108, −0.063] |
| domain | hold-arena-expert | 987 | accuracy | 0.690 | 0.634 | −0.056 [−0.086, −0.025] |
| domain | hold-mmlu-cf | 1,988 | accuracy | 0.529 | 0.495 | −0.033 [−0.053, −0.013] |
| domain | hold-mmlu-pro | 1,988 | accuracy | 0.725 | 0.660 | −0.064 [−0.083, −0.046] |
| domain | hold-mmlu-prox | 1,988 | accuracy | 0.659 | 0.628 | −0.031 [−0.050, −0.012] |
| domain | hold-supergpqa | 1,882 | accuracy | 0.587 | 0.583 | −0.003 [−0.023, +0.017] |
| domain | test | 2,996 | accuracy | 0.606 | 0.574 | −0.031 [−0.049, −0.014] |
| domain | test-exams | 1,927 | accuracy | 0.516 | 0.507 | −0.009 [−0.030, +0.012] |
| prompt guard | dev | 320 | AUC | 0.912 | 0.964 | +0.051 [+0.015, +0.095] |
| prompt guard | fresh-cyberseceval-indirect | 245 | recall@0.5 | 0.061 | 0.825 | +0.763 [+0.710, +0.816] |
| prompt guard | fresh-guardrail-hn | 244 | AUC | 0.866 | 0.933 | +0.067 [+0.022, +0.111] |
| prompt guard | fresh-ipi-arena | 71 | recall@0.5 | 0.789 | 1.000 | +0.211 [+0.113, +0.310] |
| prompt guard | fresh-sep | 2,000 | AUC | 0.643 | 0.573 | −0.071 [−0.086, −0.056] |
| prompt guard | hold-bipia | 637 | AUC | 0.664 | 0.708 | +0.044 [+0.010, +0.080] |
| prompt guard | hold-jailbreakhub-late | 1,303 | AUC | 0.652 | 0.731 | +0.079 [+0.049, +0.107] |
| prompt guard | hold-llmail | 1,152 | AUC | 0.972 | 1.000 | +0.028 [+0.019, +0.039] |
| prompt guard | hold-notinject | 339 | specificity@0.5 | 0.935 | 0.844 | −0.091 [−0.130, −0.053] |
| prompt guard | hold-promptshield-test | 2,000 | AUC | 0.763 | 0.794 | +0.031 [−0.008, +0.083] |
| prompt guard | hold-toxicchat | 1,152 | AUC | 0.915 | 0.915 | +0.001 [−0.019, +0.021] |
| prompt guard | test | 1,878 | AUC | 0.847 | 0.853 | +0.006 [−0.015, +0.027] |
| safety | dev | 537 | AUC | 0.884 | 0.917 | +0.033 [+0.003, +0.063] |
| safety | fresh-catqa-safety | 1,209 | recall@0.5 | 0.917 | 0.898 | −0.019 [−0.042, +0.003] |
| safety | fresh-cdna | 1,180 | AUC | 0.885 | 0.914 | +0.029 [+0.015, +0.046] |
| safety | fresh-indicsafe | 1,992 | AUC | 0.808 | 0.781 | −0.027 [−0.078, +0.023] |
| safety | fresh-linguasafe-multi | 2,000 | AUC | 0.791 | 0.855 | +0.064 [+0.028, +0.103] |
| safety | fresh-linguasafe-sr | 2,000 | AUC | 0.816 | 0.837 | +0.021 [−0.003, +0.044] |
| safety | fresh-turkish-overrefusal | 480 | AUC | 0.953 | 0.956 | +0.004 [−0.017, +0.025] |
| safety | hold-coconot | 897 | AUC | 0.883 | 0.955 | +0.072 [+0.049, +0.098] |
| safety | hold-coconot-contrast | 379 | specificity@0.5 | 0.757 | 0.834 | +0.076 [+0.040, +0.116] |
| safety | hold-jbb | 200 | AUC | 0.866 | 0.931 | +0.065 [+0.022, +0.108] |
| safety | hold-openai-moderation | 1,473 | AUC | 0.871 | 0.900 | +0.029 [+0.016, +0.045] |
| safety | hold-orbench-hard | 1,319 | specificity@0.5 | 0.774 | 0.434 | −0.340 [−0.373, −0.307] |
| safety | hold-toxicchat | 1,657 | AUC | 0.906 | 0.954 | +0.047 [+0.033, +0.061] |
| safety | hold-xstest | 450 | AUC | 0.782 | 0.849 | +0.066 [+0.035, +0.097] |
| safety | test | 1,120 | AUC | 0.912 | 0.931 | +0.018 [+0.003, +0.035] |
| safety | test-aegis2 | 1,749 | AUC | 0.919 | 0.931 | +0.012 [+0.003, +0.021] |
| safety | test-nemotron-v3 | 2,000 | AUC | 0.890 | 0.895 | +0.005 [−0.008, +0.019] |
| safety | test-polyguardprompts | 2,000 | AUC | 0.799 | 0.900 | +0.101 [+0.075, +0.128] |
| fact check | dev | 198 | AUC | 0.702 | 0.699 | −0.003 [−0.085, +0.084] |
| fact check | fresh-halueval-wild | 188 | AUC | 0.682 | 0.702 | +0.020 [−0.075, +0.117] |
| fact check | fresh-mbpp | 961 | specificity@0.5 | 0.999 | 0.941 | −0.058 [−0.075, −0.044] |
| fact check | fresh-mgsm | 1,991 | specificity@0.5 | 0.913 | 0.000 | −0.913 [−0.935, −0.890] |
| fact check | fresh-mintaka | 2,000 | recall@0.5 | 0.951 | 1.000 | +0.049 [+0.040, +0.059] |
| fact check | fresh-truthfulqa | 726 | recall@0.5 | 0.797 | 0.996 | +0.198 [+0.172, +0.229] |
| fact check | hold-no_robots | 2,000 | AUC | 0.962 | 0.833 | −0.129 [−0.148, −0.109] |
| fact check | hold-shipped-factcheck | 2,000 | AUC | 0.724 | 0.889 | +0.165 [+0.142, +0.187] |
| fact check | hold-simpleqa | 2,000 | recall@0.5 | 0.895 | 1.000 | +0.104 [+0.090, +0.118] |
| fact check | hold-wildbench | 414 | AUC | 0.849 | 0.776 | −0.074 [−0.143, −0.005] |
| fact check | test | 800 | AUC | 0.741 | 0.753 | +0.011 [−0.032, +0.053] |
| modality | dev | 1,500 | AUC | 0.812 | 0.701 | −0.110 [−0.141, −0.082] |
| modality | fresh-emu-edit | 2,000 | recall@0.5 | 0.386 | 0.102 | −0.284 [−0.305, −0.264] |
| modality | fresh-oneig-zh | 1,320 | recall@0.5 | 0.230 | 0.239 | +0.008 [−0.020, +0.035] |
| modality | fresh-text-requests | 743 | specificity@0.5 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| modality | hold-alpaca | 2,000 | specificity@0.5 | 0.964 | 0.994 | +0.030 [+0.021, +0.038] |
| modality | hold-arena-t2i-hard | 210 | recall@0.5 | 0.671 | 0.233 | −0.438 [−0.519, −0.357] |
| modality | hold-gedit-bench | 1,189 | recall@0.5 | 0.330 | 0.137 | −0.193 [−0.222, −0.162] |
| modality | hold-parti | 1,559 | recall@0.5 | 0.331 | 0.056 | −0.275 [−0.298, −0.253] |
| modality | hold-realmix | 3,769 | AUC | 0.892 | 0.713 | −0.180 [−0.198, −0.164] |
| modality | hold-search-arena | 2,000 | specificity@0.5 | 0.958 | 0.989 | +0.031 [+0.022, +0.040] |
| modality | test | 3,000 | AUC | 0.817 | 0.708 | −0.109 [−0.128, −0.090] |
| modality | test-dolly | 2,000 | specificity@0.5 | 0.984 | 0.999 | +0.015 [+0.009, +0.021] |
| modality | test-realedit | 2,000 | recall@0.5 | 0.316 | 0.134 | −0.182 [−0.206, −0.160] |
| pii | dev | 1,506 | AUC | 0.940 | 0.919 | −0.021 [−0.035, −0.007] |
| pii | fresh-btc | 1,342 | AUC | 0.914 | 0.904 | −0.010 [−0.028, +0.009] |
| pii | fresh-pii-trace | 1,709 | AUC | 0.998 | 0.993 | −0.005 [−0.009, −0.001] |
| pii | fresh-ru-pii | 1,842 | AUC | 0.939 | 0.952 | +0.013 [+0.000, +0.026] |
| pii | fresh-uner-ewt | 1,780 | AUC | 0.921 | 0.905 | −0.016 [−0.044, +0.007] |
| pii | hold-kaggle-essays | 1,125 | AUC | 0.989 | 0.974 | −0.015 [−0.036, +0.001] |
| pii | hold-pii-prompts | 2,000 | AUC | 0.946 | 0.970 | +0.024 [+0.014, +0.034] |
| pii | hold-wildchat | 2,000 | specificity@0.5 | 0.716 | 0.857 | +0.141 [+0.123, +0.158] |
| pii | test | 3,000 | AUC | 0.947 | 0.929 | −0.018 [−0.029, −0.007] |
| pii | test-abcd | 1,486 | AUC | 0.993 | 0.987 | −0.005 [−0.011, −0.001] |
| pii | test-mapa | 1,679 | AUC | 0.918 | 0.817 | −0.101 [−0.102, −0.069] |
| pii | test-tab | 1,461 | AUC | 0.958 | 0.965 | +0.007 [−0.009, +0.023] |
| feedback | dev | 1,199 | accuracy | 0.574 | 0.537 | −0.037 [−0.071, −0.004] |
| feedback | fresh-crosswoz | 1,995 | accuracy | 0.766 | 0.588 | −0.178 [−0.206, −0.150] |
| feedback | hold-shipped-feedback | 1,715 | accuracy | 0.311 | 0.272 | −0.038 [−0.058, −0.018] |
| feedback | test | 2,842 | accuracy | 0.530 | 0.561 | +0.031 [+0.008, +0.053] |
| feedback | test-sgd | 1,998 | accuracy | 0.688 | 0.630 | −0.059 [−0.077, −0.041] |
| hallucination | dev | 1,210 | AUC | 0.786 | 0.777 | −0.009 [−0.028, +0.009] |
| hallucination | fresh-faithbench | 523 | AUC | 0.646 | 0.678 | +0.032 [−0.023, +0.080] |
| hallucination | fresh-shroom | 749 | AUC | 0.726 | 0.699 | −0.027 [−0.070, +0.018] |
| hallucination | hold-attributionbench-ood | 1,390 | AUC | 0.780 | 0.823 | +0.043 [+0.025, +0.060] |
| hallucination | hold-hallumix | 2,000 | AUC | 0.755 | 0.793 | +0.038 [+0.020, +0.057] |
| hallucination | hold-halubench | 1,877 | AUC | 0.527 | 0.547 | +0.021 [−0.006, +0.047] |
| hallucination | hold-summedits | 578 | AUC | 0.823 | 0.794 | −0.030 [−0.063, +0.005] |
| hallucination | test | 3,000 | AUC | 0.774 | 0.753 | −0.021 [−0.034, −0.008] |
| hallucination | test-psiloqa | 1,265 | AUC | 0.884 | 0.867 | −0.016 [−0.033, +0.001] |
| hallucination | test-ragbench | 1,311 | AUC | 0.605 | 0.557 | −0.048 [−0.075, −0.020] |
| hallucination | test-ragtruth | 1,528 | AUC | 0.864 | 0.850 | −0.014 [−0.031, +0.004] |

Not measured: the files whose text the suite does not publish, since their
sources are gated or bar redistribution:

- domain: `hold-yahoo`, `fresh-kmmlu`, `fresh-eli5-engineering`;
- fact check: `hold-factbench`, `fresh-alignbench`, `fresh-belle-eval`;
- jailbreak: `hold-hackaprompt-late-levels`, `fresh-b3`, `fresh-tensor-trust`;
- modality: `hold-anyinstruct`, `hold-shipped-modality`;
- PII: `hold-ai4privacy`;
- safety: `test-wildguardmix`, `test-wildjailbreak-eval`;
- feedback: `fresh-dstc2`;
- hallucination: `fresh-aggrefact-*` and `fresh-seahorse`.

The jailbreak and safety `test` files are scored on their published rows
(1,878 of 3,000 and 1,120 of 3,003).

## Thresholds

Vela 1.0's thresholds are operating points of its own scores. The 0.3B's
probabilities are calibrated (its choice temperature is 1.379), so the same
number means a different operating point. `router_signal_ab.py calibrate` maps
each Vela 1.0 threshold to the 0.3B threshold that keeps its operating point on
the suite's `dev` split, the split the suite reserves for thresholds:

- **A binary signal** keeps its false-positive rate on the dev negatives;
- **a confidence floor** keeps its share of dev rows below it. A floor is where
  domain falls back to its fallback category, feedback abstains and the
  modality detector consults keywords.

The table gives the shipped threshold, to two decimals and measured there. It
reports the rate it keeps, and the true-positive rate (binary) or balanced
accuracy (floors) both models reach, on dev and on the in-distribution test
files:

| Signal | Vela 1.0 | 0.3B | Kept rate, dev (Vela 1.0 / 0.3B) | Dev (Vela 1.0 / 0.3B) | Test (Vela 1.0 / 0.3B) |
| --- | ---: | ---: | --- | --- | --- |
| prompt guard | 0.3 | 0.74 | FPR 0.062 / 0.059 | TPR 0.688 / 0.812 | TPR 0.606 / 0.645 |
| prompt guard | 0.45 | 0.75 | FPR 0.059 / 0.059 | TPR 0.625 / 0.812 | TPR 0.577 / 0.638 |
| prompt guard | 0.5 | 0.75 | FPR 0.059 / 0.059 | TPR 0.609 / 0.812 | TPR 0.564 / 0.638 |
| prompt guard | 0.6 | 0.75 | FPR 0.059 / 0.059 | TPR 0.609 / 0.812 | TPR 0.545 / 0.638 |
| prompt guard | 0.7 | 0.75 | FPR 0.059 / 0.059 | TPR 0.578 / 0.812 | TPR 0.518 / 0.638 |
| prompt guard | 0.8 | 0.76 | FPR 0.055 / 0.055 | TPR 0.578 / 0.812 | TPR 0.473 / 0.632 |
| prompt guard | 0.85 | 0.77 | FPR 0.051 / 0.047 | TPR 0.547 / 0.812 | TPR 0.463 / 0.630 |
| prompt guard | 0.9 | 0.77 | FPR 0.047 / 0.047 | TPR 0.531 / 0.812 | TPR 0.423 / 0.630 |
| pii | 0.4 | 0.01 | FPR 0.471 / 0.200 | TPR 0.983 / 0.933 | TPR 0.981 / 0.888 |
| pii | 0.5 | 0.01 | FPR 0.451 / 0.200 | TPR 0.983 / 0.933 | TPR 0.978 / 0.888 |
| pii | 0.6 | 0.01 | FPR 0.426 / 0.200 | TPR 0.980 / 0.933 | TPR 0.970 / 0.888 |
| pii | 0.7 | 0.01 | FPR 0.389 / 0.200 | TPR 0.975 / 0.933 | TPR 0.962 / 0.888 |
| pii | 0.85 | 0.01 | FPR 0.300 / 0.200 | TPR 0.955 / 0.933 | TPR 0.939 / 0.888 |
| pii | 0.9 | 0.01 | FPR 0.242 / 0.200 | TPR 0.927 / 0.933 | TPR 0.905 / 0.888 |
| safety | 0.5 | 0.46 | FPR 0.226 / 0.226 | TPR 0.869 / 0.895 | TPR 0.810 / 0.865 |
| fact check | 0.65 | 0.86 | FPR 0.515 / 0.515 | TPR 0.889 / 0.778 | TPR 0.828 / 0.807 |
| fact check | 0.85 | 0.91 | FPR 0.444 / 0.444 | TPR 0.869 / 0.737 | TPR 0.792 / 0.743 |
| fact check | 0.95 | 0.93 | FPR 0.414 / 0.414 | TPR 0.808 / 0.707 | TPR 0.743 / 0.698 |
| domain | 0.5 | 0.28 | below 0.049 / 0.048 | balanced accuracy 0.614 / 0.579 | balanced accuracy 0.563 / 0.540 |
| feedback | 0.5 | 0.30 | below 0.001 / 0.001 | balanced accuracy 0.563 / 0.572 | balanced accuracy 0.558 / 0.591 |
| feedback | 0.7 | 0.37 | below 0.025 / 0.031 | balanced accuracy 0.556 / 0.573 | balanced accuracy 0.555 / 0.592 |
| modality | 0.5 | 0.35 | below 0.001 / 0.000 | balanced accuracy 0.716 / 0.553 | balanced accuracy 0.677 / 0.548 |
| modality | 0.6 | 0.45 | below 0.018 / 0.019 | balanced accuracy 0.719 / 0.541 | balanced accuracy 0.678 / 0.544 |
| modality | 0.7 | 0.51 | below 0.034 / 0.036 | balanced accuracy 0.720 / 0.535 | balanced accuracy 0.677 / 0.539 |

- **Prompt guard:** Vela 1.0 Guard's scores sit near 0 and 1, so its thresholds
  from 0.3 to 0.9 keep almost the same false-positive rate, and they map to
  0.74–0.77. At those thresholds the 0.3B catches 81% of the dev attacks
  against Vela 1.0's 53–69%. The dev split holds only 256 benign prompts.
  At 0.75 on the test files, the 0.3B flags 10.7% of benign prompts against
  Vela 1.0's 8.7% at 0.5, and catches 63.8% of attacks against 56.4%.
  - On the E2E attack fixtures (`e2e/testcases/testdata/jailbreak_detection_cases.json`)
    the 0.3B scores 0.930–0.969 on the six attacks and at most 0.437 on the six
    benign prompts. So every Guard threshold from 0.44 to 0.93 blocks all six
    attacks with no benign false positive. That includes the "Unrestricted DI
    persona" attack Vela 1.0 Guard scored 0.055
    ([#4120](https://github.com/vllm-project/semantic-router/issues/4120)).
- **PII:** the 0.3B's span head keeps only the spans it is confident in, by its
  own per-label thresholds. Even with every span it returns, it flags fewer dev
  negatives (20.0%) than Vela 1.0 does at its strictest threshold, 0.9
  (24.2%). Every PII threshold therefore maps to 0.01, which accepts every span
  it returns; the least confident span it returned on dev had 0.025. On the
  test files that flags 7.5% of the negatives and catches 88.8% of the sensitive
  requests, against Vela 1.0's 12.1% and 90.5% at 0.9. At 0.9 the 0.3B would
  catch only 72.9%.
  - On the suite's 21,251 subject questions (the domain rows), the 0.3B puts a
    span on 19.1% at 0.01, against Vela 1.0's 25.2% at 0.9 and 49.6% at 0.01.
    A span made only of digits and operators appears on 4.1% against 9.2% at
    0.9: Vela 1.0 reads many numbers as `CREDIT_CARD` or `IBAN_CODE`.
    `mom-v1`'s `personal_data` rule, which allows only `NRP`, matches 18.0% of
    them at 0.01, against 37.1% on Vela 1.0 at its earlier 0.7.
  - The 0.3B still reads some bare numbers as spans: "What is 2 + 2?" gets a
    `DATE_TIME` span on "2" at 0.86, and "What is 3 + 5?" a `PHONE_NUMBER` span
    on "+" at 0.76, where Vela 1.0 finds none. A rule that denies every PII type
    blocks them.
- **Hallucination:** the Router counts the hallucination spans the 0.3B returns
  with no threshold of its own (`min_span_confidence` 0), so there is nothing
  to map.

The module defaults take these values: prompt guard 0.75, domain 0.28, PII 0.01,
fact check 0.93 and feedback 0.37. A module that runs any other model and sets
no threshold keeps the one it defaulted to before (0.5, 0.5, 0.9, 0.95, 0.7).
The maintained recipes and E2E profiles that run the defaults take the mapped
values of the thresholds they set; those that pin a model keep theirs:

| Configuration | Rules | Vela 1.0 | 0.3B |
| --- | --- | ---: | ---: |
| `mom-v1` | `prompt_attack` | 0.5 | 0.75 |
| `mom-v1` | `unsafe` | 0.5 | 0.46 |
| `mom-v1` | `personal_data`, `personal_attribute` | 0.7 | 0.01 |
| `privacy` | `jailbreak_strict` | 0.45 | 0.75 |
| `privacy` | `pii_strict` | 0.85 | 0.01 |
| `agent` | `pii_strict` | 0.9 | 0.01 |
| `config/config.yaml` | `unsafe-content` | 0.5 | 0.46 |
| `config/config.yaml`, fragments | `unsafe_completion` | 0.85 | 0.77 |
| `config/config.yaml`, fragments | `restricted_pii` | 0.85 | 0.01 |
| `config/config.yaml` | modality `confidence_threshold` | 0.7 | 0.51 |
| E2E `envoy-ai-gateway`, `routing-strategies`, `streaming`, `aibrix` | prompt guard | 0.5–0.7 | 0.75 |
| E2E `production-stack` | prompt guard | 0.3 | 0.74 |
| E2E `multi-endpoint` | prompt guard | 0.5, 0.9 | 0.75, 0.77 |
| E2E, every PII rule | PII | 0.4–0.9 | 0.01 |
| E2E `hallucination` | fact check | 0.65 | 0.86 |

## E2E profiles

Beyond their thresholds, four E2E cases changed with the default. Each tripped
on a signal the case does not test:

- **`production-stack` load tests:** after a history question the 0.3B read
  the request number as a `DATE_TIME` span, and the profile denies every PII
  type. The template asks a physics question and ends in "(request N)", and
  the load tests pass with it.
- **`envoy-ai-gateway`, `decision-priority-selection`:** one query named
  "urgent", so `urgent_request` (priority 30) outranked the thinking decision
  on either model; it no longer does. "What is 2 + 2?" became "How do I solve a
  quadratic equation?" (the `DATE_TIME` span above).
- **`envoy-ai-gateway`, `tool-selection`'s PII precedence:** the case runs on its
  own recipe, `e2e-pii-precedence`, which holds only the baseline's PII block
  and the weather tool selection. The 0.3B scores the case's
  `__TOOL_SELECTION_ADD_WEATHER__` marker before payment data as an injection
  (0.83, wherever the marker sits), so on the default routing the jailbreak
  block outranked both.
- **`envoy-ai-gateway`, `security-window-provenance`:** the 0.3B reads a prompt
  whole up to 8,192 tokens, so its score names no window. Its prepared binding
  says so (`overflow: truncate`, no `window_size`), and the case reads that
  from the model inventory: a guard that scans keeps the window checks, and one
  that reads the whole prompt must name no window.

An attack at the end of 7,821 tokens of benign text scores 0.95 on the 0.3B;
at the end of 22,541 tokens it scores 0.58, below 0.75, because the model reads
only the first 8,192. `security-long-text` places its attack at 1,677 tokens.

## Restore the Vela 1.0 specialists

One block restores them; module thresholds the configuration does not set
return to the specialists' with them:

```yaml
global:
  model_catalog:
    system:
      safety: models/Vela-1.0-Encoder-307M-Safety
      prompt_guard: models/Vela-1.0-Encoder-307M-Guard
      domain_classifier: models/Vela-1.0-Encoder-307M-Domain
      pii_classifier: models/Vela-1.0-Encoder-307M-PII
      fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck
      hallucination_detector: models/Vela-1.0-Encoder-307M-Halu
      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback
```

A modality classifier names `models/Vela-1.0-Encoder-307M-Modality` as its
`classifier.model_path`. One line brings back one signal:

| Signal | Line under `global.model_catalog` |
| --- | --- |
| Domain | `system.domain_classifier: models/Vela-1.0-Encoder-307M-Domain` |
| Prompt guard | `system.prompt_guard: models/Vela-1.0-Encoder-307M-Guard` |
| Safety | `system.safety: models/Vela-1.0-Encoder-307M-Safety` |
| Fact check | `system.fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck` |
| User feedback | `system.feedback_detector: models/Vela-1.0-Encoder-307M-Feedback` |
| PII | `system.pii_classifier: models/Vela-1.0-Encoder-307M-PII` |
| Hallucination | `system.hallucination_detector: models/Vela-1.0-Encoder-307M-Halu` |
| Modality | `modules.modality_detector.classifier.model_path: models/Vela-1.0-Encoder-307M-Modality` |

Signal rule thresholds are the configuration's own: a signal moved back takes
its Vela 1.0 rule thresholds with it, the left column of the tables above.

## Reproduce

With the suite's `text/` directory, the runtime installed and the router
binary built:

```bash
python3 tools/router_signal_ab.py rows --suite text --out rows.jsonl
python3 tools/router_signal_ab.py config --arm vela1 > vela1.yaml   # and --arm vela2
# Per arm: the recording runtime as the Router's runtime command, the Router, then the rows.
export VLLM_SRUN_COMMAND="python3 tools/router_signal_ab.py record --log-dir vela1/exchanges --real vllm-srun --append=--result-cache-entries=0"
VLLM_SRUN_RESULT_CACHE=0 router -config vela1.yaml -api-port 18080 &
python3 tools/router_signal_ab.py run --port 18080 --rows rows.jsonl --out vela1/preview.jsonl
python3 tools/router_signal_ab.py join --rows rows.jsonl --run vela1/preview.jsonl --log-dir vela1/exchanges --arm vela1 --out vela1/preds
# Hallucination, against runtimes serving Vela-1.0-Encoder-307M-Halu and Vela-2.0-0.3B:
python3 tools/router_signal_ab.py halu --side vela1 --suite text --out vela1/preds --ports 8100
python3 tools/router_signal_ab.py score --suite text --a vela1/preds --b vela2/preds --out score.json --md score.md
python3 tools/router_signal_ab.py calibrate --suite text --a vela1/preds --b vela2/preds \
  --thresholds thresholds.json --out calibration.json --md calibration.md
```

`thresholds.json` lists the Vela 1.0 thresholds to map, for example
`{"jailbreak": [0.5], "pii": [0.9]}`. Latency: as in
[router-latency-cpu.md](router-latency-cpu.md#reproduce), with the record's
configuration as the `vela2` arm and the restore block added for `vela1`.
