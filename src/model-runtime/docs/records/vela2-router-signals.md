# Vela 2.0 0.3B for the Router's built-in signals, against the Vela 1.0 defaults

[#4639](https://github.com/vllm-project/semantic-router/issues/4639) asked
whether `vllm-sr/Vela-2.0-0.3B` should become the default model of the
Router's built-in signals once Vela 2.0 is public, on two conditions measured
on a CPU: accuracy level or better for every signal on the built-in signals'
evaluation rows, and end-to-end router latency level or better.

**The latency condition fails, so the defaults stay on Vela 1.0.** Through
the Router on 12 CPU cores, a request on the 0.3B takes LAT_V2_P50 ms at the
median against LAT_V1_P50 ms on the Vela 1.0 models, and the Router serves
LAT_V2_RPS against LAT_V1_RPS requests per second; every percentile and every
concurrency is worse, with every interval on the worse side. ACCURACY_LEAD

The built-in signals can still run on Vela 2.0 as an opt-in: one deployment
bound to every signal it answers
([Choose a model](https://vllm-sr.ai/docs/model-runtime/choose-a-model#vela-20)).
[#4668](https://github.com/vllm-project/semantic-router/issues/4668) evaluates
it as the default on GPUs.

- **Date:** 2026-10-07.
- **Machine:** AMD EPYC 9575F, CPU only. The accuracy runs and the latency
  rounds ran on two hosts of this model; each run's processes ran in
  `systemd` scopes on the cores given below, with memory bound to the cores'
  NUMA node.
- **Commits:** the Router built from this change (`ROUTER_COMMIT`) in CI's
  builder image (`golang:1.25-bookworm`, the image recipe's flags; the same
  binary on both hosts, sha256 `6eaeb76d…`); the model runtime from `main`
  (`db35009da`) in a Python 3.12 environment with PyTorch 2.10.0 (CPU), as the
  router image pins it. This change touches no runtime code path.
- **Models:** the Vela 1.0 models at their pinned revisions (Domain
  `f6354f54`, Guard `087f9e40`, Safety `6e70e725`, FactCheck `99ede1ab`,
  Feedback `47434a7f`, Modality `5384b899`, PII `6d3300c4`, Halu `ca875312`)
  and Vela 2.0 0.3B at `a3209a50`, all on the `exact` profile.
- **Arms:** `tools/router_signal_ab.py config` writes both.
  - `vela1`: no model catalog, so every signal runs on its Vela 1.0 default
    (modality has no default; the arm names Vela 1.0 Modality, as
    `config/config.yaml` does).
  - `vela2`: the same signals and decisions plus one Vela 2.0 0.3B deployment
    that domain, prompt guard, safety, fact check, feedback, modality, PII and
    hallucination are bound to. Each signal asks the question the model card
    gives for it, with the Vela 1.0 model's labels; a request asks every
    question in one `/v1/decisions` call (six, or seven after an assistant
    turn).

## Latency

The [router-latency record](router-latency-cpu.md)'s method and corpus: its
configuration's five model-backed request signals (domain, prompt guard, PII,
fact check, feedback), the 539 inputs of `tools/router_latency.py corpus`,
`POST /api/v1/routing/preview`, result caches off on both layers. Each pass
sends 20 warm-up requests, then the corpus three times: sequentially, then at
concurrency 4 and 16. Five rounds alternate the arms, the order rotating each
round; each Router and its runtime processes run on cores 32–43 (12 vCPUs),
the driver on 4 others. Per metric, the median of the five rounds and the
paired difference `vela2 − vela1` per round with a 95% t interval
(`router_latency.py rounds`). Raw rounds: `vela2-router-signals.json`.

LATENCY_TABLE

The Router makes the same routing decision on LAT_DECISIONS_SAME of 539 inputs
in both arms. LAT_NOTES

**Why the 0.3B is slower.** Every request carries the questions, their
options and the 17 PII labels: at least 560 tokens of schema ahead of a prompt
whose median is about 15 tokens, through one 307M-parameter forward. Each
Vela 1.0 model is a 307M-parameter encoder of the same size, but it reads
only the prompt, and the Router runs the models in separate runtime
processes at once. Moving any one signal to the 0.3B makes it the slowest call
of the request (the PII question alone reads about 200 schema tokens), so no
subset of signals is level either. `max_speed` (the 0.3B's `float32-packed`
copy) runs the same request in 76–78 ms at the median on these cores,
still about five times Vela 1.0's.

## Accuracy

The evaluation rows are
[`vllm-sr/router-signal-suite`](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
at revision `fa08b2a6`: every test, source test, held-out and fresh held-out
file of the eight signals whose text the suite publishes. Headline per file,
as the suite scores it: AUC of the positive class for jailbreak, safety, fact
check, PII (any sensitive identifier in the request), modality (an image to
generate) and hallucination; accuracy for domain (14 subjects) and feedback (5
types). Intervals are group bootstraps (2,000 replicates; rows that share a
source group are resampled together, the same draws for both arms), so the
differences are paired. A signal's mean combines one draw of every file in its
set.

- **Request-time signals, through the Router:** each arm's Router answered
  every row through `POST /api/v1/routing/preview` with the full request
  configuration (a feedback row after an assistant turn). The model runtime
  the Router managed was `tools/router_signal_ab.py record`, which serves the
  Router's own socket and logs every exchange; the probabilities each signal
  read are those the runtime returned to the Router. Vela 1.0 Guard's score is
  its riskiest window, as the Router reads it.
- **Hallucination, at response time:** the routing preview never reaches it,
  so each row is asked as the Router's detector asks it (`router_signal_ab.py
  halu`): Vela 1.0 Halu with the grounded `{context, question, answer}` item,
  `reject` at 8,192 tokens and threshold 0.5; the 0.3B with the `halu` preset
  over `{request, context, answer}`. The score is the highest token (Vela 1.0)
  or word (0.3B) probability of the answer. The Vela 1.0 values equal the
  ones the suite card publishes for this model on every file it lists.

Means per signal (`vela2 − vela1`, 95% interval). *Held out* is the suite
card's set of corpora no model trains on; *fresh* is the suite's fresh
held-out set, built after both models' training data; *in distribution* is
the suite's own test and the training corpora's published tests:

ACCURACY_TABLE

ACCURACY_NOTES

Per file:

FILE_TABLE

Not measured: the files whose text the suite does not publish, since their
sources are gated or bar redistribution: domain `hold-yahoo` and
`fresh-kmmlu`, `fresh-eli5-engineering`; fact check `hold-factbench`,
`fresh-alignbench`, `fresh-belle-eval`; jailbreak `hold-hackaprompt-late-levels`,
`fresh-b3`, `fresh-tensor-trust`; modality `hold-anyinstruct`,
`hold-shipped-modality`; PII `hold-ai4privacy`; safety `test-wildguardmix`,
`test-wildjailbreak-eval`; feedback `fresh-dstc2`; hallucination
`fresh-aggrefact-*` and `fresh-seahorse`. The jailbreak and safety `test`
files are scored on their published rows (1,878 of 3,000 and 1,120 of 3,003).

## What stays on Vela 1.0 either way

Hazard has no trained Vela 2.0 question (the card publishes none), and Vela
2.0 has no embedding, multimodal or rerank exit, so Hazard, Embedding, Omni
and the Reranker keep their Vela 1.0 models in both arms.

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
```

Latency: as in [router-latency-cpu.md](router-latency-cpu.md#reproduce), with
`router_signal_ab.py config --set latency --arm vela1` and `--arm vela2`.
