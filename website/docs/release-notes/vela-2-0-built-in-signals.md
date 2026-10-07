# Release note: Vela 2.0 for the built-in signals

Vela 2.0 is public on Hugging Face:
[`vllm-sr/Vela-2.0-0.3B`, `-0.8B`, `-4B` and `-9B`](https://huggingface.co/collections/vllm-sr/vela-20),
Apache-2.0, with no token needed. The model runtime keeps the revisions it
pins, so nothing it loads changes.

## The built-in signals can run on Vela 2.0

Domain, jailbreak, safety, fact check, user feedback and modality now run on
a Vela 2.0 deployment, as PII and hallucination already did. Each signal asks
the question the model was trained on for it, with the labels of the Vela 1.0
model it stands in for, so rules, thresholds and policies read the answer as
before. A request asks one deployment every question about the same text in
one call. Bind the signals to one deployment to opt in; see
[Choose a model](model-runtime/choose-a-model.md#vela-20).

Hazard categories, embeddings, multimodal embeddings and reranking have no
Vela 2.0 question and keep their Vela 1.0 models.

## The defaults stay on Vela 1.0

With no model configured, every built-in signal still runs on its Vela 1.0
model. [#4639](https://github.com/vllm-project/semantic-router/issues/4639)
set two conditions for a switch, measured on a CPU: accuracy level or better
for every signal, and end-to-end router latency level or better. The latency
condition fails by a wide margin. On 12 CPU cores the Router answers a request
on the 0.3B in about 128 ms at the median, against 16 ms on the Vela 1.0
models, and serves about a fifth of the requests per second. Every 0.3B
request carries its questions, their options and the 17 PII labels (at least
560 tokens) through one 307M-parameter forward, where each Vela 1.0 model
reads only the request. The
[A/B record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)
has the latency and the accuracy of every signal on the router signal suite.
[#4668](https://github.com/vllm-project/semantic-router/issues/4668) evaluates
Vela 2.0 as the default on GPUs.

## What else changes

- A modality detector whose model comes from a `modality_detector` binding
  no longer needs `classifier.model_path`.
- A consumer window (`prompt_guard.window`, a safety module's `window`) on a
  signal bound to a Vela 2.0 deployment fails preparation with a message that
  says to remove it: the model reads the whole text.

Nothing changes for a configuration that binds no signal to a Vela 2.0
deployment.
