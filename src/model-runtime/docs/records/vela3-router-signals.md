# vela3-fast as the default of the Router's built-in signals

[#5494](https://github.com/vllm-project/semantic-router/issues/5494) brings
the Vela 3.0 packages into the model runtime, starting with
`vllm-sr/vela3-fast`, and makes it the default model of every built-in
Router signal it answers, in place of Vela 2.0 0.3B
([vela2-router-signals.md](vela2-router-signals.md)). This record is the
evidence for the module thresholds the switch sets, and how the two models
compare on the router signal suite.

- **Thresholds:** each Vela 1.0 threshold a module or a maintained rule
  defaulted to is mapped to the vela3-fast threshold that keeps its operating
  point on the suite's `dev` split, as the 0.3B's were.
- **Accuracy:** on the suite's `test` files, against the 0.3B's
  numbers in [vela2-router-signals.md](vela2-router-signals.md#thresholds),
  vela3-fast is ahead on prompt attacks, PII, domain, feedback and modality,
  and behind on safety and fact check.
- **Restore:** naming the 0.3B restores it: `global.model_catalog.deployments.primary.artifact:
  vllm-sr/Vela-2.0-0.3B` (its module thresholds follow it,
  `canonical_operating_points.go`), or the one-block Vela 1.0 restore of the
  0.3B's record.

- **Date:** 2026-10-27.
- **Model:** `vllm-sr/vela3-fast` at `7cbefd4b`, served by the vela3 family
  ([vela3-parity.md](vela3-parity.md)).
- **Suite:** [`vllm-sr/router-signal-suite`](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
  at `fa08b2a6`. The Vela 1.0 arm is the suite's `released-vela` predictions,
  the Vela 1.0 specialists scored as published (the 0.3B's record used the
  Router's own Vela 1.0 run, so its Vela 1.0 column differs slightly).
- **Machine:** an RTX 4090 (FP32, the runtime's precision for vela3-fast on any
  device), PyTorch 2.10.0. A forward on a GPU differs from one on a CPU in the
  last digits only, well below the two decimals a threshold ships with.

## How each row was asked

`tools/router_signal_direct.py` asks each suite row the question the Router's
default deployment is asked (`modelservice.BuiltinTasks`, `serving/signal_question.go`):
domain, fact check, modality and feedback as truncating Choice questions,
prompt attack and harmful request read whole, PII as the `pii` preset; the
feedback question after an assistant turn. It writes the suite's prediction
layout, which `tools/router_signal_ab.py calibrate` reads.

With `--questions own` each row is asked only its own signal's question (prompt
attack and harmful request together). vela3-fast reads each router adapter's
questions about one part with one reading policy in one sequence, and never
mixes such rows in a forward, so the answers equal the full bundle's; on the
0.3B, which reads every question in one sequence, they would not.

```bash
python3 tools/router_signal_direct.py --model vllm-sr/vela3-fast --device cuda:0 \
  --suite SUITE/text --out PREDICTIONS --questions own --splits dev
python3 tools/router_signal_direct.py ... --splits test
python3 tools/router_signal_ab.py calibrate --suite SUITE/text \
  --a SUITE/predictions/suite/released-vela --b PREDICTIONS --b-name vela3-fast \
  --thresholds THRESHOLDS.json --out calibration.json
```

## Thresholds

A binary signal keeps its false-positive rate on the dev negatives; a
confidence floor keeps its share of dev rows below it. The table reports the
true-positive rate (binary) or balanced accuracy (floors) both models reach on
dev and on the in-distribution `test` files. Every point is in
[vela3-router-signals.json](vela3-router-signals.json).

| Signal | Vela 1.0 | vela3-fast | Kept rate, dev (Vela 1.0 / vela3-fast) | Dev (Vela 1.0 / vela3-fast) | Test (Vela 1.0 / vela3-fast) |
| --- | ---: | ---: | --- | --- | --- |
| prompt guard | 0.3 | 0.87 | FPR 0.062 / 0.059 | TPR 0.688 / 0.859 | TPR 0.606 / 0.714 |
| prompt guard | 0.5 | 0.88 | FPR 0.059 / 0.059 | TPR 0.609 / 0.859 | TPR 0.564 / 0.714 |
| prompt guard | 0.85 | 0.90 | FPR 0.051 / 0.051 | TPR 0.547 / 0.859 | TPR 0.463 / 0.704 |
| prompt guard | 0.9 | 0.91 | FPR 0.047 / 0.051 | TPR 0.531 / 0.859 | TPR 0.423 / 0.701 |
| pii | 0.4–0.9 | 0.18 | FPR 0.508–0.366 / 0.283 | TPR 0.985–0.976 / 0.969 | TPR 0.987–0.971 / 0.953 |
| safety | 0.5 | 0.52 | FPR 0.226 / 0.222 | TPR 0.869 / 0.798 | TPR 0.863 / 0.830 |
| fact check | 0.65 | 0.9997 | FPR 0.515 / 0.515 | TPR 0.889 / 0.727 | TPR 0.828 / 0.695 |
| fact check | 0.95 | 0.9999 | FPR 0.414 / 0.343 | TPR 0.808 / 0.576 | TPR 0.743 / 0.542 |
| domain | 0.5 | 0.41 | below 0.047 / 0.043 | balanced accuracy 0.619 / 0.594 | balanced accuracy 0.597 / 0.584 |
| feedback | 0.7 | 0.47 | below 0.025 / 0.025 | balanced accuracy 0.556 / 0.820 | balanced accuracy 0.515 / 0.807 |
| modality | 0.7 | 0.63 | below 0.034 / 0.033 | balanced accuracy 0.720 / 0.848 | balanced accuracy 0.719 / 0.858 |

- **Prompt guard:** every Vela 1.0 threshold from 0.3 to 0.9 maps to
  0.87–0.91. At 0.88 vela3-fast catches 71.4% of the test attacks against
  Vela 1.0's 56.4% at 0.5 and the 0.3B's 63.8% at 0.75. The six E2E attack
  fixtures (`e2e/testcases/testdata/jailbreak_detection_cases.json`) score
  0.9925–1.0 and the six benign ones at most 0.336, so every threshold from
  0.34 to 0.99 blocks all six attacks with no benign false positive.
- **PII:** the span-set adapter keeps only spans above its own per-length
  thresholds, so even with every span it returns it flags fewer dev negatives
  (28.3%) than Vela 1.0 at its strictest threshold (36.6%), and every PII
  threshold maps to 0.18. On the test files it catches 95.3% of the sensitive
  requests (the 0.3B: 88.8%, Vela 1.0: 97.1% at 0.9). Of the 25 E2E PII
  fixtures it finds a span on 24; the one it misses is a time of day ("1
  o'clock").
- **Safety and fact check** regress. The fact-check adapter's probabilities
  sit near 0 and 1, so its operating points land at 0.9997 and 0.9999, and at
  Vela 1.0's 0.95 it catches 54.2% of the test rows that need a fact check
  against 74.3% (the 0.3B: 69.8%). Safety catches 83.0% against 86.3%.
- **Domain:** the 0.41 floor keeps Vela 1.0's 4.7% of dev rows below it; on
  the 261 E2E domain fixtures vela3-fast names the expected domain on 191.
- **Hallucination:** the Router counts the spans the `halu` preset returns
  with no threshold of its own (`min_span_confidence` 0); nothing to map.

## What the switch changes

The module defaults take the mapped values of the Vela 1.0 thresholds they
defaulted to: prompt guard 0.88, domain 0.41, PII 0.18, fact check 0.9999 and
feedback 0.47 (`calibratedModuleThresholds`). The configurations that run the
default take the mapped values of their rules' Vela 1.0 thresholds; those
that pin a model keep theirs (the maintained recipes pin Vela 2.0 0.8B or 4B):

| Configuration | Rules | Vela 1.0 | 0.3B | vela3-fast |
| --- | --- | ---: | ---: | ---: |
| `config/config.yaml` | `unsafe-content` | 0.5 | 0.46 | 0.52 |
| `config/config.yaml`, fragments | `unsafe_completion` | 0.85 | 0.77 | 0.90 |
| `config/config.yaml`, fragments | `restricted_pii` | 0.85 | 0.01 | 0.18 |
| `config/config.yaml` | modality `confidence_threshold` | 0.7 | 0.51 | 0.63 |
| E2E `ai-gateway`, `routing-strategies`, `streaming` | prompt guard | 0.5 | 0.75 | 0.88 |
| E2E `production-stack` | prompt guard | 0.3 | 0.74 | 0.87 |
| E2E `multi-endpoint` | prompt guard | 0.5, 0.9 | 0.75, 0.77 | 0.88, 0.91 |
| E2E, every PII rule | PII | 0.4–0.9 | 0.01 | 0.18 |
| E2E `hallucination` | fact check | 0.65 | 0.86 | 0.9997 |

Not measured here: end-to-end Router latency on CPU. vela3-fast runs a 380M
backbone and one adapter network per router question group at FP32, so a
request costs more forwards than on the 0.3B.
