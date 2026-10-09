# Decision 2.5 Vega 27B

Training and evaluation code for `vllm-sr/Decision-2.5-Vega-27B`, the 27B text decision model of the
Decision 2.5 generation. The model is a full-weight fine-tune of `Qwen/Qwen3.8-27B` that answers one
typed System One question per forward pass: every option is listed under a single-token answer code,
and a readout initialised from the language-model head scores those codes at the last prompt token.

Run modules from `src/training/decision2` (imports are `d25.vega...`).

| Directory | Contents |
| --- | --- |
| `common/` | Shared prompt, answer codes and the training-row contract (`decision_format.py`) |
| `data/` | Corpus builders, decontamination against the Decision Index public suite, mixtures, teacher labelling |
| `train/` | FSDP2 full fine-tuning trainer, warm-start and weight-averaging tools |
| `eval/` | Inference engine, public-suite runner (official kit scoring) and private-part proxies |
| `k8s/` | Job generators for training, labelling and evaluation runs |
| `release/` | Packaging, model card and upload helpers |

Evaluation follows the official Decision Index kit (`apolinario/decision-index`, edition 0.3) for the
public suite. The board's private parts are approximated by proxies calibrated on open entrants;
the calibration report lives with the evaluation records.
