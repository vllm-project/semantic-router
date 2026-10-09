# Decision 2.5 Omni 27B

Training and evaluation code for `vllm-sr/Decision-2.5-Omni-27B`, the 27B multimodal decision model
of the Decision 2.5 generation. It answers one typed System One question per forward pass about a
state made of text and up to four images. Every option is listed under a single-token answer code,
and a readout initialised from the language-model head scores those codes at the last prompt token.

The language weights come from a Decision 2.5 Vega checkpoint or from `Qwen/Qwen3.8-27B`; the vision
encoder is the native Qwen3.8-27B encoder, kept frozen or trained at a low learning rate.

Run modules from `src/training/decision2` (imports are `d25.omni...`). The prompt, answer codes and
row contract extend `d25.vega.common.decision_format`.

| Directory | Contents |
| --- | --- |
| `common/` | Image-aware prompt and row contract (`vision_format.py`) |
| `suite/` | Builders for the eleven public Vision-board benchmarks and the board scoring rule |
| `proxy/` | Private-part proxies on fresh images, calibration on open entrants, and the gate |
| `data/` | Multimodal decision corpus builders, licence registry, text and image decontamination |
| `model/` | Checkpoint assembly (Vega language weights plus the Qwen3.8-27B vision encoder) |
| `train/` | Multimodal fine-tuning on top of the Vega trainer, with text replay |
| `eval/` | Image-aware inference engine and suite runners |
| `k8s/` | Job generators |
| `release/` | Packaging and model card |

The Vision board runs every entrant through its own image path with images capped at 1.6 MP
(1,638,400 pixels). Benchmark images are never used for training, and every training image is
checked against all benchmark images, including near-duplicates.
