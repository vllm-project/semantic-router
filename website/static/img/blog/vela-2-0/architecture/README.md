# Vela 2.0 architecture figures

These SVGs describe the public `AutoModel(..., trust_remote_code=True).system_one(...)` inference path at the model revisions below. The diagrams come from static configuration and source inspection, rather than model execution or a new benchmark run. Model sizes identify independent checkpoints; a shared drawing topology does not imply shared weights.

## Sources

| Model | Fixed revision | Public sources |
| --- | --- | --- |
| 0.3B | `7162ab91b808bf36201cdd2bc87212f9d2c5c4db` | [Configuration](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/7162ab91b808bf36201cdd2bc87212f9d2c5c4db/config.json), [model and readouts](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/7162ab91b808bf36201cdd2bc87212f9d2c5c4db/modeling_vela2.py), [inference wrapper](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/7162ab91b808bf36201cdd2bc87212f9d2c5c4db/vela2_inference.py) |
| 0.8B | `f21e7359e267f7e34af02502d3bc79eb7c8244c4` | [Configuration](https://huggingface.co/vllm-sr/Vela-2.0-0.8B/blob/f21e7359e267f7e34af02502d3bc79eb7c8244c4/config.json), [model and readouts](https://huggingface.co/vllm-sr/Vela-2.0-0.8B/blob/f21e7359e267f7e34af02502d3bc79eb7c8244c4/modeling_vela2.py), [inference wrapper](https://huggingface.co/vllm-sr/Vela-2.0-0.8B/blob/f21e7359e267f7e34af02502d3bc79eb7c8244c4/vela2_inference.py) |
| 4B | `d90b6f11c13a1776e5d33c50b89f8343006f68fa` | [Configuration](https://huggingface.co/vllm-sr/Vela-2.0-4B/blob/d90b6f11c13a1776e5d33c50b89f8343006f68fa/config.json), [model and readouts](https://huggingface.co/vllm-sr/Vela-2.0-4B/blob/d90b6f11c13a1776e5d33c50b89f8343006f68fa/modeling_vela2.py), [inference wrapper](https://huggingface.co/vllm-sr/Vela-2.0-4B/blob/d90b6f11c13a1776e5d33c50b89f8343006f68fa/vela2_inference.py) |
| 9B | `dd7f4da485a3f3df5ef7c30cef3b5245660b313e` | [Configuration](https://huggingface.co/vllm-sr/Vela-2.0-9B/blob/dd7f4da485a3f3df5ef7c30cef3b5245660b313e/config.json), [model and readouts](https://huggingface.co/vllm-sr/Vela-2.0-9B/blob/dd7f4da485a3f3df5ef7c30cef3b5245660b313e/modeling_vela2.py), [inference wrapper](https://huggingface.co/vllm-sr/Vela-2.0-9B/blob/dd7f4da485a3f3df5ef7c30cef3b5245660b313e/vela2_inference.py) |

The 0.3B backbone details follow the [Transformers 4.57.6 ModernBERT SDPA implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py). The decoder backbone modules follow the [Transformers 5.17.0 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3_5/modeling_qwen3_5.py), with execution supplied by each checkpoint's `_TreeForward.tree`.

The custom source symbols that determine the readouts and execution are `_Vela2Forward.heads` for the encoder, `CandidateHead.forward`, `SpanHeadV2.forward`, `_TreeForward.tree` and `run_batch` for the decoders, and `Vela2._predict`, `span_head_for` and `calibrated` in the inference wrappers.

## Editable sources

SVG text, shapes and paths can be edited directly. The accompanying Python files regenerate all eleven diagrams with Python 3 and its standard library:

```sh
python3 generate_all.py
```

- `generate_all.py`: four model views and state-prefix execution.
- `figures_encoder.py`: encoder attention, GEGLU and readouts.
- `figures_hybrid.py`: Gated-DeltaNet, gated GQA and SwiGLU.
- `figures_heads.py`: decoder decision and span heads.
- `architecture_svg.py`: SVG drawing primitives.
- `architecture-spec.json`: checkpoint revisions and diagram dimensions.

The generator reads the local specification, writes SVGs alongside these files, and requires no model weights. Update the generator and regenerate its SVG together when changing a figure.

## Reading the diagrams

Activation flow runs from bottom to top. Dashed links denote parameter selection or sharing, rather than activation mixing. Router and broad span heads have separate weights and are selected per question. Decoder prefix reuse applies within one rendered sequence; extra span questions and long-target windows require additional sequences. The encoder uses a joint bidirectional sequence and has a different cosine-based readout.

The input limits are the public inference budgets, including schema and text, rather than the backbone position capacities. The diagrams show the selected public inference path and omit registered modules that it does not call.
