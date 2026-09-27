# About-27B first-release backbone comparison

The latest project instruction explicitly adds the **official**
`google/gemma-4-26B-A4B-it` general model as a possible starting point at the
about-27B size, alongside the already permitted official `Qwen/Qwen3.8-27B`.
This expands the permitted general-model source at this size; it does not
permit initializing from anyone else's decision-fine-tuned model. Existing
Qwen3.8-27B development results remain attached only to their exact weights.

The [official Gemma 4 model card](https://huggingface.co/google/gemma-4-26B-A4B-it)
describes a 25.2B-total, approximately 3.8B-active mixture-of-experts model,
with a separate vision encoder, 256K advertised context and Apache-2.0
license. The [official Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
describes general post-trained multimodal weights under Apache-2.0. These
are **source descriptions**, not Decision 2.0 performance claims or exact
loaded text-parameter measurements. A mixture's total loaded parameters and
active parameters must both be reported; do not compare active Gemma
parameters with total dense-Qwen parameters as if they were the same size.
Read-only HF CLI metadata at 2026-09-27 12:43 UTC identifies the official
Gemma repository revision as
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` and reports
`25,805,936,206` BF16 safetensors parameters across the complete repository.
That metadata count includes whatever tensors the complete multimodal package
stores; it is not yet a measured text-only runtime count.
At that revision, HF CLI downloaded only `config.json` for a read-only CPU
preflight; its SHA-256 is
`ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3`.
The pinned training image has Transformers 5.17.0 and imports
`Gemma4ForConditionalGeneration`. This establishes parser availability, not
weight-load or GPU execution parity. The configuration declares 30 text
layers, 128 experts with eight selected per token, and separate vision config.

Before a Gemma optimizer arm, pin its immutable HF revision using HF CLI on an
authorized experiment node, audit the exact tensor inventory and text-only
load path, then build a native Choice/Noul/Score adapter. Require identical
answers in two zero-step processes, complete no-truncation input admission,
one finite optimizer step and exact reload, all on the intended hardware.
Only then preregister a matched-data/token-budget experiment against the
existing official Qwen candidate. Development SELECT and opened DEV/pilot
may choose whether this architecture is worth a frozen JevArena v3 and public
JevBench rerun; prior Qwen or external Index scores cannot be inherited.

As of this note, no Gemma weight has been loaded, trained or evaluated by
this project. The existing Qwen3.8-27B candidate remains on release HOLD for
its Score weakness. The next useful large-model experiment is an admitted
native Gemma zero-step/runtime comparison or an independently quality-passed
Score data arm, whichever becomes ready first; neither changes the already
measured Qwen result.
