# Vela Omni runtime artifacts

The published Nano and Mini repositories contain native Python models and
weights. They do not publish the four ONNX graphs used by the router. This
directory produces those graphs during an explicit build or model preparation
step, verifies them against the pinned public implementation, and packages only
data for the native runtime. Serving never imports the published Python code.

## Prepare artifacts

From the repository root, build the standalone data image:

```sh
docker buildx build -f tools/models/vela_omni/Dockerfile \
  --build-arg VELA_OMNI_VARIANTS="nano mini" \
  --build-arg VELA_OMNI_KEEP_GOLDEN=1 \
  --output type=local,dest=./artifacts .
```

The output contains `vela-1.0-omni-nano/` and `vela-1.0-omni-mini/`. Nano is the
default when `VELA_OMNI_VARIANTS` is omitted. Set `VELA_OMNI_KEEP_GOLDEN=1` for
model-backed tests; ordinary runtime bundles omit those reference inputs and
outputs. Both variants are exported sequentially, with each modality in a fresh
process to release its temporary memory. The builder runs offline unit tests
before downloading models and requires all three modalities to pass parity.

The router Dockerfiles share this producer stage. They place immutable bundles
in `/opt/router-model-artifacts`, outside the mounted model cache; the router's
model preparation copies and validates the selected bundle into its configured
cache. Producer Python, native source weights, and temporary files are absent
from the final serving image. Source and export scratch files are removed in the
same build layer. Keep the Docker build cache to reuse unchanged artifacts.

For a pre-provisioned source snapshot, use the pinned dependencies in
`requirements.txt` and export without network access:

```sh
python tools/models/vela_omni/export.py --variant nano \
  --source /models/native-nano --output /models/exported-nano
python tools/models/vela_omni/bundle.py verify /models/exported-nano
python tools/models/vela_omni/bundle.py stage /models/exported-nano \
  /models/vela-1.0-omni-nano --keep-golden
```

`--download` explicitly provisions the revision in `sources.json`; `HF_HOME`
sets its cache. Every code, config, tokenizer, and weight file is authenticated
before loading source Python. Extra Python files are rejected. Export output
must be separate from the source and initially empty. A failed export has only
a pending manifest and cannot be staged as a runnable bundle.

## Tensor contract

| Graph | Input | Output |
|---|---|---|
| `text` | Batch-one INT64 token IDs and attention mask, dynamic sequence | Normalized native embedding |
| `image` | FP32 normalized RGB pixels, Nano 512² / Mini 384² | Normalized native embedding |
| `clap` | FP32 features `[1,1,1001,64]` for one endpoint-cover window | Normalized 512-dimensional CLAP embedding |
| `audio` | FP32 Whisper features `[1,80,3000]` and aggregated CLAP embedding `[1,512]` | Normalized native embedding |

Nano uses 384 dimensions and a 512-token public limit; Mini uses 768 dimensions
and a 32768-token public limit. Output dimensions are the published readouts,
including Mini's normalization before and after its native prefix. Mini's graph
partitions queries into blocks of 256 while retaining all keys and values. It
builds the causal and padding mask per block, avoiding a quadratic full-sequence
attention allocation. Projections, rotary positions, and decoder layers retain
their original weights and computations. No context is truncated.

CPU deployments execute dynamic sequence lengths. A GPU deployment must declare
`input.max_tokens` explicitly. The native session specializes its execution
shape to that budget and pads shorter requests with a masked suffix; requests
over the budget follow the authored overflow policy. For example, a Mini
budget of 2048 admits up to 2048 tokens while the model capability remains
32768. This budget is part of the pooled instance identity. Choose it for the
workload and available device memory: every text inference uses that execution
shape. GPU preparation fails if any compute node would fall back to CPU.

The manifest declares identity, revision, tokenizer, exact graph names, dtypes,
shapes, public budgets, preprocessing parameters, and SHA-256 file inventory.
Actual mel-filter arrays are exported from the pinned processors. Audio retains
the complete Whisper and CLAP residual branches, independent resampling from
original PCM, and the endpoint-cover windows including the final tail. Source
image decoding and resize are tested independently against native preprocessing.
Mini's optional query/document instruction formatting is a typed preprocessing
mode; the default preserves the public no-instruction behavior.

## Qualification

```sh
OMP_NUM_THREADS=4 python -m unittest discover \
  -s tools/models/vela_omni -p 'test_*.py'
python tools/models/vela_omni/export.py --variant mini --download \
  --output /models/exported-mini --full-context --threads 16 \
  --max-address-space-gib 32
```

Regular Mini builds execute a 1024-token probe and record that fact. A declared
32768-token limit alone is not evidence of full-context execution; use
`--full-context` for qualification and inspect `full_context_executed` and
`longest_text_tokens_executed` in `reference_parity.json`. The optional address
space guard fails the process before an excessive CPU allocation can affect
other workloads.

For the CPU 32K reference, the original public `encode_text` uses explicit KV
head repetition followed by Torch's independent CPU flash SDPA kernel. The
default Transformers GQA optimization otherwise selects quadratic CPU math.
Qualification first compares this equivalent kernel dispatch against the public
default on short text, causal boundaries, and padded batches; the receipt names
the backend and records those checks. The reference does not use the exported
query-block implementation.

Parity covers English, Chinese, whitespace, padding, overflow rejection, optional
instructions, non-square PNG and JPEG inputs, stereo PCM, source sample rates,
and one/two/three CLAP windows. It checks full embedding shape, finite values,
absolute/relative error and cosine similarity. The receipt includes source
identity, dependency versions, producer file hashes, and executed cases. GPU
qualification requires the requested execution provider with CPU fallback
disabled; the native model tests additionally audit the execution profile.

Run native source-reference parity explicitly with an artifact retaining goldens
and `ORT_DYLIB_PATH` pointing to the installed ONNX Runtime library:

```sh
VELA_OMNI_ARTIFACT=/models/exported-nano \
go -C onnx-binding test ./instance -run '^TestPublishedOmniParity$' -count=1 -v
VELA_OMNI_ARTIFACT=/models/exported-mini \
go -C onnx-binding test ./instance -run '^TestPublishedOmniFullContext$' -count=1 -v
```

The second command requires a Mini artifact exported with `--full-context`, or
`VELA_OMNI_FULL_CONTEXT_REFERENCE` naming a separately generated matching source
reference. These explicit reference tests are excluded from model-free Core;
routine CI does not claim Mini 32K qualification. The image-calibration CI lane
prepares Nano once, checks export parity, and uses that same immutable artifact
for owned image classification, prepared inventory, cache/memory integration,
and the complete frozen routing calibration. Candle's legacy multimodal lane
retains its separate original binding compatibility tests.

`bundle.py` uses only Python's standard library to verify the complete inventory
and receipt and to stage a new directory atomically. It rejects missing graphs,
extra unlisted files, changed bytes, failed or incomplete parity, native Python
source, and source weight files. Bundles are immutable; create a new destination
when producing a different revision or graph implementation.
