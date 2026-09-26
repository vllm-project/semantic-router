# Native published-model collectors

`python -m inference.run` reads only a gold-free `*.prompts.jsonl` and writes
one `id`/`answers`/`latency_ms` record per item. Run inference on the remote
experiment host. The model load is outside the latency clock. Results can be
continued with `--resume`; existing rows must match the same prompt payload,
adapter version, model revision, model identity, and model manifest digest.
Run one published model per process because the bundles share the Python
package name `decision`. For Hugging Face local downloads, the adapter checks
the requested immutable revision against the download metadata and records
whether that attestation was available.

## APUS OpenJev 4B and 9B

The separate post-freeze Choice+Noul final-evaluation appendix is documented
in [`APUS_FINAL_APPENDIX.md`](APUS_FINAL_APPENDIX.md). It is excluded from the
all-type rank because native APUS does not support ordinal Score.

The independent `python -m inference.apus` collector calls each released
checkpoint's bundled `openjet_runtime.OpenJet.from_pretrained(...).decide` with
`effort=high` (all 32 layers), the published chat template, BF16 weights and
full-head candidate projection. Model revisions are 4B
`422b3741f8b5c092eeefef847c1ca89d78337d45` and 9B
`82c9c56cfa9de8d36704ed91948d4726ef111635`. The adapter checks HF
download metadata plus all non-card files in each `release-manifest.json`,
including weights, tokenizer, chat template, native source and configs. The
authors updated the English and Chinese model cards after the release
manifest; their current hashes are recorded separately.

The native API accepts 2–16 Choice candidates and a canonical yes/no Noul
proposition. We serialize JSON states without changing content and place the
original candidate ID inside its description because APUS's prompt renderer
otherwise hides IDs. Original Noul true/false meanings are appended to the
question instructions before using its canonical yes/no candidates. The
portable APUS runtime has only a binary `score_level` proposition, so ordinal
Score is recorded as `unsupported_native_ordinal_score` in the shared 1,600
question denominator, and Choice/Noul coverage is reported separately. It
does not silently map the binary proposition to a multi-level distribution.
Native candidate probabilities are uncalibrated; the 4B and 9B releases both
report BF16 merge decision parity but failed probability equivalence, so
their calibration must be measured afresh. The published reference runtime
specifies CUDA PyTorch 2.8 and Transformers 5.16.1; this experiment uses
ROCm Torch 2.12 with Transformers 5.16.1 and records that runtime as
`unvalidated_rocm_apus_native`.

Use the task image built from `inference/Dockerfile.apus-rocm`, which changes
Transformers only in a separate image. Download each complete revision with
the HF CLI on the GPU host. A CPU `--verify-only` pass checks file identity
before the GPU smoke. For example:

```bash
cd /work/source
PYTHONPATH=/work/source HF_HUB_OFFLINE=1 python3 -m inference.apus \
  --size 4b --model-path /work/models/APUS-OpenJev-v1-4B \
  --model-revision 422b3741f8b5c092eeefef847c1ca89d78337d45 --verify-only

ROCR_VISIBLE_DEVICES=3 PYTHONPATH=/work/source HF_HUB_OFFLINE=1 \
  python3 -m inference.apus \
  --size 4b --model-path /work/models/APUS-OpenJev-v1-4B \
  --model-revision 422b3741f8b5c092eeefef847c1ca89d78337d45 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/apus4b-dev.predictions.jsonl
```

The 9B invocation uses `--size 9b` and its matching path/revision. A one-item
GPU smoke uses `--max-items 1`; `--resume` validates model, adapter, effort,
input hashes and answer keys before continuing. The CSS pilot uses the same
collector with a separate gold-free prompt file and output. Neither adapter
reads gold.

## Decider 4B v2.1

Use the downloaded, pinned `Mapika/decider-4b` repository. The adapter imports
that repository's bundled `decider/` v1.4 inference code and calls its native
`system_one(state, questions)` with the released per-type temperatures,
state-first layout, independent questions, and isolated Score levels. Eager
PyTorch (`use_graphs=False`) avoids a CUDA-graph dependency on ROCm. The native
API rounds reported probabilities to four decimals and the Score mean to two;
the collector preserves those values.

```bash
cd /work/source
PYTHONPATH=/work/source python -m inference.run \
  --backend decider \
  --model-path /work/models/decider-4b \
  --model-revision eb5fbdfc9448473ec25e399882912863afbdb70e \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/decider4b-dev.predictions.jsonl
```

## Kev 4B

The independent `python -m inference.kev` collector uses the [Kev-4B
checkpoint](https://huggingface.co/jaredpalmer/kev-4b) at commit
`139fdd94f1b6a6ad80cc15e08fcb99cac885a101` and the model's pinned
Qwen3.5-4B-Base revision
`1001bb4d826a52d1f399e183466143f4da7b741b`. Download both with the
Hugging Face CLI on the experiment host; the base must be in the standard
`HF_HOME` cache. The collector forces Hub offline mode when loading. Its
[source checkout](https://github.com/jaredpalmer/kev) must be at the commit
recorded in the model provenance,
`6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`; all 30 recorded source
hashes and the adapter weight hash are verified before model load.

The native path calls `Checkpoint.load(LoadOptions())`, then
`SystemOneRequest`, `to_record`, `DecisionModel.probs`, and `to_answers` from
that source checkout. This is Kev's FP32 eager evaluation path with its
released temperature (about 2.406); it uses the same typed response as the
Kev server. The server's bf16, prefix-cache, and optional fused paths are
faster and can differ slightly numerically. Inputs exceeding its 8,192-token
state limit or 8,192-token complete state-plus-question row limit are recorded
as invalid `context_overflow`. The collector uses strict encoding so there is
no silent state truncation. The released provenance records NVIDIA H200 with
PyTorch 2.8; ROCm numeric parity remains to be checked by the scheduled GPU
smoke test.

```bash
cd /work/source
PYTHONPATH=/work/source HF_HUB_OFFLINE=1 python -m inference.kev \
  --model-path /work/models/kev-4b \
  --source-path /work/external/kev-4b-repo \
  --model-revision 139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/kev4b-dev.predictions.jsonl
```

For the CSS panel, change `--input` to
`/work/runs/css-transfer-v1/css-evaluation.prompts.jsonl` and choose a
separate `--output`. A one-item smoke run can use `--max-items 1`; the full run
then uses `--resume` with the same output file.

## Laya typed-decisions 0.4B

The independent `python -m inference.laya` collector uses
[Laya's native Agent](https://github.com/NandhaKishorM/laya) `system_one`
path and the released [typed-decisions checkpoint](https://huggingface.co/convaiinnovations/laya-typed-decisions).
The exact HF revision is `1a793eb568e6718f15941d08f85432581df534e3`;
the source is pinned at `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`.
The adapter verifies the model configuration, weights and tokenizer hashes
before load. It leaves the checkpoint's per-type and per-option temperatures,
BF16 autocast, default head budget, and native answer fields unchanged.

Laya natively supports Choice, Noul and Score, but its 1024-token context
and 256-token head budget can truncate state, instructions and option text.
The collector runs the published native behavior and records per-question
truncation diagnostics. Any truncated question has an explicit invalid entry
in the standard `answers` map, so it remains in the scoring denominator;
its original native result is retained under `native_answers` for diagnosis.
If the head budget rejects a request entirely, its answer is likewise
recorded invalid. Laya's `confidence`
for Choice/Score is normalized entropy, while `answer_confidence` is its
reported probability of the selected answer; neither should be assumed to
use Jev's confidence definition. The source has no ROCm numeric
qualification, so receipts carry `runtime_qualification=unvalidated_rocm`.

The model and source are staged in the experiment workspace. The CPU-only
preflight and one-GPU launch are:

```bash
cd /work/source
PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
  python -m inference.laya \
  --model-path /work/models/laya-typed-decisions \
  --source-path /work/external/laya-source \
  --model-revision 1a793eb568e6718f15941d08f85432581df534e3 \
  --verify-only

ROCR_VISIBLE_DEVICES=7 PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python -m inference.laya \
  --model-path /work/models/laya-typed-decisions \
  --source-path /work/external/laya-source \
  --model-revision 1a793eb568e6718f15941d08f85432581df534e3 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/laya-typed-dev.predictions.jsonl
```

The same collector accepts the CSS gold-free prompt files with a distinct
output path. `--max-items 1` and `--resume` support a one-item smoke test
followed by the full run. The GPU remains reserved for scheduled inference;
the preflight above reads files and imports no model weights.

## This-That 1.0

The independent `python -m inference.this_that` collector uses the
[published 1.0 checkpoint](https://huggingface.co/flock-io/this-that-model-1.0)
at HF revision `3d927195c4f9845efe66c5715883a7a0f42b1239` and the
[authors' 1.0 source commit](https://github.com/FLock-io/this-that-model/commit/4efe782ccbb9c1979a9951c35a29a8e0b3b80bf0).
It verifies HF revision metadata, the source commit and clean tracked files,
and SHA-256 of the model configuration, weights and tokenizer. It calls the
native `TypedDecider.from_pretrained` / `Question` / `decide` API with default
BF16, temperature 1.0, and state-first rendering.

This-That has one generic declared-option output head. Choice uses that head
directly. Noul is represented as the binary declared options no/yes, and
Score as an ordered option list whose probability-weighted index is reported;
both projections are marked in each answer. This measures how the published
model handles the same user-facing decision, but it is not evidence of native
Noul or ordinal Score heads. Rank, Matrix and free-text outputs are
unsupported. The native prompt builder silently truncates state beyond 1536
tokens, so the collector detects that limit first and records an invalid
`context_overflow` result without an inferred answer. The released code was
not numerically qualified on this ROCm host; receipts say
`runtime_qualification=unvalidated_rocm` until GPU validation is completed.

Download with HF CLI on the experiment host and check out the exact source
commit in a standalone repository. The pinned artifacts are already staged
under the following paths in the experiment workspace:

```bash
cd /work/source
PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
  python -m inference.this_that \
  --model-path /work/models/this-that-model-1.0 \
  --source-path /work/external/this-that-model-1.0-source \
  --model-revision 3d927195c4f9845efe66c5715883a7a0f42b1239 \
  --verify-only

ROCR_VISIBLE_DEVICES=7 PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python -m inference.this_that \
  --model-path /work/models/this-that-model-1.0 \
  --source-path /work/external/this-that-model-1.0-source \
  --model-revision 3d927195c4f9845efe66c5715883a7a0f42b1239 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/this-that10-dev.predictions.jsonl
```

After the one-item GPU smoke passes, switch `--input` to a gold-free CSS
panel and choose a separate output. `--max-items 1` followed by `--resume`
supports a non-destructive smoke and continuation.

## Decision 1.0 Kai and Lex 0.6B

The independent `python -m inference.kai_lex` collector uses each released
bundle's `decision_runtime.load_native` and `decision_inference.SystemOne`
with the documented default physical B8 scheduling. It verifies exact
Hugging Face revision metadata and the complete native file roster before
importing bundled code. Kai is pinned to revision
`7185f514f54b8f93c55998b1e8f9c5cc67f0d029` and native manifest
`c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4`;
Lex is pinned to revision `ee8e74d912fca8328a353c11d174b44da3f91781`
and native manifest
`f288d873999832a3f37c6a7c4268c2ab309691e621794dbf7acab891acbbb7e6`.
The native loader checks source, weights and tensor inventory again.

The release validation used AMD ROCm gfx942, Python 3.12.13, PyTorch
`2.12.0+git6bbd260` / HIP `7.2.53211`, Transformers 4.57.6,
tokenizers 0.22.2, safetensors 0.8.0, NumPy 2.5.3 and FP32 SDPA.
The current shared task image has the matching Python/PyTorch/HIP but
Transformers 5.17.0, tokenizers 0.23.2 and NumPy 2.3.5. Create a separate
environment after active GPU jobs finish; leave their environment unchanged:

```bash
python -m venv --system-site-packages /work/envs/kai-lex
/work/envs/kai-lex/bin/python -m pip install \
  'transformers==4.57.6' 'tokenizers==0.22.2' \
  'safetensors==0.8.0' 'numpy==2.5.3'
```

Download the complete repositories on the experiment host with the HF CLI,
preserving local-dir metadata:

```bash
hf download llm-semantic-router/Decision-1.0-Kai-0.6B \
  --revision 7185f514f54b8f93c55998b1e8f9c5cc67f0d029 \
  --local-dir /work/models/Decision-1.0-Kai-0.6B
hf download llm-semantic-router/Decision-1.0-Lex-0.6B \
  --revision ee8e74d912fca8328a353c11d174b44da3f91781 \
  --local-dir /work/models/Decision-1.0-Lex-0.6B
```

The CPU-only bundle check needs no GPU:

```bash
cd /work/source
PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
  /work/envs/kai-lex/bin/python -m inference.kai_lex \
  --backend kai --model-path /work/models/Decision-1.0-Kai-0.6B \
  --model-revision 7185f514f54b8f93c55998b1e8f9c5cc67f0d029 \
  --verify-only
```

Run each model in a separate process with one AMD GPU visible. For example,
after GPU 7 is allocated:

```bash
cd /work/source
ROCR_VISIBLE_DEVICES=7 PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /work/envs/kai-lex/bin/python -m inference.kai_lex \
  --backend kai --model-path /work/models/Decision-1.0-Kai-0.6B \
  --model-revision 7185f514f54b8f93c55998b1e8f9c5cc67f0d029 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/kai06b-dev.predictions.jsonl

ROCR_VISIBLE_DEVICES=7 PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /work/envs/kai-lex/bin/python -m inference.kai_lex \
  --backend lex --model-path /work/models/Decision-1.0-Lex-0.6B \
  --model-revision ee8e74d912fca8328a353c11d174b44da3f91781 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/lex06b-dev.predictions.jsonl
```

For CSS transfer, change `--input` to a gold-free
`/work/runs/css-transfer-v1/css-{pilot,evaluation}.prompts.jsonl` file
and choose a distinct output. Choice, Noul and Score are supported.
SystemOne allows 2–255 Choice options and 2–10 Score levels. Native Score
records can accept more levels, but this adapter uses the published
SystemOne contract. Rank, Matrix, free-text answers and arbitrary extra
question fields are unsupported. Noul returns its native yes probability;
the no probability is its complement. Every complete
state/question/candidate sequence must fit 1024 tokens. Native overflow is
recorded as invalid for the whole request, without truncation or forced
guess, so coverage remains in the scoring denominator. By default the
adapter refuses unqualified runtime versions; `--allow-unvalidated-runtime`
is for clearly labelled exploratory runs only.

## Decision 1.0 Lux 9B

The adapter imports the downloaded Lux bundle's own
`src/decision/model.py` and calls `DecisionModel.from_pretrained(...).decide`.
This verifies the bundle manifest, applies the shipped calibration, and loads
its automatic normalization profile. It rejects any input over its 16,384
token complete-question limit without truncation.

The shared experiment container currently has the qualified Torch 2.12,
HIP 7.2, Transformers 5.17, tokenizers 0.23.2, and safetensors 0.8.0 but no
`fla`. Do not silently bypass Lux's runtime check. Build its published
`Dockerfile.runtime` from the downloaded bundle. It pins a public ROCm image
by digest and adds hash-verified FLA 0.5.2 wheels at `/opt/decision-fla`.
The build needs network access to those wheel URLs. Run the resulting image
with the model and source mounted and `PYTHONPATH=/opt/decision-fla:/work/source`;
the exact Docker launch and validation can be chosen with the GPU schedule.

```bash
cd /work/source
PYTHONPATH=/opt/decision-fla:/work/source python -m inference.run \
  --backend lux \
  --model-path /work/models/Decision-1.0-Lux-9B \
  --model-revision bd45a30aee8c84032791c245c70f86dee5389cc8 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/lux9b-dev.predictions.jsonl
```

`--allow-unvalidated-runtime` is reserved for explicit exploratory use; such
records carry the loader's runtime differences and should not be mixed with
qualified published-model scores.

## Decision 1.0 Nox 4B and Sol 2B

Nox and Sol use the same native `src/decision/model.py` loader as Lux. Their
bundle manifest verifies weights, tokenizer, calibration and packaged source;
the loader checks the FLA dispatch and exact qualified runtime before loading.
The adapter additionally checks each bundle's base-model family and returned
native model ID. Use the published `Dockerfile.runtime` from either bundle to
provide the hash-verified FLA overlay on the qualified ROCm stack, or run in
an environment proven to match its `runtime.json`. The `--backend` value is
part of the prediction receipt and resume identity.

```bash
cd /work/source
PYTHONPATH=/opt/decision-fla:/work/source python -m inference.run \
  --backend nox \
  --model-path /work/models/Decision-1.0-Nox-4B \
  --model-revision 0bb833504965c0eabdb9630b7bbd385cb2fe5cd4 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/nox4b-dev.predictions.jsonl

PYTHONPATH=/opt/decision-fla:/work/source python -m inference.run \
  --backend sol \
  --model-path /work/models/Decision-1.0-Sol-2B \
  --model-revision 0665a41108e8f0b33a9515c98311c45947b99399 \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/sol2b-dev.predictions.jsonl
```

## Decision 1.0 Eos 0.8B

Eos has an older native `decision/` package. The adapter imports its
`DecisionModel.from_pretrained(...).decide`, verifies its `MODEL_MANIFEST.json`
against local files, and checks the qualified PyTorch, HIP, Transformers,
FLA, device architecture and attention setting from `runtime.json` before
loading. Eos itself verifies the BF16 backbone, FP32 head, parameter count,
released temperature, and eight-question physical batch limit. The native
response uses `Decision-1.0-Eos` without the public repository's size suffix;
the adapter verifies this mapping and records the sized repository ID.

```bash
cd /work/source
PYTHONPATH=/opt/decision-fla:/work/source python -m inference.run \
  --backend eos \
  --model-path /work/models/Decision-1.0-Eos-0.8B \
  --model-revision 3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/eos08b-dev.predictions.jsonl
```

For any Decision backend, `--allow-unvalidated-runtime` is limited to clearly
labeled exploratory runs. Such receipts record the runtime differences and
must stay separate from qualified comparisons. The collector never changes
the native model's prompt, candidate order, probability calibration, or
truncation policy.
