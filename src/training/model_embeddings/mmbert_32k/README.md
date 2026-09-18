# mmBERT-32K training

This package trains the three related mmBERT-32K text models used for
long-context embedding and reranking.

| Model | Entry point | Configuration | Objective |
| --- | --- | --- | --- |
| `mmbert-32k-yarn` | `foundation.py` | `configs/foundation.json` | 32K masked-language continuation with YaRN |
| `mmbert-embed-32k-2d-matryoshka` | `embedder.py` | `configs/embedder.json` | Multiple-negatives ranking with 2D Matryoshka supervision |
| `mmbert-rerank-32k-2d-matryoshka` | `reranker.py` | `configs/reranker.json` | Binary relevance loss across 20 layer/dimension heads |

The public [mmBERT-32K model guide](../../../../website/docs/training/mmbert-32k-models.md)
explains when to use each architecture. This README focuses on running the
checked training configurations.

## Published Vela checkpoints

Current Vela embedding and reranking checkpoints use the shared
[Vela Base](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M).
Their model cards provide native loading examples. The BGE/AllNLI and
BGE/Quora/FEVER configurations below describe the original mmBERT training
recipes; they do not reproduce a current Vela checkpoint.

Native evaluation uses the declared encoder dtype with inference autocast
disabled. Embedding pooling/normalization and reranker heads use FP32. Training
may use FP32 master weights with BF16 forward, but development selection must
use the declared evaluation precision. Results from different arithmetic modes
remain separate.

## Train a new task from a standard Base

[`newbase_training.py`](newbase_training.py) trains the complete ModernBERT
encoder from an explicit local Hugging Face Base snapshot. It preserves the
Base's standard configuration, including its RoPE settings. An embedding task
uses masked mean pooling and normalized dimension prefixes; a reranker creates
fresh CLS scoring heads. The default `ExitSpec` supervises all 20 combinations
of layers `[3, 6, 11, 22]` and dimensions `[768, 512, 256, 128, 64]` in each
update. No previous task weights or external teacher are needed. Other exit
sets must include the encoder's full depth and width.

To continue a compatible task checkpoint with a new optimizer, set
`initialization: "continued_task"` and bind its complete loader files through
`base_directory` and `base_files`. This verifies all encoder tensors and reranker
heads without reinitializing them, and records continuation separately from a
fresh Base initialization. `--resume` instead restores an existing optimizer,
scheduler, RNG state and position in the same stream.

Prepare separate dataset directories, each containing `components.jsonl`,
`records.jsonl`, and a `manifest.json` with its `split` and both file SHA256s.
[`FrozenCorpus`](newbase_data.py) checks complete text hashes, normalized text
hashes, component references, and parent groups. Each retrieval record names a
query, all known positives, judged negatives, unjudged candidates, and its
explicit `candidate_component_ids`. Every candidate must belong to one of
those relevance partitions. Semantic-pair records instead contain two
`pair_component_ids` and a `[0,1]` label. Keep source licensing and split/group
exclusion in the data producer; loading a valid manifest does not establish
those properties. Single-text representation records use `component_id` and
no relevance or pair label.

Related, unjudged documents are masked as contrastive alternatives by default.
For a deliberately ordered pair that shares source parents, the producer can
set `contrastive_preference_component_ids` to an explicit subset of its
unjudged candidates. This enables that contrastive comparison while preserving
the unknown relevance label; it does not admit the pair to BCE or Lambda loss.
For suspected false negatives, `contrastive_ignored_component_ids` removes
explicit unjudged candidates from the contrastive denominator. It cannot
override positives, judged negatives, or explicit contrastive preferences.
The candidate pool and labels remain unchanged, and an omitted or empty list
preserves the default mask. Freeze the filtering method and teacher identity
with the data manifest; the loader does not infer exclusions from scores.
For reranking, optional `unjudged_preference_component_ids` selects which
unjudged candidates the positive should outrank under the weak preference loss.
An empty list disables that term for the query; omission retains all unjudged
alternatives. Excluded candidates remain in the complete pool for soft teacher
targets. This field cannot change human judgments.

Freeze batches before training with
[`prepare_stream`](newbase_stream.py). A stream specification contains `seed`,
`cycles`, and a `sources` mapping. Each source specifies `steps_per_cycle`,
`batch_queries`, and `strata`, a mapping from stratum names to record IDs.
Sampling cycles through independent parent groups and preserves the supplied
candidate lists:

```python
from pathlib import Path
import json
from src.training.model_embeddings.mmbert_32k.newbase_data import FrozenCorpus
from src.training.model_embeddings.mmbert_32k.newbase_stream import prepare_stream

corpus = FrozenCorpus.load(Path("/data/retrieval/train"), "train")
specification = json.loads(Path("/data/retrieval/stream.json").read_text())
prepare_stream(corpus, specification, Path("/data/retrieval/draws"))
```

The training JSON uses these fields. Paths point to local files; hashes bind
the exact model, data, stream, and imported code used by the run.

| Fields | Meaning |
| --- | --- |
| `task`, `run_id`, `seed` | `embedding` or `reranker`, a run name, and initialization/RNG seed. |
| `base_directory`, `base_files` | Standard Base snapshot and relative filename-to-SHA256 mapping. |
| `train_directory`, `development_directory`, `train_manifest_sha256`, `development_manifest_sha256` | Separate training and development corpora; optional `train_split`/`development_split` default to `train`/`validation`. |
| `draws_directory`, `draws_sha256`, `steps` | Frozen stream and its exact number of optimizer updates. |
| `code_root`, `execution_lock_path`, `execution_lock_sha256` | Code snapshot and JSON `files` mapping of relative Python paths to SHA256s; every imported task module must be included. |
| `exits` | `layers`, `dimensions`, `layer_weights`, and `dimension_weights`; each weight vector sums to one. Optional `exit_weights` replaces their product with a positive matrix summing to one. `ExitSpec().to_dict()` gives the default 20 exits. |
| `objectives` | Per-source explicit loss configurations described below. |
| `optimizer` | `warmup_steps`; optional `encoder_lr`, `head_lr`, `weight_decay`, `betas`, `eps`, and `minimum_lr_ratio` for AdamW with cosine decay. |
| `training_precision`, `gradient_checkpointing`, `gradient_clip_norm` | FP32 master weights with `float32` or `bfloat16` forward, optional non-reentrant checkpointing, and clipping norm. |
| `source_maximum_tokens`, `training_token_budget`, `evaluation_token_budget` | Per-source complete-input limit and padded microbatch budgets. Inputs are never truncated. |
| `eval_steps`, `maximum_seconds` | Evaluation steps start at zero; optional elapsed-time limit stops before the next update. A running update or evaluation may exceed that limit. |
| `evaluation`, `selection` | Optional reporting slices and numeric development selection policy. Selection also requires `baseline_path` and `baseline_sha256`. |
| `provenance`, `expected_initial_state_sha256` | Optional lineage metadata and an exact initial tensor-state check. |

Embedding objective `kind` is `retrieval`, `cosent`, or `paraphrase`. Retrieval
supports multi-positive contrastive supervision and optional
`distillation_weight` from the current, detached full exit to smaller exits.
`cosent` uses graded pair ordering; `paraphrase` uses binary labels and an
explicit margin. Reranker `kind: ranking` combines configurable
`pairwise_weight`, `bce_weight`, `preference_weight`, and `lambda_weight`.
Unjudged alternatives can contribute preference supervision, but are excluded
from BCE and LambdaLoss; `positive_bce_judged: false` also excludes an
unjudged composite positive from those absolute relevance terms. See
[`ObjectiveConfig`](newbase_batches.py) for scales, temperature, candidate
limits, and defaults. Source names do not select the loss implicitly.

An optional `teacher` configuration adds frozen external soft supervision:

```json
{
  "teacher": {
    "objective": "query_order",
    "weight": 0.25,
    "temperature": 2.0,
    "directory": "/data/retrieval/teacher-cache",
    "manifest_sha256": "CACHE_MANIFEST_SHA256"
  }
}
```

Use `query_order` for rerankers or `relational_cosine` for embeddings. Rerankers
match the teacher's per-query score distribution at the full exit by default
(`exit_supervision: "full"`). Set `teacher.exit_supervision: "all"` to supervise
every configured exit with its existing exit weight. Every exit uses the same
complete candidate pool and cached teacher scores; padding does not contribute
to the distribution. This option applies only to rerankers. Embeddings
match cosine relations at the full exit, with equal weight per eligible anchor
and relation family. Duplicate or related components do not create unrelated
negative pairs; semantic batches also retain their explicitly paired relation.
Both objectives leave the original supervised losses and relevance masks
unchanged. Teacher tensors are detached. Omitting the configuration, or setting
its weight to zero, preserves the original loss computation.

An independent embedding `anchor` configuration uses `objective: "pointwise_cosine"`
with the same cache identity fields and a same-width teacher. It matches absolute
unit-vector coordinates at every exit, explicitly normalizing each Matryoshka
prefix. The loss is mean squared vector distance divided by two, without an
additional division by width. Per-source `anchor_scale` adjusts its contribution.
By default every student depth uses the same teacher vector. Set
`anchor.target_layers` to an explicit ordered list, such as `[3, 6, 11, 22]`,
to supply a separate teacher vector for each supervised depth. The list must
cover exactly the student's trained depths. The cache manifest must declare
the same ordered `target_layers`, with unit vectors shaped
`[entries, len(target_layers), dimensions]`. Each student depth then matches
its corresponding teacher vector; dimensions still use normalized prefixes.
Layer-specific caches apply only to pointwise embedding anchors.
The `representation` objective supports single-text batches with anchors and/or
full-logical-batch relations; microbatch accumulation alone does not enlarge the
relation pool. Different-width teachers remain valid for relation distillation.

Generate the cache only from the frozen training stream. Its `manifest.json`
contains `complete: true`, `task`, `source_split`, `train_manifest_sha256`,
`draws_sha256`, `dimensions`, and a `files` mapping from `entries.jsonl` and
`values.safetensors` to SHA256s. A `teacher` object records `repo_id`, `revision`,
model `files` and the actual `representation`. Each JSONL entry has an ordered
`identity` made with [`input_identity`](newbase_teacher.py), its
`identity_digest` as `key`, a contiguous `value_index`, and a `token_sha256`
made with `token_digest(input_ids, pad_id)`. `record_inputs` enumerates every
required component or query/document pair. The safetensors file contains one
FP32 `values` tensor: unit vectors shaped `[entries, dimensions]` for embedding,
or scores shaped `[entries, 1]` for reranking. Cache loading checks exact input
coverage, corpus/stream identities, file hashes, geometry and finite values;
each lookup also checks the actual complete token sequence. Tokenization
mismatches fail rather than reusing another input's prediction. Cache generation,
source admission and teacher choice remain the caller's responsibility.

Run from the repository root with an environment that supports the Base's
actual Transformers configuration and accelerator. Expose one assigned device
for GPU training. The configuration hash below is the SHA256 of the JSON file:

```bash
export PYTHONPATH="$PWD"
python -m src.training.model_embeddings.mmbert_32k.newbase_training \
  --config /data/retrieval/task.json --config-sha256 CONFIG_SHA256 \
  --output /data/retrieval/run --device cuda

python -m src.training.model_embeddings.mmbert_32k.newbase_scoring \
  --model /data/retrieval/run/step-100 --task embedding \
  --known-dev /data/retrieval/validation --split validation \
  --output /data/retrieval/evaluation --device cpu --token-budget 32768
```

Use `--task reranker` for a reranker checkpoint. Evaluation runs in FP32,
reports every configured exit, and writes aggregate `metrics.json` and numeric
per-record scores. Optional `--evaluation-config` names source/language/length
slices or semantic metrics. Selection constraints and score weights are
explicit configuration, with identical dataset support and precision required
for baseline comparisons.

Each `step-N` saves the complete updated encoder, tokenizer, exit/representation
metadata, and `newbase_checkpoint.json`. Rerankers additionally save every
head in `classification_heads.safetensors` and `matryoshka_config.json`.
`NewBaseTask.resume()` verifies hashes and reloads all exits. Training
checkpoints also preserve optimizer, scheduler, RNG, and evaluation history;
use `--resume /data/retrieval/run/step-N` with the same configuration and a new
output directory to continue the same stream. The
[two-step CPU example](tests/test_newbase_training.py) exercises a complete
configuration, training, checkpoint creation, and exact next-update resume.
Export and engine qualification remain separate from this training interface.

## Install

Use Python 3.10+ and the patched PyTorch 2.13 runtime. For CPU validation,
install `torch==2.13.0` from `https://download.pytorch.org/whl/cpu` first.
For the pinned ROCm base in `runtime.json`, replace its old Torch stack before
installing the family dependencies (the family does not use torchaudio):

```bash
python -m pip uninstall -y torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
  'torch==2.13.0+rocm7.2' 'torchvision==0.28.0+rocm7.2'
```

Then install the family dependencies:

```bash
python -m pip install --requirement \
  src/training/model_embeddings/mmbert_32k/requirements.txt
```

Run `python -m pip check` after installation. The base-image metadata in
`runtime.json` describes the original image; `pytorch_install` is mandatory and
overrides its older Torch build. The safety workflow Dockerfile performs the
same HIP-wheel installation and checks that generic requirements did not replace
it with a CUDA wheel. CPU contract tests do not qualify a ROCm image or GPU
training run; keep the separate accelerator validation, resolved configuration,
and output checksums with each release run.

Transformers 5 uses per-attention-type ModernBERT RoPE parameters and reports a
normalized final hidden state. The trainer validates the shared YaRN frequencies
and explicitly captures the pre-normalization final exit required by legacy
reranker heads. Full task-quality retraining remains separate from these
small-model compatibility tests.

The existing Accelerate pin is retained because no patched release is available for
[GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79).
Even release 1.15 and upstream main still lack the checkpoint-loader fix,
although 1.15 falls outside the advisory's current version range. The supported
Transformers load/Trainer path is regression-tested with those Accelerate
checkpoint APIs disabled; do not use `load_checkpoint_in_model` or
`load_checkpoint_and_dispatch` for untrusted artifacts. This scoped avoidance
does not fix the installed upstream library or justify dismissing its alerts.

## Configure paths

Run from the repository root and keep datasets and checkpoints outside Git:

```bash
export PYTHONPATH="$PWD/src"
export MMBERT32K_FOUNDATION_DATA=/path/to/tokenized-cc100-32k
export MMBERT32K_FOUNDATION_OUTPUT=/path/to/mmbert-32k-yarn
export MMBERT32K_BGE_DATA=/path/to/bge-m3-data
export MMBERT32K_EMBEDDER_OUTPUT=/path/to/mmbert-embed-32k-2d
export MMBERT32K_RERANKER_OUTPUT=/path/to/mmbert-rerank-32k-2d
```

All model and dataset revisions, random seeds, objectives, and production
hyperparameters live in the JSON configurations. Command-line arguments are
applied after the config and are explicit overrides.

## Inspect commands first

The family runner can resolve a command without importing the ML stack:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage train --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/embedder.json \
  --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/reranker.json \
  --print-command
```

Remove `--print-command` to execute the selected workflow.

## Train the 32K foundation

Foundation data preparation streams the nine configured CC-100 languages,
inserts tokenizer document boundaries, and emits only complete 32,768-token
sequences. The target is 30,774 sequences: 3,420 each for `en`, `zh-Hans`, and
`de`, followed by 3,419 each for `fr`, `es`, `ru`, `ar`, `ja`, and `ko`.

Preparation uses two passes so the second pass can verify the exact compressed
source prefixes observed by the first:

```bash
export MMBERT32K_FOUNDATION_AUDIT=/path/to/mmbert32k-cc100-audit

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare --output_dir "$MMBERT32K_FOUNDATION_AUDIT"

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare \
  --source_prefix_contract_dir "$MMBERT32K_FOUNDATION_AUDIT"
```

The replay writes a packing manifest that binds source counters and hashes,
packed-token content, Arrow schema, and every shard checksum. Foundation
training validates that manifest before loading model weights:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage train
```

The training config applies official YaRN position scaling before model
construction, validates each rotary layer, uses SDPA attention, and continues
masked-language modeling for one epoch. It also flushes and correctly scales
the final partial gradient-accumulation window.

CC-100's dataset card does not declare a license and its content is derived
from Common Crawl. Preparation requires the explicit acknowledgement recorded
in the config. Review the dataset card, Common Crawl terms, privacy, and the
intended release jurisdiction before redistributing packed data or publishing
new weights.

## Train the embedder

Download the BGE-M3 training data at the revision pinned in the config, then
run:

```bash
hf download Shitao/bge-m3-data \
  --repo-type dataset \
  --revision a69db8b86e9c1767d193ee0de95e5c4001a71eae \
  --local-dir "$MMBERT32K_BGE_DATA"

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/embedder.json
```

The configuration trains normalized embeddings at dimensions 768, 512, 256,
128, and 64 and enables adaptive-layer supervision. It also includes the
pinned AllNLI training source and STS-B evaluation source.

## Train the reranker

The reranker uses the same BGE-M3 data directory:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/reranker.json
```

The trainer creates heads at layers 3, 6, 11, and 22 for dimensions 768, 512,
256, 128, and 64. Keep `classification_heads.pt` and
`matryoshka_config.json` with the encoder and tokenizer when packaging the
artifact.

## Portable launchers

These wrappers resolve through the same configs:

```bash
bash src/training/model_embeddings/mmbert_32k/run_rope_training.sh --print-command
bash src/training/model_embeddings/mmbert_32k/run_bge_style_training.sh --print-command
bash src/training/model_embeddings/mmbert_32k/run_rerank_2d_matryoshka_training.sh --print-command
```

Set `MMBERT32K_STAGE=prepare` when using `run_rope_training.sh` for data
preparation. `MMBERT32K_PYTHON_BIN` selects a non-default Python executable.

## Validate

The lightweight contract suite does not download models:

```bash
python -m unittest discover \
  -s src/training/model_embeddings/mmbert_32k/tests \
  -p 'test_*.py'
```

Before release, evaluate the embedder at every supported dimension and selected
layer, evaluate all reranker heads, and run long-context loading and loss checks
for the foundation. Record the exact revisions, resolved config, dataset
manifest, metrics, and artifact checksums.
