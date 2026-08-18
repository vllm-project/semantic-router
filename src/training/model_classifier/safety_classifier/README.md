# mmBERT-32K Safety Classifier Reconstruction

This directory is the canonical source for two hierarchical content-safety
classifiers aligned with the repository's other current mmBERT classifiers:

- Level 1: binary `safe` / `unsafe` prompt classification;
- Level 2: the historical nine-output hazard classifier, explicitly versioned
  as `legacy-9-v1`.

The earlier Hugging Face repositories were trained from
`jhu-clsp/mmBERT-base`. Their model cards and adapter metadata survive, but the
producing trainer, exact split, seed, and optimizer state do not. This workflow
therefore produces new deterministic checkpoints from the pinned
`llm-semantic-router/mmbert-32k-yarn` base. It does not claim bit-for-bit
reproduction and does not overwrite the historical models.

## Reconstructed Contract

The machine-readable source of truth is
[`configs/reconstruction-v1.json`](configs/reconstruction-v1.json). It pins:

- base-model and dataset revisions;
- raw input-file SHA-256 digests;
- prompt-only normalization, de-duplication, split precedence, and sampling;
- the `legacy-9-v1` source-taxonomy crosswalk;
- max length 512;
- disabled ModernBERT reference compilation so distributed ranks do not each
  create a large TorchInductor worker pool;
- LoRA rank 32, alpha 64, dropout 0.1, and the four ModernBERT target modules;
- global batch 64, 10 epochs, AdamW, linear warmup, and all random seeds;
- distinct adapter and merged release repositories.

Facts recovered from historical cards are kept separate from reconstruction
decisions in the contract and public documentation.

## Data Contract

`data.py prepare` downloads only pinned files and verifies their hashes. It
uses AEGIS `prompt` and `prompt_label`; response/refusal variants are excluded.
Empty and redacted prompts are removed. Fingerprints use NFKC normalization,
Unicode whitespace collapse, case folding, and SHA-256, while the original
stripped text remains the model input.

Holdout precedence is `test > validation > train`. Any lower-precedence
duplicate is removed before sampling, and conflicting-label fingerprint groups
are dropped entirely.

- Level 1 emits 10,000 training rows per label without replacement.
- Level 2 emits 2,000 rows per label. Underrepresented classes use documented,
  deterministic oversampling; validation and test remain natural-distribution
  AEGIS data.
- Multi-hazard AEGIS rows use the first mapped category in the dataset's source
  order. All mapped targets remain in each row for audit and strict-subset
  evaluation.

## Lightweight Validation

The data, taxonomy, metrics, contract, and artifact-shape tests require only
the Python standard library:

```bash
python -m unittest discover \
  -s src/training/model_classifier/safety_classifier/tests \
  -p 'test_*.py' -v
```

## ROCm Environment

The Dockerfile uses an immutable official ROCm/PyTorch base digest. Build it
from the repository root:

```bash
docker build \
  -f src/training/model_classifier/safety_classifier/Dockerfile.rocm \
  -t semantic-router-mmbert32k-safety:reconstruction-v1 .
```

The image carries conservative single-node RCCL defaults verified for the
eight-device workflow: scratch reclaim is disabled, GPU P2P is disabled in
favor of shared-memory collectives, and the channel count is capped at eight.
These settings trade some collective bandwidth for bounded startup time; LoRA
gradient communication is small relative to the frozen base model.

Mount the repository, a persistent Hugging Face cache, and credentials using
the normal secret mechanism for the environment. Do not copy tokens into the
image or repository.

## Prepare Once

Data preparation must run once, before distributed launch, rather than once per
rank:

```bash
python -m src.training.model_classifier.safety_classifier.data prepare \
  --contract src/training/model_classifier/safety_classifier/configs/reconstruction-v1.json \
  --output-dir /artifacts/data
```

The command writes deterministic JSONL splits and manifests under
`/artifacts/data/{level1,level2}`. Keep this directory outside Git.

## Eight-Accelerator Training

The checked contract reproduces global batch 64 as `8 samples × 8 ranks × 1`
gradient accumulation step:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.training.model_classifier.safety_classifier.train \
  --task level1 \
  --expected-world-size 8 \
  --data-dir /artifacts/data \
  --output-dir /artifacts/runs/level1

torchrun --standalone --nproc_per_node=8 \
  -m src.training.model_classifier.safety_classifier.train \
  --task level2 \
  --expected-world-size 8 \
  --data-dir /artifacts/data \
  --output-dir /artifacts/runs/level2
```

Use `--max-steps 2` for an accelerator smoke. Override runs are marked
non-release-eligible in `training_manifest.json`.

## Evaluate, Merge, and Publish

Run independent evaluation against the materialized test split, then merge and
compare logits:

```bash
python -m src.training.model_classifier.safety_classifier.evaluate \
  --task level1 \
  --model /artifacts/runs/level1/adapter \
  --artifact-type adapter \
  --data /artifacts/data/level1/test.jsonl \
  --output-dir /artifacts/runs/level1/evaluation

python -m src.training.model_classifier.safety_classifier.export \
  --task level1 \
  --run-root /artifacts/runs/level1 \
  --merged-dir /artifacts/runs/level1-merged
```

Repeat for Level 2. Publication refuses existing repositories by default and
performs a remote checksum plus load/inference verification:

```bash
python -m src.training.model_classifier.safety_classifier.release \
  --task level1 \
  --run-root /artifacts/runs/level1 \
  --merged-dir /artifacts/runs/level1-merged
```

Every published adapter contains `adapter_model.safetensors`; every merged
repository contains full `model.safetensors` weights. Both include label,
contract, data, dependency, metric, parity, and file-checksum manifests.
