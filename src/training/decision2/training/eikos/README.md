# Eikos native-letter architecture arm

This is a bounded Decision 2.0 architecture experiment, initialized from
[`caiovicentino1/Eikos-4B`](https://huggingface.co/caiovicentino1/Eikos-4B)
revision `582ffb13f19a4da3f455e3db198584190bd7755b`. Its released
`decision_core.py`, `letter_adapter.py`, and `serve.py` define the SemIf prompt,
the `A..Z, AA..` single-token labels, last-token LM-head readout, and
temperature feature contract. We train PEFT LoRA projections in that text
backbone while preserving the released readout and tokenizer. The source
release includes MIT contributions, the Qwen Apache-2.0 license, and `NOTICE`;
downstream releases must retain these notices and the training-data provenance.

The arm tests whether language-model letter logits transfer better than our
Decision 1.0/2.0 dynamic candidate head. Eikos's published scores are author
measurements; our unified benchmark supplies the comparison. This experiment
uses no Jev outputs or final benchmark labels in training.

**Rights status:** the initial `balanced_human_5824_v1` pilot contains upstream
TweetEval and SemEval rows with noncommercial or research-only terms. The
project's stated scope is noncommercial research. Its existing adapter and
merged package may be evaluated and considered for a release within that
scope, with exact source/provenance attestation and visible upstream conditions;
raw source rows are not redistributed. A separate rights-clean TRAIN/SELECT/CAL
split supplies an architectural control. This does not authorize commercial use.

## Data and gates

- The clean-control trainer requires a `decision2-rights-clean-splits/1`
  manifest with `publication_eligible=true`, source-rights and overlap audits,
  and exact TRAIN/SELECT/CAL row counts, byte sizes, and SHA256s. The exporter
  also accepts the original pilot with a separate noncommercial research
  attestation bound to its manifest, training provenance, and every source
  count. Frozen package inference requires this attestation for an old pilot.
- The earlier `balanced_human_5824_v1` pilot contained 5,824 TRAIN rows, a
  separate 600-row SELECT, and 900-row hard CAL. Their upstream terms must be
  disclosed; the clean-control command below does not accept them.
- The Eikos-4B release declares `max_one_pass=100`. Six TRAIN rows have 105 or
  127 options. `one_pass_quarantine` writes their IDs, family, type, source,
  original rights string, input digest, and reason to `quarantine.jsonl`.
  Exactly 5,818 rows produced gradients in the initial pilot. Each clean run
  writes its own explicit quarantine and effective counts to `provenance.json`.
  Training refuses SELECT or CAL above the native one-pass option limit.
- The longest old-pilot native TRAIN prompt had 6,240 tokens, so its limit was 8,192.
  Choice option order receives one deterministic permutation per row. Boolean
  options always render as native yes/no; ordinal levels keep their order.
- The training evaluator first scores every checkpoint on SELECT by
  family-macro accuracy, then Brier, then earliest step. `native_select` then
  reranks the durable checkpoints and original source on the **same SELECT**
  rows using released `serve.Decider`; its `NATIVE_BEST.json` is authoritative
  for calibration and deployment. Hard CAL labels are parsed for leakage
  checks during training and only opened for calibration **after** native
  SELECT has fixed a checkpoint. DEV/CSS pilot and pressure panels are external diagnostics,
  never used by this trainer's checkpoint selector. Final holdout gold is never
  accepted by these commands.

## Reproduce on one qualified GPU

Mount the task source, pinned model files and isolated data under `/work` in
the qualified ROCm image. Set `ROCR_VISIBLE_DEVICES` to one assigned device and
`PYTHONPATH=/opt/decision-fla:/work/source`, `PROMPT_STYLE=semif`,
`HF_HUB_OFFLINE=1`. The commands below refer only to paths inside the task
mount; the local trainer's code and SHA256 are captured in `provenance.json`.

```bash
python3 -m training.eikos.train \
  --model-path /work/models/Eikos-4B \
  --train /work/data/rights_clean_v1/rights_clean.train.jsonl \
  --select /work/data/rights_clean_v1/select.jsonl \
  --cal /work/data/rights_clean_v1/cal.jsonl \
  --data-manifest /work/data/rights_clean_v1/rights_clean.manifest.json \
  --output /work/runs/eikos4b-rights-clean-v1-r1 \
  --microbatch 2 --accumulation 16 --save-every 32 \
  --lora-rank 8 --lora-alpha 16 --learning-rate 2e-5 \
  --brier-weight 0.25

python3 -m training.eikos.native_select \
  --model-path /work/models/Eikos-4B \
  --run /work/runs/eikos4b-rights-clean-v1-r1 \
  --select /work/data/rights_clean_v1/select.jsonl

python3 -m training.eikos.calibrate \
  --model-path /work/models/Eikos-4B \
  --run /work/runs/eikos4b-rights-clean-v1-r1 \
  --cal /work/data/rights_clean_v1/cal.jsonl \
  --output /work/runs/eikos4b-rights-clean-v1-r1/hard-cal

python3 -m training.eikos.infer \
  --model-path /work/models/Eikos-4B \
  --run /work/runs/eikos4b-rights-clean-v1-r1 \
  --calibration /work/runs/eikos4b-rights-clean-v1-r1/hard-cal/calib.json \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/eikos4b-rights-clean-v1-r1/dev.predictions.jsonl
```

The native serving path merges the selected LoRA and applies the exact source
prompt/readout. Baseline and LoRA SELECT values come from the same training
evaluator and the same released serving evaluator. On the unadapted SELECT600,
the fast training evaluator scored 317 correct and the original serving path
scored 320; eleven answers differed (mean maximum per-item probability
difference 0.0092). This is why `native_select` is required before calibration.
The public model must use a qualified inference path and carry its own
calibration file. Do not infer performance from internal SELECT alone.

## Standalone candidate export

`export` merges the **native SELECT** checkpoint into the pinned Eikos base,
keeps the released SemIf prompt and letter-logit server, and embeds hard-CAL
temperatures. It refuses a changed source release, adapter, calibration,
training data manifest, or existing output directory. The candidate has an
immutable `decision2_provenance.json` and a complete `SHA256SUMS` covering
weights, tokenizer, serving files, and notices. It remains unreleased until
merged-model inference parity and independent final evaluation pass.

```bash
python3 -m training.eikos.export \
  --model-path /work/models/Eikos-4B \
  --run /work/runs/eikos4b-rights-clean-v1-r1 \
  --calibration /work/runs/eikos4b-rights-clean-v1-r1/hard-cal/calib.json \
  --calibration-report /work/runs/eikos4b-rights-clean-v1-r1/hard-cal/report.json \
  --data-manifest /work/data/rights_clean_v1/rights_clean.manifest.json \
  --output /work/models/dev-2.0-4b-eikos-candidate
```

The standalone folder serves with its copied `serve.py`; for example,
`python3 serve.py --model . --device cuda:0`. Its `decision_config.json`
points to the embedded `calib.json`. The source Eikos contribution carries
MIT terms, and the Qwen base carries Apache-2.0 terms; both license texts and
the upstream NOTICE are copied verbatim. The model card must name the Eikos
and Qwen lineage, the rights of the fine-tuning data, the 100-option one-pass
contract, the run-specific quarantined TRAIN records, and measured limitations.

The `published_infer` collector uses only a frozen standalone folder. It
checks every package file against `SHA256SUMS` before loading the model, and
assigns `model_sha256 = SHA256(SHA256SUMS bytes)` to each gold-free prediction.
The embedded `calib.json` hash appears on every prediction and in an atomic
prediction manifest alongside input hash, counts, native runtime, model ID,
selected checkpoint, and collector source hash. Score the prediction file only
after verifying that manifest against the frozen package.
For the original noncommercial research candidate, pass `--rights-attestation`
with the exact `decision2-noncommercial-research-attestation/1` JSON. The
collector verifies its TRAIN/SELECT/CAL and training-provenance bindings and
writes `rights_attestation_sha256` to every answer row and the prediction
manifest. This attestation is an external document; it does not alter the
functional package SHA256 or provide rights to redistribute raw source rows.
The functional file roster excludes exactly the later publication documents:
`README.md`, root score/rank/matrix files, matching `card-artifacts/` files,
and `publication-manifest.json`. Their hashes must be tracked separately;
unlisted model, code, tokenizer, calibration, or license files are rejected.

`verify_export --direct-selected` loads the selected source LoRA and the merged
package in the **same process** and compares all categorical answers and
probabilities without reading labels. The predeclared gate is zero categorical
differences, p99 maximum per-option drift no greater than 0.005, and worst
per-option drift no greater than 0.02. Run it on the full DEV and three CSS
pilot prompts, then bind both immutable reports with `parity_receipt` before
freezing the candidate. Historical predictions are a separate repeatability
check; they are not the direct parity reference.

```bash
python3 -m training.eikos.published_infer \
  --model-path /work/models/dev-2.0-4b-eikos-candidate \
  --input /work/runs/dev.prompts.jsonl \
  --output /work/runs/eikos4b-rights-clean-v1-r1/published-dev.predictions.jsonl \
  --model-id llm-semantic-router/dev-2.0-4b \
  --model-revision "$(python3 -c 'import json; print(json.load(open("/work/runs/eikos4b-rights-clean-v1-r1/NATIVE_BEST.json"))["checkpoint"])')"
```

Run CPU tests with `python3 -m unittest discover -s training/eikos/tests -q`.
