# APUS native eligible final appendix

This is a separate comparison slice for pinned [APUS OpenJev 4B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/tree/422b3741f8b5c092eeefef847c1ca89d78337d45) and [9B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-9B/tree/82c9c56cfa9de8d36704ed91948d4726ef111635). It uses the native `OpenJet` high/full-depth API. APUS's portable release supports Choice and binary Noul; a complete ordinal Score answer is unavailable. The final synthetic report must show both the full typed-question denominator, with Score invalid, and the eligible Choice+Noul denominator. APUS stays out of the all-type rank and matrix. The 15-task CSS panel has Choice questions only.

`python -m inference.apus_eligible plan` is a read-only **evaluation planner** with one exception: it writes its own new plan JSON. It requires the already saved main final plan plus its externally recorded SHA-256, the matching pre-test freeze manifest, and the gold-free final and CSS evaluation prompt files. It refuses a missing final prompt file, a changed freeze or protocol source, changed CSS bytes, unexpected question types/counts, and an existing appendix output directory or plan file. It never reads either gold file or runs a model. The main final generator must have run only **after** the Decision 2.0 checkpoints and CAL artifacts were frozen; see `scripts/FINAL_EVAL.md`.

After the main final plan is saved, its SHA is recorded, and the fresh synthetic final gold-free prompts have been generated, run this on the experiment host's exact source mirror. Keep the private plan path outside the new appendix output directory:

```bash
PYTHONPATH="$SOURCE_ROOT" PYTHONDONTWRITEBYTECODE=1 \
python3 -m inference.apus_eligible plan \
  --main-plan "$MAIN_PLAN_JSON" \
  --expected-main-plan-sha256 "$MAIN_PLAN_SHA256" \
  --freeze-manifest "$FREEZE_JSON" \
  --final-prompts "$EVAL_ROOT/final.prompts.jsonl" \
  --css-prompts "$CSS_PANEL_DIR/css-evaluation.prompts.jsonl" \
  --source-root "$SOURCE_ROOT" \
  --model-root "$MODELS_ROOT" \
  --output-root "$EVAL_ROOT/apus-eligible" \
  --plan-path "$PRIVATE_APUS_PLAN_JSON"
```

The command writes the saved plan exactly once and prints its SHA-256. Save that digest independently. Review the plan's `preflight_commands`, `inference_commands`, `raw_prediction_hash_command`, and `scoring_commands` before executing them in that order. Its inference commands must run inside the APUS Transformers 5.16.1 ROCm image described in `Dockerfile.apus-rocm`, one model per process with an assigned `GPU_ID`. The CLI downloads the pinned HF 4B/9B snapshots on the GPU host before preflight; each preflight verifies local HF revision metadata and the released weight/runtime/config hashes. The plan uses a separate new output directory and does not alter the main Decision 2.0 publication config.

The scoring commands contain the final and CSS gold **paths** for later use after the agreed final evaluation begins. They are only printed at planning time. Hash all original APUS predictions using `raw_prediction_hash_command` before scoring. The synthetic scorer is `typed-decision-report/2`; the CSS scorer is `css-transfer-score/2`. Once those reports exist, audit and generate the separate appendix JSON without opening gold:

```bash
PYTHONPATH="$SOURCE_ROOT" PYTHONDONTWRITEBYTECODE=1 \
python3 -m inference.apus_eligible summarize \
  --plan "$PRIVATE_APUS_PLAN_JSON" \
  --expected-plan-sha256 "$APUS_PLAN_SHA256" \
  > "$EVAL_ROOT/apus-eligible/APUS_ELIGIBLE_REPORT.json"
```

The audit binds both APUS revisions, adapter/version, release manifests, the main freeze, each gold-free input hash, original prediction hash ledger, and v2 score reports. It checks that every Score answer explicitly says `unsupported_native_ordinal_score`, and that the Score slice has zero valid answers. It reports Choice+Noul accuracy with **invalid or missing eligible responses counted as misses**, CSS Choice accuracy and 15-task macro statistics, and the transparent full typed-question denominator. The final generator may put Choice and Noul in the same item, so these denominators count question responses rather than only the 1,600 JSONL items. The audit reads the gold digests from score reports and requires each APUS model to use the same gold bytes; it does not open gold itself.

The APUS model cards declare Apache-2.0 weights, but their training provenance lists public sources under different licenses and does not publish all raw training examples. Example-level training overlap with a later final panel cannot be ruled out. The released runtime was qualified on CUDA PyTorch 2.8; this separate ROCm evaluation records its own runtime qualification, with no numerical parity claim. Native candidate probabilities are uncalibrated. The earlier [DEV/CSS pilot report](APUS_DEVELOPMENT_2026-09-26.md) is development evidence and must not be presented as final held-out results.
