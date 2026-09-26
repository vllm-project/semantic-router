# Locked Decision 2.0 final evaluation

This is a **future command plan**, not a request to run the held-out panels.
`plan_final_eval.py` is read-only and uses no GPU or API. It emits commands
only after every proposed Decision 2.0 checkpoint and its independent CAL
temperature report are frozen. This stage does not generate or read synthetic
final gold, invoke the 15-task CSS evaluation prompts, call Jev, or publish.

## Freeze before opening either final panel

Finish training and select each candidate on TRAIN/SELECT, synthetic DEV, or
the three CSS pilot tasks. Fit CAL only after `BEST.json` and `COMPLETE.json`
freeze the selected checkpoint. Create a private pre-test JSON declaration
with this shape; use actual SHA-256 values and paths, never placeholders:

```json
{
  "freeze_version": "decision2-pretest-freeze/1",
  "selection_sources": ["train", "select", "cal", "synthetic_dev", "css_pilot"],
  "candidates": [
    {
      "key": "d2-9b",
      "label": "dev-2.0-9b",
      "size": "9B",
      "model_id": "llm-semantic-router/dev-2.0-9b",
      "run_dir": "/ABS/complete-run",
      "cal_data": "/ABS/audited-cal.jsonl",
      "calibration": "/ABS/per-type-calibration.json",
      "source_path": "/ABS/pinned-source-model",
      "selected_checkpoint": "checkpoint-0000025",
      "model_sha256": "64 lowercase hex digits",
      "calibration_sha256": "64 lowercase hex digits",
      "best_sha256": "64 lowercase hex digits",
      "complete_sha256": "64 lowercase hex digits",
      "provenance_sha256": "64 lowercase hex digits"
    }
  ]
}
```

Omit `source_path` for a full checkpoint. Add more candidates to the same
declaration **before** final generation if the family has multiple frozen
sizes. One declaration yields one shared synthetic final set and one CSS
evaluation run per model. The planner requires the completed run's BEST and
COMPLETE receipts, audited CAL data hash, actual source+adapter+head+tokenizer
model hash, original CAL report bytes, and CAL lineage to match the
declaration. It rejects `synthetic_final` and `css_evaluation` as selection
sources. That declaration documents the decision; it cannot prove nobody
looked at a public test split, so keep the freeze timing and run logs.

For the 4B native SemIf architecture, use `architecture: "eikos_semif"` in its
candidate entry. The entry has the same identity fields, but `best_sha256`
hashes `NATIVE_BEST.json` because the released native serving SELECT rerank is
authoritative. It also requires absolute `source_path` (the pinned original
Eikos release), `package_dir` (the fully checked standalone candidate),
`calibration_report`, `training_data_manifest`, and `parity_reports` mapping
`dev` and `css_pilot` to independent gold-free same-process parity JSON files.
Each report must bind the selected adapter, package SHA256SUMS, exact CAL and
full prompt panel. The predeclared gate requires zero categorical changes,
p99 maximum-option probability drift at most 0.005, and maximum drift at most
0.02. The planner verifies these reports before opening either final panel.
The published inference command reads the exact packaged native SemIf model,
then emits a model/CAL-bound prediction manifest. Card text and ranking images
are separate documentation files and cannot alter the checked executable
package identity.

The CSS input is the frozen **gold-free** 6,547-row
`css-evaluation.prompts.jsonl` with SHA-256
`7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`.
The planner parses this prompt file and verifies the exact digest and count;
it never opens CSS labels. The synthetic final generator and both v2 scorers,
paired comparison code, inference adapters, and publication loader are pinned
by source SHA-256 inside the planner. Review and deliberately update those
pins if code changes before execution.

## Produce the command plan after freeze

Use the same path namespace that the later inference processes will see.
`EVAL_ROOT` must be a new, absent directory. The planner prints JSON or
Markdown to stdout and never creates `EVAL_ROOT` itself:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.plan_final_eval \
  --freeze-manifest "$FREEZE_JSON" \
  --css-prompts "$CSS_PANEL_DIR/css-evaluation.prompts.jsonl" \
  --source-root "$SOURCE_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  --model-root "$MODELS_ROOT" \
  --external-root "$EXTERNAL_ROOT" \
  --kai-lex-python "$KAI_LEX_PYTHON" \
  --fla-path "$QUALIFIED_FLA_PATH" \
  --format json > "$PRIVATE_PLAN_JSON"
```

Record `sha256sum "$PRIVATE_PLAN_JSON"` in the private research ledger before
running the first final-panel command. Keep the recorded 64-character digest
outside the plan and pass it as `PLAN_SHA256` to each audit. The planner also
requires absolute paths so later commands do not depend on the working
directory. Run from a complete exact source mirror that includes `publication/`
and the pinned training-model modules.

The generated plan contains exact commands for:

1. Creating a private fresh 32-byte seed and a family-disjoint synthetic
   final set (`benchmark.generate --split final`, 100 groups per family).
   **Execute only after** the freeze declaration is locked. The plan merely
   prints these commands at this stage.
2. Native inference for each frozen 2.0 model and every baseline on the same
   synthetic-final and CSS-evaluation gold-free prompts. Allocate a GPU per
   process and set `GPU_ID`; run Jev with a protected token file supplied on
   stdin. The script never reads or stores an API token.
3. Hashing all original predictions and Jev API receipts **before** scoring.
   Fresh Jev responses are normalized with `transfer.normalize_jev`, which
   verifies the API-body hash and writes the state/questions input hash;
   `clients.normalize_jev` is the legacy DEV converter and must not be used.
4. `benchmark.score` and `transfer.score`, each at the exact pinned v2 source
   hash, producing separate new reports. Missing/invalid outputs remain in
   denominators and original prediction hashes must match the reports.
5. Paired 5,000-replicate confidence intervals for every 2.0 versus baseline
   pair. The synthetic comparator resamples four-variant groups within each
   final family; the CSS comparator resamples identical item IDs within each
   of the 15 tasks.
6. A predeclared `publication_config` with every model and paired CSS report,
   followed by `publication.generate` for `score-table.md`, `ranking.svg`,
   `matrix.svg`, and their integrity manifest. No upload is included.

After prediction hashing and both v2 score sets exist, run the read-only
score audit. It checks original prediction bytes, source hashes, report
versions, model revisions, frozen 2.0 model/CAL manifests, complete panel
counts, and a common gold digest **from the reports**; it never opens gold:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.audit_final_eval \
  --plan "$PRIVATE_PLAN_JSON" --expected-plan-sha256 "$PLAN_SHA256"
```

Before generating card artifacts, copy the predeclared publication config
from the saved plan rather than editing displayed results after seeing them:

```bash
python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); open(sys.argv[2],"x").write(json.dumps(p["publication_config"],indent=2)+"\n")' \
  "$PRIVATE_PLAN_JSON" "$EVAL_ROOT/publication-config.json"
```

After paired reports and rank/matrix artifacts exist, use
`scripts.audit_final_eval --plan "$PRIVATE_PLAN_JSON" --expected-plan-sha256 "$PLAN_SHA256" --complete` to validate
their exact prediction/gold bindings, bootstrap settings, publication config,
and artifact SHA-256 values. Keep raw logs, receipts, prediction manifests,
the pre-test freeze declaration, and the saved command plan private. Review
the resulting model card and licenses before a separate publication decision.

## Native runtime catalog and comparison limits

| Model | Exact release | Native command | Qualification and type coverage |
| --- | --- | --- | --- |
| Jev 1.13 | API model `jev-1.13.0` | `clients.jev_api` then `transfer.normalize_jev` | Official hosted typed API; external latency/cost/runtime; stochastic outputs may vary. |
| Decider 4B | HF `eb5fbdfc9448473ec25e399882912863afbdb70e` | `inference.run --backend decider` | Native Choice/Noul/Score, eager ROCm; native probability rounding retained. |
| Kev 4B | HF `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`, source `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63` | `inference.kev` | Native typed API; strict context overflow invalid; ROCm numeric parity with H200 release not established. |
| This-That 1.0 | HF `3d927195c4f9845efe66c5715883a7a0f42b1239`, source `4efe782ccbb9c1979a9951c35a29a8e0b3b80bf0` | `inference.this_that` | Native declared-option Choice; Noul/Score are generic-head projections; context overflow invalid; ROCm unqualified. |
| Laya Typed | HF `1a793eb568e6718f15941d08f85432581df534e3`, source `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0` | `inference.laya` | Native typed API; 1024-token context and 256-token head can truncate; truncated answers invalid; ROCm unqualified. |
| Eikos 4B | HF `582ffb13f19a4da3f455e3db198584190bd7755b` | `inference.eikos` | Native SemIf letter-logit Choice/Noul/Score, released calibration, one-pass limit 100 options; ROCm numeric parity not author-qualified. |
| JevK5 2B | HF `7922d1f55df137b72ef763fced56fd09efc5e99d`, runtime `1e5ae1b533b9eb80c0cbe3fbd010607d0b4e26ae` | `inference.jevk5 --size 2b` | Native typed API, multi-pass option readout and expected-value Score; direct release-file hashes. |
| JevK5 4B | HF `c4f7fdb3aeab5582336406e78d3bef11bf98833d`, same runtime | `inference.jevk5 --size 4b` | Native typed API, multi-pass option readout and expected-value Score; release SHA256SUMS. |
| JevK5 9B | HF `d6521a18a86999190e9d775c915af3d6d6772fc4`, same runtime | `inference.jevk5 --size 9b` | Same native contract as JevK5 4B. |
| Decision 1.0 Eos | HF `3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd` | `inference.run --backend eos` | Native typed bundle; published qualified ROCm+FLA runtime required. |
| Decision 1.0 Kai | HF `7185f514f54b8f93c55998b1e8f9c5cc67f0d029` | `inference.kai_lex --backend kai` | Native typed bundle, 1024-token limit; isolated qualified Python 3.12.13, PyTorch 2.12/HIP 7.2, Transformers 4.57.6, tokenizers 0.22.2, safetensors 0.8.0, NumPy 2.5.3 runtime. |
| Decision 1.0 Lex | HF `ee8e74d912fca8328a353c11d174b44da3f91781` | `inference.kai_lex --backend lex` | Same native contract and qualified runtime as Kai. |
| Decision 1.0 Lux | HF `bd45a30aee8c84032791c245c70f86dee5389cc8` | `inference.run --backend lux` | Native typed bundle, 16,384-token limit; qualified ROCm+FLA runtime required. |
| Decision 1.0 Nox | HF `0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` | `inference.run --backend nox` | Native typed bundle; qualified ROCm+FLA runtime required. |
| Decision 1.0 Sol | HF `0665a41108e8f0b33a9515c98311c45947b99399` | `inference.run --backend sol` | Native typed bundle; qualified ROCm+FLA runtime required. |
| Decision 2.0 Qwen candidate(s) | Pre-test checkpoint model SHA + CAL SHA from lock | `training.model.infer --calibration` | Native Choice/Noul/Score dynamic candidates; each CAL temperature is bound to exact source+adapter+head+tokenizer bytes. |
| Decision 2.0 native SemIf 4B candidate | Checked package `SHA256SUMS` SHA + CAL SHA + gold-free direct parity reports | `training.eikos.published_infer` | Eikos letter-logit readout, original native server and independent hard CAL; source release and adaptation are verified against the package. BF16 ROCm parity and task-level transfer differences are disclosed. |

The synthetic final ranking compares native type slices and four family
accuracies on identical prompts. CSS evaluates **Choice only** and reports
the median task macro-F1/accuracy over 15 human-label tasks. Include invalid
coverage, Brier/ECE, original probability-sum deviations, runtime and output
limitations beside any ranking. Neither panel alone proves universal
superiority; training/pretraining overlap and hosted service changes remain
limitations. Do not reuse DEV/CSS pilot scores as final or select models from
either final panel.
