# Model-card artifacts from frozen reports

This generator turns **scored reports**, not raw predictions or handwritten
numbers, into model-card material:

- `score-table.md`: ranking, four family slices, three native task-type slices,
  configured 2.0 versus 1.0 differences, optional CSS transfer results, and
  exact scored model identities.
- `ranking.svg`: horizontal final-family macro-accuracy ranking.
- `matrix.svg`: final-family and native-type accuracy heat map.
- `manifest.json`: input report/prediction SHA-256 digests, frozen gold digest,
  generator source digests, metric policy, and output file digests.

Run from the repository root with Python 3.10 or newer; no plotting library is
required. Report paths in the config are resolved relative to the config file.
The generator requires final `typed-decision-report/2` scores and, when CSS
scores are supplied, `css-transfer-score/2`. Its manifest uses
`decision-model-card-artifacts/2`; regenerate cards after rescoring frozen
predictions under the normalized probability policy.

```json
{
  "title": "Decision 2.0 — frozen typed-decision benchmark",
  "models": [
    {
      "key": "d2-4b",
      "label": "Decision 2.0 4B",
      "group": "decision2",
      "size": "4B",
      "benchmark_report": "reports/d2-4b.final.score.json",
      "css_report": "reports/d2-4b.css.score.json"
    },
    {
      "key": "nox-4b",
      "label": "Decision 1.0 Nox-4B",
      "group": "decision1",
      "size": "4B",
      "benchmark_report": "reports/nox-4b.final.score.json",
      "css_report": "reports/nox-4b.css.score.json"
    },
    {
      "key": "open-baseline",
      "label": "Open baseline",
      "group": "open",
      "size": "4B",
      "benchmark_report": "reports/open-baseline.final.score.json"
    }
  ],
  "comparison_pairs": [
    {
      "new": "d2-4b",
      "old": "nox-4b",
      "css_comparison_report": "reports/d2-vs-nox.css.compare.json"
    }
  ]
}
```

`css_report`, `size`, `comparison_pairs`, and
`css_comparison_report` are optional. The config supplies display metadata;
the model ID, revision, backend, scores, and prediction hashes come from the
scorer reports. Allowed groups are `decision2`, `decision1`, `open`, `hosted`,
and `other`.

```bash
python3 -m publication.generate \
  --config /path/to/publication-config.json \
  --output-dir /path/to/new/model-card-artifacts
```

The output directory must not already exist. Copy `score-table.md`,
`ranking.svg`, `matrix.svg`, and `manifest.json` together into a model repository
or publication package. Embed the SVGs beside the corresponding tables in the
model card. Review every display label, size, license, and model attribution
before release; those fields are supplied by the config rather than measured
by the benchmark.

## Score and uncertainty policy

The main ranking uses the unweighted mean of `accuracy_all` across the four
frozen final families: constraint competition, exception stack, evidence join,
and resource ledger. `accuracy_all` includes invalid and missing predictions
as misses. The family and native Choice/Noul/Score columns refer to overlapping
views of the same questions. Brier is shown only when a scorer report contains
it, with its probability-answer coverage. These final benchmark reports do not
contain uncertainty intervals, so the generator displays point estimates only.
The ranking is relative to the configured models; it is not an external
leaderboard.

Optional CSS transfer scores are kept in a separate table. Its headline is the
median over 15 human-label evaluation tasks; the three pilot tasks are excluded.
A CSS task is Choice classification; the transfer panel does not independently
measure native Noul or Score.
A 95% interval appears **only** for a supplied `transfer.compare` paired item
bootstrap report whose gold and both prediction hashes match the two CSS score
reports. It describes the CSS panel difference, not the synthetic benchmark.

The Decision 1.0 card's historical scores used another evaluation protocol.
For 2.0 versus 1.0 claims, score both checkpoints on the same frozen final
gold and feed those reports to this generator. The generator rejects different
final gold hashes or item/question counts, malformed score summaries, missing
final families or native types, and duplicate scored revisions.

The scorer reports identify the gold and prediction bytes by digest; they do
not by themselves prove who ran the inference or which scorer source revision
was used. Retain the original predictions, gold manifest, scoring commands,
model revisions, and run logs alongside the generated artifacts for audit.

## Local checks

```bash
python3 -m unittest discover -s publication/tests -v
python3 -m py_compile publication/*.py
```

## Self-contained Decision 2.0 model bundle

`publication.bundle` packages a **materialized full** Decision 2.0 LoRA
checkpoint, its original CAL report, frozen score artifacts, and the scored
prediction manifest into a new Hugging Face repository directory. It also
requires a reviewed public training disclosure, the completed training run,
and the source-data builder manifest. It runs on CPU and does not publish
anything. The input checkpoint is produced by
`training.model.materialize` and must contain its portable
`materialization_receipt.json`. A LoRA adapter alone is rejected.

```bash
python3 -m publication.bundle \
  --checkpoint /ABS/decision2-merged \
  --calibration /ABS/decision2-calibration.json \
  --card-artifacts /ABS/model-card-artifacts \
  --scored-manifest /ABS/decision2.final.predictions.jsonl.manifest.json \
  --training-record /ABS/reviewed-public-training-record.json \
  --run-dir /ABS/completed-training-run \
  --training-data-manifest /ABS/balanced_human_5824.manifest.json \
  --score-key d2-4b \
  --model-id llm-semantic-router/dev-2.0-4b \
  --base-model-id Qwen/Qwen3.5-4B-Base \
  --license apache-2.0 \
  --output /ABS/new-decision2-hf-bundle
```

`--model-id` must be one of `llm-semantic-router/dev-2.0-9b`,
`llm-semantic-router/dev-2.0-4b`, `llm-semantic-router/dev-2.0-2b`, or
`llm-semantic-router/dev-2.0-0.8b`; the size must agree with the selected
score artifact. The Hugging Face collection title is **Decision 2.0** and is
created separately after model repositories pass release review.

The reviewed training record has this schema. The `source` strings and `rows`
must exactly match `counts.source` in the builder manifest; show **every**
source, including inherited Decision 1.0 training sources. The initialization
revision must be an immutable 40-character commit SHA and `source_name` must
match the frozen run's initialization fingerprint. License/rights statements
and attributions are reviewed declarations; a digest cannot prove legal
permission or that a Hugging Face commit contains the same bytes.

```json
{
  "record_version": "decision2-public-training-record/1",
  "initialization": {
    "model_id": "llm-semantic-router/Decision-1.0-Nox-4B",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "source_name": "Decision-1.0-Nox-4B",
    "license_status": "Review and state the source model's applicable license"
  },
  "sources": [
    {
      "source": "exact counts.source key",
      "rows": 100,
      "url": "https://example.org/original-source",
      "license_status": "State the upstream license and any separate rights limits",
      "attribution": "Original dataset authors"
    }
  ],
  "known_overlap": [
    "State known same-task supervision and inherited train/test near matches"
  ],
  "evaluation_interpretation": [
    "Distinguish cross-target, related-task and unseen-task transfer claims"
  ],
  "limitations": [
    "State data imbalance, context, calibration and untested-condition limits"
  ]
}
```

Replace the illustrative values before packaging. For the balanced-human
5,824-row arm, disclose its 3,600 TweetEval TRAIN examples and the original
task/platform rights caveat, the 120 FLUTE TRAIN examples that make CSS FLUTE
same-task supervised, sparse high Score levels, and inherited 1.0 exposure.
Keep the full source-data and run receipts outside the public model repo; the
package includes a public count/hash summary and the reviewed record.

The score key must identify the Decision 2.0 entry in the card-artifact
manifest. Its scored prediction hash, model ID, revision, checkpoint hash,
CAL file hash, and context limit must match the supplied prediction manifest.
The completed run's BEST, COMPLETE and provenance file hashes, selected
checkpoint model files, source-model fingerprint, and TRAIN/SELECT/CAL
partition hashes/counts must agree with the original CAL, merge receipt, and
data builder manifest. The package records these identities in
`training-provenance.json` and `MODEL_MANIFEST.json` and renders training
sources, optimization, selection, calibration, overlap, and limitations in the
card. The source-data manifest is used for verification, not copied wholesale:
it can contain private row IDs and audit details.
The score may have been measured on the selected LoRA checkpoint or the
published merged weights. The model card states which; a pre-merge score is
clearly marked as still needing a separate numerical parity check. The
packager never rewrites the original CAL report to force its model hash to
match the merged weights. It verifies the source-to-merged receipt instead.

The output layout is:

```text
README.md                         HF model card with generated table/figures
MODEL_MANIFEST.json               SHA-256 inventory and source/score identity
calibration.json                  original CAL report bytes
model/decision_config.json        dynamic-candidate architecture and prompt
model/decision_head.safetensors   FP32 candidate head
model/backbone/                   full merged Qwen3.5 text weights and config
model/tokenizer*                  tokenizer files
model/materialization_receipt.json
decision2/                       portable native Choice/Noul/Score inference
card-artifacts/                   frozen report-derived table, figures, manifest
score-table.md, ranking.svg, matrix.svg
scored-predictions.manifest.json  exact scored run receipt
training-record.json               reviewed public source/rights disclosure
training-provenance.json           verified public counts, hashes, optimization, limitations
requirements.txt                  required package names; choose device builds
```

The copied runtime imports no training checkout. `Decision2.from_pretrained`
verifies every packaged file, the merged model identity, and calibration
lineage before loading. `system_one(state=..., questions=...)` returns Choice
label plus distribution, Noul `P(true)`, and Score expected index plus level
distribution; over-budget questions receive an explicit invalid answer.
CUDA/ROCm uses BF16 backbone compute and an FP32 head; CPU uses FP32 and may
require substantial RAM. The packager's tests verify contract and byte
integrity without loading weights or claiming numerical parity.

Only reviewed SPDX license metadata and public Hugging Face IDs may be used.
The packager rejects credential-like strings, absolute workstation paths,
symlinks, non-model checkpoint files, mismatched score/calibration/training
hashes, changed selected weights, and unaccounted training sources.
Review licensing and run a real merged-weight inference comparison before
uploading the generated directory.
