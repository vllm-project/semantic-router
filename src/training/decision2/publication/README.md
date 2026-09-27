# Model-card artifacts from frozen reports

## Slim first-release Hugging Face tree

The private v3 package retains the full release gate, prediction bindings,
review receipts and all generated artifacts. After its exact verifier passes,
`publication.export_hf_v3` creates a separate, smaller 4B repository directory.
Weights, tokenizer, model configuration, calibration, native inference code,
licenses and charts retain their exact bytes. The export leaves private
non-runtime training provenance in the verified source package and regenerates
the public `model/SHA256SUMS` over the reduced inference file set:

```text
README.md                    Apache-2.0 model card and owl banner link
LICENSE                      Apache-2.0 text
LICENSE-Eikos, LICENSE-Qwen   inherited model license texts
NOTICE, ATTRIBUTIONS.md      inherited notice and source credits
model/                       byte-identical inference files, new SHA256SUMS
assets/                      owl banner and six frozen rank/matrix/Pareto figures
evaluation/EVALUATION.md     concise first-release protocol
evaluation/manifest.json     public file hashes and private-package binding
```

The original Decision 1.0 Nox-4B mosaic owl is reused **without changing its
pixels**, with a new SVG `DEV2.0-4B` name-card layout. Obtain the pinned banner
through HF CLI from `llm-semantic-router/Decision-1.0-Nox-4B` at revision
`cde2a68dbaa557ea65dc458104d410a0802ee259`, file
`assets/decision-nox-4b-header.png`. The exporter requires its exact SHA-256.
Run the full private verifier first, then:

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.export_hf_v3 \
  --private-package /ABS/verified-private-v3-package \
  --owl-banner /ABS/decision-nox-4b-header.png \
  --output /ABS/new-DEV2.0-4B-hf-repository
PYTHONPATH=src/training/decision2 python3 -m publication.export_hf_v3 \
  --verify /ABS/new-DEV2.0-4B-hf-repository
```

The exporter checks the private package's complete SHA-256 inventory and
passes only a package whose 4B release gate and Apache-2.0 declarations are
already present. The public manifest records both the original scored model
identity and the new public runtime manifest digest. It refuses symlinks,
altered inference files, altered charts, unrecognized owl pixels and extra
public root files. Revalidate native predictions on the exported directory
against the frozen gold-free outputs before publication; the old provenance-
requiring `published_infer.package_identity` does not apply to this reduced
public runtime. Use the **same ROCm/PyTorch/Transformers/FLA image and GPU
architecture as the scored run**. The parity command checks the original
prediction manifest, full prompt/prediction hashes, deterministic gated-delta
backend, runtime versions and every categorical/probability output. Run it for
each scored panel (typed FINAL, human transfer and public JevBench), keeping
the resulting receipt in private storage:

```bash
PYTHONPATH=src/training/decision2 python3 \
  src/training/decision2/scripts/verify_public_hf_runtime_parity_v3.py \
  --private-package /ABS/verified-private-v3-package \
  --public-export /ABS/new-DEV2.0-4B-hf-repository \
  --prompts /ABS/frozen-panel.prompts.jsonl \
  --scored-predictions /ABS/frozen-panel.predictions.jsonl \
  --scored-native-manifest /ABS/frozen-panel.predictions.jsonl.manifest.json \
  --output /ABS/private-panel-parity-receipt.json \
  --device cuda:0
```

The private verification bundle remains untouched. The public manifest names
both the scored native identity and the reduced runtime identity; it does not
claim that the regenerated SHA256SUMS is byte-identical to the scored one.

## JevArena v3 first release

The separate [`generate_arena_v3`](generate_arena_v3.py) and
[`bundle_arena_v3`](bundle_arena_v3.py) commands implement the prospective
8,147-item typed FINAL + 15-task human transfer first-release path. A pinned
231-item public JevBench rerun is required but scored and graphed separately.
Authored questions are reserved for v3.1; Decision Bench and Decision Index
are not first-release gates. See [BUNDLE_ARENA_V3.md](BUNDLE_ARENA_V3.md)
for the exact inputs, pre-key freeze, numeric comparison and package audit.
The six-axis v2 commands below remain unchanged for historical reproducibility.

## JevArena six-axis release artifacts

`python3 -m publication.generate_arena` creates the Decision 2.0 release
tables and six SVG figures: separate JevArena and 231-item public JevBench
rank/Pareto charts, plus JevArena axis and model-by-task matrices. It accepts
only a completed `jevarena-ranking/2` **release** report and a matched
`jevarena-jevbench-public-rank/1` report containing exactly the same models,
revisions, parameter counts and public scorer digests. Every matched 2.0/1.0
size pair needs typed group-bootstrap and transfer paired-bootstrap reports
bound to the exact ranked prediction hashes. It refuses to overwrite an
output directory and screens generated files for credentials, host paths and
IP addresses.

The configuration names relative report paths:

```json
{
  "arena_rank": "reports/jevarena-release-rank.json",
  "jevbench_public_rank": "reports/jevbench-public-rank.json",
  "comparison_pairs": [
    {
      "new": "DEV2.0-0.8B",
      "old": "decision-1.0-eos",
      "typed_comparison": "reports/typed-pair.json",
      "transfer_comparison": "reports/transfer-pair.json",
      "new_typed_report": "reports/DEV2.0-0.8B-typed.json",
      "old_typed_report": "reports/eos-typed.json",
      "new_transfer_report": "reports/DEV2.0-0.8B-transfer.json",
      "old_transfer_report": "reports/eos-transfer.json"
    }
  ]
}
```

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.generate_arena \
  --config /path/to/release-config.json \
  --output-dir /path/to/new/release-card-artifacts
```

The generated table discloses the public subsets and distinguishes the 231
exposed JevBench questions from the upstream closed benchmark. It never
claims an official closed-set ranking. The older `/2` generator and portable
Qwen3.5 LoRA bundle described below are retained for audit; they are not a
Decision 2.0 JevArena release artifact and must not be used as the final card
without the six-axis `/3` report and package parity gate.

## Earlier four-family artifact generator

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
  --model-id llm-semantic-router/DEV2.0-4B \
  --base-model-id Qwen/Qwen3.5-4B-Base \
  --license apache-2.0 \
  --output /ABS/new-decision2-hf-bundle
```

`--model-id` must be one of `llm-semantic-router/DEV2.0-0.6B`,
`llm-semantic-router/DEV2.0-0.8B`, `llm-semantic-router/DEV2.0-2B`,
`llm-semantic-router/DEV2.0-4B`, `llm-semantic-router/DEV2.0-9B`, or
`llm-semantic-router/DEV2.0-27B`; the size must agree with the selected
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
