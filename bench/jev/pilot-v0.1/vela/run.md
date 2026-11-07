# Vela arm — pilot v0.1

This is a supplementary contribution to research PR #4336. It was run from the Yuki branch tip `332dd26b7ed7886599e2d492eaaf0c976f330254` after verifying that the shared `inputs.jsonl`, `question.json`, and `protocol.json` were unchanged and byte-identical to the canonical files used for the run.

## Configuration

- Model: `llm-semantic-router/Vela-1.0-Encoder-307M-Domain`
- Revision: `f6354f54adcf38770f635ad903be2b00577f6c11`
- Shared fixture: `pilot-v0.1`, English, six cases
- Label mapping: complete 14-label checkpoint mapping
- Interface: Vela fixed-label classifier; case text is sent without candidate descriptions
- Attempts: one measured attempt per case, one separate warmup, no automatic retries
- Device/runtime: CPU, Transformers 5.17.0, PyTorch 2.14.0, Python 3.13.15, Apple M5
- Stop policy: continue and record; no inference failures occurred
- Contract policy: exact labels, finite probabilities in [0,1], sum within 0.001; no renormalization or threshold/fallback
- Protocol status: `shared_protocol_frozen: false` in the shared protocol metadata

The runner input adapter (`runner-input.jsonl`) adds only the current runner field names (`case_id`, `text`, `expected_label`) to the unchanged shared cases. Case 006 has no expected label and is diagnostic-only.

## Results

| Case | Expected | Vela top-1 | Top-1 p | Scored |
|---|---|---|---:|---|
| pilot-001 | biology | biology | 0.9998 | yes |
| pilot-002 | computer science | computer science | 1.0000 | yes |
| pilot-003 | math | math | 0.9998 | yes |
| pilot-004 | history | history | 0.9678 | yes |
| pilot-005 | health | biology | 0.8153 | yes |
| pilot-006 | diagnostic-only | computer science | 0.9075 | no |

All 6/6 records were contract-valid and there were no inference failures. Primary accuracy is 4/5 (80%). All records preserve the complete 14-label probability vector; each vector sums to one within 1e-5.

Offline review on 2026-09-30 corrected the displayed top-1 probabilities for cases 004 and 005 from the unchanged raw records. The saved summary's P50 15.952 ms and P95 17.582 ms match the third and sixth sorted `latency_wall_ms` observations (consistent with nearest-rank percentiles); its generation method is not included in the submitted runner. The ordinary six-observation median of that field is 16.141 ms. Both timing fields bracket the forward pass, excluding tokenization and softmax; despite its name, `latency_wall_ms` is not end-to-end latency. These six observations are not a stable tail-latency estimate or a basis for cross-arm speed ranking. The original summary is preserved.

The metadata phrase “one unrecorded warmup” is stale: `vela-warmup.jsonl` contains the separate warmup capture. The original metadata is preserved rather than silently rewritten.

The full record file is `vela-results.jsonl`; `vela-raw.jsonl` is the un-enriched runner output and `vela-warmup.jsonl` is the separate warmup capture. Since this is a local Transformers run, HTTP status, server timeout, and raw HTTP response body are unavailable and are recorded as null/unavailable rather than inferred.

## Reproduction

From the repository root, with the evaluation environment used for this run:

```bash
.venv-semantic-router-eval/bin/python bench/jev/pilot-v0.1/vela/run-vela.py \
  --input bench/jev/pilot-v0.1/vela/runner-input.jsonl \
  --output /tmp/vela-pilot-v0.1.jsonl \
  --warmup-output /tmp/vela-pilot-v0.1-warmup.jsonl \
  --metadata-output /tmp/vela-pilot-v0.1-metadata.json \
  --config bench/jev/pilot-v0.1/vela/baseline.yaml \
  --mapping bench/jev/pilot-v0.1/vela/category_mapping.json \
  --device cpu --warmup-runs 1
```

The shared fixture and question hashes are recorded in every result and in `run-metadata.json`. `SHA256SUMS` covers every file in this arm directory.

### Reproduction packaging fix (2026-09-30)

The two dependencies above are copied byte-for-byte from lyy26299's earlier
[pinned evaluation configuration](https://github.com/lyy26299/semantic-router/tree/53526a3edfa7afaae6760670a68033692cc112cc/tools/eval/category-comparison).
The mapping hash `48ce5b71a0d3479e460af7e5a39dd05a7c66728380f201a1f794b335de915b1a`
matches every measured record. The baseline model/revision matches the run metadata;
the original run did not record its baseline YAML hash, so byte identity with that
run's local YAML cannot be established. This supplies reproduction dependencies,
not a new model run. Runner source and all captured result/metadata files are unchanged.

Subsequent formatting update (2026-10-01): repository-pinned Black 25.1.0 reformatted `run-vela.py`, and Ruff cleanup reordered its standard-library imports. Apart from import order, its parsed AST is unchanged. Its current SHA-256 is `df490857e8a6f82e72973970b51969af3c735c1cce655312ab34c3dfa5a7c3ea`. The historical `runner_source_sha256` in `run-metadata.json` is preserved and refers to the original source at commit `efb0e7143fa52f715b3de31f84625c2031ceb6b2`, not the formatted file. `SHA256SUMS` covers current packaged files. Captured results, metadata and summary are unchanged; no model was rerun.

With Python and PyYAML available, check configuration, mapping and inputs without
Torch, downloaded model weights or inference:

```bash
python3 bench/jev/pilot-v0.1/vela/run-vela.py \
  --input bench/jev/pilot-v0.1/vela/runner-input.jsonl \
  --output /tmp/vela-mapping-check-unused.jsonl --mapping-only
```

The output argument is required by the CLI but is not written in mapping-only mode.
For inference, use the environment versions recorded in `run-metadata.json`;
the `.venv-semantic-router-eval` directory is an operator-created environment,
not a bundled checkout dependency. Use fresh output paths: this runner overwrites
existing output files. Updated `SHA256SUMS` reflects this run-note edit and the two
added configuration files; hashes of the captured evidence remain unchanged.

### Checkpoint label-index review fix (2026-10-02)

The runner normalizes integer/string `id2label` keys, requires the complete contiguous index range and verifies each semantic label against the pinned mapping at its exact index. Complete `LABEL_n` metadata remains supported; the external mapping supplies its semantic names. Empty/invalid metadata, non-integer indices, duplicate normalized indices, invalid labels, incomplete maps and semantic permutations fail before inference.

The 11 offline regressions are in `bench/jev/pilot-v0.1/tests/test_vela_runner.py`:

```bash
python3 -m unittest discover -s bench/jev/pilot-v0.1/tests -p test_vela_runner.py -v
```

The current runner SHA-256 is `e4cce78707ab3d5b360d64c178b4ff0eeab6dec84a38cea08590d5322f8fdd92`. The earlier formatting hash above describes the 2026-10-01 snapshot. The saved metadata's `runner_source_sha256` still identifies the historical captured-run source; this fix has not been exercised by rerunning the model. Saved records and scores remain unchanged.
