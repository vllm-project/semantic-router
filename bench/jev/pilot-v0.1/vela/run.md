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
| pilot-004 | history | history | 0.9998 | yes |
| pilot-005 | health | biology | 0.9998 | yes |
| pilot-006 | diagnostic-only | computer science | 0.9075 | no |

All 6/6 records were contract-valid and there were no inference failures. Primary accuracy is 4/5 (80%). Model-inference latency was P50 15.952 ms and P95 17.582 ms on the local CPU. All records preserve the complete 14-label probability vector; each vector sums to one within 1e-5.

The full record file is `vela-results.jsonl`; `vela-raw.jsonl` is the un-enriched runner output and `vela-warmup.jsonl` is the separate warmup capture. Since this is a local Transformers run, HTTP status, server timeout, and raw HTTP response body are unavailable and are recorded as null/unavailable rather than inferred.

## Reproduction

From the repository root, with the evaluation environment used for this run:

```bash
.venv-semantic-router-eval/bin/python bench/jev/pilot-v0.1/vela/run-vela.py \
  --input bench/jev/pilot-v0.1/vela/runner-input.jsonl \
  --output /tmp/vela-pilot-v0.1.jsonl \
  --warmup-output /tmp/vela-pilot-v0.1-warmup.jsonl \
  --metadata-output /tmp/vela-pilot-v0.1-metadata.json \
  --config tools/eval/category-comparison/baseline.yaml \
  --mapping tools/eval/category-comparison/category_mapping.json \
  --device cpu --warmup-runs 1
```

The shared fixture and question hashes are recorded in every result and in `run-metadata.json`. `SHA256SUMS` covers every file in this arm directory.
