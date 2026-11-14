# Decision 3 quantized-package qualification

`tools/decision3_quantized_qualification.py` turns measured evidence into one
machine-readable pass/fail record. It is intentionally separate from package
loading: the tool does not read weights or select/promote a package.

The record reuses Decision 3's package identity instead of inventing another
artifact digest:

- `model_sha256` is the scored checkpoint identity declared by
  `MODEL_MANIFEST.json`.
- `manifest_sha256` is the SHA-256 of that exact manifest.
- published inputs additionally pin the package repository and full revision.

A record contains one baseline/candidate pair, quantization and calibration
provenance, an immutable evaluation dataset reference, and the exact harness,
code, dependency, and device versions. Measurements include:

- exact aggregate and `choice` / `noul` / `score` correct and changed-answer counts;
- calibration/evaluation duplicate-state rows, any of which forces FAIL;
- errors, NaNs, OOMs, and missing results, any of which forces FAIL;
- package, post-load device, and peak device bytes;
- load time, median/p95 latency, and throughput at concurrency 1, 4, and 16.

Its quantization vocabulary matches the runtime contract from #5066: `float8`,
or `int4` with its group size and symmetry; `method` is one of
`max-per-channel`, `awq`, or `gptq` and must agree with those runtime fields.
Int4 records require calibration provenance; float8 records explicitly use
`null`. Exact counts avoid rounded-accuracy ambiguity. Evidence from #5066 is
an intended first input, but a record for a community-published package remains
third-party experimental evidence and does not make that package an official
`vllm-sr` model.

## Finalize

Author the evidence as JSON without `outcome`, then bind it to the exact package
manifests and compute the verdict:

```bash
python3 tools/decision3_quantized_qualification.py finalize evidence.json record.json \
  --baseline-manifest /models/d3/MODEL_MANIFEST.json \
  --candidate-manifest /models/d3-awq/MODEL_MANIFEST.json
```

The command writes both passing and failing records: exit 0 means PASS, 1 means
FAIL, and 2 means the evidence or package identity is invalid. It rejects short
revisions, unknown fields, missing calibration or evaluation provenance, reused
calibration rows, inconsistent quality counts, mismatched package identities,
and a hand-written verdict.

Validate a stored record. Omitting the two optional manifest arguments checks the
record's structure and arithmetic only; supplying them rebinds it to the package
bytes:

```bash
python3 tools/decision3_quantized_qualification.py check record.json \
  --baseline-manifest /models/d3/MODEL_MANIFEST.json \
  --candidate-manifest /models/d3-awq/MODEL_MANIFEST.json
```

`policy.issue` points to the accepted issue that fixed the gates before the
candidate was scored. Qualification does not imply upload, runtime support,
automatic selection, or promotion.
