## Reports, manifests and promotion drafts

The ledger is the single replay source for every human-facing artifact: the
same ledger renders byte-identically, and mixed dataset hashes are flagged as
not directly comparable.

```bash
recipe-lab report --ledger ledger.jsonl --output report.md
recipe-lab provenance --ledger ledger.jsonl --output manifest.json
recipe-lab record-draft --ledger ledger.jsonl --output promotion-draft.md
recipe-lab next --spec spec.yaml --ledger ledger.jsonl --output ctx.md
```

`report` also renders explicitly selected receipts from outside the ledger
(`--evidence path.json`, repeat in workflow order, `--background` for the
operator's own context); unknown or malformed schemas are refused and the
output is published atomically - a failed render never replaces an existing
report. `provenance` emits a `recipe-lab-run-v1` manifest whose vocabulary
aligns with the upstream provenance schemas (record IDs, ledger hash,
bench_refs). `record-draft` produces a docs/records draft from the latest
PASS live record for human review; drafts are never merged automatically.
`next` exports the agent context with the untrusted-material header and the
citable refs described in the report section above.

## Speed depth receipts

`recipe-lab suggest-depth` prints and saves one
`recipe-lab-sr-bench-depth-v1` receipt: `status` (`proposed` / `keep-parent`
/ `inconclusive`), the run and target identity (`dev_run`, `parent_target`,
`fast_target`), `parent_sha256`, `deployed_source_sha256`,
`parent_runtime_hash` and `output_policy`, the per-case `observations`
(`role` `move` / `protect` / `skip`, both finish reasons, the comparable
opening-request hash and whole-task seconds; case IDs only, never task
text), and when a threshold separates the observations `before`, `after`,
`candidate` and `candidate_sha256`. `report --evidence` renders the receipt
in supplied order; the saved receipt never changes shape even when Markdown
delivery fails (`report_error` on the receipt, evidence still on disk).
