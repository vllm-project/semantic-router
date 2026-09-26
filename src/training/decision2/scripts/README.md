# Frozen v2 DEV and CSS pilot rescore

`rescore_v2.py` scores the completed native prediction panel on CPU. It pins
the 1,600-item synthetic DEV and 1,430-item CSS pilot gold SHA-256 values,
published model revisions, and Lux checkpoint25 model SHA-256. It requires
complete prediction ID coverage for both panels, then creates a new output
directory with 24 v2 score reports, `summary.json`, and `summary.md`.

Run from the repository root on the experiment host (or an exact source
mirror), where `RUNS_ROOT` points to the frozen run directory:

```bash
PYTHONPATH=. python3 -m scripts.rescore_v2 \
  --runs-root "$RUNS_ROOT" \
  --output-dir "$RUNS_ROOT/dev-css-pilot-v2-20260926-v1"
```

The destination must not exist. The script builds a temporary directory and
renames it only after every model and integrity check succeeds. Earlier
reports, raw predictions, API receipts, and gold files are never rewritten.
`summary.json` records complete SHA-256 values for both gold files, all raw
and scored predictions, each new score report, and scorer source files. The
Markdown companion is suitable for copying into the research progress gist.

Jev's original DEV prediction file used the full API request-body SHA-256 as
`source_input_sha256`. The v2 scorer requires the gold-free `state/questions`
payload SHA-256. The script derives a separate Jev prediction copy after
verifying each of the 1,600 prompt IDs against the original API receipt hash,
requested/returned model, HTTP status, response answers, and unchanged legacy
prediction. The derived copy is included in the versioned output directory.

CSS pilot is for adapter checks and training decisions. Its three-task median
must not be presented as the 15-task held-out CSS evaluation headline.
