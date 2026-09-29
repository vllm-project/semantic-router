# HT-DEV v2 preregistration, amendment 5 (2026-09-30): the GPU cap, before scoring and analysis

Committed after the collection lanes stopped and before any HT-DEV v2 prediction is scored or compared with formal
H. Scoring, metrics and the decision rule are unchanged.

1. **The collection reached the 1.5 GPU-h cap before the last four jobs.** HT-DEV v2 items are longer on average than
   the pilot's (216 news articles, 216 talk-page conversations and 216 CGA openings per model), so a job cost
   0.016 GPU-h (0.6B) to 0.036 GPU-h (4B, 9B), above the estimates. The lanes stopped at 1.454 GPU-h (collections
   1.414, smokes 0.040) with 50 of 54 models collected.
2. **Not collected:** the four 0.6B own-1.0 and peer models, `kai1`, `lex`, `bosun06`, `gliner25` (last in the
   preregistered queue order 9B → 4B → 2B → 0.8B → 0.6B; about 0.08 GPU-h). Their jobs stay ready under
   `/data/dev2/runs/eval/htdev2/ops/jobs/` for a later top-up.
3. **Validation set = the 50 collected models** (`records/htdev2/spec-validated.json`): 0.6B 8 (all 0.6B-track
   soups), 0.8B 12, 2B 5, 4B 16, 9B 9.
