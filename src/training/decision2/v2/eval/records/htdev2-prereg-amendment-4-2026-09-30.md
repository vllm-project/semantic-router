# HT-DEV v2 preregistration, amendment 4 (2026-09-30): validation models, before any collection

Committed after the panel freeze and five 20-item smoke runs, and before any full collection; no model has been
scored on HT-DEV v2. Scoring, metrics and the decision rule are unchanged.

1. **Four models are excluded from the validation set:** `dec-m5-N5B-soup`, `dec-m5-N5BN-soup`, `dec-m5-N5N-soup`
   (4B) and `dec-m6-2b-S6X-b1_3` (2B). Their formal package directories were removed from node A after the formal
   runs; the only copies are on node B (16 GB per M5 soup, 7.1 GB for S6X). The allocation has no node-B GPU, and
   the workstation link cannot move 55 GB in time. Like 27B, they cannot be collected on node A.
2. **Validation set = 54 models** (`records/htdev2/spec-collected.json`): 0.6B 12, 0.8B 12, 2B 5, 4B 16, 9B 9.
   Two of them (`dec-m4-N4LX-soup`, `dec-m4-N4XF-soup`) collect `css-pilot` in the same job; the others use the
   stored pilot readouts named in the spec.
3. **Runtime notes, disclosed:** four formal autotune caches were not on node A (the four excluded models only), so
   every collected model uses a copy of its formal cache or its v1 job's; the smoke mode cannot run the 0.6B
   `training.model.infer` collector (it has no `--max-items`), so those jobs run without a smoke, as their formal
   runs did.
