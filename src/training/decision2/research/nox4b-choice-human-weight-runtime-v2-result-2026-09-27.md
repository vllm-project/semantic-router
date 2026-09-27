# Own-Nox 4B Choice-weight v2: second zero-step STOP

The separately frozen [runtime-corrected v2 protocol](nox4b-choice-human-weight-runtime-v2-prereg-2026-09-27.md)
stopped **before its first optimizer step**. The original v1 numerical gate
was not relaxed, and no v2 one-step or 128-step arm, DEV/CSS pilot, formal
JevArena, public JevBench or publication followed.

The pinned container retained its original
`PYTHONPATH=/opt/decision-fla`, imported both the exact read-only mirrored
trainer and optimized `fla`, and exposed one BF16 GPU. A fresh idle-card
reservation was used and released after the process ended. The v2
`--zero-step-only` process exited normally. Its metrics events were exactly
`baseline` and `zero_step_only`; it wrote **no optimizer event or
checkpoint**. Source file fingerprints, 7,455 TRAIN rows, 4,194,465
unpadded tokens, maximum length 6,596, 700 SELECT/CAL rows and exactly
2,240 weighted Choice rows matched the frozen input contract.

The all-700 native SELECT comparison found zero ID, prompt-hash or
token-ID-hash mismatches, but **one categorical change** and maximum
absolute offered-option probability difference **0.01503915** against the
preserved control. The preregistered limits were zero changes and at most
`1e-4`; thus the gate failed. The new baseline contained 524/700 correct
versus the historical 523/700, but neither accuracy is admissible evidence
for this experiment because the native probability parity failed.

| Private receipt | SHA-256 |
| --- | --- |
| v2 provenance | `24b70d40e28147f5495bb1486d760baadc0b28585b422c309d31f9b4e79f712d` |
| v2 metric events | `ee94928ffda1927418b923074c328401acc9bb31167ce3e32490313567cbf0ce` |
| v2 zero-step SELECT predictions | `6b8d25a0097efde6fe22610a2f36d1d2190214b09296317cc9a25945ff402707` |
| archived control zero-step predictions | `eabd788e4e656974e3f380cdff0d32f1ad0887ddb2034235b2c43b9c3f994de3` |

Provenance creation and the final metrics-file update were about 37 seconds
apart, approximately 0.0103 allocated GPU-hour excluding container setup.
The first failed probe had omitted optimized `fla`; restoring it reduced
the worst drift from 0.02677 to 0.01504 and category changes from four to
one, but did **not** establish parity. The remaining difference may arise
from cross-process/GPU numerical behavior or another runtime detail.
That is a hypothesis, not a verified root cause.

Any subsequent 4B experiment needs a **new prospective matched-runtime
comparison design**. It could create a new fixed-step control and treatment
under the same runtime, with both zero steps checked against each other,
but should not claim equivalence to the already completed historical
control merely because model and data files match. Do not reuse this
failed v2 score or change the frozen numerical threshold after seeing it.
