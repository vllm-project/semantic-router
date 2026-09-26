# Qwen3.5 0.8B initialization control — preregistration

This arm tests whether the poor external transfer of the completed 0.8B Base
run is due to its initialization rather than the typed decision head alone.
The source is `Qwen/Qwen3.5-0.8B` (posttrained, Apache-2.0), immutable revision
`2fc06364715b967f1860aea9cf38778875588b17`. It is a different source
from the previously trained `Qwen3.5-0.8B-Base`; the two runs share the same
rights-clean v2 data and trainer, and source identity must be disclosed.

The comparison control is the completed Base run: TRAIN 7,455, SELECT 700,
CAL 700; one epoch, 466 updates, full backbone, native 256-dimensional
candidate head, CE + 0.5 Brier, max length 8,192, microbatch 1, accumulation
16, seed 20260926, learning rates 2e-5 backbone and 2e-4 head, save interval
64. The posttrained arm uses exactly those settings with `--init-kind
posttrained`, changing only the pinned source and output directory. Its
checkpoint selector uses SELECT family-macro accuracy, then normalized Brier,
then earliest step. CAL is audited but never used for gradient steps or model
selection. Inference and scoring use the same native adapter and prompt panel
as the Base, Decision 1.0 Eos, and Joyfox reference runs.

The Base arm's SELECT rose to 567/700 yet independent typed DEV was only
392/1,600, so SELECT cannot be taken as transfer evidence. Complete the arm
unless a runtime, loss, overlap, or checkpoint integrity failure occurs.
After `BEST.json` and `COMPLETE.json` freeze, fit separate native-type
temperatures on CAL. Then run typed DEV1,600, CSS pilot1,430, and public
JevBench subset231 once each with the frozen checkpoint. Compare by matched
item/task and report invalid or overlength questions as wrong. Keep sealed
FINAL and the authored release candidate closed; no JevArena release rank is
computed from these development results.

A useful continuation must beat Eos 1.0 at 0.8B on typed DEV (49.5%) and
public231 (142/231), and show transfer improvement beyond the Base arm's CSS
pilot median macro-F1 .27046. Joyfox's pinned native open baseline is a
stronger reference: typed DEV 993/1,600, CSS pilot median macro-F1 .30119,
public231 136/231 with 41 length-invalid inputs under its 1,024-token cap.
Check at least task-level CSS direction, transition cases and calibration;
a SELECT-only gain or a single-public-subset gain is insufficient to name a
`dev-2.0-0.8b` release. This run is an initialization control, not an
automatic release candidate.
