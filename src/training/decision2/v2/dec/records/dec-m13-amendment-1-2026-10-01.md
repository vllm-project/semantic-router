# Decoder Milestone 13 — amendment 1 (4B formal entries with relative readout paths; 2026-10-01)

Committed before the formal entries are regenerated and before any formal GPU job. Prereg
[`dec-m13-prereg-2026-10-01.md`](dec-m13-prereg-2026-10-01.md), "Formal, successor items, hand-offs".

## What happened

At 15:18Z the 4B formal entries (`m13/select/formal/4b-finalists.json`, `m10_formal_select.py --tier 4b`; slot 0 the LH
soup as the formal-path parity reference, slot 1 the finalist `4b-LHA10SD`) were written on node F with the points'
16K typed DEV / CSS pilot readout paths **relative** to `m13/` (`lines/<point>/dev/dev.predictions.jsonl`). The formal
library resolves those paths from its own working directory, so both smokes stopped at the library's input check
(`missing lines/4b-LH-f/dev/dev.predictions.jsonl`, `missing lines/4b-LHA10SD/dev/dev.predictions.jsonl`) within the
same second, and the wrapper marked both points `FAILED` ("smoke failed").

No formal job ran: no container started, `formal/m13/stage-cal` is empty, no CAL698 fit, no package, no collection.
Nothing was measured, so there is no result to select on or to hide.

## Change

- The entries are regenerated with the same tool, the same points, checkpoints and readouts, in the same slot order,
  with **absolute** host paths (`/data/dev2/runs/dec/m13/lines/...`). The first entries and the two `FAILED` markers and
  launch locks are kept aside under `formal/m13/logs/amendment-1/` (not deleted).
- The two formal chains are relaunched as prereg'd (node F GPU6 `4b-LH`, GPU7 `4b-LHA10SD`). Every later formal entry
  file (2B / 0.8B finalists, if any) uses absolute paths.
- Nothing else changes: the development rules output (`m13/select/4b-finalists.json`, run once at 15:17Z), the
  finalist, the bars and the stop rules are untouched. A failure of the relaunched chains is final (no rerun).
