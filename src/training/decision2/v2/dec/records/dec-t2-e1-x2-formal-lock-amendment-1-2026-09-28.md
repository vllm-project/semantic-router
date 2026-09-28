# X2 formal lock, amendment 1: same-node collection

Status: **committed before any E1-X2 formal prediction was scored or read.**
Amends [`dec-t2-e1-x2-formal-lock-2026-09-28.md`](dec-t2-e1-x2-formal-lock-2026-09-28.md)
(commit `f13c03006`).

The coordinator's cross-track note of 13:35 UTC+8 (read after the lock was
committed) sets a same-node rule: Qwen3.5/FLA models pick different Triton
autotune configurations across nodes (Lux 1.0 differed on 43 of 8,778 answers),
so a candidate must be collected on the same node as its comparator, with the
autotune cache persisted. The Nox 1.0 comparator run (`m1-adopt/nox1`) is on
node A. Therefore:

1. The **formal** E1-X2 collection is the frozen-runner run on **node A GPU5**
   (track allocation), after C1 releases the GPU, with
   `TRITON_CACHE_AUTOTUNING=1` and a run-specific `TRITON_CACHE_DIR` mounted
   read-write and kept with the run. Everything else in the lock is unchanged:
   candidate identity (`model_sha256` `de41a699…`, calibration `71bab01e…`),
   adapter spec, panels, comparator, pre-stated reading and the one-candidate
   limit.
2. The node B GPU0 collection already started under the original lock is kept
   as a **cross-node repeat** only. It is sealed and its agreement with the
   node A collection is reported; it is never the formal result and it is not
   scored before the node A collection is sealed.
