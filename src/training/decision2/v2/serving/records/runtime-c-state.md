# Runtime C (inference optimization) — state

Owner: inference-optimization worker 885d85cc (`track=runtime-c`), started 2026-10-03 11:35 UTC+8 after the
user stopped all training (COORDINATION 11:00 / 11:12). Branch `xunzhuo/decision-2-runtime-c` (from integration
`3d8826ff3`), worktree `vllm-sr-dev2-runtime-c`. The only writer to the six public model repos. Budget 60 GPU-h.
The Index submission worker (f38ee089) has first call on every GPU. Times are UTC unless marked.

## Current handoff

- **Running:** first profiles of the released runtimes (node B GPU0–3, `results/prof0-*`).
- **Leases:** node A GPU0–1, node B GPU0–3 (`owner.runtime-c`).
- **Next:** remote-code pass-through; Transformers 5.18 parity; tree mode with fused kernels; card; rollout.
- **Repos:** untouched. Current `main`: Kai `881bee41`, Eos `ad0aa724`, Sol `4b75b521`, Nox `ce1bdc9d`,
  Lux `214ffa43`, Vega `9b067a95`.

## Log

- 03:35 started; leases written; six released packages downloaded on node B
  (`/data/dev2/runs/runtime-c/pkgs/<name>@<rev8>`, no links).

## GPU-hours

| Job | Node / GPU | Wall | GPU-h |
| --- | --- | --- | --- |
