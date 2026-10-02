# open-jev-fast study — state (worker 2d541b40)

Study of the MIT-licensed `open-jev-fast` inference backend (commit `c52b8bb`) for ideas portable to the Decision 2.0
runtime on AMD MI325X. No third-party code, weights or data enter this repository; prototypes live in a private
scratch directory; no released package is changed. Times are UTC.

## Log (newest first)

- 2026-10-02 04:13 — **PAUSED by user decision** (speed-up work to be planned for all Decision 2.0 sizes first).
  All study containers stopped, the three study GPU leases released (`owner.prev-ojf-<time>`), GPUs verified idle.
  Write-up: [`open-jev-fast-study-2026-10-02.md`](open-jev-fast-study-2026-10-02.md).
  - Measured and final: latency panel at 0.8B / 4B / 27B (table below); 27B best bit-identical stack (lean LoRA +
    trims + graphs) p50 / p95 75.2 / 80.6 ms; 0.8B graph + trims 9.5 / 11.4 ms; kernels per request 0.8B 1,661 →
    1,494 (trims), 27B 7,914 → 5,384 (lean LoRA + trims); launch calls per request about 20 with graphs; scored-panel
    fidelity 0.8B and 4B 0 changes on all four panels; TunableOp not adopted.
  - Not completed: 27B scored-panel pass of the stack (released-runtime pass done), 27B and full private-panel runs,
    a batch-composition control, 2B / 9B, `torch.compile`, graphed candidate heads, length bucketing, batch-invariant
    kernels.
  - Resume: on go-ahead, lease one GPU per job again, redeploy the private harness and rerun its queued jobs (27B
    scored panels with the stack, 27B private-panel sample, 0.8B full private panel); then the rollout of
    recommendations 1–3 through the standard parity gate.

- 2026-10-02 03:50 — **Formal-panel fidelity of the bit-identical stack (graphs + host masks + exact kernel trims):
  0.8B and 4B answer all four scored panels bit-identically** (typed-final 2,000, css15 6,547, public231 231,
  mlx-diag 2,275 answers; 0 changes, drift 0.0; 525 captured shapes each). 27B (lean LoRA + trims + graphs) on
  the latency panel: p50 93.4 → 75.2 ms, 400 / 400 bit-identical; its formal-panel run is in progress.
  - Kernel trims: one BF16 cast per shared Linear input and RMSNorm's `1 + w` computed once (0.8B graph
    10.7 → 9.5 ms, still bit-identical). Open-Jev's fused RMSNorm on top: 8.6 ms but 3 / 400 decisions changed.
  - TunableOp (27B, hipBLASLt + rocBLAS search, 77 GEMM shapes tuned): released path −1%, full stack −4%, and
    3 / 400 decisions changed (max dp 0.11). Not adopted.

- 2026-10-02 03:30 — Latency panel (first 400 typed-final prompts, single requests, 400 warm-up), one leased GPU each:

  | Tier | Released runtime p50 / p95 ms | Prototype | p50 / p95 ms | Answers vs released |
  | --- | --- | --- | --- | --- |
  | 0.8B | 22.3 / 22.5 | HIP graph per exact shape, host-built masks | 10.1 / 11.9 | 400 / 400 bit-identical |
  | 4B | 26.8 / 30.2 | same | 19.7 / 22.4 | 400 / 400 bit-identical |
  | 27B | 90.2–94.3 / 95.5–98.5 | same | 88.4–88.6 / 94.1–94.4 | 400 / 400 bit-identical |
  | 27B | (same) | lean unmerged LoRA (BF16 factors, folded 2x scale, one input cast) | 79.9 / 85.1 | 400 / 400 bit-identical |
  | 27B | (same) | lean LoRA + HIP graph | 77.2 / 82.4 | 400 / 400 bit-identical |
  | 27B | (same) | LoRA merged into the BF16 base (+ graph) | 66.7 / 71.7 | **4 / 400 decisions changed, max dp 0.36** |

  - Host-built masks alone (the sync removal) give no gain in eager mode at any size.
  - Kernels per request: about 1,650 (0.8B), 2,180 (4B), 7,900 (27B; the unmerged LoRA path adds about 3,600).
    GPU-busy fraction: about 47% (0.8B), 73% (4B), 97% (27B): the small tiers are launch-bound like Open-Jev with
    FLA, the 27B is GPU-bound with many tiny kernels.
  - The prefix-state handoff (shared token prefix once, questions' suffixes from the expanded cache) is not
    bit-identical (different GEMM shapes); on a private multi-question sample at 0.8B it changed 0.4% of decisions
    and was slower except on very long shared contexts. Numbers private.
- 2026-10-02 03:10 — Profiling and first prototypes on one leased GPU per node (node A GPU7: 0.8B / 4B; node B GPU7:
  27B), scored images, frozen Triton autotune caches, `--network none`.
- 2026-10-02 02:40 — Static catalogue of the repository; request-structure analysis of the private eval panel
  (numbers kept private). Our runtime already reads every candidate of a question in one row (option endpoints +
  a global query), so Open-Jev's per-candidate prefix tree does not apply; only the context shared by a request's
  questions is recomputed (once per question).
