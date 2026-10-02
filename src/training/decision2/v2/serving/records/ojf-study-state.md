# open-jev-fast study — state (worker 2d541b40)

Study of the MIT-licensed `open-jev-fast` inference backend (commit `c52b8bb`) for ideas portable to the Decision 2.0
runtime on AMD MI325X. No third-party code, weights or data enter this repository; prototypes live in a private
scratch directory; no released package is changed. Times are UTC.

## Log (newest first)

- 2026-10-02 03:10 — Profiling and first prototypes on one leased GPU per node (node A GPU7: 0.8B / 4B; node B GPU7:
  27B), scored images, frozen Triton autotune caches, `--network none`.
  - 0.8B, latency panel (first 400 typed-final prompts, single requests): the shipped runtime is launch-bound
    (about 800 kernel launches per request). One HIP graph per exact padded shape, with the attention masks built
    from host-known lengths, cuts p50 22.3 → 10.1 ms and p95 22.5 → 11.9 ms; **all 400 answers bit-identical**
    to the released runtime. Host-built masks alone (sync removal) give no gain in eager mode.
  - 27B latency panel with the LoRA merge, and the prefix-state handoff for multi-question requests, are running.
- 2026-10-02 02:40 — Static catalogue of the repository; request-structure analysis of the private eval panel
  (numbers kept private). Our runtime already reads every candidate of a question in one row (option endpoints +
  a global query), so Open-Jev's per-candidate prefix tree does not apply; only the context shared by a request's
  questions is recomputed (once per question).
