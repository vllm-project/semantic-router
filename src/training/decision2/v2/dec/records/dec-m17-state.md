# Decoder M17 — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17` (restart worker; the first M17 worker
stopped after creating the worktree, nothing had run). Prereg `da770d98a`, ops `1ce8b2220` (mirror on nodes E / F),
data lock `4f68f1eb0`.

## 2026-10-02 02:00Z

- Data built on E and F (01:58Z), byte-identical: `4b-LHS10SD` 66,004 rows / T + 64 tokens (IB .100, 17.1% of English
  tokens swapped out), `4b-LHS17SD` 71,088 rows / T + 68 (IB .170, 29.0%); multilingual share .415 (LH .414);
  `sentfin` dropped from the IB pool (COORDINATION 01:05). `READY-m17.json` written on F.
- Leases node F GPU2, 3, 6, 7 `track=dec-m17`; chains launched 01:59:48Z from `1ce8b2220`: GPU6 `4b-LHS10SD` s1
  (pre-warm, zero-step running), GPU7 s2, GPU2 / GPU3 `4b-LHS17SD` s1 / s2 waiting for the pre-warm marker.
- Node F prep: Triton caches 4b-train / 4b-read copied from M15; reference readouts `4b-LH-f` and `4b-LHA10SD-m13`
  copied from M15 (8 panels each, plus their old MLX-DEV reads); old MLX-DEV 4B panel copied (report only).
- Node E: data build only; GPU0–3 untouched (Index runs only, if any). GPU-h so far: 0.
