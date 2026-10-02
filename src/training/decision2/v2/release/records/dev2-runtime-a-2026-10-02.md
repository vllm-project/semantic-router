# Decision 2.0 runtime speed-up, phase A (2026-10-02)

Owner: runtime phase A worker 2d541b40 (`track=runtime-a`), worktree `vllm-sr-dev2-runtime-a`, branch
`xunzhuo/decision-2-runtime-a`. User approval of phase A 2026-10-02 18:00 UTC+8 (gist `6b59c3be`, COORDINATION
18:00). Everything stays private. Ops, evidence, specs, decisions and receipts:
[`dev2-runtime-a-2026-10-02/`](dev2-runtime-a-2026-10-02/).

## Result

**Five repositories carry the phase A runtime as runtime-only revisions; Vega 27B is measured and prepared and is
held for the 27B model release.** Every size answers all four scored panels (typed-final 1,600, css15 6,547,
public231 231, mlx-diag 2,275: 10,653 prompts) identically to the released runtime on the released weights: 0 answer
changes, 0.0 drift.

| Model | Superseded | New `main` | Parity (identical / prompts, drift) | p50 before → after | p95 before → after | Request peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Decision-2.0-Kai-0.6B | `c441862b` | `51b7b4740c8c70282648d3849d233a962234e4ec` | 10,653 / 10,653, 0.0 | 15.96 → 4.89 ms | 16.15 → 6.05 ms | 1.50 → 1.93 |
| Decision-2.0-Eos-0.8B | `25f0914a` | `1d380452ef703b283438732ec6393dd11d38ff24` | 10,653 / 10,653, 0.0 | 21.51 → 5.99 ms | 21.72 → 6.87 ms | 2.01 → 2.51 |
| Decision-2.0-Sol-2B | `951e7f7f` | `6a62b3198f3bcf87259cfc2d72185b4861884394` | 10,653 / 10,653, 0.0 | 21.09 → 7.18 ms | 21.41 → 7.21 ms | 4.57 → 5.07 |
| Decision-2.0-Nox-4B | `36596d27` | `137e28ceebd52b9611bb72e3645536867e0ec1ec` | 10,653 / 10,653, 0.0 | 27.26 → 12.94 ms | 28.36 → 14.58 ms | 9.25 → 9.74 |
| Decision-2.0-Lux-9B | `6af07f36` | `f77b41f5c1a98974548d929c03e7bce008f71bd0` | 10,653 / 10,653, 0.0 | 31.76 → 19.48 ms | 32.86 → 19.79 ms | 16.83 → 17.32 |
| Decision-2.0-Vega-27B | `e60bd8e3` | held (section 5) | 10,653 / 10,653, 0.0 | 109.80 → 69.18 ms | 113.87 → 71.50 ms | 57.36 → 61.49 |

Latency: `runtime_bench.py`, 400 single typed-final requests after 400 warm-up requests (two passes), one MI300X GPU on
node A, the tier's release image, the old and the new package in turn from the same autotune cache; all 400 answers
bit-identical between the two runtimes in every bench. The card's Speed line is the new p50 (Kai 4.9 ms, Eos 6.0,
Sol 7.2, Nox 12.9, Lux 19.5).

**Kai against the projection.** The phase A gist projected about 5 ms for Kai 0.6B; measured p50 4.89 ms (p95
6.05 ms), 3.3× faster than the released 15.96 ms. A per-request profile of the new runtime totals 5.28 ms, 4.11 ms of
it in the backbone.

## 1. Runtime (`runtime/fast.py`, `runtime/fast_kernels.py`; commit `54303117b`)

Installed by `decision2/qwen.py` on ROCm with Transformers 5.17 for `qwen3`, `qwen3_5_text` and `qwen3_5`; every
other setup keeps the released path unchanged.

- **HIP graphs per exact padded shape**: captured on the second sighting of `(rows, T, padded,
  output_hidden_states)`, LRU cap of 512 graphs and 4 GiB of outputs, eager above 65,536 tokens, `graphs=False` turns
  them off. Every bench forward after warm-up replayed a graph (1,194–1,195 replays, 0 failures).
- **Host-built masks**: the BF16 additive mask (0 / −inf) built once per shape on a 16-aligned row stride, instead of
  SDPA converting the boolean mask in every layer.
- **Exact trims**: one shared BF16 cast of each Linear's input (`shared_cast_linears`), RMSNorm's `1 + w` cached
  (`rmsnorm_plus_one`).
- **27B: the lean unmerged LoRA path** (496 LoRA Linears; FP32 factors stay FP32 and unmerged).
- **Bit-exact fused Triton kernels** on gfx942 (residual add + RMSNorm, attention prep with RoPE, SiLU-mul): Qwen3.5
  and, since `cb6a73d14`, Qwen3 (Kai: plain-weight RMSNorm, head_dim 128, round before the weight). 24–64 layers
  fused per size.
- **Frozen, pre-warmed autotune cache** per size: the new side's cache after the parity run, read-only under
  `release/triton/runtime-a-<key>` (digest in `<key>/triton.json`); every release run copies it.

Tests: `tests/gpu_fast_path.py` (8 cases, 0 differences on node A GPU0) and the release suite in the release image
(222 passed, 2 skipped).

## 2. Left out (not shown bit-identical)

- **The FLA gfx942 retune** of `xunzhuo/decision-2-rocm-kernels` `c065bf18e`: not evaluated for bit-identity; the
  released FLA call and its autotune configurations stay.
- **GVA-prenorm** from the same branch: not evaluated; left out.

## 3. Releases

`ops/make_fast.py release` derives `specs/dev2-<key>-ra.json` and `<name>.decision.ra.json` from the org card-only
revision (spec, sealed gate and decision of its release on node A) and this record's parity and bench: only
`runtime_source`, `card.speed`, one runtime sentence, the gate receipt and `_release` change. `ops/ra.sh` publishes:

- before the upload: `main` equals the superseded revision; the fresh build differs from `main` only in the runtime
  (`api.py`, `qwen.py` changed; `fast.py`, `fast_kernels.py` added) and the README's Speed line; identity,
  parameters, profile and `max_input_tokens` are equal; `gate profile` passes; no other `release.sh` for the repo;
- `release.sh --upload --collect --already-collected`: native examples (two processes), the card example, remote code
  on the package and on the download, the card example from the Hub in fresh caches under Transformers 5.17 and 5.18,
  readback, gate seal, collection;
- after the upload: weights byte-identical to the superseded revision (`ra_diff.py`), examples bit-identical to the
  superseded package, card HTTP 8 / 8, links 8 / 8, the read-only collection check (private, nothing moved), `gate
  evaluate`.

All five passed; storage 58.17 GB of 100. Receipts: `<key>/release/`.

**Examples against the superseded package.** The org revisions ran their examples on another device (Kai: CPU) or
another GPU with the org run's own autotune cache (Nox, Lux). Those receipts are not comparable bit for
bit: Nox's differs by up to 0.0088 in probability with 0 category changes, while the same comparison on the same GPU
and cache is bit-identical. `ra.sh` (`2bc554c58`) therefore runs the superseded package's examples on the same GPU,
image and copy of the frozen cache and requires bit-identity there; the receipt comparison is kept as
`examples-vs-superseded-receipt.json`. Nox's post-checks were repeated that way with `--post-only` (`aeded84e1`);
its first two attempts stay in `4b/release/extra.before-*`.

## 4. Coordination

- Publishes ran one at a time on node A GPU0 (Kai, Sol, Nox, Eos, Lux; 11:22Z–12:40Z). `ra.sh` re-read `main`
  right before each upload and refused while another `release.sh` for the repo ran on the node.
- 9B M10 merged this branch and derives its KIB4-a33 release from `specs/dev2-9b-ra.json` (superseding `f77b41f5`,
  with the frozen cache `runtime-a-9b`), so Lux keeps the fast runtime.
- The release-gate commits `5e2c3fa92` and `c6c754a41` are cherry-picked here (`f71d7a0f6`, `64d951e1c`).
- Shared-module changes (called out): `runtime/fast.py`, `runtime/fast_kernels.py` (new), `decision2/qwen.py` (the
  hook), `tests/gpu_fast_path.py`.

## 5. Vega 27B (held)

Parity and bench on `e60bd8e3`'s weights pass (table above; 64 layers fused, lean LoRA). `specs/dev2-27b-ra.json` and
`Decision-2.0-Vega-27B.decision.ra.json` supersede `e60bd8e3`, frozen cache `runtime-a-27b` (`c42bbdfb…`). Not
published: at 21:20 UTC+8 COORDINATION directed the 27B worker (f5779f55) to release the cross-arm point on top of
`e60bd8e3`, and a runtime-only revision now would make that release re-derive. Either the 27B release derives from
`dev2-27b-ra.json` as 9B M10 did, or, after it lands, `ra.sh 27B` publishes a runtime-only revision on top of it
(re-derived against the new `main`, with a parity run on the new weights).

GPU use: about 4–5 GPU-h of the 20 budgeted (parity and bench 1.7 GPU-h; the rest releases, post-checks and tests).
