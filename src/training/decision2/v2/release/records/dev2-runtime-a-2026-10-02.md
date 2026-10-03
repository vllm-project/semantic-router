# Decision 2.0 runtime speed-up, phase A (2026-10-02)

Owner: runtime phase A worker 2d541b40 (`track=runtime-a`), worktree `vllm-sr-dev2-runtime-a`, branch
`xunzhuo/decision-2-runtime-a`. User approval of phase A 2026-10-02 18:00 UTC+8 (gist `6b59c3be`, COORDINATION
18:00). The six repositories and the collection are public since 2026-10-03 01:25 UTC+8 (user decision); runtime
revisions on them were allowed again at 02:38 (COORDINATION). Ops, evidence, specs, decisions and receipts:
[`dev2-runtime-a-2026-10-02/`](dev2-runtime-a-2026-10-02/).

## Result

**All six repositories carry the phase A runtime as runtime-only revisions** (Vega 27B since 2026-10-03, together
with the opt-in shared-context switch; section 5), and Kai, Eos, Sol and Vega also ship the switch (off by default;
section 6). Every size answers all four scored panels (typed-final 1,600, css15 6,547,
public231 231, mlx-diag 2,275: 10,653 prompts) identically to the released runtime on the released weights: 0 answer
changes, 0.0 drift.

| Model | Superseded | New `main` | Parity (identical / prompts, drift) | p50 before → after | p95 before → after | Request peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Decision-2.0-Kai-0.6B | `c441862b` | `51b7b4740c8c70282648d3849d233a962234e4ec` | 10,653 / 10,653, 0.0 | 15.96 → 4.89 ms | 16.15 → 6.05 ms | 1.50 → 1.93 |
| Decision-2.0-Eos-0.8B | `25f0914a` | `1d380452ef703b283438732ec6393dd11d38ff24` | 10,653 / 10,653, 0.0 | 21.51 → 5.99 ms | 21.72 → 6.87 ms | 2.01 → 2.51 |
| Decision-2.0-Sol-2B | `951e7f7f` | `6a62b3198f3bcf87259cfc2d72185b4861884394` | 10,653 / 10,653, 0.0 | 21.09 → 7.18 ms | 21.41 → 7.21 ms | 4.57 → 5.07 |
| Decision-2.0-Nox-4B | `36596d27` | `137e28ceebd52b9611bb72e3645536867e0ec1ec` | 10,653 / 10,653, 0.0 | 27.26 → 12.94 ms | 28.36 → 14.58 ms | 9.25 → 9.74 |
| Decision-2.0-Lux-9B | `6af07f36` | `f77b41f5c1a98974548d929c03e7bce008f71bd0` | 10,653 / 10,653, 0.0 | 31.76 → 19.48 ms | 32.86 → 19.79 ms | 16.83 → 17.32 |
| Decision-2.0-Vega-27B | `5c85c127` | `9b067a95560284dac8c98ef4130fd5a2c5a92ff9` (section 5) | 10,653 / 10,653, 0.0 | 104.12 → 71.42 ms | 110.07 → 73.80 ms | not measured |

Latency: `runtime_bench.py`, 400 single typed-final requests after 400 warm-up requests (two passes), one MI300X GPU on
node A, the tier's release image, the old and the new package in turn from the same autotune cache; all 400 answers
bit-identical between the two runtimes in every bench. The card's Speed line is the new p50 (Kai 4.9 ms, Eos 6.0,
Sol 7.2, Nox 12.9, Lux 19.5, Vega 71.4). Nox and Lux have since been superseded by their publishers' model releases,
which build on this runtime.

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

## 5. Vega 27B (released 2026-10-03 on `5c85c127`)

First measured on `e60bd8e3`'s weights (p50 109.80 → 69.18 ms, p95 113.87 → 71.50 ms; evidence `27b/e60bd8e3/`) and
held while the 27B model release ran. That release landed as `5c85c127` with the old runtime; COORDINATION 02:38 /
03:18 UTC+8 then gave the Vega slot to this worker with the shared-context switch merged in. Parity and bench were
run again on `5c85c127` through the switch runtime (`27b/switch/`): 10,653 / 10,653 identical, 0.0 drift; 400 bench
items bit-identical, p50 104.12 → 71.42 ms, p95 110.07 → 73.80 ms; frozen cache `runtime-a-27b-ras` (`b406a73a…`,
2,788 files). `ra.sh 27B --switch` published `9b067a95` (spec `dev2-27b-ras.json`, decision
`Decision-2.0-Vega-27B.decision.ras.json`): weights byte-identical, the README's Speed line 93.4 → 71.4 ms, every
`release.sh` check, examples bit-identical to the superseded package on the same GPU, card HTTP, links, collection,
and the 86-request gate. The first post-check pass failed only in `ra_diff.py`, whose file listing (shared with the
rename record) still required a private repository; `ra_diff.py` now checks visibility against `hub.expected_private`
(`6aabf62d3`) and `--post-only` passed (the first pass is kept as `27b/switch/release/extra.before-*`).

## 6. Opt-in shared-context switch (`runtime/shared_ctx.py`, `9d90afd10`; COORDINATION 2026-10-03 02:38 / 03:18)

Runtime-only revisions with the runtime of `9cffe606c` (phase A plus `share_context`, off by default) on Kai, Eos and
Sol's phase A revisions, and on Vega's together with phase A. Same parity rollout: `fast.sh <tier> --switch` (old side:
current `main` with its release image, spec and autotune cache), `make_fast.py --kind switch` (`specs/dev2-<key>-ras.json`,
`<name>.decision.ras.json`), `ra.sh <tier> --switch` (`decision2/shared_ctx.py` must be added; the README may be
unchanged because its Speed line rounds to 0.1 ms). After each upload `ops/gate86.sh` runs the Index harness's
86-request parity gate on the download: the steps of `v2/eval/ix1/launch.sh parity` (the package's entry point, then
the kit runner `87d4650b` with the release engine, then `v2.eval.ix1.parity`) with the package's own `vllm-sr`
repo_id, since `launch.sh` still names the former organization. The phase A revisions of Kai, Eos and Sol passed it
first (86 / 86, max |dp| 0.0).

| Model | Superseded | New `main` | Parity (default path) | p50 before → after | p95 before → after | 86-request gate |
| --- | --- | --- | --- | --- | --- | --- |
| Decision-2.0-Kai-0.6B | `51b7b474` | `881bee413681d80ebeac86afcda8b4138dae516e` | 10,653 / 10,653, 0.0 | 4.86 → 4.88 ms | 6.05 → 6.04 ms | 86 / 86, 0.0 |
| Decision-2.0-Eos-0.8B | `1d380452` | `ad0aa724c924f7c4194be94b1b8441caf2d61c01` | 10,653 / 10,653, 0.0 | 5.96 → 6.00 ms | 6.84 → 6.89 ms | 86 / 86, 0.0 |
| Decision-2.0-Sol-2B | `6a62b319` | `4b75b52114583b4519001492e8dfb0926c89cfe1` | 10,653 / 10,653, 0.0 | 7.17 → 7.16 ms | 7.21 → 7.20 ms | 86 / 86, 0.0 |
| Decision-2.0-Vega-27B | `5c85c127` | `9b067a95560284dac8c98ef4130fd5a2c5a92ff9` | 10,653 / 10,653, 0.0 | 104.12 → 71.42 ms | 110.07 → 73.80 ms | 86 / 86, 0.0 |

All four: weights byte-identical (`ra_diff.py --switch`), post-checks OK, still public, Nox and Lux untouched. Kai,
Eos and Sol's README is unchanged (only `api.py`, `qwen.py`, the new `shared_ctx.py` and the manifest differ). The
first Kai and Eos benches ran while Vega's post-checks shared the GPU and are not used; both were repeated alone
(`<key>/switch/bench/`). Frozen caches `runtime-a-<key>-ras` (`<key>/switch/triton.json`). Receipts:
`<key>/switch/release/`.

GPU use: about 9.5 GPU-h of the 20 budgeted (phase A about 6; the switch rollout about 3.5, most of it Vega).

## 7. ROCm hotfix: `attn_prep` above 2 GiB and the eager fallback (2026-10-03; hotfix owner 490b6f72)

COORDINATION 2026-10-03 12:03 UTC+8: the Index submission's release spot check found that Nox `ce1bdc9d`, Lux
`214ffa43` and Vega `9b067a95` return `error` on 5 of 2,160 sample requests (`PassManager::run failed` in AMD
`make_ttgir`, compiling `_attn_prep_kernel`). Branch `xunzhuo/decision-2-runtime-hotfix`, worktree
`vllm-sr-dev2-runtime-hotfix`; runtime commit `3c8cd1a84`.

**Root cause.** Triton 3.7.1 marks a pointer argument `tt.pointer_range = 32` only when its tensor's storage is at most
2 GiB. `_attn_prep_kernel` chose its source, weight and destination pointers with a runtime branch
(`if hid < NH`: query, else key). On a long request with many questions, the query projection
(B × T × heads × 2 × D BF16; 16 KiB per token for Nox and Lux) is above 2 GiB while the key projection is below, so the
two arms carry differently canonicalized pointers and `TritonAMDGPUCanonicalizePointers` asserts "ifOp types must
match in both arms". Every failing log shows the same signature: the query pointer is the only one without
`pointer_range`. That is why a standalone call with small batches compiled, why the autotune cache did not matter, and
why Vega (wider query rows) fails on more requests. Eos and Sol (8 KiB per token) fail the full-Index request
`reversechain_query_51` the same way; Kai's narrower rows (4 KiB per token) did not reach the bound on the Index.

**Fix (`runtime/fast_kernels.py`).** `attn_prep` launches the kernel once per tensor (query, then key); each launch is
straight-line code with the same per-row arithmetic, so no pointer passes through a runtime branch.

**Guard (`runtime/fast.py`).** A fused decoder-layer call that raises (compile or launch failure) runs the layer's
eager forward instead, which computes the same values (the fused kernels round exactly like the ops they replace);
that input shape then stays eager in that layer, and the receipt counts it (`fused_fallbacks`). A failure during HIP
graph capture propagates to `Graphs` as before, which runs that shape eagerly.

**Kernel and unit evidence (node B).**

- `ops/attn_prep_2gib.py` at Nox's widths, 9 × 16,384 tokens (query projection 2.42 GB, key 0.30 GB): the released
  kernel fails with `PassManager::run failed`; the new kernel runs and equals the released kernel run row by row on
  copies below 2 GiB (9 / 9 rows bit-identical).
- `tests/gpu_fast_path.py`: 14 / 14 cases bit-identical to the eager backbone, including the > 2 GiB Nox-width case
  (0 fallbacks) and every fused kernel forced to fail (each layer on the fallback, still bit-identical).

**Rollout (`fast.sh` / `make_fast.py` / `ra.sh` / `ra_diff.py --hotfix`).** The old side is each repository's current
`main` with the image, spec and autotune cache of the release that built it. `--parity` adds a third side, the new
package with every fused kernel raising (`ops/fallback_parity.py`), compared with the old side at tolerance 0, so the
fallback is shown byte-identical on every prompt. The parity and bench ran on node D (Kai on node A) from copies of
the previews and caches; the benches were repeated alone. `card.speed` stays the current main's (README byte-identical);
the bench is recorded in each decision. `ra.sh --hotfix` accepts exactly `decision2/fast.py` and
`decision2/fast_kernels.py` changing.

| Model | Superseded | New `main` | Parity default / fallback (identical of 10,653, drift) | Bench p50 old → new (400 bit-identical) | 86-request gate |
| --- | --- | --- | --- | --- | --- |
| Decision-2.0-Nox-4B | `ce1bdc9d` | `7fc0023a8c51ffaf0f4d4a8f1eb1fb54fa2451cc` | 10,653 / 10,653, 0.0 | 13.94 → 13.72 ms | 86 / 86, 0.0 |
| Decision-2.0-Lux-9B | `214ffa43` | `7c6792f7e59dce7a64115bfe750ee636e7a63bd4` | 10,653 / 10,653, 0.0 | 18.71 → 18.73 ms | 86 / 86, 0.0 |
| Decision-2.0-Vega-27B | `9b067a95` | `477e90f537eb5bd62e90d5e5361c7b654e69cf45` | 10,653 / 10,653, 0.0 | 74.67 → 74.54 ms | 86 / 86, 0.0 |
| Decision-2.0-Eos-0.8B | `ad0aa724` | `34e2db970f2f49ef431237aa3218e0688a1fe992` | 10,653 / 10,653, 0.0 | 6.12 → 6.16 ms | 86 / 86, 0.0 |
| Decision-2.0-Sol-2B | `4b75b521` | `23cbe9f96dde4a7e1e6a238d576128bf73c5b09e` | 10,653 / 10,653, 0.0 | 7.55 → 7.61 ms | 86 / 86, 0.0 |
| Decision-2.0-Kai-0.6B | `881bee41` | `d06cf74b1304c6454b9de6e0adb8eaad5ac719fa` | 10,653 / 10,653, 0.0 | 4.91 → 4.95 ms | 86 / 86, 0.0 |

Every release: weights byte-identical (`ra_diff.py --hotfix`), examples bit-identical to the superseded package on
the same GPU, image and frozen cache, the Hub `trust_remote_code` smoke under Transformers 5.17 and 5.18, readback,
gate seal, collection order, card HTTP, links 8 / 8, `gate evaluate`. Frozen caches `runtime-a-<key>-rah`
(`<key>/hotfix/triton.json`). Receipts: `/data/dev2/runs/release/dev2-rah-<tier>-*` on node A.

**Real calls (Index harness, the stored runs' frozen caches).**

- Before upload, the staging builds answered the 5 failing requests `ok` (Nox, Lux; Nox's answers byte-identical to
  the stored pre-release answers, max |Δp| 0.0), and the Vega staging build answered all 2,160 sample requests `ok`
  (its 5 formerly failing requests byte-identical to the stored answers; 2 near-tie flips elsewhere).
- Release spot checks of the downloads (2,160 requests, `v2.eval.ix1.submission spotcheck` + `repin.py spotcheck`):
  Nox `7fc0023a` 2,160 `ok`, 0 flips, max |Δp| 0.0143; Lux `7c6792f7` 0 flips, 0.0249; Eos `34e2db97` 6 near-tie
  flips, 0.0153; Sol `23cbe9f9` 0 flips, 0.0141; Vega `477e90f5` 1 near-tie flip, 0.0103; Vega `5c85c127` (the PR
  pin before the fix) 0 flips, 0.0103.
- The 60 heaviest Index requests (questions × longest row): the released Eos `ad0aa724` and Sol `4b75b521` each
  error on `reversechain_query_51`; the fixed Nox, Lux, Eos and Sol return 59 `ok` and 1 over the input limit, and
  the fixed Vega 60 `ok`.

**Index submission.** Dataset `vllm-sr/decision-2.0-decision-index` and PR
https://github.com/apolinario/decision-index/pull/48 pin the fixed revisions once each passes the spot check
(`v2/eval/ix1/repin.py` recomputes `weights-vs-release.json` for the new revision; `publish_dataset.py --update`):
dataset commits `0fac8ed9` (Nox, Lux) and `bf87c73e` (Vega, Sol, Eos); PR head `64768747`. Kai stays pinned at
`881bee41`, which never reached the bound on the Index.

Shared-module changes (called out): `runtime/fast.py`, `runtime/fast_kernels.py`, `tests/gpu_fast_path.py`; the ops
above; `v2/eval/ix1/launch.sh` entries, `repin.py`, `publish_dataset.py --update`. The opt-in shared-context switch is
unchanged.
