# BF16-resident runtime: runtime-only revisions of the six DEV2.0 repositories (2026-10-01)

Coordinator note 2026-10-01 10:30 UTC+8 (b5f60b33): roll the shipped runtime's BF16-resident Linear weights out to
all six released private repositories as runtime-only revisions, each only with 0 answer changes on every scored
prompt and on mlx-diag, and measure latency and memory old vs new. Release worker, worktree `vllm-sr-dev2-release`
(branch `xunzhuo/decision-2-training-release`); state [`dev2-bf16-resident-state.md`](dev2-bf16-resident-state.md).
Node receipts are copied under [`dev2-bf16-resident-2026-10-01/`](dev2-bf16-resident-2026-10-01/); node paths stay
in records only. Times are UTC. Everything stays private.

## Result

- **All six repositories are republished as runtime-only revisions** with the BF16-resident runtime (`5dc962b00`),
  each with **0 answer changes on every scored prompt and on mlx-diag** (11,053 answers per model) before the upload
  and again on the real download. All private; the collection "🎲 Decision 2.0" is unchanged: the six in order (0.6B, 0.8B, 2B, 4B, 9B, 27B), then
  item 7, a collaborator's `DEV2.0-Route-0.6B`, already there before this rollout and not touched by it.

  | Model | New `main` | Final decision (gate items) | Replaces | Answer changes (max drift) | p50 latency; peak GPU memory, old → new |
  | --- | --- | --- | --- | --- | --- |
  | DEV2.0-0.6B | `def20a1c62f2941bb727d079d7f895800e9d8808` | `330f3aef…` (successor, 12 / 12) | `476fe984` | 0 of 11,053 (0) | 19.2 → 16.6 ms; 2.3 → 1.5 GiB |
  | DEV2.0-0.8B | `e13a40f848b3b27489fa4083d5e75950750f4856` | `d0b8347b…` (own 1.0, 6 / 6) | `bede7938` | 0 of 11,053 (8.9e-16) | 22.7 → 21.6 ms; 2.9 → 2.0 GiB |
  | DEV2.0-2B | `56950ec50d74e0a6d34a04070eb02c05a19f6775` | `efee9bf0…` (own 1.0, 6 / 6) | `a53cf66a` | 0 of 11,053 (8.9e-16) | 24.1 → 23.5 ms; 7.1 → 4.6 GiB |
  | DEV2.0-4B | `4f560ae5d26d378cd8db0a93a06c1603ea76b635` | `b33fb92a…` (own 1.0, 6 / 6) | `fadbba4f` | 0 of 11,053 (1.9e-14) | 28.5 → 28.1 ms; 15.8 → 9.3 GiB |
  | DEV2.0-9B | `b4f65fa8ec56c1930d3c52b29a9335f89ea10544` | `36a468e7…` (own 1.0, 6 / 6) | `e51f9881` | 0 of 11,053 (8.9e-16) | 33.8 → 27.2 ms; 29.8 → 16.8 GiB |
  | DEV2.0-27B | `4e89288d6146034743a14e3fbb98b5864e693c52` | `cc65becb…` (successor, 13 / 13) | `53233103` | 0 of 11,053 (0) | 122.9 → 93.3 ms; 97.6 → 52.1 GiB |

- **Weights are byte-identical.** On the Hub, only `decision2/api.py`, `decision2/qwen.py`, `README.md` and
  `MODEL_MANIFEST.json` changed against each replaced revision; every weight file keeps its size and LFS SHA-256 and
  the loaded parameter counts are unchanged. The release examples give bit-identical answers to the replaced
  revisions' recorded pre-upload examples.
- **Max drift** is the largest absolute difference of any reported number (probabilities, Score expected values)
  against the scored predictions. Panel by panel it equals the drift of the replaced revisions' own release parity
  (0 for 0.6B and 27B, ≤ 1.9e-14 elsewhere).
- **Card:** one line under Model details, e.g. "Runtime update: BF16-resident weights; answers unchanged; latency p50
  122.9 → 93.3 ms, p95 127.5 → 96.8 ms; peak GPU memory 97.6 → 52.1 GiB (400 single requests, 346 input tokens on
  average, one AMD MI325X GPU)." `MODEL_MANIFEST.json` `runtime_equivalence` gains one runtime sentence.
- **Latency and memory (§3):** p50 falls 0.4–2.6 ms up to 4B, 6.7 ms at 9B (−20%) and 29.6 ms at 27B (−24%); peak
  GPU memory falls 31–47% (27B: 97.6 → 52.1 GiB).
- **Storage:** 52.02 → 52.13 GB of 100 GB (headroom 47.87 GB). The six revisions add only small git files and no LFS objects (every weight blob is shared with the replaced revision, so nothing needs purging); the +0.11 GB is another track's `decision-2.0-training-data`.
- **GPU:** 2.68 GPU-hours (node A GPU0–1 1.61, node B GPU2 1.07; §5).

## 1. Runtime change (`5dc962b00`, shared module, separate commit)

`v2/release/runtime/qwen.py` used to load every Qwen package with `model.float()` and run the backbone under BF16
autocast, so every Linear weight was held in FP32 and cast to BF16 before every matmul. On a GPU the runtime now
calls `keep_linear_bf16(model.backbone)` before moving the model to the device:

- every `nn.Linear` weight (and bias) of the backbone whose value BF16 represents exactly is stored in BF16 — the
  exact tensor autocast would have produced, so every product is unchanged;
- everything else stays FP32: the embedding table, all norms, gated-delta `A_log` / `dt_bias`, conv filters, the
  decision head (outside the backbone), any weight shared with a non-Linear module (tied embeddings), and any
  Linear weight BF16 cannot hold exactly — the 27B adapter's FP32-trained LoRA factors, which stay unmerged PEFT
  modules (autocast still casts them per call, as before);
- autocast is unchanged (`torch.autocast("cuda", torch.bfloat16)` around the backbone, head in FP32 with autocast
  off); CPU loading is unchanged (FP32, no autocast);
- `Decision2.from_pretrained(..., bf16_resident=False)` keeps the FP32 copies (`examples.py run --fp32-master`);
  example and parity receipts now record the residency counts and the loaded elements by dtype.

Converting on the host before `.to(device)` also means the GPU never holds the FP32 copies: memory after loading
falls by the size of the Linear weights in FP32 minus BF16.

## 2. Tests

- `v2.release.tests.test_bf16_resident` (image, CPU): only exact Linear tensors move (tied / inexact stay FP32,
  values unchanged); repeat calls are stable; tiny Qwen3 dense, Qwen3.5 hybrid (gated-delta tensors stay FP32) and
  unmerged-LoRA backbones give **bitwise-equal outputs** under BF16 autocast before and after the conversion. 5 / 5.
- `v2.release.tests.gpu_bf16_resident` (image, node A GPU0): tiny real `qwen-full` Qwen3 and Qwen3.5 packages and a
  `qwen-adapter` LoRA package, built exactly as releases are, answer the release examples in fresh isolated
  processes BF16-resident vs FP32-master: **byte-identical answers** for all three; the resident runs hold every
  backbone Linear weight in BF16 except the LoRA factors (Qwen3 14 / 0, Qwen3.5 15 / 0, adapter 14 / 28); the
  FP32-master and CPU runs hold every parameter in FP32. 3 / 3.
- The release suite (146 stdlib tests) passes; `check_no_private.sh` before every commit.
- Image limits found on the way: Transformers 5.17 binds the image's CUDA-only `causal_conv1d` even for CPU tensors,
  and the image's PyTorch has no CPU LAPACK for the chunked gated-delta reference, so the Qwen3.5 CPU unit test uses
  the torch references (recurrent form) and the GPU fixture runs its CPU leg for the Qwen3 packages only (the
  Qwen3.5-family cards already say CPU is not verified).

## 3. Latency and memory, old vs new runtime (same GPU)

`v2/release/runtime_bench.py`: one isolated container process per runtime (old = the verified download of the
released revision with its own vendored runtime; new = the preview build of the runtime-only spec), the scored
image and kernels, a fresh copy of the frozen autotune cache each, the first 400 typed-final prompts (one question
each) as single in-process `system_one` calls with `torch.cuda.synchronize()` around each. The 400 prompts run once
untimed (first use of each input shape), then the same 400 are timed. Memory is `torch.cuda` allocated bytes: after
loading, and the peak during the timed requests (weights + activations). One run at a time per node (node A GPU1;
27B node B GPU2).

| Model | p50 ms | p95 ms | mean ms | items/s | after load GiB | peak GiB | Linear BF16 / FP32 | identical answers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 19.2 → 16.6 | 19.4 → 16.9 | 19.0 → 16.5 | 52.5 → 60.6 | 2.22 → 1.40 | 2.32 → 1.50 | 196 / 0 | 400 / 400 |
| DEV2.0-0.8B | 22.7 → 21.6 | 22.9 → 22.7 | 25.3 → 24.7 | 39.5 → 40.5 | 2.81 → 1.90 | 2.91 → 2.01 | 186 / 0 | 400 / 400 |
| DEV2.0-2B | 24.1 → 23.5 | 24.8 → 23.7 | 24.2 → 23.4 | 41.4 → 42.7 | 7.02 → 4.46 | 7.14 → 4.57 | 186 / 0 | 400 / 400 |
| DEV2.0-4B | 28.5 → 28.1 | 29.0 → 28.9 | 32.8 → 28.2 | 30.5 → 35.5 | 15.68 → 9.12 | 15.83 → 9.25 | 248 / 0 | 400 / 400 |
| DEV2.0-9B | 33.8 → 27.2 | 53.3 → 27.6 | 36.6 → 27.8 | 27.3 → 36.0 | 29.58 → 16.70 | 29.79 → 16.83 | 248 / 0 | 400 / 400 |
| DEV2.0-27B | 122.9 → 93.3 | 127.5 → 96.8 | 124.3 → 94.0 | 8.0 → 10.6 | 97.26 → 51.90 | 97.56 → 52.07 | 496 / 992 | 400 / 400 |

- Every tier answered all 400 bench prompts **bit-identically** (0 changes, drift 0.0) old vs new.
- Peak memory falls 31–47%: the Linear weights are held once in BF16 instead of in FP32 (and the per-call BF16
  copies disappear). The gain in latency grows with size: −0.4 to −2.6 ms up to 4B, −6.7 ms at 9B, −29.6 ms (−24%)
  at 27B (the coordinator's estimate was ~10 ms at 9B and ~30 ms at 27B). The old 9B runtime also had a slow tail
  (p95 53 ms vs p50 34 ms) that the new one does not have.
- Superseded bench runs (kept on the nodes, counted in GPU-hours): a first pass with 20 warm-up requests (p95
  dominated by first-use shape spikes) and a second pass whose node-A runs overlapped on GPU0 and GPU1 (a model load
  on one GPU disturbed timings on the other). Their answers were also 400 / 400 identical.

## 4. Releases

Runner [`ops/rollout.sh`](dev2-bf16-resident-2026-10-01/ops/rollout.sh) `--release`: `release.sh --upload --collect
--already-collected` with the final spec and decision, the scored image (`f83b1d10` for 0.6B–9B, the kernel image
`dbe5f32b` for 27B), the image's FLA / causal-conv1d kernels required (Qwen3.5 family) and a fresh copy of each scoring
run's frozen Triton autotune cache (digest checked first; 0.6B is dense Qwen3 and needs neither, its parity runs at
tolerance 0). Parity covers every scored prompt — typed-final 1,600 prompts (2,000 answers), css15 6,547, public231
231 — and mlx-diag 2,275, before the upload and again on the real `hf download`; parity-pre blocks the upload on any
change. The 4B's mlx-diag ran in its own no-upload run with the mlx scoring run's cache, as in its BF16-storage
release. Then: [`ops/runtime_diff.py`](dev2-bf16-resident-2026-10-01/ops/runtime_diff.py) against the replaced
revision on the Hub, the new pre-upload examples against the replaced revision's recorded `pre-a.json` (tolerance 0),
collection order and title, card HTTP, links, `gate evaluate`, storage.

Answer changes (max drift of any reported number), parity-pre / parity-post:

| Model | typed-final (2,000) | css15 (6,547) | public231 (231) | mlx-diag (2,275; 4B: its own no-upload run) |
| --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| DEV2.0-0.8B | 0 (8.9e-16) / 0 (8.9e-16) | 0 (3.3e-16) / 0 (3.3e-16) | 0 (4.4e-16) / 0 (4.4e-16) | 0 (4.4e-16) / 0 (4.4e-16) |
| DEV2.0-2B | 0 (8.9e-16) / 0 (8.9e-16) | 0 (3.3e-16) / 0 (3.3e-16) | 0 (2.2e-16) / 0 (2.2e-16) | 0 (6.7e-16) / 0 (6.7e-16) |
| DEV2.0-4B | 0 (1.2e-15) / 0 (1.2e-15) | 0 (2.2e-16) / 0 (2.2e-16) | 0 (1.8e-14) / 0 (1.8e-14) | 0 (1.9e-14) |
| DEV2.0-9B | 0 (8.9e-16) / 0 (8.9e-16) | 0 (3.3e-16) / 0 (3.3e-16) | 0 (4.4e-16) / 0 (4.4e-16) | 0 (4.4e-16) / 0 (4.4e-16) |
| DEV2.0-27B | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |

Release checks:

| Model | Re-hashed files | Loaded parameters | Weights vs replaced revision | Changed files | Examples vs replaced | Links | Wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 33 | 597,103,104 | 2 files, 1.51 GB: identical | api.py, qwen.py + MODEL_MANIFEST.json, README.md | identical | 14 / 14 | 510 s |
| DEV2.0-0.8B | 30 | 753,446,208 | 2 files, 2.02 GB: identical | api.py, qwen.py + MODEL_MANIFEST.json, README.md | identical | 14 / 14 | 620 s |
| DEV2.0-2B | 33 | 1,883,930,944 | 3 files, 4.79 GB: identical | api.py, qwen.py + MODEL_MANIFEST.json, README.md | identical | 15 / 15 | 733 s |
| DEV2.0-4B | 36 | 4,208,383,488 | 6 files, 9.70 GB: identical | api.py, qwen.py + MODEL_MANIFEST.json, README.md | identical | 17 / 17 | 851 s |
| DEV2.0-9B | 40 | 7,940,895,744 | 10 files, 17.93 GB: identical | api.py, qwen.py + MODEL_MANIFEST.json, README.md | identical | 15 / 15 | 1161 s |
| DEV2.0-27B | 32 | 26,096,775,168 | 2 files, 1.89 GB: identical | api.py, qwen.py + MODEL_MANIFEST.json, README.md | identical | 12 / 12 | 3424 s |

- Every run: repeat-pre and repeat-post bit-identical, card example executed before and after the download, readback
  private with no card problems, card HTTP passed (anonymous access refused), collection "🎲 Decision 2.0" unchanged
  (0.6B, 0.8B, 2B, 4B, 9B, 27B, then the collaborator's `DEV2.0-Route-0.6B`), `gate evaluate` all items passed; the gate sealed each final decision to its revision.
- Order: 0.6B → 0.8B → 2B → 4B → 9B on node A GPU0, one after another; the 27B on node B GPU2 started with the 9B run
  and uploaded after it (the 9B's scored image `f83b1d10` and inputs exist only on node A; node B holds the 27B
  inputs and the kernel image). For the 27B on node B, three small successor-gate evidence files were relayed from
  node A to the same paths (SHA-256 equal on both nodes), the A20r checkpoint's SHA-256 list was compared across the
  nodes, and the old-runtime side of the bench is a fresh real download of `5323310` (`ops/relay_27b.sh`). Node B's HF
  cache links the base blobs into a shared store, which the containers mount read-only.
- Decisions ([`ops/make_bf16r.py`](dev2-bf16-resident-2026-10-01/ops/make_bf16r.py)): each copies the superseded final
  decision's judgement (identity, report, paired comparison, calibration, licence, disclosures, C1 line; the 0.6B and
  27B successor profile with the same current revision and evidence) and changes only the action, the rationale and
  the supersedes chain; `runtime_revision` names the runtime commit and the bench comparison.

## 5. GPU-hours

**2.684 GPU-hours** (node A GPU0–1 1.611, node B GPU2 1.073), all under the shared lease `owner.release-bf16r`
(released). Wall-clock × GPUs; release runs from their `RELEASE-RECEIPT.json`, tests and benches from their start
stamp to their last write. Preview builds, relays, diffs and storage checks were CPU only.

| Item | Where | Runs | GPU-h |
| --- | --- | --- | ---: |
| Tests (`test_bf16_resident` on CPU, `gpu_bf16_resident`) | node A GPU0 | 3 (the first stopped at the CPU unit test, the second was superseded) | 0.056 |
| Benches reported in §3 (one at a time) | node A GPU1 (0.6B–9B), node B GPU2 (27B) | 6 | 0.267 |
| Benches superseded (20 warm-up; overlapping on two GPUs; 27B with the blob store unmounted) | node A GPU0 / GPU1, node B GPU2 | 10 | 0.258 |
| Releases | node A GPU0 (0.6B 0.142, 0.8B 0.172, 2B 0.204, 4B 0.236, 9B 0.323), node B GPU2 (27B 0.951) | 6 | 2.028 |
| 4B mlx-diag parity (no upload) | node A GPU1 | 1 | 0.076 |
| **Total** | | | **2.684** |

## 6. Commits

| Commit | What |
| --- | --- |
| `5dc962b00` | runtime (shared modules, separate commit): `keep_linear_bf16` in `runtime/qwen.py`, `bf16_resident` in `runtime/api.py`, `examples.py run --fp32-master` and residency / dtype receipt fields; `test_bf16_resident`, `gpu_bf16_resident` |
| `fe666a1ac`, `9a8d72b91` | tests: the Qwen3.5 CPU parity test on the torch references; the GPU fixture's CPU leg for the Qwen3 packages |
| `91d36d213` | `v2/release/runtime_bench.py`; ops `make_bf16r.py`, `rollout.sh`, `runtime_diff.py`, `relay_27b.sh`; draft specs and decisions |
| `79bff5da9` | bench: timed second pass; node B blob-store mount |
| `45e097444` | final specs `specs/dev2-*-bf16r.json`, decisions, bench receipts — the mirror every release ran from (`runtime_source` = the `5dc962b00` mirror) |
| `3c4e51ed1`, `c7fb109e7` | progress receipts (0.6B–4B) |
| this record | receipts of 9B and 27B, the record, state, `ops/tables.py`, `ops/fetch_receipts.sh` |
