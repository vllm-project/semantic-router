# 0.6B M6 formal runs: comparability facts (2026-09-29)

Checked on node A with CPU-only containers (no GPU devices, no network) before any M6 formal run.
Rule applied: the 23:40 comparability rule (candidate and comparator on the same kernel-equipped
image with the same frozen, persisted autotune cache).

## Images

| Image | Id | Created (UTC) | FLA | causal-conv1d | Triton / torch / Transformers |
| --- | --- | --- | --- | --- | --- |
| `decision20-train-fast:host2` | `sha256:f83b1d10f14d…2d54` | 2026-09-26 12:45 | 0.5.2 (`/opt/decision-fla`) | 1.7.0 + `causal_conv1d_cuda` extension | 3.7.1 / 2.12.0+git6bbd260 / 5.17.0 |
| untagged | `sha256:dbe5f32b2263…40b1` | 2026-09-26 08:09 | 0.5.2 (`/opt/decision-fla`) | 1.7.0 + extension | 3.7.1 / 2.12.0+git6bbd260 / 5.17.0 |

- Both images are on node A. `dbe5f32b` is present untagged (no repo tag or digest on node A).
- Their package trees are identical: `/opt/decision-fla` (479 files), `causal_conv1d` (4 files) and its
  extension `.so`, `triton` (353 files), `torch/lib` (71 files) and `transformers` (2,682 files)
  give the same per-file SHA-256 trees in both images. So `f83b1d10` is an attested equivalent of
  `dbe5f32b` with FLA + causal-conv1d.
- `run_same_panel.sh` defaults to `decision20-train-fast:host2` (`f83b1d10`), and so does
  `run_container.sh`. All M4/M5 0.6B formal runs and the adopted 0.6B comparators recorded
  `f83b1d10` in `COLLECT.json` (GLiNER2.5-Decide uses its own image `6a0c206a`, as recorded by the
  eval track). `m6_formal.sh` keeps the runner default.
- FLA is importable only with `/opt/decision-fla` on `sys.path`, which `same_panel collect` adds by
  default (`--pythonpath /opt/decision-fla`).

## Frozen autotune cache

- Chosen: the eval track's frozen node-A cache `/data/dev2/runs/eval/m1/lux1-triton-cache-frozen`
  (the Lux1 D1 cache, read-only, used by the Lux1 comparator run `m1/d1-lux1-autotune-cache`, the
  node-B cross-check and the 9B track). Tree SHA-256
  `e215f8bd5145181404bae7c502c084033d85c2e767ee4651942e870dcc72a94f` (equal to the eval records and
  `lux1-triton-cache-frozen.tree.sha256`), 303 files (12,746,118 bytes), six FLA autotune keys.
- Snapshot: `/data/dev2/runs/06b/m6/triton-cache-frozen/` (read-only, `chmod -R a-w`), manifest of
  every file and the tree hash in `/data/dev2/runs/06b/m6/triton-cache-frozen.MANIFEST.json`
  (SHA-256 `8355004d06c4f5f7e5b1f8f32c84c0591210c8269f52d016a06374dc960007e4`, read-only).
  Tree hash of the snapshot: `e215f8bd…a94f` (identical to the source).
- Tree rule (the eval track's): SHA-256 of the `sha256sum` listing sorted by path, i.e.
  `find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum` (`v2.06b.m6_cache tree`).
- Every M6 formal collection (main and `mlx-diag`) gets a fresh writable copy
  `<run-dir>.triton-cache`, passed with `--env TRITON_CACHE_AUTOTUNING=1
  --env TRITON_CACHE_DIR=<copy> --mount-rw <copy>`; `<run-dir>/M6-CACHE.json` records the snapshot
  tree before, the copy's tree after and every added, removed or changed file.

## Does the 0.6B inference path autotune?

No, as far as code and a CPU probe can show:

- The adapter `dev2-06b-causal-8k` runs `training.model.infer`, which loads the package through
  `DecisionModel.from_checkpoint`. For `backbone_model_type: qwen3` that is Transformers' `Qwen3Model`
  with `attn_implementation="sdpa"`: plain attention and MLP, no gated-delta or conv layers, no
  `torch.compile`, and no `triton.autotune` kernels in `training/model`.
- CPU probe inside `f83b1d10` with the collector's `PYTHONPATH` (including `/opt/decision-fla`),
  loading the `m5-z-soup` package and running a backbone forward: `fla` and `causal_conv1d` are never
  imported. `triton` is imported once as a module, by `torch._dynamo` (via `torch.utils._triton`)
  during weight loading, and no Triton cache file is written.
- On the GPU, SDPA dispatches to precompiled ROCm kernels, not JIT Triton autotuning. The per-run
  `M6-CACHE.json` confirms this for each formal run: an unchanged tree (no added autotune entry)
  means no autotuning took place.
- So the frozen cache is a no-op for our model and for the adopted non-FLA 0.6B comparators (the eval
  track's C1-event-2 record says the same for Kai1, Lex, Bosun and GLiNER2.5-Decide). It is passed
  anyway so the formal runs meet the rule as written. The M4 `m4-t-a7-soup` run had no persisted
  cache; the control run `m6-control-released` checks, answer by answer, that this changes nothing.
