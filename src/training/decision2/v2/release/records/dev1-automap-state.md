# Decision 1.0 auto_map / trust_remote_code — worker state (keep current; newest first)

Assignment: coordinator note 2026-10-01 11:25 (user request 11:10: add HF `auto_map` / `trust_remote_code`
to every Decision 1.0 model, uniform with Decision 2.0). Worker `1.0-automap` (2a4d413a).
Branch `xunzhuo/decision-1-automap` (worktree `vllm-sr-dev2-automap1`, from `origin/xunzhuo/decision-2-training`
`9ff8ae938`). Gist file `07-decision-2-release.md` (release engineering; update in place).
GPU: lease-polite sharing on nodes C–F only; never node C GPU0, node E GPU4–5, node F GPU0–1.

## Now

- 2026-10-01 14:30 UTC+8 (06:30Z; earlier entries' clock labels ran up to ~30 min ahead) — **All seven 1.0 repos
  live; DEV2.0-Route PR being verified.**
  - **Eos:** native Eos (`3c2d6326` code via `inference.run`) and the remote code on node D GPU1 with the same
    pinned FLA l2norm configs: bit-identical on all 8,378 prompts (0 changes, drift 0.0). Native Eos itself vs the
    stored set: 46 (pinned) / 56 (unpinned) changes; unpinned vs pinned native: 30 typed-final changes. So the
    stored set is not reproducible; the gate is met against the native runtime. Eos #3 merged → `bbdc2221`;
    readback and fresh-cache smoke (card block) pass.
  - **DEV2.0-Route-0.6B:** the 2.0 worker published its remote code to DEV2.0-0.6B (`25669e2d`). Staged the same
    three files, `config.json` fields, manifest `remote_code`, and the 2.0 `api.py` prompt fix applied as one change
    (`stage_dev2route1.py`; work `pub-ef74a012/DEV2.0-Route-0.6B`). CPU checks running (`verify/`): native original
    vs native staged vs AutoModel staged, 120 prompts. PR only after they pass; never merged by us.
  - Kai CPU parity (`p2-3684c2d8/kai-cpu`) still running (~6 cores effective).

- 2026-10-01 14:15 UTC+8 (06:15Z) — **Record, gist 07 and integration done; waiting for a GPU for Eos.**
  - Record [`dev1-automap-2026-10-01.md`](dev1-automap-2026-10-01.md); gist `07-decision-2-release.md` entry
    (≈14:05); `xunzhuo/decision-2-training` fast-forwarded to `ffea82d96` (privacy check clean over 105 items).
  - Fresh-cache smoke (clean venv, Transformers 5.18.0, CPU, empty cache, card block verbatim): all six pass.
  - Eos facts (image): the GatedDeltaNet forward matches the native Eos pinned hash (so its native convolution was
    installed in the scored run) and `causal_conv1d` is installed (the small-batch reference). Remaining suspect:
    FLA l2norm autotuning, unpinned in Eos's native runtime. Watcher on node D
    (`/data/dev2/runs/release/dev1-automap/eos-d/investigate.sh`, polls every 60 s, takes / releases its own
    leases): native Eos pinned → remote code pinned (vs that) → native Eos unpinned. No lease on C–F free so far.
  - DEV2.0-Route-0.6B: no DEV2.0 repo has the 2.0 remote code on the Hub yet (2.0 worker still verifying).
  - Kai CPU parity (`p2-3684c2d8/kai-cpu`, 48 threads) still running.
- 2026-10-01 13:40 UTC+8 (05:40Z) — **Six repos merged and read back; Eos open (needs a GPU).**
  - Merged own PRs (all checks passed, no other open PRs): Kai #1 → `69aef406`, Lex #1 → `1f9750a7`, Route #3 →
    `a5b21dff`, Sol #2 → `fc210c8f`, Nox #2 → `f098bdec`, Lux #2 → `a31b9e2c`. Readback: every staged file by
    SHA-256, every other file (weights included) by LFS SHA-256 / blob ID: all pass. Work root
    `/data/dev2/runs/release/dev1-automap/pub-ef74a012`.
  - Checks before merging: full-panel GPU parity (above), CPU equivalence of the parity-tested vs uploaded code
    (`equiv-f6b8bfdf`: identical responses; only Eos-gated code and the Hub helper differ), PR-revision smoke in a
    clean venv (Transformers 5.18.0, torch 2.14.1 CPU): AutoConfig / tokenizer / AutoModel / pipeline / errors OK.
  - Fresh-cache smoke with the card's own block running (`smoke-main.log`, `smoke-main-lux.log`).
  - Lux with the published FLA l2norm profile is bit-identical (drift 0.0 everywhere): the plain-path CSS15 drift of
    Lux / Nox is FLA autotuning.
  - **Eos:** eos3 (with its native convolution) 71 changes, i.e. worse than eos2 (47). Files and model code are
    identical to Sol's; the native Eos runtime never pinned FLA's l2norm configs, so its scored predictions carry
    that run's autotune picks. Planned: native Eos (`3c2d6326` code) and the remote code on the same GPU with the same
    pinned configs; native Eos unpinned vs the stored set. Node E GPU6–7 are now the 2.0 worker's; no lease on C–F
    is free (polling). Eos PR not opened.
- 2026-10-01 12:55 UTC+8 (04:55Z) — **Parity: Kai, Lex, Route, Sol, Nox, Lux pass; Eos needs its native convolution
  (rerun queued).** Node E GPU7, work root `/data/dev2/runs/release/dev1-automap/p2-3684c2d8`.
  - 0 answer changes / 0 missing on all 8,378 scored prompts: Kai and Lex (max drift 2.4e-7 / 2.6e-7, Score
    expected value only; CSS15 bit-identical), Route vs its native Kai-runtime reference (`reference-route`, same GPU;
    2.4e-7), Sol (bit-identical, drift 0.0), Nox (0.0105, CSS15 only), Lux (0.0103, CSS15 only). Nox: only 16 CSS15
    prompts differ at all, all 6.1k–8.1k tokens; Lux rerun with the published FLA l2norm profile running.
  - **Eos failed on the plain path:** 47 answer changes (14 / 32 / 1), drift 0.031. Cause: the native Eos runtime
    (`3c2d6326`) swaps in its own ROCm Triton causal conv + SiLU (FP64 accumulation) for batch × length ≥ 2048 on
    gfx942. Vendored as `decision1_rocm_conv.py` (Eos profile only, same conditions, Triton optional), commit
    `f1fddc759`; rerun `eos3` queued.
  - The 2.0 worker pushed `origin/xunzhuo/decision-2-automap`; its API.md is the draft I aligned with. Mirrored its
    `HF_HUB_OFFLINE` fix. 1.0 section kept in `automap/API-decision1.md` until their API.md is integrated.
  - Transformers 4.57.6 (`/data/dev2/tools/envs/tf4576`) and a clean venv with Transformers 5.18.0 / torch 2.14.1 CPU
    (`/data/dev2/tools/envs/fresh-cpu`) both load the staged Lex repo; pipeline, forward, error answers, refusals OK.
- 2026-10-01 12:25 UTC+8 (04:25Z) — **Remote code aligned with the 2.0 draft API; Kai and Lex pass parity; decoder runs
  queued on node E GPU7.**
  - Code (`v2/release/automap/decision1/`, commit `3684c2d8b`): `configuration_decision1.py`, `modeling_decision1.py`,
    `pipeline_decision1.py` plus `decision1_system_one.py`, `decision1_vela.py` (vendored ModernBERT from Transformers
    4.57.6, YaRN RoPE, three paths, type-sorted B8, published flags: no MHA fast path, no TF32, ROCm contiguous QKV),
    `decision1_qwen.py` (Qwen3.5 backbone via Transformers, FP32 candidate head, segment-exact prompts, B8 padded to
    32, temperature softmax; CPU rebinds gated-delta layers to the PyTorch references like 2.0).
    `config.json` gains `model_type: decision1`, `architectures`, `auto_map`, `custom_pipelines.decision`.
  - Aligned with the 2.0 worker's draft API (their worktree, uncommitted `automap/API.md`): file and class names per
    family, `system_one` / `forward`, error answers (`invalid_question`, `max_length_exceeded`), device moves reload,
    casts / `train()` / `save_pretrained()` refused, `threads`, no `choice` / `noul` / `score` helpers (2.0 has none).
    1.0 keeps the native request-level admission: if any question is over the limit, every question gets
    `max_length_exceeded` (2.0 Qwen answers per question) — divergence to report.
  - **Parity (node E GPU7 / GPU6, Transformers 5.17 image, draft with the fast path on):** Kai and Lex 0 answer
    changes, 0 missing on all 8,378 scored prompts (typed-final 1,600, CSS15 6,547, public 231); over-length rejections
    agree on 404 + 44 prompts; token counts equal on every prompt; max drift 4.5e-7 (Kai), 6.6e-7 (Lex).
  - Running (`/data/dev2/runs/release/dev1-automap/p2-3684c2d8`, mirror `3684c2d8b`): GPU7 chain Lex → Sol → Lux →
    Kai → Eos → Nox with the final flags; Kai on CPU (48 threads). Node E GPU6 is now leased by the 2.0 worker; every
    other GPU on nodes C–F is leased (IX1, M10).
  - Compatibility finding (unchanged): the vLLM-SR runtime `58cd660b5` `parse_decision_config` accepts only the
    Decision keys in root `config.json`, so a future catalog bump to the new revisions needs that parser to ignore the
    four Transformers keys. Deployed catalogs pin older revisions.
- 2026-10-01 11:30 UTC+8 — Survey done (heads, references, stored predictions); see git history.

## Next

1. Finish decoder parity; Route reference = the native Kai runtime (`7185f514` code) with Route's weights, then parity.
2. Transformers 4.57.6 check (`/data/dev2/tools/envs/tf4576` on node E) and a fresh-venv CPU smoke test.
3. HF PRs (based on current heads), merge own 1.0 PRs when checks pass; DEV2.0-Route PR after the 2.0 code lands.
4. Hub smoke + hash readback; record, gist 07, API.md 1.0 section, merge into `xunzhuo/decision-2-training`.
