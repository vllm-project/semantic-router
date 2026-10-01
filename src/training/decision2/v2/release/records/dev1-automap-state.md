# Decision 1.0 auto_map / trust_remote_code — worker state (keep current; newest first)

Assignment: coordinator note 2026-10-01 11:25 (user request 11:10: add HF `auto_map` / `trust_remote_code`
to every Decision 1.0 model, uniform with Decision 2.0). Worker `1.0-automap` (2a4d413a).
Branch `xunzhuo/decision-1-automap` (worktree `vllm-sr-dev2-automap1`, from `origin/xunzhuo/decision-2-training`
`9ff8ae938`). Gist file `07-decision-2-release.md` (release engineering; update in place).
GPU: lease-polite sharing on nodes C–F only; never node C GPU0, node E GPU4–5, node F GPU0–1.

## Now

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
