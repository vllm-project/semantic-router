# Decision 1.0 auto_map / trust_remote_code — worker state (keep current; newest first)

Assignment: coordinator note 2026-10-01 11:25 (user request 11:10: add HF `auto_map` / `trust_remote_code`
to every Decision 1.0 model, uniform with Decision 2.0). Worker `1.0-automap` (2a4d413a).
Branch `xunzhuo/decision-1-automap` (worktree `vllm-sr-dev2-automap1`, from `origin/xunzhuo/decision-2-training`
`9ff8ae938`). Gist file `07-decision-2-release.md` (release engineering; update in place).
GPU: lease-polite sharing on nodes C–F only (prefer node E GPU6–7 when the 2.0 auto_map worker is idle);
never node C GPU0, node E GPU4–5, node F GPU0–1.

## Now

- 2026-10-01 11:30 UTC+8 — **Survey done; designing the remote code.**
  - Hub heads (all public except DEV2.0-Route): Kai `9d6872cd`, Lex `6c5e3d48`, Route `c7a31eb0` (subin, new
    weights 03:08Z), Eos `a66df1b5`, Sol `a1c9f252`, Nox `eab48e99`, Lux `8db79130` (subin's neutral-wording PRs
    merged 03:08Z), DEV2.0-Route-0.6B `72a2d317` (private, 2.0 `qwen-full` package by subin).
    No open PRs on any repo; Eos has one open non-PR discussion (fontlab), untouched.
  - The 2.0 worker's branch `xunzhuo/decision-2-automap` is not pushed yet; drafting against the System One
    schema, will reconcile before publishing.
  - References: vLLM-SR decision runtime `58cd660b5` (branch `xunzhuo/decision-runtime`, `src/vllm-sr/decision_runtime`),
    native 1.0 runtimes of the runtime-bearing revisions (Kai `7185f514`, Lex `ee8e74d9`, Eos `3c2d6326`,
    Sol `0665a411`, Nox `0bb83350`, Lux `bd45a30a`), stored scored predictions on node A
    (`runs/eval/m1/r{1..4}-*`, `runs/eval/m1-adopt/{kai1,sol1,nox1,lux1}`).
  - **Compatibility finding:** the vLLM-SR runtime's `parse_decision_config` accepts only the eight descriptor keys
    (+ `calibration`) in root `config.json`; adding `auto_map` there makes a *future* catalog bump to the new
    revisions fail until the runtime tolerates standard HF keys. Deployed catalogs pin older revisions, so serving is
    unaffected. To be reported to the coordinator.

## Next

1. Read the native 1.0 code; write `v2/release/automap/` remote code (configuration / modeling / pipeline) and
   `API.md`; unit tests on CPU.
2. Parity: encoders on CPU (node E), decoders on a free leased GPU; 0 answer changes vs stored scored predictions.
3. HF PRs (based on current heads), merge own 1.0 PRs when checks pass; DEV2.0-Route PR only, never merged.
4. Fresh-environment Hub smoke test + hash readback; record, gist 07, merge into `xunzhuo/decision-2-training`.
