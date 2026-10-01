# auto_map / trust_remote_code for DEV2.0 — state (keep current; newest first)

Assignment: coordinator note 2026-10-01 11:20 UTC+8 (worker 4c0a68cd), user request 10:53: every DEV2.0 repo loads with
stock `transformers` (`AutoConfig` / `AutoTokenizer` / `AutoModel` / `pipeline("decision")`, `trust_remote_code=True`),
wrapping the vendored runtime; 0 answer changes vs native on the BF16 rollout's parity set plus mlx-diag; 27B fetches its
pinned base (PEFT auto-detection tested). Worktree `/home/xunliu/code/vllm-sr-dev2-automap`, branch
`xunzhuo/decision-2-automap`. GPUs: node E GPU6–7 only (render node per container, `dock.sh`). Started 03:07Z.

## Now

- 05:55Z — **Rollout DONE** (all six, record `587c0e490` in integration; mains 0.6B `def20a1c`, 0.8B `e13a40f8`,
  2B `56950ec5`, 4B `4f560ae5`, 9B `b4f65fa8`, 27B `4e89288d`). Integration merged (`eb359fdbc`); all six finals
  derived (`07e2c3ffe`, mirrored on node E); `automap.sh --release` refuses unless `main` = the superseded revision.
  - **Verify: all six tiers pass** — AutoModel vs native **0 of 11,053 answers changed, max drift 0.0** for 0.6B,
    0.8B, 2B, 4B (mlx-diag in its own run), 9B; 27B AutoModel parity finishing on GPU7.
  - **Transformers 5.18.0** (`--tf518`, AutoModel parity vs the 5.17 native answers): 0.6B 10,653 / 10,653
    prompts identical, drift 0.0.
  - **Publishing** (queues on node E, logs `/data/dev2/logs/automap-gpu{6,7}-publish-20261001T054050Z.log`):
    GPU6 `for t in 0.6B 0.8B 2B 4B 9B: --tf518 && --release` (0.6B release running from 05:52Z), then the GPU
    integration re-run; GPU7 after the 27B verify: 27B `--tf518 && --release`.
  - Handoff if not done by ~08:00Z: wait for both chains, `ops/fetch_receipts.sh <node-e>`, `ops/summary.py` into
    the record's Result table, card HTTP / links are in each `<key>/release/extra/`, gist 07 entry, merge into
    integration (`git merge origin/xunzhuo/decision-2-training`, tests, push HEAD:xunzhuo/decision-2-training).
- 05:20Z — **Verify runs (draft specs, no upload; native parity + AutoModel steps, prompt-by-prompt compare):**
  0.6B, 0.8B, 2B **passed**: 0 answer changes vs native on typed-final 1,600 / css15 6,547 / public231 231 /
  mlx-diag 2,275, max drift **0.0** (bit-identical); native vs sealed 0 changes (drift ≤ 9e-16; 0.6B 0.0).
  Running: GPU6 4B → 4B mlx → 9B, then the 5.18 AutoModel parity chain (0.6B → 9B, queued); GPU7 27B verify
  (from 04:56Z; the Decision 1.0 auto_map worker held GPU7 04:37–04:56Z, my watcher took it when its lease ended).
  - CPU integration (mirror `9c3c860e0`): all pass — Qwen3 full + LoRA adapter bit-identical on CPU; offline
    cache by repo ID through the hard-link view (removed after load); refusals; **PEFT pitfall shown: a root
    adapter copy makes AutoModel load `Qwen3Model` (the named base + adapter); our `adapter/` layout loads
    Decision2Model**. Qwen3.5 CPU skipped in the image (no LAPACK).
  - CPU spot checks with a standard CPU PyTorch 2.12.0 overlay (`/data/dev2/tools/cpu-torch212`): 0.8B (Qwen3.5,
    FLA on the AutoModel path → 18 layers rebound) and 0.6B: examples bit-identical, 200/200 typed-final prompts
    identical (drift 0); CPU vs the GPU-scored predictions differ (0.8B 4/200, 0.6B 3/200 decisions; drift ≤ 0.005).
  - Rollout: 0.6B `def20a1c`, 0.8B `e13a40f8`, 2B `56950ec5`, 4B `4f560ae5` released; 9B and 27B releases running
    since 04:26Z; its record not final yet. Finals derived for 0.6B / 0.8B / 2B / 4B (`d265f81f2`).
- 04:35Z — Code complete through the pipeline; draft specs/decisions for all six, finals for 0.6B / 0.8B (their
  rollout gates exist). Builder inputs relaying node A → node E (109 paths, 36 GB); all pinned mirrors created on
  node E (runtime/automap `8e808244`, vendor `2f21790b`, `33de83ce`, `f8f52c69`, `3277dec9`, `ff660322`, plus
  `95a683e5`, `c68de36a`). Next: preview builds (CPU) → native vs AutoModel full parity on GPU6/7 (5.17), then
  AutoModel parity under 5.18 (overlay `/data/dev2/tools/tf518`), then releases after the rollout's record.

## Publish gate (do not upload before all hold)

- BF16-resident rollout results record on origin (`records/dev2-bf16-resident-2026-10-01.md` final, all six) and its
  final revision for the repo is `main` (0.6B `def20a1c`, 0.8B `e13a40f8` released 04:00Z, 2B `56950ec5` and 4B `4f560ae5` 04:28Z; 9B and 27B running).
- Merge integration, `make_automap.py final`, rebuild via `release.sh --upload --collect --already-collected`
  (full parity + automap steps + Hub smoke under 5.17 and `--hub-site tf518=/data/dev2/tools/tf518`),
  `hf_headroom.sh` first, `rewrite_history=False`.

## Facts (verified)

- Remote code works end to end: 0.6B overlay (old main + remote code) on CPU: AutoConfig / AutoTokenizer (same IDs) /
  AutoModel / forward / pipeline, answers bit-identical to native; tiny-package integration on GPU6 (Qwen3, Qwen3.5
  hybrid with FLA kernels, Qwen3 LoRA adapter): native vs AutoModel bit-identical, card block passes.
- Found and fixed: (1) a root `config.json` with `auto_map` makes the vendored loader's `AutoTokenizer` (via AutoConfig,
  no trust_remote_code) print Transformers' remote-code prompt on stdout / wait up to 15 s → runtime fix
  `0cbf1033e` (`Decision2.from_pretrained` sets `TIME_OUT_REMOTE_CODE = 0` while loading; same tokenizer fallback as
  before). (2) hub 1.31 `snapshot_download` with a commit revision still lists files offline → `local_files_only`
  under HF_HUB_OFFLINE (`4d5e55021`).
- Limits found: the image's PyTorch has no CPU LAPACK, so Qwen3.5 (0.8B–27B) CPU inference cannot run in the image
  (native or AutoModel); CPU spot checks there are 0.6B (Qwen3) only; Qwen3.5 CPU needs a standard PyTorch.
- PEFT: our 27B layout keeps `adapter_config.json` under `adapter/` → `find_adapter_config_file(repo)` is None; the
  integration test shows a root copy redirects AutoModel to the named base (test pending re-run after the offline fix).
- transformers: 5.18.0 (2026-09-30) is current stable, 5.17.0 is in the images; 5.18 requires hub ≥ 1.31 (image has
  1.31.0), tokenizers 0.23.x → installable `--no-deps` as an overlay.
- Current mains before the rollout: 0.6B `476fe984`, 0.8B `bede7938`, 2B `a53cf66a`, 4B `fadbba4f`, 9B `e51f9881`,
  27B `5323310`. Profiles: 0.6B–9B `qwen-full`, 27B `qwen-adapter`.

## Commits (branch `xunzhuo/decision-2-automap`)

`6382faf46` remote code + API.md · `83e04b0dc` builder / config.json / card · `08f183855` examples automap modes ·
`343197f9d` merge of the rollout branch · `0cbf1033e` runtime prompt fix (shared module) · `9923cc505` integration
test · `4d5e55021` offline fix · `8e8082440` release.sh automap steps + one render node per GPU (shared script) + gate
item 7.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| API spec `v2/release/automap/API.md` (shared with the 1.0 worker) | worktree | done (`6382faf46`) |
| Remote code: configuration / modeling / pipeline | worktree | done |
| Builder, config.json fields, card section, examples modes, release.sh, gate item | worktree | done |
| Tests: stdlib (`test_automap`), image integration CPU + GPU | node E | GPU passed; hub-layout leg re-run |
| Preview builds + full native vs AutoModel parity (5.17) per tier | node E GPU6–7 | next |
| AutoModel parity under 5.18 | node E GPU6–7 | |
| Publish after the rollout record: final decisions, release.sh with Hub smoke (5.17 + 5.18) | node E | |
