# auto_map / trust_remote_code for DEV2.0 — state (keep current; newest first)

Assignment: coordinator note 2026-10-01 11:20 UTC+8 (worker 4c0a68cd), user request 10:53: every DEV2.0 repo loads with
stock `transformers` (`AutoConfig` / `AutoTokenizer` / `AutoModel` / `pipeline("decision")`, `trust_remote_code=True`),
wrapping the vendored runtime; 0 answer changes vs native on the BF16 rollout's parity set plus mlx-diag; 27B fetches its
pinned base (PEFT auto-detection tested). Worktree `/home/xunliu/code/vllm-sr-dev2-automap`, branch
`xunzhuo/decision-2-automap`. GPUs: node E GPU6–7 only (render node per container, `dock.sh`). Started 03:07Z.

## Now

- 05:55Z — Code complete through the pipeline; draft specs/decisions for all six, finals for 0.6B / 0.8B (their
  rollout gates exist). Builder inputs relaying node A → node E (109 paths, 36 GB); all pinned mirrors created on
  node E (runtime/automap `8e808244`, vendor `2f21790b`, `33de83ce`, `f8f52c69`, `3277dec9`, `ff660322`, plus
  `95a683e5`, `c68de36a`). Next: preview builds (CPU) → native vs AutoModel full parity on GPU6/7 (5.17), then
  AutoModel parity under 5.18 (overlay `/data/dev2/tools/tf518`), then releases after the rollout's record.

## Publish gate (do not upload before all hold)

- BF16-resident rollout results record on origin (`records/dev2-bf16-resident-2026-10-01.md` final, all six) and its
  final revision for the repo is `main` (0.6B `def20a1c`, 0.8B `e13a40f8` released at 05:55Z; 2B/4B/9B/27B pending).
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
