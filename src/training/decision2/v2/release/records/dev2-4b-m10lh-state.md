# DEV2.0-4B successor `m10-4b-LH` release — state (keep current; newest first)

Assignment: user request 2026-10-01 (worker b5f60b33, after the BF16-resident rollout): package `m10-4b-LH`
(release format, BF16 storage, BF16-resident runtime), parity 0 changes, bench; successor item 8 (C1 post-key,
custodial, one attempt); final decision items 1–8; coordinate with the 2.0 auto_map worker (never concurrent
DEV2.0-4B pushes; carry its remote code; re-run its Hub trust_remote_code smoke); card; publish (private), purge
superseded 4B weights; gist 07; merge; then a private Index run (IX1 harness, node C free GPUs, never GPU0; results
private only). GPUs: node A GPU0–1 (release / C1). Worktree `/home/xunliu/code/vllm-sr-dev2-release`. Started 06:37Z.

## Now

- 08:50Z — **Released** `13d4214361d0` (manifest `598cad2d`, gate seal `be7b02b4`, 14 / 14 items; Hub smoke 5.17 +
  5.18 pass). Purge of `3785b7b9`'s 6 weight blobs done (9.70 GB; DEV2.0-4B 9.72 GB; headroom 47.53 GB). C1 backup
  eval-artifacts `18d45a76`; registry 4B = LH post-key run. Record `dev2-4b-m10lh-2026-10-01.md`. Next: merge, gist
  07. Private Index: node C GPU6 / GPU7 (released by eval-ix1 08:18Z; their released markers archived as
  `owner.prev-*`), parity gate PASS (86 / 86, max |Δp| 0.0); full run 7 shards via private `lh-run.sh` (GPU6 shards
  0, 2, 4, 6; GPU7 1, 3, 5) from 08:40Z, ETA ≈ 10:00Z; then `score.sh --model DEV2.0-4B-LH --size 4B` (private only).
- 08:25Z — auto_map 4B published `3785b7b9` (gate seal `1f1872e8`, copied to `dev2-4b-m10lh-2026-10-01/current/`);
  IX1 token-budget fix merged (integration `0ad342663`). LH rebuilt on top: BASE_SPEC `dev2-4b-automap.json`,
  runtime_source = automap_source = mirror `08ec0834e`; draft `.draft2` (the C1-frozen `.draft` 2998debf kept).
  New package manifest `2ee61ea0`, 40 files (only README / config.json / remote code / decision2 api.py, qwen.py
  differ from the C1 package; weights identical): parity exact on all four panels, AutoModel vs native 0 changes
  (drift 0.0). Bench (new vs 4f560ae5 download, same weights as 3785b7b9): p50 26.4 vs 29.2 ms, p95 26.9 vs 30.3,
  peak 9.3 GiB both. Transformers 5.18 site installed on node A (`transformers` tree = node E's `5c3a59d6`; full
  digest `93df9002`). Final spec / decision `4947fda6` committed (`32d9fc5e5`); **release running** node A GPU0
  (`/data/dev2/logs/dev2-4b-lh-release-20261001T081057Z.log`, release.sh checks main == 3785b7b9 first). Next:
  purge (ops/purge_superseded.py plan → apply, node copy = bf16r download), C1 backup (a20r backup_c1.py), registry
  baseline_entry, record, gist 07, merge; Index: node C GPU1–5 leased 9B M9 until ~11:00Z, GPU6–7 eval-ix1 (idle,
  leased) — poll; add an LH named entry to ix1/launch.sh (do not overwrite DEV2.0-4B outputs).
- 07:30Z — pre-release package (draft spec, no upload) manifest `7517d321`, 37 files, identity `6a555335`,
  verify_bundle ok. Parity exact vs the T = 1 derivation: typed-final 1,600 / css15 6,547 / public231 231 / mlx-diag
  2,275, 0 answer changes, max drift 1.7e-14, predictions byte-equal (`c148d77a` / `56cf468f` / `aebe679d` /
  `94124564`). Item 8: C1 v1.2 post-key 52.71 vs DEV2.0-4B 452f1332 48.38: +4.32 [+2.60, +6.00] PASS (spec
  `v2/eval/sealed/c1-postkey/dev2-4b-m10lh.json`, job `c1-postkey/4B/dev2-4b-m10lh-20261001T071945Z`, summary
  `05af1b0f`, seal `e2217e6c`, one attempt). Bench running on GPU0. auto_map 4B revision `3785b7b9` uploaded 07:27Z
  by the node E chain (post-upload checks running) — wait for its record before the LH publish; carry its
  remote code.
- 06:50Z — read M10 hand-off / results; auto_map 4B revision is queued (node E GPU6 chain 0.6B → 0.8B → 2B → 4B →
  9B; 0.6B published `08b00e07`). T = 1 policy (13:40): ship T = 1; parity reference = `formal/m10/m10-4b-LH-t1-derived`.
  IX1 long-input runtime fix not merged yet (no runtime change since `5dc962b00` except label_token `de275f9fb`).
