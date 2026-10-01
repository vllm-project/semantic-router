# DEV2.0-4B successor `m10-4b-LH` release — state (keep current; newest first)

Assignment: user request 2026-10-01 (worker b5f60b33, after the BF16-resident rollout): package `m10-4b-LH`
(release format, BF16 storage, BF16-resident runtime), parity 0 changes, bench; successor item 8 (C1 post-key,
custodial, one attempt); final decision items 1–8; coordinate with the 2.0 auto_map worker (never concurrent
DEV2.0-4B pushes; carry its remote code; re-run its Hub trust_remote_code smoke); card; publish (private), purge
superseded 4B weights; gist 07; merge; then a private Index run (IX1 harness, node C free GPUs, never GPU0; results
private only). GPUs: node A GPU0–1 (release / C1). Worktree `/home/xunliu/code/vllm-sr-dev2-release`. Started 06:37Z.

## Now

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
