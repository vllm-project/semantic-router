# DEV2.0-4B successor `m10-4b-LH` release — state (keep current; newest first)

Assignment: user request 2026-10-01 (worker b5f60b33, after the BF16-resident rollout): package `m10-4b-LH`
(release format, BF16 storage, BF16-resident runtime), parity 0 changes, bench; successor item 8 (C1 post-key,
custodial, one attempt); final decision items 1–8; coordinate with the 2.0 auto_map worker (never concurrent
DEV2.0-4B pushes; carry its remote code; re-run its Hub trust_remote_code smoke); card; publish (private), purge
superseded 4B weights; gist 07; merge; then a private Index run (IX1 harness, node C free GPUs, never GPU0; results
private only). GPUs: node A GPU0–1 (release / C1). Worktree `/home/xunliu/code/vllm-sr-dev2-release`. Started 06:37Z.

## Now

- 06:50Z — read M10 hand-off / results; auto_map 4B revision is queued (node E GPU6 chain 0.6B → 0.8B → 2B → 4B →
  9B; 0.6B published `08b00e07`). T = 1 policy (13:40): ship T = 1; parity reference = `formal/m10/m10-4b-LH-t1-derived`.
  IX1 long-input runtime fix not merged yet (no runtime change since `5dc962b00` except label_token `de275f9fb`).
