# 2B release candidate meets the first-release threshold: S2T seed soup (Sol 1.0 + v2 full-M, own-Sol trust region)

> **Erratum (2026-09-29, per coordinator 05:25 note).** The staging revision cited below (`545a6784`) is no longer in the
> `dev2-dec-staging` history, because LFS cleanups rewrote it. The S2T weights are no longer on the staging repo, on
> purpose. S2T identity = hash list node B `m3/hf-staging/S2T-soup.sha256`
> (`22e7fe86fba8b925d4930e2691c046a10235dc30ae45b9933560d8e81c9a47dd`, 11 files) plus the verified node copies. It is
> released as `llm-semantic-router/DEV2.0-2B@5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38`. See
> [dec-staging-citation-errata-2026-09-29.md](dec-staging-citation-errata-2026-09-29.md).

Status: **qualifies under the coordinator's 2B threshold and the M3 release reading**
(prereg `5c8bbc569` rule 4). Written 2026-09-28 ≈23:00 UTC+8 right after scoring.
The recommendation between 2B artifacts is final once the matched AutoJev (S2J)
and own-Lux (S2L) artifacts are read (amendment 2 rule, fixed before any 2B
formal result). All numbers are **post-key same-panel** (node A, frozen runner)
unless labelled development.

## Result (node A GPU5, image `f83b1d10…`, 16,384-token package, CAL698)

| | S2T soup | Sol 1.0 same-limit 16K (**comparator**) | Sol 1.0 adopted | Decider 2B (best 2B peer) |
| --- | ---: | ---: | ---: | ---: |
| **JevArena v3** | **53.437** | 45.781 | 45.580 | 49.499 |
| Paired Δ (95% CI) | — | **+7.66 [+3.26, +10.81]** | +7.86 [+3.36, +10.80] | +3.94 [−2.48, +5.67] |
| T (typed FINAL) | .5434 | .4253 | — | — |
| H (CSS15 median macro-F1) | .5255 | .4928 | — | (H Δ +.105) |
| Choice / Noul / Score | 445 / 567 / 175 | 374 / 438 / 155 | — | — |
| Public JevBench 231 (easy/standard/hard) | **171** (48/66/57) | 160 (48/66/46) | 161 | 175 |
| Typed Brier / ECE | .300 / .174 | .345 / .249 | — | — |
| Invalid typed / CSS15 / public | 0 / 4 / 0 | 0 / 4 / 0 | — | — |

Release reading: the paired lower bound against the stricter comparator is +3.26 > 0 ✓.
It is above 90% of the best measured 2B peer (53.44 ≥ 44.5) ✓, and its human transfer
is not below Decider 2B (H +.105) ✓. No decision type collapsed ✓: Choice, Noul and
Score all rise, and typed-FINAL families are up on all four (constraint competition
88 vs 64, evidence join 635 vs 573, exception stack 289 vs 175, resource ledger 175
vs 155).

**Disclose:**

- CSS15 tasks below Sol 1.0 16K: `mrf` −.094, `wiki_corpus` −.045, `flute` −.032,
  `ibc` −.026, `wiki_politeness` −.024; `persuasion`, `talklife` and `conv_go_awry`
  are within .01 (7 of 15 tasks up; largest gains: `reddit_humor` +.107, `emotion`
  +.079, `tempowic` +.058).
- `mlx-diag` (development diagnostic, 16K): type macro .7085 vs Sol 1.0 .7105;
  English .7496 vs .7636; non-English .7016 vs .7016 (Choice .665 vs .671, Noul
  .658 vs .680, Score .781 vs .754); weakest language ko .59 vs .63.
- Teacher provenance: own Sol 1.0 soft targets only (a clean own-weight teacher).

## Artifact for release engineering

- **Weights:** private `llm-semantic-router/dev2-dec-staging` commit
  `545a6784175ce7a62319abeaec9502d775e586e2`, folder `m3/S2T-soup/checkpoint/`. [corrected 2026-09-29: revision no
  longer exists in the repo history; identity by hash list `S2T-soup.sha256` `22e7fe86…`, released as
  DEV2.0-2B@`5ad3e9a3`, see errata record.]
  It is a full Decision 2.0 checkpoint in FP32: Qwen3.5 text backbone (two safetensors
  shards) plus decision head. Loaded parameters: **1,883,930,944** (safetensors
  header count). `model_sha256` `073bd1f2fe62e39fe993f57006bab17ece50a7e6fc7c5ee72107fefddeb81da4`,
  head `33b6541bb6636677eb91a4d8e06acd4db81152b11097088840796f11d49c707a`. 11 files,
  verified on node A against node B's list.
- **Release calibration:** `m3/S2T-soup/cal698-16k/calibration.json`
  `d73e9ce464efcb7fe00def6680a0065488fc6e650c8ea42b479ddef9c04392e7` (CAL698
  `19cc1a8c…`, `frozen_checkpoint`, 16,384 tokens; temperatures Choice 0.74244 /
  Noul 0.81608 / Score 0.15391). The development CAL698 fit at 8K is
  `m3/S2T-soup/cal698-8k-dev/`, not for release.
- **Packaging:** profile **`qwen-full`**, `max_input_tokens` 16,384, adapter
  `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`; the full checkpoint
  carries every weight, `--source-path` is recorded only).
- **Scored run directory (node A):** `/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA`.
  Files and hashes:
  - `REPORT.json` `c04ca52a8107390913843bc4287bf2d30b5948e50e98c97d15910082298e4f4d`
  - `SEAL.json` `71e641886ba1f4e501c7cdf1220e4a6942b32b95508d0a1ccbeac8fb984bf9e8`
  - `PAIRED-vs-same-limit-16k.json` `499590052ef4bcaf6adaed31344ab658f8cdcd8cdb92ea7633e57175ea0428c3`
  - `PAIRED-vs-adopted-1.0.json` `8ecbebb5eb082fc357089220a101c0884fc4dc96453f1d82a7fab665a637b9c5`
  - `PAIRED-vs-decider2b.json` `69f0ae9a52cca6bc879b67b5c15161f29a6ccebafe8d5fbf1fc8018890ab768f`

  Persisted autotune cache: `m3-S2T-soup-nodeA-triton`. `mlx-diag` run:
  `m3-S2T-soup-mlx` (`mlx-diag.score.json` `b4fd0aa8…`). Comparators:
  `/data/dev2/runs/dec/formal/m3/sol1-16k` (45.781) and
  `/data/dev2/runs/eval/m1-adopt/sol1` (45.580).
- **Weight origin:** uniform FP32 average of three seeds (20260926 / 27 / 28) of
  recipe M3F from own `llm-semantic-router/Decision-1.0-Sol-2B@ce0c018a`
  (Apache-2.0; from Qwen3.5-2B, Apache-2.0). Recipe:
  - full fine-tuning, backbone lr 5e-6, head lr 5e-5, one epoch (729–741
    updates of ≥ 64 rows);
  - objective CE + 0.5 Brier + 0.5 KL to own Sol 1.0 soft targets
    (`947bc65b…`, package temperature 1.30036) on every row;
  - mixture `13804ac6…` (56,198 rows / 29.25M tokens): pk1 A0s with the two A7h
    shortcut families excluded, the data-v2 mx-v2-full-M recipe pools, and A7 v3
    `dec10-stage4v2` retention at 0.5×;
  - SELECT700 selection per seed: checkpoints 366 / 741 / 638, SELECT 575 / 613 /
    620 of 700; the soup scores 606 / 700;
  - code `5c8bbc569`, image `dbe5f32b…`, node B GPU3 / GPU4 / GPU3.

  Seed `model_sha256`: `91577451…`, `4f017a3e…`, `61eb7f69…`.
- **Development evidence (node B, same image; development only):**

  | Arm | P | Choice / Noul / Score (typed DEV) |
  | --- | ---: | --- |
  | Sol 1.0 | 43.27 | 410 / 218 / 311 |
  | S2T seeds s1 / s2 / s3 | 46.17 / 42.61 / 48.14 (mean 45.64) | — |
  | Soup | **47.14** (+3.87 [−0.04, +6.50] vs Sol 1.0) | 489 / 230 / 257 |

  Soup ≥ seed mean → the soup is the artifact (prereg rule 1). It is a finalist:
  P > 39.27 and every type is above its 75% floor.
