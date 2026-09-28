# M3a — AutoJev-27B soft targets: production (2026-09-28)

After the passed [qualification v2](m3a-autojev-qualification-v2-2026-09-28.md), per [prereg v2](m3a-prereg-v2-2026-09-28.md).
All three files were produced on **node A GPU2–4** (one node, one runtime, the frozen 11-entry autotune cache
`cff772eb…`) by `v2/data/m3/produce_nodeA.sh` from the exact mirror of `e48564a47`. Node B was not used.
Reports (count-only): [`m3a/autojev27/`](m3a/autojev27/). Private dataset
`llm-semantic-router/decision-2.0-training-data`, tree `m3/teachers/autojev27/` with `PROVENANCE.md`
(teacher provenance caveat) and `qualification-v2.report.json`; readback at `780d2743` verified all files.

| Wave | Prompts / targets | Shards (GPU2 / 3 / 4) | Targets SHA-256 | Attestation SHA-256 | HF revision |
| --- | --- | --- | --- | --- | --- |
| AJ-M — M recipes (`rp-v2/aj-m.*`) | 84,523 / 84,523 | 28,098 / 28,245 / 28,180 | `ba52dd86ff7fc7c422bb097c7036c344399de549597cec60c352fef02c21038a` | `99412ea41222ff722259af153bf98b027a2a24b0bd009ad705721fae66832110` | `3a99bf1c26ac7924ec1216cdf1f703e1a5f21ae6` |
| AJ-A0s — pk1 A0s (`pk1/A0s-train.*`) | 7,299 / 7,299 | 2,425 / 2,416 / 2,458 | `97a071affe820d148f7cfe6b50aa59b04511523b68f2d5bf3a4026b7811be138` | `d846600e760c53654be19e897ed3b71e54f1262d188528f82e56851241583406` | `9bb9790b014d17d7dad50677b4f057f20e50460c` |
| AJ-SL — S and L minus M (`rp-v2/aj-sl.*`) | 37,219 / 37,219 | 12,536 / 12,319 / 12,364 | `1254cb4703a10810acdef695a8218866fe7a664f5901074e8e8dffe68486aa75` | `2bc00a8786223969e8edc347ce80f14daa1e928878c877099d7578b7465677e7` | `780d27439c5a07650a37b9a0ce3369dae3359baf` |

- **Coverage:** AJ-M + AJ-SL = every non-A0s row of every D10 recipe (121,742); AJ-A0s = every pk1 A0s row.
  No prompt went unanswered (none over AutoJev's input limit).
- **Argmax vs gold** (Choice / Noul / Score): AJ-M 0.844 / 0.827 / 0.612; AJ-A0s 0.735 / 0.838 / 0.640; AJ-SL
  0.855 / 0.833 / 0.613 (own-Lux on the same RP-v2 rows: about 0.79 / 0.79 / 0.55).
- **Checks per wave:** guard PASS (0 shared ids, input hashes or prompt digests with SELECT700, CAL700, CAL698,
  every v1 / v2 AHO and SHO slice and every gold-free eval panel; AJ-M 129 shared normalized states,
  informational); production repeat check 128 / 128 bitwise, autotune unchanged; every shard kept the frozen
  autotune digest; per-row attestation (model id, attested revision, adapter, backend, config / package / source
  hashes, prompt digest bound to the training row, shard, GPU, image).
- **Provenance:** every file carries the caveat — AutoJev's public training reportedly used
  closed-model-generated SFT data; disclose on every distilled card; each AutoJev-distilled candidate needs a
  matched own-Lux-target control; own-Lux stays the clean default.

## GPU-hours (node A)

Qualification v2 0.180; shards AJ-M 1.693, AJ-A0s 0.232, AJ-SL 0.809; production repeat checks 0.051.
**Total 2.965 GPU-h.** GPU2–4 held 12:18–13:31 UTC and released to the 9B track at 13:31 UTC.
