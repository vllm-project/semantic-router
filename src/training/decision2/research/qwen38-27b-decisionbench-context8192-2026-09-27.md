# Qwen3.8-27B Decision Bench v4 native context ablation

This development experiment was [preregistered in the unified research
log](https://gist.github.com/Xunzhuo/cd90fce0fa548616d8a4f1b2d2398dea#file-decision-2-qwen38-27b-context8192-2026-09-27-md)
before running the 8,192-token adapter. It is a public Choice-only diagnostic,
not a sealed JevArena score or release qualification.

The matched 4,096-token reference used selected checkpoint `0000368` of the
rights-clean Qwen3.8-27B clean-v2 research run, unchanged source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, model fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`,
and CAL700 temperature receipt
`e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78`.
The frozen public text-readable panel has 1,041 prompts, SHA-256
`41c0e4728202d800972edb31375e92e92148906856b2f20396154c8d0a9c80da`.
The 30 visual-only cases remain N/E. The reference answered 935/1,041
correctly; nine over-budget `AGT-3` cases were invalid and counted wrong.

The changed variable is only native `--max-length 8192` (4,096 reference),
with no truncation, training, prompt edit, checkpoint change or calibration
refit. All 1,041 prompts are rerun. Input, model, adapter and CAL identities
must match their sidecar receipts. Invalid and missing answers count wrong.

## Native result and paired analysis

The 8,192-token collector completed all 1,041 prompts with **zero truncation**
and **zero invalid answers**. The same frozen scorer counted **939/1,041 =
90.20% correct**, versus the 4,096-token reference's **935/1,041 = 89.82%**.
The task macro score moves 89.668% to 90.060%, and category macro 89.134%
to 89.301%. The 30 visual-only cases remain N/E in both reports.

| Paired subset | 4,096-token reference | 8,192-token run | Observation |
| --- | ---: | ---: | --- |
| Nine formerly over-budget `AGT-3` items | 0/9, all invalid | 6/9, all valid | Six repaired answers; three remain wrong |
| Other 1,032 items | 935/1,032 | 933/1,032 | Two `COM-1` answers flip correct to wrong |
| All 1,041 text-readable items | 935/1,041 | 939/1,041 | Net four additional correct |

All other 1,030 formerly valid items preserve their categorical answers: 933
remain correct and 97 remain wrong. `AGT-3` changes 17/30 to 23/30, while
`COM-1` changes 26/30 to 24/30; the other 32 tasks retain their hard-decision
counts. Both changed `COM-1` cases had narrow top-two probability margins in
at least one run (0.0025–0.0295). Their drift occurs despite unchanged model,
adapter, CAL, prompt and encoded-input contracts and is an observed numerical
reproducibility limitation, not a measured benefit or regression caused by
the admission limit. The model's encoder uses `max_length` only to reject
over-budget inputs; it does not pad or alter accepted token IDs by that value.

Valid-only normalized multiclass Brier is .14018 in the 4,096 run and .14293
in the 8,192 run; these denominators differ by nine harder cases. On the
matched 1,032 formerly valid items it moves .14018 to .14033, and log loss
.25175 to .25213. The full-run p50/p95 item latencies were 148.7/313.5 ms
and 210.3/522.9 ms respectively, with runtime variance and the same slow
reference linear-attention fallback; these are run diagnostics, not a
cross-model efficiency comparison.

The 8,192-token native sidecar confirms the same model ID/revision,
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`
model fingerprint, `4d204f9923cc1c9d29ea7385cf1268ee9c0be7d8578c8ae24d3d2c518854127c`
adapter hash, prompt SHA-256 and CAL receipt as the reference. The exact
inference source SHA-256 is
`147026f0988b7d35f1f204466fc2b453ebabf43e1414b4ab5f2ba8fffb94b163`,
the corrected scorer source is
`0864db1462a671f12a266476053a8fb8e156cecd0db0f3bc99ddbfca20a25b44`,
and the image digest is
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
New prediction, sidecar and score SHA-256 are respectively
`aaa6679efb829d74a58bb9282c220e1499b0ef5130cb45e4791b89b02b5fd6e2`,
`30b968ddb56c5b516da5565c9668bf5be78fb486832546040f349debef34122d`,
and `205be9f5fc867ba81d0f937f59f2d0fc1567d5e7007bbb7c497132175765d3b2`.

One environment preflight with both `HIP_VISIBLE_DEVICES` and
`ROCR_VISIBLE_DEVICES` set found no GPU and ended before model load. The
corrected run used one visible GPU and completed; no failed preflight produced
predictions or training updates. No source weights or checkpoint were changed.

This is an exposed-public ablation. The small net gain and two under-budget
answer flips do not establish a broadly improved 27B model. The 8,192-token
adapter is not a frozen release adapter and needs a complete same-panel rerun
and package parity if adopted. No sealed JevArena labels were opened.
