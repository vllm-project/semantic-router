# Official Qwen3.5-9B short-TRAIN full development arm

## Frozen training outcome

The single prospective arm in [the full-arm preregistration](qwen35-9b-shorttrain-full-prereg-2026-09-28.md) used fresh official `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, not the earlier smoke checkpoint or a third-party Decision model. Its complete-group filtered rights-clean v2 TRAIN held 7,324 rows / 5,261 groups / 3,579,176 native tokens at SHA-256 `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`. The unchanged SELECT and CAL input hashes matched the preregistration. CAL was audited for split identity only; this run did not select on, calibrate with, or score CAL.

Training exited zero on the single pinned GPU/image after **458/458 consecutive finite optimizer updates**, with no resume or duplicate run and all eight prescribed SELECT checkpoints at steps 64, 128, 192, 256, 320, 384, 448 and 458. All eight SELECT passes returned 700/700 valid answers. The full runtime, including source load, training, SELECT and saving, was **0.836667 one-GPU-hour**, under the 4-hour cap. `COMPLETE.json`, `BEST.json`, the saved checkpoint receipt and training log have SHA-256 values `2c938aba095c5b0edcc859af94e49ef403ef42d08b8ca2e39f19c5a008adea74`, `adab3ca4692f678219ef4454a2f9df301e9da1eda124a55ca9b594e880ad8a45`, `ddbff66f8172150e89255c9e1566d892bd8e4a0f12a172d63d866fbff0bd87ea` and `4bd109bf56f6f48f9db90f138ca81bd8843e011015de3cef087051fe509e8380` respectively.

| SELECT step | Correct / 700 | Family macro accuracy | Normalized Brier |
| ---: | ---: | ---: | ---: |
| 64 | 531 | .738519 | .170597 |
| 128 | 548 | .748333 | .149438 |
| 192 | 566 | .771204 | .143795 |
| 256 | 611 | .855093 | .098408 |
| 320 | 621 | .863426 | .090585 |
| 384 | 626 | .871944 | .080555 |
| 448 | 625 | .871111 | .080662 |
| **458** | **625** | **.873426** | **.081687** |

The frozen selector correctly chose **BEST458** by family macro accuracy, despite its one fewer correct answer than step 384. Its SELECT source slices were Choice 168/200, Noul 179/200, narrative 130/130, abstention 40/40, string composition 22/40 and quantized Score 86/90. The saved SELECT prediction SHA-256 is `743b7eb03291b821c937f02e912387da6a15d1499ad7a8a3170655e194b75005`.

An independent fresh-process native reload of BEST458 passed on the fixed first 32 SELECT rows (13 Choice, 14 Noul, five Score): **zero category changes, zero p99 probability drift and zero maximum drift**. This validates saved-checkpoint fidelity only; SELECT is development evidence.

## One sealed development readout: HOLD

After the reload passed, the unchanged native adapter collected **one** gold-free typed DEV 1,600 and CSS pilot 1,430 readout at temperature 1.0 and 4,096-token cap. Typed DEV returned 1,600/1,600 valid; CSS pilot returned 1,428/1,430 valid, with two untruncated over-budget failures in discourse. Both prediction files and manifests were bound into a private pre-score seal, SHA-256 `4b618d9f083ef4892c1f8869a7b85bce62ec386c05db8728d79a38fbe0fba5f9`. Their input hashes were the same as the earlier same-panel controls, `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` and `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`; the saved model fingerprint was `a221efd24cbabcd33cdacbf1797beaae6d28d84680df32f1d3cfe563c688ca66` for both. Prediction hashes were `aa6649101e56be6c11439876d9650ab19ef5cfeb9e38376b9e471cdf63560b7c` and `e06dbaccd9e173d427918ae6cbd34db69cca33a903aa20812d848565db65a376`. Only after that seal was written were the fixed DEV and pilot gold files read by their unchanged scorers, SHA-256 `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc` and `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`. The private score reports have SHA-256 `4491ae1cfe1e5cdc6d9465a63a704f958ce4625625eb9e752e668b2d0a9ad840` and `34ffc493a44200ef113648a7ba0c09584d2428bb263e7e59bc5db7712f689246`.

| Development metric | Own Lux 1.0 frozen same-panel reference | Official Posttrained BEST458 | Change |
| --- | ---: | ---: | ---: |
| Typed DEV four-family macro `T` | .867500 | **.683125** | **−.184375** |
| Typed DEV correct / 1,600 | 1,388 | 1,093 | −295 |
| Choice / 800 | 799 | 726 | −73 |
| Noul / 400 | 258 | 193 | −65 |
| Score / 400 | 331 | 174 | −157 |
| Typed normalized Brier ↓ | .09040 | .24611 | +.15571 |
| Typed ECE10 ↓ | .04462 | .19151 | +.14689 |
| CSS pilot median task macro-F1 `H` | .57011 | .54927 | −.02084 |
| CSS pilot micro correct / 1,430 | approximately 769 from reported 53.78% | 813 | higher micro accuracy, lower task median F1 |
| Proxy `100 × sqrt(T × H)` | approximately **70.326** | **61.255** | **−9.070** |

The Lux figures are its previously frozen same-input development receipt, not newly rerun predictions. The preregistered promotion margin was Lux proxy +2.0 points, approximately `72.326`, and invalid/overbudget excess at most one percentage point. The model fails the first condition by a wide margin; no favorable alternate checkpoint was selected. Its CSS pilot task macro-F1 was discourse `.54927`, implicit hate `.41348`, and stance `.75438`, with two discourse over-budget cases. The pilot median Brier was `.62382` and ECE15 `.17699`; typed Brier and pilot Brier use different normalization and should not be directly compared. Against the pinned JPT-9B native development peer, this candidate is also far behind on typed accuracy (1,093 versus 1,436), although it has higher pilot median F1 (.54927 versus .52544); the peer comparison is descriptive, not the preregistered selector.

### Failure anatomy and next experiment

The failure is valid-output transfer, not parse loss: typed DEV had zero invalid answers. Its Choice attribute gate stayed at 400/400, but transition-table Choice was 326/400; Noul rule precedence was 193/400 and Score set reconciliation 174/400. On Noul, the candidate chose `true` for 315/400 questions despite almost balanced truth labels: of 192 gold-false cases, 157 were predicted true. On the three-level Score panel, the candidate chose only levels 0 or 2, never level 1: all 107 gold-middle cases went to level 0, and 119 of 208 gold-high cases also went to level 0. Score Brier `.47431` and ECE10 `.49369` expose severe probability error. The filtered TRAIN contained only **99 three-level Score rows** among 507 Score rows, while its 90 Score SELECT examples were all five-level. This cardinality and mechanism mismatch explains why SELECT success was insufficient evidence of typed transfer; it does not prove the initialization alone caused the failure. The 4,096-token group filter also removed long TRAIN exposure, so long-context improvement is unproven.

The next most discriminating arm is an **initialization-only contrast** from the official `Qwen/Qwen3.5-9B-Base`, whose [official repository](https://huggingface.co/Qwen/Qwen3.5-9B-Base) exists. First pin its exact revision and independently check tokenizer/native-token equivalence, zero-step decision output, source hashes and one-update numerical stability. Only if those pass, run one fresh Base-initialized arm on the **same filtered TRAIN/SELECT/CAL, head, seed, update/token budget, loss and checkpoints**; do not reuse this Posttrained checkpoint or choose among later steps using typed DEV. This isolates the promising Base-versus-Posttrained hypothesis suggested by the completed 4B contrast. If Base still fails, a separately preregistered source-disjoint rule-precedence and three-level Score data intervention is more informative than repeating the same current curriculum. Neither proposal has started.

**Disposition:** this 9B arm remains a private research HOLD. No CAL fitting, typed FINAL, 15-task human FINAL, public JevBench, Hugging Face model upload or release claim followed this failed development gate. Training used 0.836667 one-GPU-hour; the independent reload and two native development collectors used approximately another 0.086 one-GPU-hour by task-file creation and completion timestamps, about 0.923 one-GPU-hour total. The positive SELECT result and this negative development result retain their distinct meanings.
