# Own-Nox 4B clean-v2 first-release screen

The preregistered one-epoch arm started from our published Decision 1.0
Nox-4B revision `0bb833504965c0eabdb9630b7bbd385cb2fe5cd4`, not a
third-party decision model. It completed all 466 planned optimizer updates.
The training provenance, completion and BEST hashes are respectively
`496eff65241c0bd0d0f8b83cf377885994c6ceb5b8c22fb1ece5fd4c0def65a1`,
`99b04fecb472a7201656c0106d22c359819aba941ea1020dc7d5a2eeceebf017`,
and `e96277b9e0b07b7ef8c8df4b2fcbb71b7bdb9e8b4b8ed58816d8381dc221e3b6`.
Elapsed training time from provenance creation to completion was 39 minutes
21 seconds, approximately 0.656 GPU-hour on one card. The original
[pre-registration](nox4b-cleanv2-lineage-prereg-2026-09-27.md) and later
[owner-directed aggregate policy](first-release-aggregate-policy-2026-09-27.md)
remain separate and unchanged.

| SELECT700 stage | Correct | Family macro | Family Brier |
| --- | ---: | ---: | ---: |
| Zero-step own Nox 1.0 | 523 | — | — |
| Step 128 | 574 | 0.80519 | 0.12977 |
| Step 256 | 595 | 0.83824 | 0.10268 |
| Step 384 | 609 | 0.85935 | 0.09231 |
| Step 466, frozen BEST | **611** | **0.86204** | **0.08891** |

CAL700 was fit only after BEST was fixed. Its report SHA-256 is
`eed55d196a085b3fd0a918686747bf042b3f39233d5a39d0c71ae7bcb2248f05`,
with Choice/Noul/Score temperatures 1.27230, 1.08770 and 0.48580.
The calibrated native checkpoint then ran one complete, same-protocol
development diagnostic on typed DEV1,600, CSS pilot1,430 and the separate
public JevBench231. All answers on these panels were valid.

| Development diagnostic | Result |
| --- | ---: |
| Typed DEV | 1,049/1,600; four-family macro 0.655625 |
| Choice / Noul / Score | 454/800 · 228/400 · 367/400 |
| CSS pilot task-median macro-F1 | 0.443807; 693/1,430 micro correct |
| JevBench public231 | 173/231; easy 48/48, standard 67/72, hard 58/111 |

The DEV and pilot composite `100 × sqrt(0.655625 × 0.443807)` is **53.94**,
below the already retained development promotion threshold **56.0**. The
Nox-1.0 public231 control also scored 173/231 on the same public panel. The
typed/CSS/public score report SHA-256 values are respectively
`0d7bdd36d7d3f0edd8098edadc1cec1c6e39bec00590cc64e0b2c43f38ce4167`,
`9a6b0d29c27d49f7d35c942ceb44511bff1e5e14ccd64d1952590ff90473dc76`,
and `9f7ae720a6d4f17dc3e23c8d6ab20662095db0664e6c0882463504f2aa586246`.

**Decision: HOLD.** Do not run this candidate on typed FINAL or CSS15, place
it in the first-release roster, reuse the old third-party-origin 4B scores,
or publish it. The SELECT rise did not produce enough cross-task transfer;
Choice is the weakest typed slice while Score remains strong. A subsequent
4B arm should test an independent mix of genuine
human-labeled task families and verifiable, more diverse Choice reasoning,
against a matched own-Nox control, before spending on formal evaluation.
