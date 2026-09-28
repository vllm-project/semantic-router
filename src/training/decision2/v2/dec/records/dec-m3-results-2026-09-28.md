# Decoder track Milestone 3 results (2B / 4B on data v2 full-M; 0.8B release support and follow-up), 2026-09-28/29

Development readouts (SELECT700, CAL698, typed DEV1600, CSS pilot1430) are never
release scores. JevArena v3 and public JevBench 231 are **post-key same-panel**
comparisons (node A, the eval track's frozen runner, 16,384-token packages,
persisted per-run autotune caches). Proxy P = 100·√(T·H); |ΔP| < 4 is a tie.
Intervals are 10,000-draw paired bootstraps over items/groups (no seed variance).

Records:

| Record | Commit |
| --- | --- |
| [0.8B release support](dec-m3-0p8b-release-support-2026-09-28.md) | `5c8bbc569` |
| [prereg](dec-m3-prereg-2026-09-28.md) | `5c8bbc569` |
| [amendment 1](dec-m3-amendment-1-2026-09-28.md) (E8V) | `271502d8f` |
| [amendment 2](dec-m3-amendment-2-2026-09-28.md) (own-Lux arms) | `74ce7d0d0` |
| [amendment 3](dec-m3-amendment-3-2026-09-28.md) (N4LK) | `00ffa2791` |
| [2B candidate](dec-m3-2b-candidate-2026-09-28.md) | `48965e26d` |

Code:

- `5e2d6f094`: recipe-id mixtures, sharded teacher labels, label merge, M3 spec.
- `5c8bbc569`: merge subset mode; decoder-checkpoint teachers.
- `8ea171315`: E8V spec.

Operational wrappers, verbatim: [`m3-ops/`](m3-ops/).

## Milestone 3 items

| Item | Outcome |
| --- | --- |
| 1. DEV2.0-0.8B release support | Done. One record with every identity (soup, three seeds, calibrations, scored runs, hashes), the recipe (no teacher targets), disclosures, and one new disclosure: 19 training rows that A7 v3 later quarantined. DEV2.0-0.8B was then approved and released privately by release engineering and the coordinator. |
| 2. 4B Nox full fine-tuning on v2 full-M + own-Nox trust region + A7 retention | **HOLD.** N4T soup v3 56.506 vs adopted Nox1 56.470 (+0.04 [−4.13, +4.27]). |
| 3. 2B Sol, same recipe | **QUALIFIES.** S2T soup v3 **53.437** vs the stricter Sol 1.0 16K control 45.781 (**+7.66 [+3.26, +10.81]**). |
| 4. AutoJev-27B variant (matched own-teacher control) + the coordinator's matched own-Lux control | Done at both tiers: AutoJev (J) and own-Lux (L) arms, three seeds + soup each. 4B: N4J 54.212 and N4L 58.903 (+2.43 [−4.03, +5.44]). The follow-up N4LKr (own Lux at KL 1.0; amendment 3) is **59.539** (+3.07 [−4.90, +5.00]), the best 4B result but still HOLD. 2B: S2J and S2L tie with S2T on development readouts and fail the typed-DEV Score floor. |
| 5. 0.8B E8F + v2-M follow-up | Done: E8V soup v3 48.585 (+6.04 [+2.59, +13.81] vs Eos 1.0). It is below the E8F soup (−1.65 [−3.30, +3.68]), so it is not an improvement; the E8F soup (DEV2.0-0.8B) stays. |

## Data and teachers (hash-verified, node B)

- **Mixture `m3-v2m-ret`**: `13804ac6…`, 56,198 rows / 29,249,047 tokens (Choice / Noul /
  Score 16,645 / 26,782 / 12,771). Components:
  - pk1 A0s 6,547 (the 752 rows of the two A7h shortcut families excluded);
  - the mx-v2-full-M recipe pools 41,375 (H5 12,328, H1 7,879, G2 4,325, H6 4,179,
    V1S 3,664, G6 3,465, E11 2,882, H3 1,534, G4h 1,119);
  - A7 v3 `dec10-stage4v2` retention 8,276 rows / 9.75M tokens, whole groups
    (2,545 A0s duplicates dropped).
- **Teacher files** (KL 0.5 unless stated; every row covered):
  - own Nox `2afe3048…` (T = 1.32314) / own Sol `947bc65b…` (T = 1.30036);
  - AutoJev composites `23bba20b…` / `932565cf…`: AJ-A0s `97a071af…` + AJ-M `ba52dd86…` on
    the 47,922 recipe rows, own-1.0 on the 8,276 retention rows;
  - own-Lux composites `f1df0549…` / `ea75981f…`: pk1 A0 `56627939…` + RP-v2 waves 1/2/4
    on the same 47,922 rows, own-1.0 on retention.
  - Argmax = gold on the recipe rows (Choice / Noul / Score): own Nox .748 / .769 /
    .508, own Lux .789 / .794 / .565, AutoJev .799 / .825 / .607.
- **Recipe M3F:** full fine-tuning, backbone lr 5e-6, head lr 5e-5, objective CE,
  0.5 Brier and the KL term, one epoch (≈ 730 updates of ≥ 64 rows), SELECT700
  matrix-v1, CAL698.
- **Same-limit 16K 1.0 controls** (node A, shared renderer, package temperatures):
  - Nox 1.0 **55.689**, equal to its 8K control, so the 4B comparator stays the adopted run
    **56.470**;
  - Sol 1.0 **45.781**, +0.20 [−0.54, +0.80] vs adopted 45.580, so the 2B
    comparator is the 16K control.

## Arms (development vs post-key, labelled)

Development = node B, image `dbe5f32b…`, same-image 1.0 controls from M2.
Post-key = node A, image `f83b1d10…`, 16,384 tokens.

### 4B (start Nox 1.0 `@cde2a68d`)

| Arm | Dev P seeds (s1 / s2 / s3; mean) | Dev soup P [Δ vs Nox 1.0] · T / H · C/N/S | Post-key v3 [Δ vs adopted Nox1 56.470] | Post-key T / H · C/N/S · public231 | `mlx-diag` type macro (non-English) |
| --- | --- | --- | --- | --- | --- |
| Nox 1.0 | — | 52.46 · .6663 / .4131 · 460/228/378 | 56.470 (adopted); 55.689 (16K control) | .6144 / .5190 · 552/653/178 · 173 | .7945 (.7891) |
| N4T own Nox | 57.49 / 54.38 / 58.72; 56.86 | **58.49** [+6.03; +3.26, +8.51] · .6875 / .4976 · 523/234/343 | **56.506** [+0.04; −4.13, +4.27] → HOLD | .6450 / .4950 · 569/689/174 · 173 | .7635 (.7545) |
| N4J AutoJev | 59.55 / 55.27 / 61.26; 58.70 | **60.09** [+7.63; +4.82, +10.17] · .6694 / .5394 · 500/225/346 | **54.212** [−2.26; −5.97, +2.65] → HOLD | .6072 / .4840 · 558/644/169 · 173 | .7675 (.7583) |
| N4L own Lux | 60.81 / 56.57 / 56.79; 58.05 | **58.27** [+5.81; +3.09, +8.30] · .6375 / .5327 · 460/229/331 | **58.903** [+2.43; −4.03, +5.44] → HOLD | .6462 / .5369 · 565/695/174 · 173 | .7626 (.7545) |
| **N4LKr own Lux, KL 1.0** | 57.60 / 55.61 / 57.88; 57.03 | **58.52** [+6.06; +3.32, +8.58] · .6456 / .5305 · 505/225/303 | **59.539** [+3.07; −4.90, +5.00] → HOLD | .6294 / **.5632** · 554/664/**189** · 171 | .7731 (.7650) |

Paired development contrasts (soups): N4J − N4T +1.60 [−0.01, +3.29]; N4L − N4T
−0.21 [−1.70, +1.32]. Post-key vs Decider 4B (61.882): N4L −2.98 [−11.28, +0.80];
N4LKr −2.34 [−12.20, +0.81], with H +.008 (level with the tier leader). N4LKr axis
intervals vs Nox1: H +.044 [−.095, +.076], T +.015 [−.007, +.037]. N4LKr raised 9 of
15 CSS15 tasks (`tempowic` +.080, `emotion` +.078, `flute` +.062) and lowered
`wiki_corpus` −.109, `reddit_humor` −.070 and `talklife` −.044. On typed FINAL,
Score rose +11 and exception stack +11. It loses on `mlx-diag`: .7731 vs .7945,
non-English Noul .723 vs .800, ko .63 vs .69.

### 2B (start Sol 1.0 `@ce0c018a`)

| Arm | Dev P seeds (s1 / s2 / s3; mean) | Dev soup P [Δ vs Sol 1.0] · T / H · C/N/S | Rule outcome | Post-key v3 |
| --- | --- | --- | --- | --- |
| Sol 1.0 | — | 43.27 · .5869 / .3191 · 410/218/311 | — | 45.580 (adopted); **45.781** (16K control) |
| **S2T own Sol** | 46.17 / 42.61 / 48.14; 45.64 | **47.14** [+3.87; −0.04, +6.50] · .6100 / .3643 · 489/230/257 | finalist | **53.437** [**+7.66; +3.26, +10.81** vs 16K; +7.86 vs adopted; +3.94 [−2.48, +5.67] vs Decider 2B] → **QUALIFIES** |
| S2J AutoJev | 46.79 / 43.02 / 48.69; 46.16 | 47.00 · .6106 / .3617 · 542/240/**195** | Score floor 233 failed → not a finalist | — |
| S2L own Lux | 47.55 / 43.60 / 48.08; 46.41 | 47.20 · .6094 / .3656 · 522/239/**214** | Score floor 233 failed → not a finalist | — |

Paired development contrasts (soups): S2J − S2T −0.14 [−1.84, +1.63]; S2L − S2T
+0.06 [−1.53, +1.73]. S2T post-key: T .5434 vs .4253, H .5255 vs .4928,
Choice / Noul / Score 445 / 567 / 175 vs 374 / 438 / 155, public231 171 vs 160,
`mlx-diag` .7085 vs .7105.

### 0.8B follow-up E8V (start Eos 1.0; amendment 1)

Mixture `60f1841b…`: E8F's data with A7 v3 and pk1 A0s, plus the v2-M pools
(200,469 rows / 152.96M tokens). Recipe: E8F.

| Arm | Dev P seeds (s1 / s2 / s3; mean) | Dev soup P · C/N/S | Post-key v3 | Post-key T / H · C/N/S · public231 | `mlx-diag` type macro (non-English Noul) |
| --- | --- | --- | --- | --- | --- |
| Eos 1.0 | — | 30.58 · 510/198/85 | 42.547 (adopted); 42.361 (16K) | .3925 / .4612 · 315/410/120 · 142 | .6648 (.592) |
| E8F soup (DEV2.0-0.8B) | 37.72 / 39.44 / 33.96; 37.04 | 40.73 · 610/212/159 | 50.236 | .5734 / .4401 · 529/611/107 · 156 | .6521 (.542) |
| E8V soup | 35.45 / 42.61 / 40.33; 39.46 | **43.01** · 657/233/155 (finalist) | **48.585**: +6.04 [+2.59, +13.81] vs Eos 1.0; **−1.65 [−3.30, +3.68] vs the E8F soup** | .5641 / .4185 · 569/518/**120** · 160 | .6606 (.542) |

Reading (amendment 1): (a) the lower bound vs Eos 1.0 is > 0 ✓; (b) v3 ≥ the E8F
soup ✗. So E8V is not an improvement candidate. It recovers the typed Score loss
(120, back to Eos 1.0; resource ledger +13) and slightly improves `mlx-diag`
(+0.85 type macro). It does not recover multilingual Noul (.542 = E8F), and it
loses evidence join (−50) and CSS15 H (−.022). The first `mlx-diag` attempt was
refused by the runner (GPU5 not idle at 1% VRAM) and re-collected once.

## Findings

1. **At 2B, full fine-tuning on data v2 with an own-1.0 trust region is the
   first recipe to move Sol on v3 (+7.7):** every typed family and 7 of 15 CSS15
   tasks improved.
2. **At 4B, the three-task CSS pilot does not rank v2-trained arms.** Each v2-M
   soup raised the pilot's `discourse` (+.08 to +.13) and `semeval_stance` (+.11
   to +.15); development H is the median of the three tasks, so it is essentially
   `discourse`. On CSS15, human transfer moved −.024 (own Nox), −.035 (AutoJev)
   and +.018 (own Lux). The same split appeared at 0.6B (coordinator note 21:30).
   Research & data should check these two pilot tasks against data v2 at dataset level.
3. **The teacher matters at 4B, not at 2B.** At 2B all three teachers tie on the
   development proxy. Only own Sol kept typed-DEV Score above the floor; the
   AutoJev and Lux targets pushed three-level Score toward level 0. At 4B only Lux
   raised CSS15 H, and AutoJev lowered it most despite the best agreement with
   gold on the training rows.
4. **Data v2 lowers 4B multilingual ability on `mlx-diag`** (type macro −2.7 to −3.2;
   non-English Noul −6 to −10) with every teacher, even though v2-M holds 11,456
   multilingual Noul rows. At 2B `mlx-diag` is level.
5. **A stronger own-Lux trust region helps 4B transfer.** Raising KL from 0.5 to
   1.0 (N4LKr vs N4L) moved v3 58.90 → 59.54 and CSS15 H .537 → .563; typed Score
   also rose (+15). The two own-Lux soups are the only 4B recipes where both T
   and H exceed Nox 1.0.
6. **The 4B paired v3 interval is about ±5**, and wider on the low side because H is a
   median over 15 tasks. A 4B candidate needs a v3 gain of about +5 (≈ 61.5,
   Decider 4B's level) to clear a lower bound > 0. N4LKr's +3.07 is not enough.
7. **Adding v2-M to E8F's data does not fix the 0.8B multilingual-Noul loss**,
   and it costs typed evidence join. Typed Score does recover.

## GPU-hours (wall-clock × GPUs from every job receipt)

| Use | GPU-h |
| --- | ---: |
| Teacher labels (own Nox 3 shards, own Sol 2 shards) | 0.35 |
| 4B arms N4T / N4J / N4L / N4LKr (3 seeds each, incl. preflights and postruns) | 2.80 / 2.82 / 2.83 / 2.90 |
| 2B arms S2T / S2J / S2L | 1.39 / 1.39 / 1.39 |
| 0.8B E8V (3 seeds) | 3.98 |
| Soups, CAL698 fits, soup readouts, 16K staging fits (excl. N4LKr's, counted above) | 0.47 |
| Formal post-key collections, node A GPU5 (16K controls ×2, formal ×6, `mlx-diag` ×7, smokes) | 1.20 |
| **Total** (node B GPU0–4 20.31; node A GPU5 1.20) | **21.51** |

The aborted N4LK zero-steps (about 0.05 GPU-h) left no receipt and are not counted.
CPU builds and merges are excluded.

## Failures and operational incidents (recorded; none changes a result)

- **N4LK duplicate launch (aborted before any training step):** the launch command
  was executed twice, 26 s apart. The duplicate containers collided on their names,
  and their error receipts made the real zero-step jobs fail at receipt writing. All
  N4LK artifacts moved to `arms/aborted-20260928T1700Z-n4lk-duplicate-launch/`. The arm
  was relaunched once as **N4LKr** (same configuration) under a `mkdir` launch lock.
  Every launcher after that is lock-guarded.
- **Record times:** the prereg says ≈21:50 and amendment 1 ≈22:00. The commit times
  are 21:36:13 and 21:48:15 UTC+8. The M3 arms launched at 21:39:48, and E8V's
  first job ran after amendment 2. The order is as required; only the approximate
  labels were wrong.
- **E8V re-queue:** E8V's waiters were stopped at 22:17 before any E8V job. They
  were re-queued behind the own-Lux arms (amendment 2), because the coordinator's
  21:35 note requires a matched Lux control.
- **Repository hooks:** this worktree had a stale local `.venv-agent` directory, and
  the hooks now expect the shared venv of the primary checkout. The directory was
  moved out and replaced with the standard symlink; requirement stamps matched, so
  nothing was installed. A first integration push was rejected (non-fast-forward)
  and redone after merging.
- **A `pkill -f` pattern matched its own remote shell** and dropped that session.
  No job was affected.
- **Hugging Face private storage limit reached** (18:08 UTC): the N4LKr staging
  upload was rejected ("Private repository storage limit reached"). I freed private
  storage in my own staging repo only, with `permanently_delete_lfs_files`. That
  deletes LFS objects; git history and every commit SHA stay valid. Every node B
  copy was re-hashed first.
  - Deleted: the HOLD / non-improvement artifacts m3/N4T-soup, m3/N4J-soup,
    m3/N4L-soup, m3/E8V-soup, m2/B8F-s1 and m2/B8F-s2, 59.6 GB in total.
  - Receipts: `m3/hf-staging/free-lfs-{1,2}.receipt.json` on node B.
  - Kept in staging: m3/S2T-soup (2B candidate), m3/N4LKr-soup (4B borderline), and
    the m2 E8F soup and seeds (provenance of the released DEV2.0-0.8B).
  - `dev2-dec-staging` went from 96.5 GB at its peak to 36.7 GB.
  - The N4LKr upload was then retried once and succeeded (`784a894f`).

## Items for the coordinator

1. **2B release candidate: S2T soup** — private `llm-semantic-router/dev2-dec-staging@545a6784`
   `m3/S2T-soup/`, profile `qwen-full`, 16,384 tokens, scored run
   `/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA`. Details and disclosures are in
   the [candidate record](dec-m3-2b-candidate-2026-09-28.md). The teacher is own Sol
   1.0 (clean). Needs: release engineering, and C1 event 2 (with DEV2.0-0.6B).
2. **4B HOLD, with one borderline case for your judgment: N4LKr soup.**
   - Result: v3 59.539, +3.07 [−4.90, +5.00] vs adopted Nox1.
   - Gains: T +.015, H +.044 (level with Decider 4B), typed Score +11.
   - Losses: public231 171 vs 173; `mlx-diag` .773 vs .795 (non-English Noul −7.7).
   - It is staged at `llm-semantic-router/dev2-dec-staging@784a894f`
     `m3/N4LKr-soup/` (profile `qwen-full`, 16K), scored run
     `/data/dev2/runs/dec/formal/m3/m3-N4LKr-soup-nodeA`.
   - My recommendation is HOLD: the preregistered rule needs a lower bound > 0, and
     there is a multilingual regression.
3. **Hugging Face private storage is at its plan limit** (org-level). I freed 59.6 GB
   from my own staging repo without rewriting history. Release engineering's 2B
   package and every other track's upload draw on the same quota, so the program
   may need a quota decision or cleanup elsewhere.
4. **Cross-track:** the CSS pilot's `discourse` / `semeval_stance` inflation from data v2
   (finding 2), and the 4B `mlx-diag` loss from v2-M (finding 4).
5. **GPU:** node A GPU5 served the M3 formal runs once its lease returned to the
   decoder track (21:58 UTC+8 onward). Node B GPU0–4 and node A GPU5 are idle.

## Next step (proposed Milestone 4)

- **2B:** release support for the S2T soup: release engineering on one node A GPU,
  C1 event 2, and a card with the disclosures above.
- **4B:** push the own-Lux trust-region line that N4LKr opened, aiming at the
  ≈ +5 v3 needed for a lower bound > 0.
  - Own-Lux targets on the retention rows too (Lux-label the 8,276 A7 rows,
    ≈ 0.2 GPU-h), and KL 1.0–2.0.
  - Add the data-v2 L recipe (Lux waves cover it), 3 seeds + soup.
  - To stop the `mlx-diag` loss, add multilingual Choice/Score retention and A7
    human multilingual Score (A7q/A7k/A7s).
  - Select finalists on typed DEV plus CSS15-free signals; the CSS pilot does not
    rank v2-trained arms at 4B.
- **0.8B:** keep the released E8F soup. A later try at multilingual Noul should
  target the `mlx-diag` Noul families' source types directly (the PAWS-X-style
  paraphrase Noul that H5's answerability Noul does not cover), with an own-Eos-2.0
  (E8F soup) trust region.
