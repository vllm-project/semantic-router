# 9B Milestone 7, amendment 1: a half-dose arm Q after P's early stop (before any Q GPU job)

Amends the [preregistration](lux9b-m7-prereg-2026-09-30.md) (`3bbdd4349`). Written 2026-09-30 ~11:20 UTC+8, after P's
preregistered early rule and before any job of the new arm. P's outcome is final: it runs no further member and no
line. The C chain continues unchanged.

## What happened

- Preflights PASS for both arms (every gate). P-m1 and C-m1 each ran about 16 minutes (266 / 166 updates).
- **Early rule for P: STOP** (`m7/rules/early-P.json`). Member 1 at α 1, PN1 dev (T = 1):

  | | clean gold-no yes | hop yes | PAWS-X-6 pooled yes | accuracy | final SELECT700 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | P-m1 | **.047** | .949 | .511 | .965 | .874 |
  | C-m1 (control) | .714 | .992 | .843 | .656 | .859 |
  | R = K-a13 (reference) | .262 | .987 | .626 | .867 | — |

  Rules (a) clean gold-no −.667 ≤ −.02 and (c) SELECT +.015 passed; **(b) hop −.042 < −.03 failed**.
- The lever is far stronger than needed: at this dose the continuation learns to say "no" to nearly every near-miss
  and swap construction, and gives up 4 points of true paraphrases for it.
- R's re-read reproduces M6 (1,600 / 1,600 typed-DEV answers identical); its HT-DEV v2 on the M7 path equals the eval
  reference exactly (Δ 0.0000, H_dev2 .5636); MLX-DEV-9B for R: Noul-ML .846, Choice-ML .921, Score-ML .547.

## Amendment: arm Q (half dose), same control, same rules

- **Q** = the continuation of each K seed on **x60 replay + PN1-r2 once** (4,364 rows, 491,023 tokens), with the
  replay bisected so that Q's native tokens equal C's; every other setting as P (members, seeds 20260930 + k, lr
  5e-6 / 5e-5, own-Lux KL on replay rows, PN1 gold only, one final checkpoint, preflights on member 1). Halving the
  PN1 exposure is the smallest change that should keep the in-family effect far beyond the early rule's −.02 while
  lowering the hop cost.
- Build `m7-topup-q` (spec `m7-topup-q.json`; code `df6dcccc9`; manifest `2713d6d5…`):

  | TRAIN | Rows | Native tokens | C / N / S token share | train.jsonl | teacher.jsonl |
  | --- | ---: | ---: | --- | --- | --- |
  | **Q** | 15,808 (11,444 replay + 4,364 PN1-r2) | 6,198,155 | .286 / .500 / .214 | `82496294ccf5196aee330b05e94a2dd49522655961cf29cf57fcf790e23697d0` | `f75c150fae119055e4d855b2b742c3c6c2b1bfd24553c804482f045eec5aa3eb` |

  - Q and C differ by 47 tokens (0.0008%). The same build rebuilds C byte for byte (`eb55dbb2…`, teacher
    `babd72c5…`), so **C stays Q's matched-token control**; Q's replay groups are contained in C's (checked).
  - Exposure `exposure/Q.json` `01bb1817…`: `groups: []`, `methods_agree: true`. Every builder guard passed.
- **Early rule for Q:** exactly P's three conditions against C-m1 (already read): clean gold-no yes ≥ .02 below
  C-m1's, hop not more than .03 below, final SELECT700 not more than .02 below. A stopped Q runs nothing further.
- **Line, screens and α rule:** as preregistered for P, on the line Q5 (soup of Q-m1..Q-m5) at α ⅓, ½, ⅔.
- **Finalists:** at most three, the non-dropped picks of **P5 (empty: stopped), Q5, C5** in that priority. Slots are
  not refilled. This is the only added arm; no further arm is added in M7.
- **Chain:** `m7-gpu6q` (`lux9b/m7/chains/m7-g6q.sh`, mirror `df6dcccc9`) on GPU6: Q-m1 with preflights → Q-m1-e1 →
  early rule → (if Q continues) Q-m2..Q-m5 → Q5 line. Uploaded, size + SHA-256 verified, then launched separately.
- **Budget:** + ≈ 3.5 GPU-h (preflights 0.2, five members ≈ 1.6, member-1 readout 0.05, line ≈ 1.0, a formal run
  ≈ 0.4). Projection ≈ 10 of 24 GPU-h; the caps and stop rules are unchanged.
- Everything else (data of P and C, the screens, formal, successor items 1–8, choice among passers, disclosures)
  is as preregistered. P-m1 and its readouts stay on node A as records.
