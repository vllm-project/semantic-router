# 9B M10 amendment 6: extra seeds KIB4-s3 and KX-s4 / s5 on node B (2026-10-02)

Written ≈18:10 UTC+8 (10:10Z), after COORDINATOR WATCHDOG 18:00. Node B GPU3 / 6 / 7 were idle after the KUP and
KIBM seeds, and the watchdog asked for extra seeds of the most promising arm, to enlarge its soup. This was written
before any KIB4 or KX point was measured.

## Seeds

| GPU (node B) | Run | Seed | Why |
| --- | --- | --- | --- |
| GPU3 | m10-KIB4-s3 | 2 | KIB4 is a two-seed arm (amendment 3); this makes it a three-seed arm like K-a13IB |
| GPU6 | m10-KX-s4 | 3 | KX targets 9B's largest Index deficits (amendment 4) |
| GPU7 | m10-KX-s5 | 4 | as above |

- **Recipe.** Every seed uses its arm's recipe and TRAIN unchanged (`chains.sh` with `M10_PHASE=3`, the same stop
  rules and the 4.5 GPU-h seed cap).
- **KX data.** KX's TRAIN and teacher were built on node B (`prep.sh kx`). They enter node B's lock through
  `prep.sh kx-lock` with the SHA-256 values in node A's lock (`f1d9ecf8…`, `2a7ac626…`), so node B trains on the exact
  files that node A trains on.
- **Pre-warm.** Node B's pre-warm marker comes from m10-KUP-s1.

## Candidates

- **KIB4P** is the uniform FP32 soup of KIB4-s1 / s2 / s3's BEST checkpoints. Its points are KIB4P-a33 / a25 / a40.
- **KXP** is the uniform soup of KX-s1…s5's BEST checkpoints. It is built as the FP32 soup of [KX × 3, KX45 × 2],
  where KX45 is the soup of s4 / s5. This equals the uniform five-seed soup. Its points are KXP-a33 / a25 / a40.
- **Gate.** Each point is measured once, on its BF16 release copy, with the same gate as every other candidate. The
  two-seed or three-seed soup already measured stays a separate candidate.
- **Failures.** A failed or capped seed is not rerun, and the soup uses the seeds that finished (disclosed).

## Budget

Node B's M10 total was 26.1 GPU-h at launch; its 50 GPU-h gate still holds. The three seeds cost ≈ 8–9 GPU-h. The
M10 limit stays at 90 GPU-h, and no new Index or formal run starts above 85 GPU-h (amendment 5).
