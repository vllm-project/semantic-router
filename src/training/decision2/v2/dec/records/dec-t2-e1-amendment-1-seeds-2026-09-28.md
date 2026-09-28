# E1 amendment 1: second data-order seed for the 4B pair

Status: **committed before launch.** Amends [`dec-t2-e1-prereg-2026-09-28.md`](dec-t2-e1-prereg-2026-09-28.md).

The 0.8B N0 replicate showed that a data-order change alone moves the dev proxy
by 1.19 points (typed Choice 531 → 454/800) while the item bootstrap calls the
change significant, and the coordinator asks for two seeds before any finalist
claim. The only E1 contrast with a positive direction is X2 − X0 = +1.05
(single seed). Two replicates are added, identical to X0 and X2 except the
data-order seed:

| Arm | Replicates | Seed |
| --- | --- | --- |
| E1-X0s2 | E1-X0 (Nox 1.0 control) | 20260927 |
| E1-X2s2 | E1-X2 (Nox 1.0 + own-Lux KL 0.5, labels `752b7c8f…`) | 20260927 |

Same code (`a4c6f0000`), image, data, budget, preflight, selection, CAL and
single dev readout as E1; node B GPU1 and GPU2.

Decision rule for the 4B factor claim: σ_seed(4B) = |X0 − X0s2| / √2 on the
proxy; the factor is **confirmed** only if the mean of (X2 − X0, X2s2 − X0s2)
is ≥ +1.0, exceeds 2σ_seed(4B), and both per-seed differences are positive;
otherwise it stays suggestive or becomes negative. These replicates do not
change the frozen X2 formal candidate or its lock; they are not scored on v3.
