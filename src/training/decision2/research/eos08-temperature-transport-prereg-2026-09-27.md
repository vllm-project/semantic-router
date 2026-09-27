# 0.8B fixed-checkpoint temperature transport diagnostic

Status: **preregistered before scoring** in signed commit `ad0f286dd`;
the completed result is in
[the fixed-checkpoint diagnostic](eos08-temperature-transport-result-2026-09-27.md).
This is a read-only development
diagnostic of three already completed checkpoints. It does not fit a new
temperature, select a checkpoint, change the release scalar, reopen FINAL, or
use the public JevBench subset. No optimizer or model inference is planned.

## Fixed evidence

All three runs already used their own audited CAL partition to fit one positive
temperature per native type by hard-label NLL. Their original native DEV1,600
and CSS pilot1,430 predictions are frozen. The identical typed DEV gold has
SHA-256 `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`;
the identical CSS pilot gold has SHA-256
`9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391`.
These gold files stay private. The 0.8B runs have distinct source models and
training partitions; only Base versus Posttrained is a matched initialization
control, while Eos is a separate architecture/data comparison.

| Run | Model fingerprint | CAL receipt SHA-256 | Frozen DEV predictions SHA-256 | Frozen CSS predictions SHA-256 |
| --- | --- | --- | --- | --- |
| Decision 1.0 Eos warm-start continuation, BEST160, hard CAL900 | `305d757d4dc53622894b08e5738568799220a0bf184f41100cba00acdb258f86` | `6047fb2724175d7b286f60dcb187012b138a1571b93b9238a872f34e22a6b228` | `808fa373159c69455a03b9445df02005c052e50b1a549c66f1b280662b41f641` | `1d3b156543759f216854ddf09469fcb0a3aaa8024b0bfbef4c66b0e37a723e80` |
| Qwen3.5 0.8B Base, BEST466, CAL700 | `bfa87b54fac58799efe2a26c9082166cbfedc2890c63d01298a73b0e55d497bd` | `18653158dfc6426e2f5aa139612a4c59979d2435b456cb16e3759ff8fc44280f` | `aeef9c6b82627371854723446d72ebb4e7920af2391d71a1c0864c4e46dbd057` | `3ae26ad1faf2b7b7bab83efb1ec9e109a13f6f3bfde8074f68db929ca85c37ee` |
| Qwen3.5 0.8B Posttrained, BEST466, CAL700 | `469795c06cd37152ace164fb91904b727813389e27a6cfc6a19e47ee57d5a8ac` | `0f6630b362f92dbc1f2ba7c4704e1a584f73233fad98be93222d62e3a8903af9` | `9f1f2efc32db919a4a2e512a8325b547b4a3662ce5b9e6f8f51021d4ee1c0b28` | `af7119d64428c35686a5f6a1e8b98b628f216148a33511426e75d1d1831588ad` |

Each prediction sidecar must bind its exact model, calibration and prediction
bytes. This document records only hashes and counts, never private raw examples.

## Frozen method and interpretation

For every strictly positive probability vector in the existing calibrated
predictions, reconstruct its uncalibrated counterpart as
`p_T=1(i) = p_CAL(i)^T / sum_j p_CAL(j)^T`, using that run's previously fitted
type temperature `T`. For Noul, apply the same operation to `[1-p,p]`.
Recompute Score's returned expectation from the reconstructed level map.
Reject a run if any probability is zero, one, non-finite, missing or if the
prediction/sidecar/model/CAL/gold fingerprints disagree: reconstructing
underflowed logits would be unreliable. No replacement or fallback probability
is permitted. Check native categorical predictions, validity and counts are
unchanged. Re-score the original calibrated and reconstructed `T=1` predictions
with the same frozen typed and CSS scorers, both panels, and report overall,
per-type and per-task accuracy, Brier, NLL and ECE where available. Keep
derived row-level predictions and reports private; report only aggregates,
fingerprints and failure counts.

The primary question is whether the **same checkpoint** improves or worsens
probability quality when its CAL temperature is transported to independent
typed DEV and human CSS pilot. Accuracy should be invariant under a positive
temperature. Any difference in accuracy or validity is an integrity failure,
not a treatment effect. A calibrated-to-`T=1` Brier change of at least `0.005`
on DEV is a useful diagnosis, but **neither DEV nor the CSS pilot can select a
release temperature** because their results have already been inspected.
Disagreeing signs between CAL and external panels warrant a future, separately
preregistered source-disjoint calibration study; they do not justify a post-hoc
temperature change or a `dev-2.0-0.8b` release. The Base/Posttrained transfer
reversal is a point-decision issue and cannot be repaired by monotone
temperature scaling.
