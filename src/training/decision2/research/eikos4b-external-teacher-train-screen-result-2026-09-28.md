# Eikos-4B external teacher: frozen TRAIN-only signal screen

**Decision: do not launch a three-type distillation arm from this screen.** This
is a 96-row descriptive TRAIN sample, not a development, JevArena or public
benchmark score. The third-party model was used only for native inference;
none of its weights initialized a Decision 2.0 student.

The [prospective screen](eikos4b-external-teacher-train-screen-prereg-2026-09-28.md)
froze the published teacher revision, source release hashes, existing TRAIN
bytes, gold-free roster, native adapter, output checks and 15-minute GPU cap
before inference. The only GPU invocation finished successfully. The private
aggregate receipt is SHA-256
`ad39a3dd9759dbb2b099e24b3b8b78c9f6e1872d078de00b72d451d8f7224a72`
and mode 0600. Docker start/die events were 27 seconds apart, about 0.0075
GPU-hours on one card. Its roster, source code and release hashes match the
frozen preregistration. No output contains raw TRAIN text or per-row targets.

| Native type | Independent TRAIN groups | Valid | Hard-label agreement | Mean gold probability | Normalized Brier | Ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Choice | 32 | 32 | 21/32 = 65.6% | 0.538 | 0.219 | 0 |
| Noul | 32 | 32 | 26/32 = 81.2% | 0.740 | 0.128 | 1 |
| Score | 32 | 32 | 10/32 = 31.2% | 0.300 | 0.384 | 3 |

All 96 responses passed the native shape, finite-probability and option-key
checks; there were no overflow failures. Structural validity alone is not a
reason to distill. Score is the present weakness across several eligible
students, yet this teacher's sampled Score signal is weak and tied more often
than Choice/Noul. These 32 groups per type do not establish a general error
rate or compare teacher and student on an independent panel. A separate,
source-disjoint screen would be required even for a Choice/Noul-only teacher
arm, with a matched budget and no Score target contamination.

**Next:** prioritize independently labeled, semantically native Score evidence
data and shortcut controls. Keep this teacher as a research comparator; do not
attach this TRAIN screen to a release card or present it as a model gain.
