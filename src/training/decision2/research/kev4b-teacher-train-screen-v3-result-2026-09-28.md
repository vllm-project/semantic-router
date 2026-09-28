# Kev-4B external teacher: disjoint TRAIN-only signal result

**Decision: no three-type distillation arm from this screen.** The single
permitted v3 run completed on its frozen, v1/v2-disjoint TRAIN roster. The
teacher produced 96/96 structurally valid native decisions, but missed the
preregistered Choice, Score agreement and Score probability-quality gates.
This is a source-signal diagnostic, not a Decision 2.0 student or release
evaluation. No student was trained, and no protected benchmark label was read.

The [prospective protocol and pre-GPU roster lock](kev4b-teacher-train-screen-v3-prereg-2026-09-28.md)
were signed before teacher inference. The roster contained 96 independent
rights-clean TRAIN groups, 32 per type, with zero overlap with either previous
screen and ordered identity SHA-256
`f35e56d4127d9c573671e232880b9de76abe67a69115e97a22f6187259b2cdae`.
TRAIN SHA-256 was
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
The mirrored native script SHA-256 was
`5bd612d7e0db655ba6bda6681cd5cb192a40173062042bcad59e14f157dac4f9`.
The model revision was `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`;
model fingerprint
`8e2c7fff2ef6ad7b195443fac287fb1ae4cd83c8c9af1a12dfb743501a3ec3e9`.
The publisher source revision, 30 source hashes and the official Qwen base
revision were checked offline with the **same absolute-root mount** used in
the GPU run. The image ID was
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.

| Native type | TRAIN groups | Valid | Hard-label agreement | Mean gold probability | Half multiclass Brier | Ties | Frozen requirement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Choice | 32 | 32 | 20/32 = 62.5% | 0.499 | 0.237 | 0 | ≥22/32: **fail** |
| Noul | 32 | 32 | 26/32 = 81.2% | 0.728 | 0.111 | 0 | ≥26/32: pass |
| Score | 32 | 32 | 15/32 = 46.9% | 0.300 | 0.348 | 0 | ≥20/32 and Brier ≤0.25: **fail** |

The run used the publisher's native typed Checkpoint path and checkpoint
temperature, with the preregistered four-decimal probability tolerance and
normalization. No invalid output or context overflow occurred. Results are
descriptive for these 32 independent groups per type; they neither establish
cross-source transfer nor give a student gain. The private aggregate receipt
has mode `0600` and SHA-256
`f7402ccfea866694f2be99e6e55df443b33c474450fee84ede1e54739287e3a6`.
It contains aggregates and hashes, not raw TRAIN rows or per-item predictions.

The sole GPU invocation took **37.02 seconds**, at most **0.01029 GPU-hours**
on one isolated card. Its container exited, and the card returned to zero
allocated memory. There was no changed-roster, threshold, prompt or model
retry. A type-restricted future teacher investigation would require its own
prospective source audit, budget-matched student/control plan and independent
development gate; this screen authorizes none. Prioritize independent Score
evidence rather than distilling this teacher's weak sampled Score signal.
