import { describe, expect, it } from 'vitest'

import type { EvaluationGate } from '../types/evaluationReport'
import { decodeEvaluationComparison } from './evaluationComparisonContract'

const BASELINE = '10000000-0000-4000-8000-000000000001'
const CANDIDATE = '10000000-0000-4000-8000-000000000002'

function gates(g3Verdict: 'pass' | 'fail' | 'unavailable' = 'unavailable'): EvaluationGate[] {
  return Array.from({ length: 10 }, (_, index) => ({
    id: `G${index}`,
    name: `Gate ${index}`,
    disposition: 'required',
    verdict: index === 3 ? g3Verdict : 'pass',
    change_profile: 'recipe',
    contract_version: 'evaluation-release-gates.v2',
    evidence_refs:
      index === 3
        ? [
            'server-reduction:comparative-g3.v1',
            `run:baseline:${BASELINE}`,
            `run:candidate:${CANDIDATE}`,
            'comparison-statistic:joint.normalized_regret',
          ]
        : [`gate:G${index}`],
    evidence_level: index === 3 ? 'E0' : 'E5',
    ...(index === 3
      ? {
          owner: 'recipe-and-model-pool',
          sample_count: 20,
        }
      : {}),
  }))
}

function comparison() {
  return {
    schema_version: 'evaluation.v1',
    attestation_revision: 'evaluation-server-attestation.v2',
    baseline_run_id: BASELINE,
    candidate_run_id: CANDIDATE,
    verdict: 'unavailable',
    summary: 'Server-reduced comparison remains diagnostic.',
    metrics: [],
    statistics: [
      {
        id: 'joint.normalized_regret',
        track_id: 'joint',
        estimator_id: 'paired-bootstrap-case-clustered-delta',
        estimator_version: 'v1',
        analysis_unit: 'case_normalized_regret',
        direction: 'lower_is_better',
        non_inferiority_margin: 0.05,
        baseline_value: 0.2,
        candidate_value: 0.1,
        delta: -0.1,
        confidence_level: 0.95,
        delta_confidence_interval: [-0.12, -0.08],
        candidate_confidence_interval: [0.09, 0.11],
        sample_count: 20,
        verdict: 'pass',
      },
    ],
    gates: gates(),
    recommendations: [],
    created_at: '2026-08-30T10:00:00Z',
  }
}

describe('evaluation comparison scientific contract', () => {
  it('binds G3 to the server-reduced E0 diagnostic without making a release claim', () => {
    expect(decodeEvaluationComparison(comparison(), BASELINE, CANDIDATE)).toEqual(comparison())
  })

  it('rejects retired waived comparison verdicts', () => {
    expect(() =>
      decodeEvaluationComparison(
        { ...comparison(), verdict: 'waived' },
        BASELINE,
        CANDIDATE,
      ),
    ).toThrow(/did not match the requested pair/i)
  })

  it('rejects a gate verdict that contradicts its applicability', () => {
    const payload = comparison()
    payload.gates = payload.gates.map((gate) =>
      gate.id === 'G0' ? { ...gate, verdict: 'not_applicable' } : gate,
    )
    expect(() => decodeEvaluationComparison(payload, BASELINE, CANDIDATE)).toThrow(
      /did not match the requested pair/i,
    )
  })

  it('never lets a tiny identical pair pass', () => {
    const payload = comparison()
    payload.statistics[0] = {
      ...payload.statistics[0],
      sample_count: 2,
      delta_confidence_interval: [],
      candidate_confidence_interval: [],
      verdict: 'unavailable',
    }
    payload.gates = gates().map((gate) => (gate.id === 'G3' ? { ...gate, sample_count: 2 } : gate))
    expect(decodeEvaluationComparison(payload, BASELINE, CANDIDATE).gates[3].verdict).toBe(
      'unavailable',
    )
    payload.gates[3] = { ...payload.gates[3], verdict: 'pass' }
    expect(() => decodeEvaluationComparison(payload, BASELINE, CANDIDATE)).toThrow(
      /overclaims its E0 diagnostic reduction/i,
    )
  })

  it('rejects statistic and server-reduction tampering', () => {
    const payload = comparison()
    expect(() =>
      decodeEvaluationComparison(
        {
          ...payload,
          statistics: [{ ...payload.statistics[0], delta: 0.5 }],
        },
        BASELINE,
        CANDIDATE,
      ),
    ).toThrow(/statistic is invalid/i)
    expect(() =>
      decodeEvaluationComparison(
        {
          ...payload,
          statistics: [
            {
              ...payload.statistics[0],
              track_id: 'routing',
              direction: 'higher_is_better',
              non_inferiority_margin: 1,
            },
          ],
        },
        BASELINE,
        CANDIDATE,
      ),
    ).toThrow(/not registered/i)
    expect(() =>
      decodeEvaluationComparison(
        {
          ...payload,
          gates: payload.gates.map((gate) =>
            gate.id === 'G3' ? { ...gate, evidence_refs: ['metrics.json'] } : gate,
          ),
        },
        BASELINE,
        CANDIDATE,
      ),
    ).toThrow(/not server-owned/i)

    expect(() =>
      decodeEvaluationComparison(
        {
          ...payload,
          gates: payload.gates.map((gate) =>
            gate.id === 'G3' ? { ...gate, verdict: 'pass' } : gate,
          ),
        },
        BASELINE,
        CANDIDATE,
      ),
    ).toThrow(/overclaims its E0 diagnostic reduction/i)

    for (const g3Patch of [
      { evidence_level: 'E4' },
      { observed: 0.1 },
      { threshold: { operator: '<=', value: 0.25, unit: 'fraction' } },
      { sample_count: 19 },
    ]) {
      expect(() =>
        decodeEvaluationComparison(
          {
            ...payload,
            gates: payload.gates.map((gate) => (gate.id === 'G3' ? { ...gate, ...g3Patch } : gate)),
          },
          BASELINE,
          CANDIDATE,
        ),
      ).toThrow(/not server-owned|overclaims its E0 diagnostic reduction/i)
    }
  })

  it('accepts a live diagnostic without a replay-derived G3 sample count', () => {
    const payload = comparison()
    payload.gates = payload.gates.map((gate) =>
      gate.id === 'G3' ? { ...gate, sample_count: undefined } : gate,
    )
    expect(decodeEvaluationComparison(payload, BASELINE, CANDIDATE).statistics).toHaveLength(1)
  })

  it('requires a not-applicable G3 diagnostic to carry no synthetic sample claim', () => {
    const payload = comparison()
    payload.gates = payload.gates.map((gate) =>
      gate.id === 'G3'
        ? {
            ...gate,
            disposition: 'not_applicable',
            verdict: 'not_applicable',
            sample_count: undefined,
          }
        : gate,
    )
    expect(decodeEvaluationComparison(payload, BASELINE, CANDIDATE).gates[3].verdict).toBe(
      'not_applicable',
    )
    payload.gates[3] = { ...payload.gates[3], sample_count: 20 }
    expect(() => decodeEvaluationComparison(payload, BASELINE, CANDIDATE)).toThrow(
      /not-applicable result is invalid/i,
    )
  })
})
