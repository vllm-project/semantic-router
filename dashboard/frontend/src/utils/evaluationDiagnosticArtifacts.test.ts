import { describe, expect, it } from 'vitest'

import type {
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
} from '../types/evaluationPlane'
import { evaluationDiagnosticArtifactIssue } from './evaluationDiagnosticArtifacts'
import { decodeEvaluationCapacityProfile } from './evaluationCapacityProfileContract'
import { InvalidEvaluationDiagnosticArtifactError } from './evaluationDiagnosticArtifactValidation'
import { decodeEvaluationFailureSummary } from './evaluationFailureSummaryContract'

const validFailureSummary = {
  schema_version: 'evaluation.v1',
  total_records: 4,
  failed: 1,
  unavailable: 1,
  by_track: [{ track_id: 'routing', succeeded: 2, failed: 1, unavailable: 1 }],
}

const protocol: EvaluationCapacityLoadProtocol = {
  schema_version: 'evaluation.v1',
  kind: 'closed-loop',
  concurrency_levels: [1, 2],
  warmup_request_multiplier: 2,
  measurement_requests_per_repetition: 100,
  repetitions_per_level: 3,
  minimum_measurement_clusters_per_level: 3,
  confidence_level: 0.95,
  max_error_rate_cluster_range: 0.05,
  max_throughput_cv: 0.2,
  max_latency_p95_cv: 0.2,
}

const slo: EvaluationCapacitySLO = {
  schema_version: 'evaluation.v1',
  required_concurrency: 2,
  max_latency_p95_ms: 200,
  max_error_rate: 0.05,
  min_throughput_rps: 15,
  min_throughput_scaling_efficiency: 0.7,
}

const wilsonZ = 1.6448536269514722
const errorRateUpperBound = wilsonZ ** 2 / (100 + wilsonZ ** 2)

function repetitions(concurrency: number, elapsed: number, latencyP95: number) {
  return [1, 2, 3].map((repetition) => ({
    concurrency,
    repetition,
    requests: 100,
    successes: 100,
    errors: 0,
    elapsed_seconds: elapsed,
    throughput_rps: 100 / elapsed,
    latency_p95_ms: latencyP95,
    error_rate: 0,
    error_rate_upper_bound: errorRateUpperBound,
  }))
}

const levelOne = {
  concurrency: 1,
  warmup_requests: 2,
  warmup_errors: 0,
  warmup_elapsed_seconds: 1,
  measurement_requests: 300,
  successes: 300,
  errors: 0,
  elapsed_seconds: 30,
  throughput_rps: 10,
  throughput_cv: 0,
  latency_p50_ms: 80,
  latency_p95_ms: 100,
  latency_p99_ms: 120,
  latency_p95_cv: 0,
  error_rate: 0,
  error_rate_upper_bound: errorRateUpperBound,
  measurement_cluster_count: 3,
  error_rate_cluster_range: 0,
  input_tokens: 1200,
  output_tokens: 600,
  runtime_cost_usd: 0.04,
  repetitions: repetitions(1, 10, 100),
  throughput_scaling_efficiency: null,
  warmup_passed: true,
  latency_slo_passed: true,
  cluster_coverage_passed: true,
  error_rate_stability_passed: true,
  error_slo_passed: true,
  throughput_slo_passed: true,
  scaling_slo_passed: true,
  throughput_stability_passed: true,
  latency_stability_passed: true,
  qualified: true,
}

const levelTwo = {
  ...levelOne,
  concurrency: 2,
  warmup_requests: 4,
  warmup_elapsed_seconds: 1.5,
  elapsed_seconds: 18,
  throughput_rps: 100 / 6,
  latency_p50_ms: 120,
  latency_p95_ms: 150,
  latency_p99_ms: 180,
  repetitions: repetitions(2, 6, 150),
  throughput_scaling_efficiency: 100 / 6 / 10 / 2,
}

const validCapacityProfile = {
  schema_version: 'evaluation.v1',
  kind: 'repeated-closed-loop-capacity',
  protocol,
  levels: [levelOne, levelTwo],
  slo,
  assessment: {
    qualified_concurrency: 2,
    saturation_concurrency: null,
    slo_headroom: 0,
    verdict: 'pass',
    failure_reasons: [],
  },
}

describe('evaluation diagnostic artifact decoding', () => {
  it('returns rebuilt values for exact failure and repeated-load schemas', () => {
    expect(decodeEvaluationFailureSummary(validFailureSummary)).toEqual(validFailureSummary)
    expect(decodeEvaluationCapacityProfile(validCapacityProfile, slo, protocol)).toEqual(
      validCapacityProfile,
    )
  })

  it.each([
    ['null root', null],
    ['unknown root field', { ...validFailureSummary, unexpected: true }],
    ['non-finite aggregate', { ...validFailureSummary, total_records: Number.POSITIVE_INFINITY }],
    ['null track collection', { ...validFailureSummary, by_track: null }],
    [
      'unknown track',
      { ...validFailureSummary, by_track: [{ ...validFailureSummary.by_track[0], track_id: 'x' }] },
    ],
    [
      'duplicate track',
      {
        ...validFailureSummary,
        total_records: 8,
        failed: 2,
        unavailable: 2,
        by_track: [validFailureSummary.by_track[0], validFailureSummary.by_track[0]],
      },
    ],
    ['inconsistent aggregate', { ...validFailureSummary, failed: 0 }],
  ])('rejects an invalid failure summary: %s', (_name, value) => {
    expect(() => decodeEvaluationFailureSummary(value)).toThrow(
      InvalidEvaluationDiagnosticArtifactError,
    )
  })

  it.each([
    ['null root', null],
    ['legacy profile kind', { ...validCapacityProfile, kind: 'bounded-concurrency-sweep' }],
    ['unknown root field', { ...validCapacityProfile, unexpected: true }],
    ['null levels', { ...validCapacityProfile, levels: null }],
    ['empty levels', { ...validCapacityProfile, levels: [] }],
    [
      'non-geometric protocol',
      { ...validCapacityProfile, protocol: { ...protocol, concurrency_levels: [1, 3] } },
    ],
    [
      'missing repetition',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, repetitions: levelOne.repetitions.slice(0, 2) }, levelTwo],
      },
    ],
    [
      'tampered repetition throughput',
      {
        ...validCapacityProfile,
        levels: [
          {
            ...levelOne,
            repetitions: [
              { ...levelOne.repetitions[0], throughput_rps: 11 },
              ...levelOne.repetitions.slice(1),
            ],
          },
          levelTwo,
        ],
      },
    ],
    [
      'tampered error upper bound',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, error_rate_upper_bound: 0 }, levelTwo],
      },
    ],
    [
      'negative error ratio',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, error_rate: -0.01 }, levelTwo],
      },
    ],
    [
      'negative throughput coefficient of variation',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, throughput_cv: -0.01 }, levelTwo],
      },
    ],
    [
      'negative aggregate latency',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, latency_p50_ms: -1 }, levelTwo],
      },
    ],
    [
      'negative latency coefficient of variation',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, latency_p95_cv: -0.01 }, levelTwo],
      },
    ],
    [
      'negative repetition latency',
      {
        ...validCapacityProfile,
        levels: [
          {
            ...levelOne,
            repetitions: [
              { ...levelOne.repetitions[0], latency_p95_ms: -1 },
              ...levelOne.repetitions.slice(1),
            ],
          },
          levelTwo,
        ],
      },
    ],
    [
      'tampered stability decision',
      {
        ...validCapacityProfile,
        levels: [{ ...levelOne, throughput_stability_passed: false }, levelTwo],
      },
    ],
    [
      'tampered assessment',
      {
        ...validCapacityProfile,
        assessment: { ...validCapacityProfile.assessment, slo_headroom: 1 },
      },
    ],
  ])('rejects an invalid capacity profile: %s', (_name, value) => {
    expect(() => decodeEvaluationCapacityProfile(value, slo, protocol)).toThrow(
      InvalidEvaluationDiagnosticArtifactError,
    )
  })

  it('binds both capacity contracts to the frozen report run', () => {
    expect(() =>
      decodeEvaluationCapacityProfile(
        validCapacityProfile,
        { ...slo, max_latency_p95_ms: 201 },
        protocol,
      ),
    ).toThrow(/differs from the frozen report run contract/i)
    expect(() =>
      decodeEvaluationCapacityProfile(validCapacityProfile, slo, {
        ...protocol,
        repetitions_per_level: 4,
      }),
    ).toThrow(/differs from the frozen report run contract/i)
    expect(() =>
      decodeEvaluationCapacityProfile(validCapacityProfile, undefined, undefined),
    ).toThrow(/frozen report capacity contracts are unavailable/i)
  })

  it('classifies schema failures as invalid without hiding load errors', () => {
    expect(evaluationDiagnosticArtifactIssue('capacity-profile.json', new SyntaxError())).toEqual({
      kind: 'invalid',
      artifactName: 'capacity-profile.json',
      message: 'capacity-profile.json did not match the required evaluation.v1 diagnostic schema.',
    })
    expect(
      evaluationDiagnosticArtifactIssue(
        'failure-summary.json',
        new InvalidEvaluationDiagnosticArtifactError('failure-summary.json', 'bad shape'),
      ).kind,
    ).toBe('invalid')
    expect(
      evaluationDiagnosticArtifactIssue('failure-summary.json', new Error('HTTP 404')),
    ).toEqual({
      kind: 'unavailable',
      artifactName: 'failure-summary.json',
      message: 'failure-summary.json could not be loaded. HTTP 404',
    })
  })
})
