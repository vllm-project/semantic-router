import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationCapacityProfile } from '../../types/evaluationCapacityReport'
import type { EvaluationFailureSummary, EvaluationMetric } from '../../types/evaluationReport'
import EvaluationReportDiagnostics from './EvaluationReportDiagnostics'

const wilsonZ = 1.6448536269514722
const zeroErrorClusterUpper = wilsonZ ** 2 / (100 + wilsonZ ** 2)

const failureSummary: EvaluationFailureSummary = {
  schema_version: 'evaluation.v1',
  total_records: 4,
  failed: 0,
  unavailable: 0,
  by_track: [{ track_id: 'routing', succeeded: 4, failed: 0, unavailable: 0 }],
}

function repetitions(concurrency: number, throughput: number, latency: number) {
  return [1, 2, 3].map((repetition) => ({
    concurrency,
    repetition,
    requests: 100,
    successes: 100,
    errors: 0,
    elapsed_seconds: 100 / throughput,
    throughput_rps: throughput,
    latency_p95_ms: latency,
    error_rate: 0,
    error_rate_upper_bound: zeroErrorClusterUpper,
  }))
}

const capacityProfile: EvaluationCapacityProfile = {
  schema_version: 'evaluation.v1',
  kind: 'repeated-closed-loop-capacity',
  protocol: {
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
  },
  levels: [
    {
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
      latency_p50_ms: 8,
      latency_p95_ms: 10,
      latency_p99_ms: 12,
      latency_p95_cv: 0,
      error_rate: 0,
      error_rate_upper_bound: zeroErrorClusterUpper,
      measurement_cluster_count: 3,
      error_rate_cluster_range: 0,
      input_tokens: 40,
      output_tokens: 20,
      runtime_cost_usd: 0.01,
      repetitions: repetitions(1, 10, 10),
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
    },
    {
      concurrency: 2,
      warmup_requests: 4,
      warmup_errors: 0,
      warmup_elapsed_seconds: 1.5,
      measurement_requests: 300,
      successes: 300,
      errors: 0,
      elapsed_seconds: 18,
      throughput_rps: 100 / 6,
      throughput_cv: 0,
      latency_p50_ms: 11,
      latency_p95_ms: 13,
      latency_p99_ms: 14,
      latency_p95_cv: 0,
      error_rate: 0,
      error_rate_upper_bound: zeroErrorClusterUpper,
      measurement_cluster_count: 3,
      error_rate_cluster_range: 0,
      input_tokens: 40,
      output_tokens: 20,
      runtime_cost_usd: 0.01,
      repetitions: repetitions(2, 100 / 6, 13),
      throughput_scaling_efficiency: 100 / 6 / 10 / 2,
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
    },
  ],
  slo: {
    schema_version: 'evaluation.v1',
    required_concurrency: 2,
    max_latency_p95_ms: 20,
    max_error_rate: 0.05,
    min_throughput_rps: 15,
    min_throughput_scaling_efficiency: 0.7,
  },
  assessment: {
    qualified_concurrency: 2,
    saturation_concurrency: null,
    slo_headroom: 0,
    verdict: 'pass',
    failure_reasons: [],
  },
}

describe('EvaluationReportDiagnostics', () => {
  it('isolates an invalid capacity artifact while preserving valid outcome diagnostics', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportDiagnostics, {
        metrics: [],
        failureSummary,
        capacityProfile: null,
        failureSummaryIssue: null,
        capacityProfileIssue: {
          kind: 'invalid',
          artifactName: 'capacity-profile.json',
          message:
            'capacity-profile.json did not match the required evaluation.v1 diagnostic schema.',
        },
        loading: false,
      }),
    )
    expect(markup).toContain('Outcome accounting by evaluation area')
    expect(markup).toContain('Capacity profile diagnostic error')
    expect(markup).toContain('Diagnostic could not be verified')
    expect(markup).not.toContain('This run did not publish aggregate diagnostics')
  })

  it('distinguishes an unavailable artifact from an invalid artifact', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportDiagnostics, {
        metrics: [],
        failureSummary: null,
        capacityProfile: null,
        failureSummaryIssue: {
          kind: 'unavailable',
          artifactName: 'failure-summary.json',
          message: 'failure-summary.json could not be loaded. HTTP 404',
        },
        capacityProfileIssue: null,
        loading: false,
      }),
    )
    expect(markup).toContain('Outcome accounting diagnostic error')
    expect(markup).toContain('Diagnostic is not available')
    expect(markup).not.toContain('Diagnostic could not be verified')
  })

  it('renders the frozen protocol, repeated observations, UCB, and stability evidence', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportDiagnostics, {
        metrics: [],
        failureSummary,
        capacityProfile,
        failureSummaryIssue: null,
        capacityProfileIssue: null,
        loading: false,
      }),
    )
    expect(markup).toContain('Capacity target passed')
    expect(markup).toContain('Supported concurrency')
    expect(markup).toContain('Within target')
    expect(markup).toContain('Recorded capacity load plan')
    expect(markup).toContain('c1 → c2')
    expect(markup).toContain('100 requests × 3 repetitions')
    expect(markup).toContain('Errors / upper confidence estimate')
    expect(markup).toContain('Throughput / variation')
    expect(markup).toContain('3 independent windows')
    expect(markup).toContain('<th scope="col">Service checks</th>')
    for (const label of [
      'Warmup requests',
      'Response-time target',
      'Error-rate target',
      'Throughput target',
      'Scaling efficiency',
      'Throughput stability',
      'Response-time stability',
    ]) {
      expect(markup).toContain(`${label}: passed`)
      expect(markup).toContain(`>${label}</span>`)
    }
    expect(markup).not.toContain('Checks W / L / E / T / S / Tσ / Lσ')
    expect(markup).toContain('40 / 20')
  })

  it('retains distinct analysis methods that share an estimator release', () => {
    const metric = (id: string, analysisUnit: string, clusterUnit: string): EvaluationMetric => ({
      id,
      name: id,
      value: 1,
      unit: 'fraction',
      analysis_provenance: {
        contract_version: 'metric-analysis.v1',
        estimator_id: 'paired-bootstrap',
        estimator_version: '1.0.0',
        analysis_unit: analysisUnit,
        cluster_unit: clusterUnit,
        weighting: 'uniform_pair',
        missingness: 'fail_closed',
        exclusion_policy: 'exclude_unavailable_evidence',
        observed_exclusions: 0,
      },
    })
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportDiagnostics, {
        metrics: [
          metric('task-success', 'agent_attempt', 'task'),
          metric('tool-success', 'tool_call', 'agent_attempt'),
        ],
        failureSummary: null,
        capacityProfile: null,
        failureSummaryIssue: null,
        capacityProfileIssue: null,
        loading: false,
      }),
    )

    expect(markup).toContain('2 methods')
    expect(markup).toContain('agent_attempt / task')
    expect(markup).toContain('tool_call / agent_attempt')
  })

  it('keeps adversarial artifact responses behind closed technical details', () => {
    const backendMessage = 'decoder://artifact E5 schema-path=private.internal.field'
    const artifactName = 'worker-capacity-profile.internal.json'
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportDiagnostics, {
        metrics: [],
        failureSummary: null,
        capacityProfile: null,
        failureSummaryIssue: null,
        capacityProfileIssue: {
          kind: 'invalid',
          artifactName,
          message: backendMessage,
        },
        loading: false,
      }),
    )
    const boundaryIndex = markup.indexOf('data-evaluation-technical-details="true"')

    expect(markup.slice(0, boundaryIndex)).toContain('Diagnostic could not be verified')
    expect(markup.slice(0, boundaryIndex)).not.toContain(backendMessage)
    expect(markup.slice(0, boundaryIndex)).not.toContain(artifactName)
    expect(markup.slice(boundaryIndex)).toContain(backendMessage)
    expect(markup.slice(boundaryIndex)).toContain(artifactName)
    expect(markup).not.toContain('<details open')
  })
})
