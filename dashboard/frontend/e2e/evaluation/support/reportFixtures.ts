import type { EvaluationComparison } from '../../../src/types/evaluationComparison'
import type { EvaluationReport } from '../../../src/types/evaluationReport'
import { EVALUATION_ATTESTATION_REVISION } from '../../../src/types/evaluationPlane'
import { evaluationCatalog } from './catalog'
import { EVALUATION_RUN_IDS } from './mixtureFixture'
import { defaultEvaluationRuns } from './runFixtures'
import {
  diagnosticMetric,
  evaluationGates,
  evaluationMetricAnalysisProvenance,
  routingRecipeReport,
} from './reportMetricFixtures'

const evaluationTrackNames = new Map(
  evaluationCatalog.tracks.map((track) => [track.id, track.name]),
)

export function evaluationReport(run = defaultEvaluationRuns[0]): EvaluationReport {
  const totalRecords = run.track_ids.length * 4
  const coverage = {
    evaluated: totalRecords,
    total: totalRecords,
    fraction: 1,
    unavailable: 0,
  }
  const gates = evaluationGates(run)
  const metrics = run.track_ids.map(diagnosticMetric)
  const digest = 'sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
  return {
    schema_version: 'evaluation.v1',
    attestation_revision: EVALUATION_ATTESTATION_REVISION,
    run,
    summary: {
      verdict: gates.some(
        (gate) => gate.disposition === 'required' && gate.verdict === 'unavailable',
      )
        ? 'unavailable'
        : 'pass',
      quality_score: null,
      latency_p95_ms: null,
      runtime_cost: null,
      capacity_tco: null,
      coverage,
      passed_gates: gates.filter((gate) => gate.verdict === 'pass').length,
      failed_gates: gates.filter((gate) => gate.verdict === 'fail').length,
      unavailable_gates: gates.filter((gate) => gate.verdict === 'unavailable').length,
    },
    tracks: run.track_ids.map((trackID) => ({
      track_id: trackID,
      status: 'completed' as const,
      evidence_level: run.evidence_level,
      summary: `${evaluationTrackNames.get(trackID) || trackID} diagnostic observation completed.`,
      coverage: { evaluated: 4, total: 4, fraction: 1, unavailable: 0 },
      metrics: [diagnosticMetric(trackID)],
      gates: gates.filter((gate) => gate.track_id === trackID),
    })),
    metrics,
    gates,
    costs: {
      runtime: {
        amount: 0.03195,
        currency: 'USD',
        input_tokens: 12000,
        output_tokens: 6000,
        gpu_seconds: 0.39,
      },
      evaluation_overhead: { amount: 0.00165, currency: 'USD' },
      capacity_tco: { amount: 0.039, currency: 'USD', gpu_seconds: 0.39, energy_kwh: 0.0018 },
    },
    recommendations: [
      'Treat these E0 observations as diagnostics, not a promotion claim.',
      'Collect benchmark-native receipts and qualified robustness evidence before promotion.',
    ],
    provenance: {
      schema_version: 'evaluation.v1',
      generated_at: '2026-08-29T00:10:00Z',
      code_revision: '0123456789abcdef0123456789abcdef01234567',
      benchmark_revisions: Object.fromEntries(
        run.suite_ids.map((id) => {
          const suite = evaluationCatalog.suites.find((candidate) => candidate.id === id)
          if (!suite) throw new Error(`Missing catalog suite ${id}.`)
          return [id, suite.revision]
        }),
      ),
      workload_snapshot_digest: digest,
      policy_snapshot_digest: run.baseline_run_id
        ? 'sha256:1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
        : digest,
      binding_snapshot_digest: run.baseline_run_id
        ? 'sha256:2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
        : digest,
      pool_snapshot_digest: digest,
      environment_snapshot_digest: digest,
      target_id: run.target_id,
      seed: 42,
      redaction_policy: 'public-safe-v1',
    },
    // Method reports are intentionally an empty, non-null collection when
    // this E0 fixture has no eligible raw-coordinate method reduction.
    method_reports: [],
    routing_recipe_report: routingRecipeReport(run),
    artifacts: [
      {
        id: 'metrics-json',
        name: 'metrics.json',
        kind: 'json',
        uri: 'metrics.json',
        digest,
        media_type: 'application/json',
        size_bytes: 1024,
      },
      {
        id: 'gates-json',
        name: 'gates.json',
        kind: 'json',
        uri: 'gates.json',
        digest,
        media_type: 'application/json',
        size_bytes: 1024,
      },
      {
        id: 'provenance-json',
        name: 'provenance.json',
        kind: 'json',
        uri: 'provenance.json',
        digest,
        media_type: 'application/json',
        size_bytes: 512,
      },
      {
        id: 'failure-summary-json',
        name: 'failure-summary.json',
        kind: 'json',
        uri: 'failure-summary.json',
        digest,
        media_type: 'application/json',
        size_bytes: 512,
      },
      {
        id: 'checksums-sha256',
        name: 'checksums.sha256',
        kind: 'sha256',
        uri: 'checksums.sha256',
        digest,
        media_type: 'text/plain',
        size_bytes: 325,
      },
    ],
  }
}

export const evaluationComparison: EvaluationComparison = {
  schema_version: 'evaluation.v1',
  attestation_revision: EVALUATION_ATTESTATION_REVISION,
  baseline_run_id: EVALUATION_RUN_IDS.baseline,
  candidate_run_id: EVALUATION_RUN_IDS.candidate,
  verdict: 'unavailable',
  summary: 'Diagnostic deltas are favorable, but E0 evidence cannot support promotion.',
  metrics: [
    {
      id: 'joint.realized_quality',
      name: 'System quality',
      track_id: 'joint',
      value: 0.91,
      unit: 'fraction',
      direction: 'higher_is_better',
      baseline_value: 0.88,
      delta: 0.03,
      sample_count: 4,
      analysis_provenance: evaluationMetricAnalysisProvenance('joint.realized_quality'),
    },
    {
      id: 'capacity.latency_p95_ms',
      name: 'P95 latency',
      track_id: 'capacity',
      value: 342,
      unit: 'ms',
      direction: 'lower_is_better',
      baseline_value: 370,
      delta: -28,
      sample_count: 4,
      analysis_provenance: evaluationMetricAnalysisProvenance('capacity.latency_p95_ms'),
    },
  ],
  statistics: [
    {
      id: 'joint.normalized_regret',
      track_id: 'joint',
      estimator_id: 'paired-bootstrap-case-clustered-delta',
      estimator_version: 'v1',
      analysis_unit: 'case_normalized_regret',
      direction: 'lower_is_better',
      non_inferiority_margin: 0.05,
      baseline_value: 0.12,
      candidate_value: 0.1,
      delta: -0.02,
      confidence_level: 0.95,
      delta_confidence_interval: [],
      candidate_confidence_interval: [],
      sample_count: 4,
      verdict: 'unavailable',
    },
  ],
  gates: evaluationReport().gates.map((gate) =>
    gate.id === 'G3'
      ? {
          ...gate,
          verdict: 'unavailable',
          evidence_refs: [
            'server-reduction:comparative-g3.v1',
            `run:baseline:${EVALUATION_RUN_IDS.baseline}`,
            `run:candidate:${EVALUATION_RUN_IDS.candidate}`,
            'comparison-statistic:joint.normalized_regret',
          ],
          evidence_level: 'E0',
          observed: undefined,
          threshold: undefined,
          sample_count: 4,
          owner: 'recipe-and-model-pool',
          rationale:
            'Server-reduced synthetic replay regret is retained as an E0 diagnostic only; it cannot pass or fail G3.',
        }
      : gate,
  ),
  recommendations: ['Collect qualified robustness evidence before a guarded live trial.'],
  created_at: '2026-08-29T00:10:00Z',
}
