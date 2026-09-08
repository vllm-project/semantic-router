import type {
  EvaluationRun,
  EvaluationTrackId,
  EvidenceLevel,
} from '../../../src/types/evaluationPlane'
import type {
  EvaluationGate,
  EvaluationMetric,
  EvaluationMetricAnalysisProvenance,
} from '../../../src/types/evaluationReport'
import type { EvaluationRoutingRecipeReport } from '../../../src/types/evaluationRoutingRecipeReport'
import { evaluationCatalog } from './catalog'

type FrozenMetricAnalysisSpecification = Pick<
  EvaluationMetricAnalysisProvenance,
  | 'estimator_id'
  | 'estimator_version'
  | 'analysis_unit'
  | 'cluster_unit'
  | 'weighting'
  | 'missingness'
  | 'exclusion_policy'
>

function analysisSpecification(
  estimatorID: string,
  analysisUnit: string,
  clusterUnit: string,
  weighting: FrozenMetricAnalysisSpecification['weighting'],
): FrozenMetricAnalysisSpecification {
  return {
    estimator_id: estimatorID,
    estimator_version: 'v1',
    analysis_unit: analysisUnit,
    cluster_unit: clusterUnit,
    weighting,
    missingness: 'fail_closed',
    exclusion_policy: 'exclude_unavailable_evidence',
  }
}

// Frozen server wire specifications. These intentionally do not call the
// browser metric catalog resolver: a catalog/decoder drift must break E2E.
const METRIC_ANALYSIS_SPECIFICATIONS: Readonly<
  Record<string, FrozenMetricAnalysisSpecification>
> = {
    'routing.accuracy': analysisSpecification(
      'deterministic-routing-case-observed-ratio',
      'route_case',
      'case',
      'uniform_case',
    ),
    'model_pool.oracle_gain': analysisSpecification(
      'model-pool-dense-case-mean',
      'pool_case',
      'case',
      'uniform_case',
    ),
    'joint.realized_quality': analysisSpecification(
      'deterministic-joint-case-observed-mean',
      'routed_case',
      'case',
      'uniform_case',
    ),
    'agentic.success_rate': analysisSpecification(
      'deterministic-trajectory-observed-ratio',
      'trajectory',
      'case',
      'uniform_case',
    ),
    'multimodal.support_rate': analysisSpecification(
      'deterministic-multimodal-case-observed-ratio',
      'multimodal_case',
      'case',
      'uniform_case',
    ),
    'preference.agreement': analysisSpecification(
      'offline-preference-case-observed-ratio',
      'preference_case',
      'case',
      'uniform_case',
    ),
    'safety.violation_rate': analysisSpecification(
      'deterministic-policy-case-observed-mean',
      'policy_case',
      'case',
      'uniform_case',
    ),
    'capacity.latency_p95_ms': analysisSpecification(
      'capacity-measurement-request-quantile',
      'measurement_request',
      'repetition',
      'uniform_request',
    ),
}

const CAPACITY_LEVEL_THROUGHPUT_SPECIFICATION = analysisSpecification(
  'capacity-level-repetition-observed-mean',
  'load_repetition',
  'repetition',
  'uniform_repetition',
)

function frozenMetricAnalysisSpecification(metricID: string): FrozenMetricAnalysisSpecification {
  const specification = METRIC_ANALYSIS_SPECIFICATIONS[metricID]
  if (specification) return specification
  if (/^capacity\.level\.[1-9][0-9]{0,5}\.throughput_rps$/.test(metricID)) {
    return CAPACITY_LEVEL_THROUGHPUT_SPECIFICATION
  }
  throw new Error(`Missing frozen E2E metric analysis specification for ${metricID}.`)
}

const foundationalGateContracts = [
  {
    id: 'G0',
    name: 'Reproducibility',
    description: 'The evaluation can be traced to pinned inputs, settings, and outputs.',
    disposition: 'required',
    evidenceLevel: 'E0',
  },
  {
    id: 'G1',
    name: 'Static correctness',
    description: 'The saved evaluation bundle is complete, valid, and internally consistent.',
    disposition: 'required',
    evidenceLevel: 'E0',
  },
] as const

const gateOwners = [
  'evaluation-platform',
  'evaluation-platform',
  'router-policy',
  'recipe-and-model-pool',
  'evaluation-workload',
  'router-and-serving-runtime',
  'agent-runtime',
  'serving-capacity',
  'release-operations',
  'online-learning',
]

const gateEvidenceRefs = [
  ['run-manifest.json', 'lineage.json', 'provenance.json', 'checksums.sha256'],
  ['run-manifest.json', 'records.jsonl'],
  ['records.jsonl', 'metric:safety.violation_rate'],
  ['metrics.json', 'metric:joint.normalized_regret'],
  ['records.jsonl', 'metric:routing.accuracy'],
  ['records.jsonl', 'provenance.json'],
  ['records.jsonl', 'metric:agentic.success_rate'],
  ['records.jsonl', 'metrics.json'],
  ['run-manifest.json', 'records.jsonl'],
  ['records.jsonl', 'metric:preference.propensity_coverage'],
]

export function evaluationGates(run: EvaluationRun): EvaluationGate[] {
  const coverage = {
    evaluated: 4,
    total: 4,
    fraction: 1,
    unavailable: 0,
    confidence_level: 0.95,
    confidence_interval: [0.51, 1] as [number, number],
  }
  const notApplicableCoverage = {
    evaluated: 0,
    total: 0,
    fraction: 0,
    unavailable: 0,
  }
  const profile = evaluationCatalog.change_profiles.find(
    (candidate) => candidate.id === run.change_profile,
  )
  if (!profile) throw new Error(`Missing canonical change profile ${run.change_profile}.`)
  const gateContract = [
    ...foundationalGateContracts,
    ...profile.campaign_slots.map((slot) => ({
      id: slot.gate_id,
      name: slot.name,
      description: slot.description,
      disposition: slot.disposition,
      track_id: slot.track_id,
      evidenceLevel: slot.minimum_evidence_level,
    })),
  ]
  return gateContract.map((gate, index) => {
    const foundational = gate.id === 'G0' || gate.id === 'G1'
    const isNotApplicable = gate.disposition === 'not_applicable'
    return {
      id: gate.id,
      name: gate.name,
      description: gate.description,
      track_id: 'track_id' in gate ? gate.track_id : undefined,
      disposition: gate.disposition,
      verdict: isNotApplicable
        ? ('not_applicable' as const)
        : foundational
          ? ('pass' as const)
          : ('unavailable' as const),
      change_profile: run.change_profile,
      contract_version: evaluationCatalog.gate_contract_version,
      evidence_refs: gateEvidenceRefs[index],
      evidence_level: gate.evidenceLevel as EvidenceLevel,
      observed: isNotApplicable || !foundational ? null : 1,
      threshold:
        isNotApplicable || !foundational
          ? undefined
          : { operator: '>=', value: 1, unit: index === 0 ? 'fraction' : 'boolean' },
      sample_count: isNotApplicable ? 0 : 4,
      coverage: isNotApplicable ? notApplicableCoverage : coverage,
      owner: gateOwners[index],
      evaluated_at: '2026-08-29T00:10:00Z',
      rationale: foundational
        ? 'The server-validated bundle satisfies this foundational gate.'
        : isNotApplicable
          ? 'This gate is not required by the selected change profile.'
          : 'The E0 run produced diagnostics, but no server-owned qualified attestation exists; this gate cannot pass.',
    }
  })
}

export function diagnosticMetric(trackID: EvaluationTrackId): EvaluationMetric {
  const metrics: Record<EvaluationTrackId, EvaluationMetric> = {
    routing: {
      id: 'routing.accuracy',
      name: 'Routing accuracy',
      track_id: 'routing',
      value: 0.75,
      unit: 'fraction',
      direction: 'higher_is_better',
    },
    model_pool: {
      id: 'model_pool.oracle_gain',
      name: 'Pool oracle gain',
      track_id: 'model_pool',
      value: 0.08,
      unit: 'score',
      direction: 'higher_is_better',
    },
    joint: {
      id: 'joint.realized_quality',
      name: 'System quality',
      track_id: 'joint',
      value: 0.91,
      unit: 'fraction',
      direction: 'higher_is_better',
    },
    agentic: {
      id: 'agentic.success_rate',
      name: 'Agent success rate',
      track_id: 'agentic',
      value: 0.75,
      unit: 'fraction',
      direction: 'higher_is_better',
    },
    multimodal: {
      id: 'multimodal.support_rate',
      name: 'Multimodal support rate',
      track_id: 'multimodal',
      value: 0.75,
      unit: 'fraction',
      direction: 'higher_is_better',
    },
    preference: {
      id: 'preference.agreement',
      name: 'Preference agreement',
      track_id: 'preference',
      value: 0.8,
      unit: 'fraction',
      direction: 'higher_is_better',
    },
    safety: {
      id: 'safety.violation_rate',
      name: 'Safety violation rate',
      track_id: 'safety',
      value: 0,
      unit: 'violations/case',
      direction: 'lower_is_better',
    },
    capacity: {
      id: 'capacity.latency_p95_ms',
      name: 'P95 latency',
      track_id: 'capacity',
      value: 342,
      unit: 'ms',
      direction: 'lower_is_better',
    },
  }
  const metric = metrics[trackID]
  return {
    ...metric,
    baseline_value: null,
    delta: null,
    confidence_interval:
      metrics[trackID].unit === 'fraction' ? ([0.51, 1] as [number, number]) : undefined,
    sample_count: 4,
    analysis_provenance: evaluationMetricAnalysisProvenance(metric.id),
  }
}

export function evaluationMetricAnalysisProvenance(
  metricID: string,
): EvaluationMetric['analysis_provenance'] {
  return {
    contract_version: 'metric-analysis.v1',
    ...frozenMetricAnalysisSpecification(metricID),
    observed_exclusions: 0,
  }
}

export function denseReportMetric(index: number): EvaluationMetric {
  const ordinal = index + 1
  const id = `capacity.level.${ordinal}.throughput_rps`
  return {
    id,
    name: `Diagnostic metric ${ordinal}`,
    track_id: 'capacity',
    value: 30 + ordinal,
    unit: 'requests/s',
    direction: 'higher_is_better',
    baseline_value: null,
    delta: null,
    sample_count: 4,
    analysis_provenance: evaluationMetricAnalysisProvenance(id),
  }
}

export function routingRecipeReport(run: EvaluationRun): EvaluationRoutingRecipeReport | null {
  const plan = run.mixture?.routing_recipe_plan
  if (run.mode !== 'live' || !run.track_ids.includes('routing') || !plan) return null
  const inputAvailability = (id: string) => ({
    id,
    expected: 4,
    present: 3,
    missing: 0,
    error: 0,
    timeout: 1,
    latency: { available: false, reason: 'insufficient_latency_samples', sample_count: 1 },
  })
  const reliability = Array.from({ length: 10 }, (_, index) => ({
    lower: index / 10,
    upper: (index + 1) / 10,
    count: index < 4 ? 1 : 0,
    ...(index < 4 ? { mean_prediction: index / 10 + 0.05, observed_frequency: index % 2 } : {}),
  }))
  return {
    contract_version: 'routing-recipe-eval.v1',
    plan_digest: plan.plan_digest,
    e1: {
      expected_decisions: 4,
      observed_decisions: 4,
      signals: plan.signals.map((input) => inputAvailability(input.id)),
      projections: plan.projections.map((input) => inputAvailability(input.id)),
      eligibility_complete: 3,
      selected_feasible: 3,
    },
    e2: {
      projection_outcomes: plan.projections.map((projection) => ({
        projection_id: projection.id,
        spearman: { available: false, reason: 'insufficient_outcome_pairs', sample_count: 1 },
        brier: { available: true, value: 0.11, sample_count: 4 },
        ece_10: { available: true, value: 0.08, sample_count: 4 },
        reliability_bins: reliability,
      })),
      top_k: plan.top_k.map((k) => ({
        k,
        feasible_oracle_recall:
          k === 1
            ? { available: false, reason: 'oracle_outcome_missing', sample_count: 0 }
            : { available: true, value: 1, sample_count: 4 },
      })),
      oracle_regret: { available: false, reason: 'oracle_outcome_missing', sample_count: 0 },
    },
  }
}
