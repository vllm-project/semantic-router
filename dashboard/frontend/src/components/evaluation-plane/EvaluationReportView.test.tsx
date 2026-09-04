import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type {
  EvaluationMetricAnalysisProvenance,
  EvaluationReport,
} from '../../types/evaluationReport'
import type { EvaluationReportDiagnosticsState } from '../../types/evaluationReportDiagnostics'
import { EVALUATION_ATTESTATION_REVISION } from '../../types/evaluationPlane'
import { metricAnalysisSpecification } from '../../test/evaluationMetricAnalysisFixture'
import { buildEvaluationRoutingRecipePlan } from '../../test/evaluationRoutingRecipeFixture'
import EvaluationReportView from './EvaluationReportView'

const EMPTY_DIAGNOSTICS: EvaluationReportDiagnosticsState = {
  failureSummary: null,
  capacityProfile: null,
  failureSummaryIssue: null,
  capacityProfileIssue: null,
  loading: false,
}

function expectInsideCollapsedDetails(markup: string, value: string, summary: string) {
  const valueIndex = markup.indexOf(value)
  expect(valueIndex).toBeGreaterThan(-1)

  const detailsStart = markup.lastIndexOf('<details', valueIndex)
  const openingTagEnd = markup.indexOf('>', detailsStart)
  const detailsEnd = markup.indexOf('</details>', valueIndex)
  expect(detailsStart).toBeGreaterThan(-1)
  expect(openingTagEnd).toBeGreaterThan(detailsStart)
  expect(detailsEnd).toBeGreaterThan(valueIndex)
  const openingTag = markup.slice(detailsStart, openingTagEnd + 1)
  expect(openingTag).toContain('data-evaluation-technical-details="true"')
  expect(openingTag).not.toMatch(/\sopen(?:=|\s|>)/)
  expect(markup.slice(openingTagEnd + 1, valueIndex)).toMatch(
    new RegExp(`<summary[^>]*>${summary}</summary>`),
  )
  expect(markup.slice(0, detailsStart)).not.toContain(value)
}

function analysisProvenance(metricID: string): EvaluationMetricAnalysisProvenance {
  return {
    contract_version: 'metric-analysis.v1',
    ...metricAnalysisSpecification(metricID),
    estimator_version: 'v1',
    missingness: 'fail_closed',
    exclusion_policy: 'exclude_unavailable_evidence',
    observed_exclusions: 0,
  }
}

function withRoutingPlan<T extends Parameters<typeof buildEvaluationRoutingRecipePlan>[0]>(
  mixture: T,
) {
  return {
    ...mixture,
    routing_recipe_plan: buildEvaluationRoutingRecipePlan(
      mixture,
      [{ id: 'domain:reasoning', value_kind: 'numeric' }],
      [
        {
          id: 'projection:oracle-probability',
          value_kind: 'probability',
          outcome_binding: 'selected_is_oracle',
        },
      ],
    ),
  }
}

const report: EvaluationReport = {
  schema_version: 'evaluation.v1',
  attestation_revision: EVALUATION_ATTESTATION_REVISION,
  run: {
    schema_version: 'evaluation.v1',
    id: 'run-current',
    client_request_id: 'run-current',
    name: 'Current evaluation',
    description: 'Server-attested diagnostic report',
    status: 'completed',
    mode: 'replay',
    evidence_level: 'E0',
    track_evidence_levels: { safety: 'E0' },
    target_id: 'target-a',
    change_profile: 'recipe',
    suite_ids: ['suite-a'],
    track_ids: ['safety'],
    sample_limit: 4,
    concurrency: 1,
    seed: 7,
    progress: { percent: 100, completed: 4, total: 4 },
    created_at: '2026-08-30T00:00:00Z',
    completed_at: '2026-08-30T00:01:00Z',
  },
  summary: {
    verdict: 'unavailable',
    quality_score: null,
    latency_p95_ms: null,
    runtime_cost: 0.01,
    capacity_tco: null,
    coverage: { evaluated: 4, total: 4, fraction: 1 },
    passed_gates: 0,
    failed_gates: 0,
    unavailable_gates: 0,
  },
  tracks: [
    {
      track_id: 'safety',
      status: 'completed',
      evidence_level: 'E0',
      summary: 'Diagnostic safety observations',
      coverage: { evaluated: 4, total: 4, fraction: 1 },
      metrics: [],
      gates: [],
    },
  ],
  metrics: [
    {
      id: 'safety.violation_rate',
      name: 'Safety violation rate',
      track_id: 'safety',
      value: 0,
      unit: 'violations/case',
      sample_count: 4,
      analysis_provenance: analysisProvenance('safety.violation_rate'),
    },
  ],
  method_reports: [],
  routing_recipe_report: null,
  gates: [],
  costs: {
    runtime: { amount: 0.01, currency: 'USD' },
    evaluation_overhead: { amount: 0, currency: 'USD' },
    capacity_tco: { amount: null, currency: 'USD' },
  },
  recommendations: [],
  provenance: {
    schema_version: 'evaluation.v1',
    generated_at: '2026-08-30T00:01:00Z',
    target_id: 'target-a',
    seed: 7,
  },
  artifacts: [],
}

describe('EvaluationReportView evidence language', () => {
  it('renders sealed R2 curves with the exact method readiness boundary', () => {
    const methodReport: EvaluationReport = {
      ...report,
      method_reports: [
        {
          method: {
            schema_version: 'evaluation-method.v2',
            id: 'r2.compound-model-budget.v2',
            version: 'evaluation-method.v2',
            status: 'exploratory-import',
            execution_owner: 'server',
            input_schema: 'r2-compound-input',
            export_schema: 'r2-compound-report',
            live_input_complete: false,
            live_grader: false,
            applicable_tracks: ['model_pool'],
            live_tracks: [],
            produced_metric_ids: ['r2.compound_model_budget.audc'],
            evidence_ceiling: 'E5',
            native_parity: 'source_qualified',
            required_artifact_ids: ['curves'],
            analysis_plan: {
              schema_version: 'evaluation-method.v2',
              id: 'r2-compound-case-action-budget',
              analysis_unit: 'case_action_budget',
              cluster_unit: 'case',
              slices: [{ schema_version: 'evaluation-method.v2', id: 'all' }],
              curve_domain: 'shared_budget',
              missingness: 'fail_closed',
            },
          },
          analysis_plan: {
            schema_version: 'evaluation-method.v2',
            id: 'r2-compound-case-action-budget',
            analysis_unit: 'case_action_budget',
            cluster_unit: 'case',
            slices: [{ schema_version: 'evaluation-method.v2', id: 'all' }],
            curve_domain: 'shared_budget',
            missingness: 'fail_closed',
          },
          action_refs: [{ schema_version: 'evaluation-method.v2', id: 'small' }],
          slice_refs: [{ schema_version: 'evaluation-method.v2', id: 'all' }],
          raw_shared_domain_curve: [
            { action: { id: 'small' }, budget: 100, mean_score: 0.5, case_count: 2 },
          ],
          audc: 50,
          nauc: 0.5,
          peak: 0.5,
          qnc: 0.5,
          missing_case_action_budget_cells: 0,
        },
      ],
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportView, {
        report: methodReport,
        diagnostics: EMPTY_DIAGNOSTICS,
      }),
    )
    expect(markup).toContain('Benchmark-specific analysis')
    expect(markup).toContain('Exploratory import only')
    expect(markup).toContain('Pinned-source normalized import · native parity not verified')
    expect(markup).toContain('Model pool benchmark analysis')
    const methodStart = markup.indexOf('Model pool benchmark analysis')
    const technicalStart = markup.indexOf('Technical details', methodStart)
    expect(technicalStart).toBeGreaterThan(methodStart)
    expect(markup.slice(methodStart, technicalStart)).not.toContain('r2.compound-model-budget.v2')
    expect(markup.slice(methodStart, technicalStart)).not.toContain('case_action_budget')
    expect(markup.slice(methodStart, technicalStart)).not.toContain('r2.compound_model_budget.audc')
    expect(markup.indexOf('r2.compound-model-budget.v2')).toBeGreaterThan(technicalStart)
  })

  it('explains one frozen Mixture across recipe, pool-arm, and joint outcomes', () => {
    const mixtureReport: EvaluationReport = {
      ...report,
      run: {
        ...report.run,
        mode: 'live',
        target_id: 'mom-balanced',
        track_ids: ['routing', 'model_pool', 'joint'],
        mixture: withRoutingPlan({
          id: 'mom-balanced',
          entrypoint_model: 'vllm-sr/auto',
          aliases: ['vllm-sr/auto'],
          recipe_name: 'balanced',
          recipe_description: 'Balanced routing.',
          recipe_digest: `sha256:${'1'.repeat(64)}`,
          pool_digest: `sha256:${'2'.repeat(64)}`,
          selector_policy_digest: `sha256:${'4'.repeat(64)}`,
          selector_digest: `sha256:${'5'.repeat(64)}`,
          adaptation_digest: `sha256:${'6'.repeat(64)}`,
          binding_digest: `sha256:${'3'.repeat(64)}`,
          model_arms: [
            {
              id: 'fast',
              model: 'models/fast',
              provider_model_id_digest: `sha256:${'4'.repeat(64)}`,
              input_cost_per_million_tokens_usd: 0.1,
              output_cost_per_million_tokens_usd: 0.2,
            },
            {
              id: 'strong',
              model: 'models/strong',
              provider_model_id_digest: `sha256:${'5'.repeat(64)}`,
              input_cost_per_million_tokens_usd: 0.4,
              output_cost_per_million_tokens_usd: 0.8,
            },
          ],
          support_models: [],
          fallback_arm_id: 'fast',
          decisions: [{ name: 'reasoning', algorithm: 'confidence', arm_ids: ['fast', 'strong'] }],
        }),
      },
      metrics: [
        {
          id: 'routing.accuracy',
          name: 'Routing accuracy',
          track_id: 'routing',
          value: 0.75,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('routing.accuracy'),
        },
        {
          id: 'model_pool.oracle_quality',
          name: 'Pool oracle quality',
          track_id: 'model_pool',
          value: 1,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('model_pool.oracle_quality'),
        },
        {
          id: 'model_pool.arm.fast.quality',
          name: 'Fast quality',
          track_id: 'model_pool',
          value: 0.5,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('model_pool.arm.fast.quality'),
        },
        {
          id: 'model_pool.arm.strong.quality',
          name: 'Strong quality',
          track_id: 'model_pool',
          value: 1,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('model_pool.arm.strong.quality'),
        },
        {
          id: 'joint.realized_quality',
          name: 'Realized quality',
          track_id: 'joint',
          value: 0.75,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('joint.realized_quality'),
        },
        {
          id: 'joint.oracle_regret',
          name: 'Oracle regret',
          track_id: 'joint',
          value: 0.25,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('joint.oracle_regret'),
        },
        {
          id: 'joint.normalized_regret',
          name: 'Normalized oracle regret',
          track_id: 'joint',
          value: 0.25,
          unit: 'fraction',
          analysis_provenance: analysisProvenance('joint.normalized_regret'),
        },
      ],
    }
    const plan = mixtureReport.run.mixture?.routing_recipe_plan
    if (!plan) throw new Error('test Mixture must bind a routing recipe plan')
    const unavailable = {
      available: false,
      reason: 'insufficient_complete_pool_outcomes',
      sample_count: 1,
    }
    mixtureReport.routing_recipe_report = {
      contract_version: 'routing-recipe-eval.v1',
      plan_digest: plan.plan_digest,
      e1: {
        expected_decisions: 4,
        observed_decisions: 4,
        signals: [
          {
            id: 'domain:reasoning',
            expected: 4,
            present: 3,
            missing: 1,
            error: 0,
            timeout: 0,
            latency: { available: true, sample_count: 3, p50_ms: 2, p95_ms: 4 },
          },
        ],
        projections: [
          {
            id: 'projection:oracle-probability',
            expected: 4,
            present: 3,
            missing: 0,
            error: 0,
            timeout: 1,
            latency: {
              available: false,
              reason: 'insufficient_latency_samples',
              sample_count: 1,
            },
          },
        ],
        eligibility_complete: 3,
        selected_feasible: 3,
      },
      e2: {
        projection_outcomes: [
          {
            projection_id: 'projection:oracle-probability',
            spearman: unavailable,
            brier: unavailable,
            ece_10: unavailable,
            reliability_bins: [],
          },
        ],
        top_k: plan.top_k.map((k) => ({ k, feasible_oracle_recall: unavailable })),
        oracle_regret: unavailable,
      },
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportView, {
        report: mixtureReport,
        diagnostics: EMPTY_DIAGNOSTICS,
      }),
    )

    expect(markup).toContain('Evaluated system boundary')
    expect(markup).toContain('01 · Routing recipe')
    expect(markup).toContain('02 · Model pool')
    expect(markup).toContain('03 · Routed system')
    expect(markup).toContain('Per-model outcome matrix')
    expect(markup).toContain('models/fast')
    expect(markup).toContain('models/strong')
    expect(markup).toContain('Fallback')
    expect(markup).toContain('Normalized quality gap')
    expect(markup).toContain('Read left to right')
    expect(markup).toContain('Routing behavior')
    expect(markup).toContain('Recipe and pool pinned')
    expect(markup).toContain('Saved with this run')
    expect(markup).toContain('Decision coverage')
    expect(markup).toContain('Eligibility complete')
    expect(markup).toContain('Selected feasible')
    expect(markup).toContain('Projection outcome calibration')
    expect(markup).toContain('Outcome estimate 1')
    expect(markup).not.toContain('Insufficient complete pool outcomes')
    expectInsideCollapsedDetails(markup, 'insufficient_latency_samples', 'Technical details')
    expect(markup).toContain('Quality gap to the best feasible model')
    expectInsideCollapsedDetails(markup, 'Oracle regret', 'Technical details')
    expect(markup.indexOf('Diagnostic result only')).toBeLessThan(
      markup.indexOf('Routing behavior'),
    )
    expectInsideCollapsedDetails(markup, plan.plan_digest, 'Reproducibility details')
    expectInsideCollapsedDetails(markup, plan.target_snapshot_digest, 'Reproducibility details')
    for (const digest of [
      mixtureReport.run.mixture?.recipe_digest,
      mixtureReport.run.mixture?.pool_digest,
      mixtureReport.run.mixture?.selector_digest,
      mixtureReport.run.mixture?.adaptation_digest,
      mixtureReport.run.mixture?.binding_digest,
    ]) {
      if (!digest) throw new Error('test Mixture must include every reproducibility identity')
      expectInsideCollapsedDetails(markup, digest, 'Reproducibility details')
    }
  })

  it('renders current attested E0 evidence without manufacturing promotion readiness', () => {
    const diagnostic = {
      ...report,
      summary: {
        ...report.summary,
        verdict: 'unavailable' as const,
        unavailable_gates: 1,
        coverage: { evaluated: 4, total: 6, fraction: 2 / 3, unavailable: 2 },
      },
      gates: [
        {
          id: 'G0',
          name: 'Reproducibility',
          description: 'Reproducibility evidence is incomplete.',
          disposition: 'required' as const,
          verdict: 'unavailable' as const,
          change_profile: 'recipe' as const,
          contract_version: 'evaluation-release-gates.v2' as const,
          evidence_refs: [],
          coverage: { evaluated: 4, total: 6, fraction: 2 / 3, unavailable: 2 },
        },
      ],
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportView, { report: diagnostic, diagnostics: EMPTY_DIAGNOSTICS }),
    )

    expect(markup).toContain('Diagnostic run — no release recommendation')
    expect(markup).toContain('Diagnostic result only')
    expect(markup).toContain('0/1 required checks passed')
    expect(markup).toContain('Incomplete')
    expect(markup.match(/2 not measured/g)).toHaveLength(3)
    expect(markup).toContain('Safety violation rate')
    expect(markup).toContain('Verified result · Diagnostic')
    expect(markup).toContain('Verified artifacts')
    expect(markup).toContain('Evaluation coverage')
    expect(markup).toContain('Recorded costs')
    expect(markup).not.toContain(EVALUATION_ATTESTATION_REVISION)
    expect(markup).not.toContain('E0')
    expect(markup).not.toContain('evaluation-release-gates.v2')
  })

  it('presents actionable next steps while retaining raw service notes in technical details', () => {
    const rawServiceNote =
      'Resolve G8 from the sealed all-arm receipt under evaluation-release-gates.v2.'
    const diagnostic = {
      ...report,
      gates: [
        {
          id: 'G8',
          name: 'Shadow / canary',
          disposition: 'required' as const,
          verdict: 'unavailable' as const,
          change_profile: 'recipe' as const,
          contract_version: 'evaluation-release-gates.v2' as const,
          evidence_refs: [],
        },
      ],
      recommendations: [rawServiceNote],
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportView, { report: diagnostic, diagnostics: EMPTY_DIAGNOSTICS }),
    )
    const rawNoteIndex = markup.indexOf(rawServiceNote)
    const technicalDetailsStart = markup.lastIndexOf('<details', rawNoteIndex)
    const technicalSummary = markup.indexOf('Technical details · 1', technicalDetailsStart)

    expect(markup).toContain('Next evaluation steps')
    expect(markup).toContain(
      'Shadow / canary: Run a guarded shadow or canary with exposure, stop, and rollback monitoring.',
    )
    expect(technicalDetailsStart).toBeGreaterThan(-1)
    expect(technicalSummary).toBeGreaterThan(technicalDetailsStart)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(rawServiceNote)
  })
})
