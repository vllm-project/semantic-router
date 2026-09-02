import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationComparison } from '../../types/evaluationComparison'
import type { EvaluationMetricAnalysisProvenance } from '../../types/evaluationReport'
import { EVALUATION_ATTESTATION_REVISION } from '../../types/evaluationPlane'
import { metricAnalysisSpecification } from '../../test/evaluationMetricAnalysisFixture'
import { buildEvaluationRoutingRecipePlan } from '../../test/evaluationRoutingRecipeFixture'
import EvaluationCompare from './EvaluationCompare'

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

const mixtureBase = {
  id: 'mom',
  entrypoint_model: 'vllm-sr/auto',
  aliases: ['vllm-sr/auto'],
  recipe_name: 'balanced',
  recipe_description: '',
  recipe_digest: `sha256:${'1'.repeat(64)}`,
  pool_digest: `sha256:${'2'.repeat(64)}`,
  selector_policy_digest: `sha256:${'3'.repeat(64)}`,
  selector_digest: `sha256:${'4'.repeat(64)}`,
  adaptation_digest: `sha256:${'5'.repeat(64)}`,
  binding_digest: `sha256:${'6'.repeat(64)}`,
  model_arms: [
    {
      id: 'arm',
      model: 'model',
      provider_model_id_digest: `sha256:${'7'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 0,
      output_cost_per_million_tokens_usd: 0,
    },
  ],
  support_models: [],
  fallback_arm_id: 'arm',
  decisions: [{ name: 'route', algorithm: 'static', arm_ids: ['arm'] }],
}

const evaluationMixture = {
  ...mixtureBase,
  routing_recipe_plan: buildEvaluationRoutingRecipePlan(mixtureBase),
}

const baseline: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: 'baseline',
  client_request_id: 'baseline',
  name: 'Baseline',
  description: '',
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
}

const candidate: EvaluationRun = {
  ...baseline,
  id: 'candidate',
  client_request_id: 'candidate',
  name: 'Candidate',
  baseline_run_id: baseline.id,
}

const comparison: EvaluationComparison = {
  schema_version: 'evaluation.v1',
  attestation_revision: EVALUATION_ATTESTATION_REVISION,
  baseline_run_id: baseline.id,
  candidate_run_id: candidate.id,
  verdict: 'unavailable',
  summary: 'Diagnostic deltas are favorable, but E0 evidence cannot support promotion.',
  metrics: [
    {
      id: 'safety.violation_rate',
      name: 'Safety violation rate',
      track_id: 'safety',
      value: 0,
      unit: 'violations/case',
      analysis_provenance: analysisProvenance('safety.violation_rate'),
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
      baseline_value: 0.2,
      candidate_value: 0.1,
      delta: -0.1,
      confidence_level: 0.95,
      delta_confidence_interval: [],
      candidate_confidence_interval: [],
      sample_count: 4,
      verdict: 'unavailable',
    },
  ],
  gates: [],
  recommendations: ['Resolve G8 with evaluation-release-gates.v2 before promotion.'],
  created_at: '2026-08-30T00:02:00Z',
}

function renderComparison(
  value: EvaluationComparison,
  runs: EvaluationRun[] = [candidate, baseline],
  baselineID = baseline.id,
  candidateID = candidate.id,
): string {
  return renderToStaticMarkup(
    createElement(EvaluationCompare, {
      runs,
      baselineID,
      candidateID,
      comparison: value,
      runLedgerAvailable: true,
      runLedgerComplete: true,
      totalRuns: 2,
      hasMoreRuns: false,
      loadingMoreRuns: false,
      resourcesLoading: false,
      resourcesError: null,
      loading: false,
      error: null,
      onPairChange: () => undefined,
      onCompare: () => undefined,
      onLoadMoreRuns: () => undefined,
      onRetryResources: () => undefined,
    }),
  )
}

function visibleText(markup: string): string {
  return markup
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function expectProductLanguage(markup: string): void {
  const defaultSurface = markup.replace(
    /<details[^>]*data-evaluation-technical-details="true"[^>]*>[\s\S]*?<\/details>/g,
    '',
  )
  const text = visibleText(defaultSurface)
  expect(text).not.toMatch(/\bE[0-5]\b/)
  expect(text).not.toMatch(/\bG[0-9]\b/)
  expect(text).not.toContain('evaluation.v1')
  expect(text).not.toContain('metric-analysis.v1')
}

describe('EvaluationCompare evidence labels', () => {
  it('keeps service failures actionable while retaining raw diagnostics in closed details', () => {
    const rawError = 'evaluation.v1 decoder failed at worker://private-stack G8'
    const markup = renderToStaticMarkup(
      createElement(EvaluationCompare, {
        runs: [candidate, baseline],
        baselineID: baseline.id,
        candidateID: candidate.id,
        comparison: null,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        totalRuns: 2,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        resourcesLoading: false,
        resourcesError: rawError,
        loading: false,
        error: rawError,
        onPairChange: () => undefined,
        onCompare: () => undefined,
        onLoadMoreRuns: () => undefined,
        onRetryResources: () => undefined,
      }),
    )

    expect(markup).toContain('Run details could not be loaded.')
    expect(markup).toContain('The comparison could not be calculated.')
    expect(markup.match(/data-evaluation-technical-details="true"/g)).toHaveLength(2)
    expect(markup).toContain(rawError)
    expectProductLanguage(markup)
  })

  it('keeps the empty comparison workspace focused on one next action', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationCompare, {
        runs: [baseline],
        baselineID: '',
        candidateID: '',
        comparison: null,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        totalRuns: 1,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        resourcesLoading: false,
        resourcesError: null,
        loading: false,
        error: null,
        onPairChange: () => undefined,
        onCompare: () => undefined,
        onLoadMoreRuns: () => undefined,
        onRetryResources: () => undefined,
        onCreateRun: () => undefined,
      }),
    )

    expect(markup).toContain('No comparable candidate exists')
    expect(markup).toContain('Create candidate run')
    expect(markup).not.toContain('<select')
    expect(markup.match(new RegExp(`<${'button'}`, 'g'))).toHaveLength(1)
    expectProductLanguage(markup)
  })

  it('renders the current attested comparison contract', () => {
    const markup = renderComparison(comparison)

    expect(markup).toContain('Diagnostic comparison · not a release decision')
    expect(markup).toContain('More matched results are needed before drawing a conclusion.')
    expect(markup).toContain('Paired scientific statistics')
    expect(markup).toContain('Normalized quality gap')
    expect(markup).toContain('Normalized gap to the best model')
    expect(markup).toContain('Needs at least 20 independent case units; observed 4.')
    expect(markup).toContain('Not estimable')
    expect(markup.match(/<select/g)).toHaveLength(1)
    expect(markup).toContain('Next comparison steps')
    expect(markup).toContain('Technical details')
    expect(markup).toContain('Diagnostic deltas are favorable, but E0 evidence cannot support')
    expect(markup).toContain('Resolve G8 with evaluation-release-gates.v2 before promotion.')
    expect(markup).toContain('Diagnostic')
    expectProductLanguage(markup)
  })

  it('states that server-owned Routing Recipe aggregates are not generic comparison metrics', () => {
    const routingBaseline: EvaluationRun = {
      ...baseline,
      id: 'routing-baseline',
      client_request_id: 'routing-baseline',
      mode: 'live',
      track_ids: ['routing'],
      track_evidence_levels: { routing: 'E3' },
      evidence_level: 'E3',
      mixture: evaluationMixture,
    }
    const routingCandidate: EvaluationRun = {
      ...routingBaseline,
      id: 'routing-candidate',
      client_request_id: 'routing-candidate',
      baseline_run_id: routingBaseline.id,
    }
    const markup = renderComparison(
      { ...comparison, baseline_run_id: routingBaseline.id, candidate_run_id: routingCandidate.id },
      [routingCandidate, routingBaseline],
      routingBaseline.id,
      routingCandidate.id,
    )
    expect(markup).toContain('Routing details stay in each run report.')
    expect(markup).toContain(
      'decision coverage, calibration, top-choice coverage, and the quality gap to the best model',
    )
    expectProductLanguage(markup)

    const replayBaseline = { ...routingBaseline, mode: 'replay' as const }
    const replayCandidate = {
      ...routingCandidate,
      mode: 'replay' as const,
      baseline_run_id: replayBaseline.id,
    }
    expect(
      renderComparison(
        {
          ...comparison,
          baseline_run_id: replayBaseline.id,
          candidate_run_id: replayCandidate.id,
        },
        [replayCandidate, replayBaseline],
        replayBaseline.id,
        replayCandidate.id,
      ),
    ).not.toContain('Routing details stay in each run report.')
  })

  it('uses readable cohort labels instead of raw profile and target identifiers', () => {
    const cohortBaseline: EvaluationRun = {
      ...baseline,
      id: 'baseline-internal-target',
      target_id: 'internal-baseline-deployment',
      change_profile: 'agent_multimodal',
    }
    const cohortCandidate: EvaluationRun = {
      ...candidate,
      id: 'candidate-internal-target',
      target_id: 'internal-candidate-deployment',
      baseline_run_id: cohortBaseline.id,
      change_profile: 'agent_multimodal',
      mixture: evaluationMixture,
    }

    const markup = renderComparison(
      {
        ...comparison,
        baseline_run_id: cohortBaseline.id,
        candidate_run_id: cohortCandidate.id,
      },
      [cohortCandidate, cohortBaseline],
      cohortBaseline.id,
      cohortCandidate.id,
    )

    expect(markup).toContain('Agents and multimodal')
    expect(markup).toContain('vllm-sr/auto')
    expect(markup).not.toContain('agent_multimodal')
    expect(markup).not.toContain('internal-candidate-deployment')
    expect(markup).not.toContain('internal-baseline-deployment')
    expectProductLanguage(markup)
  })

  it('offers an attested controlled-pair candidate despite its intentional target treatment', () => {
    const controlledBaseline: EvaluationRun = {
      ...baseline,
      id: '00000000-0000-4000-8000-000000000101',
      client_request_id: '00000000-0000-4000-8000-000000000101',
      mode: 'live',
      target_id: 'baseline-deployment',
      mixture: evaluationMixture,
      controlled_pair: { pair_id: 'pair', role: 'baseline' },
    }
    const controlledCandidate: EvaluationRun = {
      ...candidate,
      id: '00000000-0000-4000-8000-000000000102',
      client_request_id: '00000000-0000-4000-8000-000000000102',
      mode: 'live',
      target_id: 'candidate-deployment',
      mixture: evaluationMixture,
      baseline_run_id: controlledBaseline.id,
      controlled_pair: { pair_id: 'pair', role: 'candidate' },
    }

    const markup = renderComparison(
      {
        ...comparison,
        baseline_run_id: controlledBaseline.id,
        candidate_run_id: controlledCandidate.id,
      },
      [controlledCandidate, controlledBaseline],
      controlledBaseline.id,
      controlledCandidate.id,
    )

    expect(markup).toContain('>Candidate</option>')
    expect(markup).not.toContain('No comparable candidate exists')
    expect(markup).not.toContain('Cohort mismatch')
    const compareButton = markup.match(
      new RegExp(`<${'button'}[^>]*>Compare results</${'button'}>`),
    )?.[0]
    expect(compareButton).toBeDefined()
    expect(compareButton).not.toContain('disabled')
    expectProductLanguage(markup)
  })
})
