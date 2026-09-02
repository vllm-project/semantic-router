import { describe, expect, it } from 'vitest'

import type { EvaluationMixture, EvaluationRun } from '../types/evaluationPlane'
import type { EvaluationRoutingRecipeReport } from '../types/evaluationRoutingRecipeReport'
import { reportFor } from '../test/evaluationPlaneApiFixture'
import { buildEvaluationRoutingRecipePlan } from '../test/evaluationRoutingRecipeFixture'
import { decodeEvaluationReport } from './evaluationReportContract'
import {
  decodeEvaluationRoutingRecipeReport,
  isEvaluationRoutingRecipePlan,
} from './evaluationRoutingRecipeContract'

const mixtureBase: Omit<EvaluationMixture, 'routing_recipe_plan'> = {
  id: 'mom-c0905088ba5b80677849d4824f21b8b075ab49c6a7f58b5b84815ccd69f3b0a5',
  entrypoint_model: 'vllm-sr/mom-primary',
  aliases: ['vllm-sr/mom-primary', 'vllm-sr/mom-alias'],
  recipe_name: 'balanced',
  recipe_description: 'Balanced test mixture',
  recipe_digest: `sha256:${'1'.repeat(64)}`,
  pool_digest: 'sha256:1ddd0ce32961e4b4c7e8de072d1b6a74b41db254a2b8295d3df1619d903becfa',
  selector_policy_digest: `sha256:${'4'.repeat(64)}`,
  selector_digest: 'sha256:e2544ed8aba06b8de5d05234b16dfd3b34c64b563b125ddf732ae6cdf2b28131',
  adaptation_digest: `sha256:${'5'.repeat(64)}`,
  binding_digest: `sha256:${'3'.repeat(64)}`,
  model_arms: [
    {
      id: 'fast',
      model: 'org/fast',
      provider_model_id_digest: `sha256:${'b'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 0,
      output_cost_per_million_tokens_usd: 1,
    },
    {
      id: 'vision',
      model: 'org/vision',
      provider_model_id_digest: `sha256:${'c'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 1,
      output_cost_per_million_tokens_usd: 2,
    },
  ],
  support_models: [],
  fallback_arm_id: 'fast',
  decisions: [{ name: 'default', algorithm: 'static', arm_ids: ['fast', 'vision'] }],
}

const plan = buildEvaluationRoutingRecipePlan(
  mixtureBase,
  [
    { id: 'complexity:complexity', value_kind: 'numeric' },
    { id: 'modality:modality', value_kind: 'numeric' },
  ],
  [
    {
      id: 'projection:route-confidence',
      value_kind: 'probability',
      outcome_binding: 'selected_is_oracle',
    },
  ],
)
const mixture: EvaluationMixture = { ...mixtureBase, routing_recipe_plan: plan }

const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: '11111111-1111-4111-8111-111111111111',
  client_request_id: '11111111-1111-4111-8111-111111111111',
  name: 'Routing recipe evaluation',
  description: '',
  status: 'completed',
  mode: 'live',
  evidence_level: 'E3',
  track_evidence_levels: { routing: 'E3' },
  target_id: mixture.id,
  mixture,
  change_profile: 'recipe',
  suite_ids: ['live-mom-core'],
  track_ids: ['routing'],
  sample_limit: 4,
  concurrency: 1,
  seed: 42,
  progress: { percent: 100, completed: 1, total: 1 },
  created_at: '2026-08-31T00:00:00Z',
  completed_at: '2026-08-31T00:01:00Z',
}

function inputAvailability(id: string) {
  return {
    id,
    expected: 4,
    present: 3,
    missing: 1,
    error: 0,
    timeout: 0,
    latency: {
      available: true,
      sample_count: 3,
      p50_ms: 2,
      p95_ms: 4,
    },
  }
}

function routingReport(): EvaluationRoutingRecipeReport {
  return {
    contract_version: 'routing-recipe-eval.v1',
    plan_digest: plan.plan_digest,
    e1: {
      expected_decisions: 4,
      observed_decisions: 4,
      signals: plan.signals.map((item) => inputAvailability(item.id)),
      projections: plan.projections.map((item) => inputAvailability(item.id)),
      eligibility_complete: 3,
      selected_feasible: 3,
    },
    e2: {
      projection_outcomes: [
        {
          projection_id: 'projection:route-confidence',
          spearman: { available: true, value: 0.7, sample_count: 4 },
          brier: { available: true, value: 0.1, sample_count: 4 },
          ece_10: { available: true, value: 0.08, sample_count: 4 },
          reliability_bins: Array.from({ length: 10 }, (_, index) => ({
            lower: index / 10,
            upper: (index + 1) / 10,
            count: index < 4 ? 1 : 0,
            ...(index < 4
              ? { mean_prediction: index / 10 + 0.05, observed_frequency: index % 2 }
              : {}),
          })),
        },
      ],
      top_k: [
        { k: 1, feasible_oracle_recall: { available: true, value: 0.75, sample_count: 4 } },
        { k: 2, feasible_oracle_recall: { available: true, value: 1, sample_count: 4 } },
      ],
      oracle_regret: { available: true, value: 0.12, sample_count: 4 },
    },
  }
}

function publishedReport(routingRecipeReport: unknown, reportRun: EvaluationRun = run) {
  return {
    ...reportFor(reportRun),
    routing_recipe_report: routingRecipeReport,
  }
}

describe('routing recipe plan contract', () => {
  it('matches the cross-language canonical target and plan digests', () => {
    expect(plan.target_snapshot_digest).toBe(
      'sha256:6d670e837cd9a9bd65f73e1cede60230388e9c1b0e0acc6640730fd3c3cbc958',
    )
    expect(plan.plan_digest).toBe(
      'sha256:7cb3a0b6d29e82c0e4608b186d9a1d1fef90a69c52e143e81364043d26cbfa95',
    )
    expect(isEvaluationRoutingRecipePlan(plan, mixtureBase)).toBe(true)
    expect(
      isEvaluationRoutingRecipePlan(
        {
          ...plan,
          arm_ids: [...plan.arm_ids].reverse(),
          signals: [...plan.signals].reverse(),
        },
        mixtureBase,
      ),
    ).toBe(true)
  })

  it('rejects unknown fields, unbound digests, foreign arms, invalid inputs, and limits', () => {
    expect(isEvaluationRoutingRecipePlan({ ...plan, future: true }, mixtureBase)).toBe(false)
    expect(
      isEvaluationRoutingRecipePlan(
        { ...plan, target_snapshot_digest: `sha256:${'0'.repeat(64)}` },
        mixtureBase,
      ),
    ).toBe(false)
    expect(
      isEvaluationRoutingRecipePlan({ ...plan, arm_ids: ['fast', 'foreign'] }, mixtureBase),
    ).toBe(false)
    expect(
      isEvaluationRoutingRecipePlan(
        { ...plan, signals: [{ id: 'Projection:future', value_kind: 'numeric' }] },
        mixtureBase,
      ),
    ).toBe(false)
    expect(
      isEvaluationRoutingRecipePlan(
        { ...plan, projections: Array.from({ length: 129 }, () => plan.projections[0]) },
        mixtureBase,
      ),
    ).toBe(false)
  })

  it('accepts every value and outcome kind in the versioned plan contract', () => {
    const generalPlan = buildEvaluationRoutingRecipePlan(
      mixtureBase,
      [{ id: 'metadata:optional', value_kind: 'none' }],
      [
        {
          id: 'projection:pool-quality',
          value_kind: 'numeric',
          outcome_binding: 'selected_pool_quality',
        },
      ],
    )
    expect(isEvaluationRoutingRecipePlan(generalPlan, mixtureBase)).toBe(true)
    expect(
      isEvaluationRoutingRecipePlan(
        {
          ...generalPlan,
          projections: [
            { ...generalPlan.projections[0], outcome_binding: 'future_outcome_binding' },
          ],
        },
        mixtureBase,
      ),
    ).toBe(false)
  })

  it('requires the Mixture producer top-k schedule rather than an arbitrary increasing subset', () => {
    const incomplete = { ...plan, top_k: [1] }
    expect(isEvaluationRoutingRecipePlan(incomplete, mixtureBase)).toBe(false)
    expect(plan.top_k).toEqual([1, 2])
  })
})

describe('routing recipe report contract', () => {
  it('requires the server-owned aggregate exactly for live Mixture routing reports', () => {
    const report = routingReport()
    expect(decodeEvaluationRoutingRecipeReport(report, run)).toEqual(report)
    expect(() => decodeEvaluationRoutingRecipeReport(undefined, run)).toThrow(/server-owned/)
    expect(decodeEvaluationRoutingRecipeReport(null, { ...run, mode: 'replay' })).toBeNull()
    expect(() =>
      decodeEvaluationRoutingRecipeReport(undefined, { ...run, track_ids: ['model_pool'] }),
    ).toThrow(/explicit null/)
    expect(() => decodeEvaluationRoutingRecipeReport(report, { ...run, mode: 'replay' })).toThrow(
      /explicit null/,
    )
    expect(decodeEvaluationReport(publishedReport(report), run.id).routing_recipe_report).toEqual(
      report,
    )
    expect(() =>
      decodeEvaluationReport(
        (() => {
          const payload = publishedReport(report)
          delete (payload as { routing_recipe_report?: unknown }).routing_recipe_report
          return payload
        })(),
        run.id,
      ),
    ).toThrow(/server-owned/)
  })

  it('fails closed on nested mutations, incomplete matrices, and detached plan identity', () => {
    const report = routingReport()
    expect(() =>
      decodeEvaluationRoutingRecipeReport(
        { ...report, e1: { ...report.e1, selected_feasible: 5 } },
        run,
      ),
    ).toThrow(/server-owned/)
    expect(() =>
      decodeEvaluationRoutingRecipeReport(
        {
          ...report,
          e2: {
            ...report.e2,
            projection_outcomes: [{ ...report.e2.projection_outcomes[0], worker_score: 1 }],
          },
        },
        run,
      ),
    ).toThrow(/server-owned/)
    expect(() =>
      decodeEvaluationRoutingRecipeReport(
        { ...report, plan_digest: `sha256:${'0'.repeat(64)}` },
        run,
      ),
    ).toThrow(/server-owned/)
    expect(() =>
      decodeEvaluationRoutingRecipeReport(
        {
          ...report,
          e2: {
            ...report.e2,
            top_k: report.e2.top_k.slice(0, 1),
          },
        },
        run,
      ),
    ).toThrow(/server-owned/)
  })

  it('preserves explicit unavailable reasons without accepting fabricated reliability bins', () => {
    const report = routingReport()
    const unavailable = {
      available: false,
      reason: 'insufficient_complete_pool_outcomes',
      sample_count: 1,
    }
    report.e2.projection_outcomes[0] = {
      projection_id: 'projection:route-confidence',
      spearman: unavailable,
      brier: unavailable,
      ece_10: unavailable,
      reliability_bins: [],
    }
    report.e2.top_k = report.e2.top_k.map(({ k }) => ({
      k,
      feasible_oracle_recall: unavailable,
    }))
    report.e2.oracle_regret = unavailable
    expect(decodeEvaluationRoutingRecipeReport(report, run)).toEqual(report)
    report.e2.projection_outcomes[0].reliability_bins.push({
      lower: 0,
      upper: 0.1,
      count: 1,
      mean_prediction: 0.5,
      observed_frequency: 1,
    })
    expect(() => decodeEvaluationRoutingRecipeReport(report, run)).toThrow(/server-owned/)
  })
})
