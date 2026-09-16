import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationCampaign } from '../../types/evaluationCampaign'
import type { EvaluationModelArm, EvaluationRun } from '../../types/evaluationPlane'
import EvaluationCampaignDecision from './EvaluationCampaignDecision'

const digest = (character: string) => `sha256:${character.repeat(64)}`

function campaignRun(id: string, modelArms: EvaluationModelArm[]): EvaluationRun {
  return {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name: id === 'baseline-live' ? 'Current pool' : 'Candidate pool',
    description: '',
    status: 'completed',
    mode: 'live',
    evidence_level: 'E5',
    track_evidence_levels: { model_pool: 'E5' },
    target_id: `${id}-target`,
    change_profile: 'model_pool',
    suite_ids: ['pool-suite'],
    track_ids: ['model_pool'],
    sample_limit: 20,
    concurrency: 1,
    seed: 42,
    progress: { percent: 100, completed: 20, total: 20 },
    created_at: '2026-08-30T00:00:00Z',
    completed_at: '2026-08-30T00:01:00Z',
    mixture: {
      id: 'mom-balanced',
      entrypoint_model: 'vllm-sr/auto',
      aliases: ['auto'],
      recipe_name: 'balanced',
      recipe_description: 'Balance quality and cost.',
      recipe_digest: digest('1'),
      pool_digest: digest('2'),
      selector_policy_digest: digest('3'),
      selector_digest: digest('4'),
      adaptation_digest: digest('5'),
      binding_digest: digest('6'),
      model_arms: modelArms,
      support_models: [],
      fallback_arm_id: 'arm-incumbent',
      decisions: [
        { name: 'balanced', algorithm: 'confidence', arm_ids: modelArms.map((arm) => arm.id) },
      ],
      routing_recipe_plan: {
        contract_version: 'routing-recipe-plan.v1',
        plan_digest: digest('7'),
        target_snapshot_digest: digest('8'),
        arm_ids: modelArms.map((arm) => arm.id),
        fallback_arm_id: 'arm-incumbent',
        signals: [],
        projections: [],
        top_k: [1],
      },
    },
  }
}

function productSurface(markup: string): string {
  return markup.replace(/\sdata-[\w-]+="[^"]*"/g, '')
}

function splitTechnicalDetails(markup: string): {
  decisionSurface: string
  technicalDetails: string
} {
  const disclosureStart = markup.indexOf('<details')
  if (disclosureStart < 0) throw new Error('Technical details disclosure was not rendered.')
  return {
    decisionSurface: markup.slice(0, disclosureStart),
    technicalDetails: markup.slice(disclosureStart),
  }
}

describe('EvaluationCampaignDecision model-pool reliability', () => {
  it('keeps release outcomes primary and reproducibility records in one closed disclosure', () => {
    const campaign = {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-campaign.v2',
      id: 'campaign-a',
      name: 'Model pool promotion',
      description: 'Paired replay and live pool treatment.',
      change_profile: 'model_pool',
      status: 'decided',
      gate_bindings: {
        g3_controlled_pair: {
          baseline_run_id: 'baseline-live',
          candidate_run_id: 'candidate-live',
        },
      },
      manifest_digest: digest('a'),
      created_at: '2026-08-30T00:00:00Z',
      decision: {
        schema_version: 'evaluation.v1',
        contract_version: 'evaluation-campaign.v2',
        attestation_revision: 'evaluation-server-attestation.v2',
        campaign_id: 'campaign-a',
        campaign_digest: digest('a'),
        verdict: 'fail',
        summary: 'server_attested_campaign_decision.v99 :: deny_on_g7',
        decision_digest: digest('b'),
        created_at: '2026-08-30T00:00:00Z',
        gates: [
          {
            id: 'G7',
            name: 'Cost / latency / capacity',
            disposition: 'required',
            verdict: 'fail',
            evidence_level: 'E5',
            source: 'campaign_slot:g7',
            evidence_refs: ['method:capacity.slo-envelope.v1'],
            observed: 1,
            sample_count: 20,
            rationale: 'campaign_slot:g7 reducer capacity.slo-envelope.v1 signed fail_closed',
          },
        ],
        evidence: [
          {
            slot_id: 'g3',
            gate_id: 'G3',
            binding_role: 'candidate',
            run_id: 'candidate-live',
            manifest_semantic_digest: digest('1'),
            manifest_artifact_digest: digest('2'),
            report_digest: digest('3'),
            private_receipt_digest: digest('4'),
            execution_attestation_digest: digest('5'),
          },
        ],
        recommendations: ['rerun_executor=server_brokered_live with receipt://internal-secret'],
        paired_live_evidence: {
          schema_version: 'evaluation.v1',
          contract_version: 'evaluation-campaign-paired-live.v3',
          controlled_pair_session_id: '10000000-0000-4000-8000-000000000001',
          controlled_pair_protocol: 'abba-interleaved.v1',
          baseline_run_id: 'baseline-live',
          candidate_run_id: 'candidate-live',
          candidate_subject_digest: digest('d'),
          digest: digest('c'),
          baseline_target_id: 'baseline--mom-balanced',
          candidate_target_id: 'candidate--mom-balanced',
          mixture_id: 'mom-balanced',
          recipe_name: 'balanced',
          track_ids: ['model_pool'],
          workload_snapshot_digest: digest('d'),
          benchmark_revisions: { 'pool-suite': digest('e') },
          bootstrap_samples: 1000,
          confidence_level: 0.95,
          promotion_policy: {
            candidate_normalized_regret_maximum: 0.25,
            paired_normalized_regret_margin: 0.05,
            minimum_no_information_frontier_lift: 0.05,
            minimum_joint_reliability: 0.8,
            maximum_all_arm_failure_rate: 0.2,
            minimum_candidate_arm_reliability: 0.8,
          },
          promotion_statistics: [
            {
              id: 'campaign.g3.all_arm_failure_rate',
              direction: 'lower_is_better',
              estimate: 0.25,
              confidence_level: 0.95,
              confidence_interval: [0.15, 0.3],
              threshold: { operator: '<=', value: 0.2, unit: 'fraction' },
              sample_count: 20,
              missing_cases: 0,
              verdict: 'fail',
            },
          ],
          seed: 42,
          baseline_manifest_digest: digest('f'),
          candidate_manifest_digest: digest('1'),
          baseline_execution_attestation_digest: digest('2'),
          candidate_execution_attestation_digest: digest('3'),
          baseline_policy_snapshot_digest: digest('4'),
          candidate_policy_snapshot_digest: digest('4'),
          baseline_binding_snapshot_digest: digest('5'),
          candidate_binding_snapshot_digest: digest('5'),
          baseline_pool_snapshot_digest: digest('6'),
          candidate_pool_snapshot_digest: digest('7'),
          baseline_environment_snapshot_digest: digest('8'),
          candidate_environment_snapshot_digest: digest('8'),
          baseline_backend_topology_digest: digest('9'),
          candidate_backend_topology_digest: digest('9'),
          baseline_code_revision: 'baseline-code',
          candidate_code_revision: 'candidate-code',
          statistics: [
            {
              id: 'campaign.g3.model_pool.worst_arm_reliability_non_inferiority',
              gate_id: 'G3',
              track_id: 'model_pool',
              analysis_unit: 'pool_worst_arm_reliability',
              direction: 'higher_is_better',
              margin: 0.02,
              baseline_value: 1,
              candidate_value: 0,
              delta: -1,
              confidence_level: 0.95,
              confidence_interval: [-1, -1],
              candidate_confidence_interval: [0, 0],
              sample_count: 20,
              missing_pairs: 0,
              verdict: 'fail',
            },
          ],
          model_pool_arm_reliability: [
            {
              arm_id: 'arm-incumbent',
              cohort: 'paired',
              direction: 'lower_is_better',
              margin: 0.02,
              baseline_failure_rate: 0,
              candidate_failure_rate: 0,
              delta: 0,
              confidence_level: 0.95,
              confidence_interval: [0, 0],
              candidate_confidence_interval: [0, 0],
              baseline_sample_count: 20,
              candidate_sample_count: 20,
              verdict: 'pass',
            },
            {
              arm_id: 'arm-new',
              cohort: 'candidate_only',
              direction: 'lower_is_better',
              margin: 0.2,
              candidate_failure_rate: 1,
              confidence_level: 0.95,
              confidence_interval: [],
              candidate_confidence_interval: [1, 1],
              baseline_sample_count: 0,
              candidate_sample_count: 20,
              verdict: 'fail',
            },
          ],
        },
      },
    } satisfies EvaluationCampaign

    const incumbent = {
      id: 'arm-incumbent',
      model: 'Balanced 8B',
      provider_model_id_digest: digest('8'),
      input_cost_per_million_tokens_usd: 0,
      output_cost_per_million_tokens_usd: 0,
    } satisfies EvaluationModelArm
    const runs = [
      campaignRun('baseline-live', [incumbent]),
      campaignRun('candidate-live', [
        incumbent,
        {
          id: 'arm-new',
          model: 'Reasoning 70B',
          provider_model_id_digest: digest('9'),
          input_cost_per_million_tokens_usd: 0,
          output_cost_per_million_tokens_usd: 0,
        },
      ]),
    ]
    const markup = renderToStaticMarkup(
      createElement(EvaluationCampaignDecision, {
        campaign,
        runs,
        onStartAnother: () => undefined,
      }),
    )

    const { decisionSurface, technicalDetails } = splitTechnicalDetails(markup)

    expect(decisionSurface).toContain(
      'Release is blocked because 1 required check did not meet the release criteria.',
    )
    expect(decisionSurface).toContain('1 blocked · 0 incomplete')
    expect(decisionSurface).toContain('The measured result did not satisfy this check.')
    expect(decisionSurface).toContain('What to do next')
    expect(decisionSurface).toContain(
      'Review the 1 blocked required check and address the measured regressions.',
    )
    expect(decisionSurface).toContain(
      'Run the affected evaluation again before making a release decision.',
    )
    expect(decisionSurface).toContain('Pool availability')
    expect(decisionSurface).toContain('≤ 20.0%')
    expect(decisionSurface).toContain('25.0%')
    expect(decisionSurface).toContain('[15.0%, 30.0%]')
    expect(decisionSurface).not.toContain('server_attested_campaign_decision.v99')
    expect(decisionSurface).not.toContain('capacity.slo-envelope.v1 signed fail_closed')
    expect(decisionSurface).not.toContain('receipt://internal-secret')
    expect(decisionSurface).not.toContain('&lt;= 0.2 fraction')
    expect(decisionSurface).toContain('Worst-model reliability')
    expect(decisionSurface).toContain('Per-model failure limits')
    expect(decisionSurface).toContain('Balanced 8B')
    expect(decisionSurface).toContain('Matched model')
    expect(decisionSurface).toContain('Added model')
    expect(decisionSurface).toContain('Reasoning 70B')
    expect(decisionSurface).not.toContain('arm-incumbent')
    expect(decisionSurface).not.toContain('arm-new')
    expect(decisionSurface).toContain('Not pairable')
    expect(decisionSurface).toContain(
      'A newly added model must also meet an absolute reliability limit',
    )
    expect(decisionSurface).toContain('Cost / latency / capacity')
    expect(decisionSurface).toContain('Bound evaluation run · End-to-end validation')
    expect(decisionSurface).not.toMatch(/data-evaluation-tag="true"[^>]*>Bound evaluation run/)
    expect(decisionSurface).toContain('data-check-id="G3"')
    expect(decisionSurface).toContain('data-evidence-level="E5"')
    expect(decisionSurface).not.toContain('baseline--mom-balanced')
    expect(decisionSurface).not.toContain('10000000-0000-4000-8000-000000000001')
    expect(decisionSurface).not.toContain('sha256:')
    expect(decisionSurface).not.toContain('Run records')
    expect(decisionSurface).not.toContain('Private execution receipt')

    expect(technicalDetails).toMatch(
      /^<details[^>]*data-evaluation-technical-details="true"[^>]*><summary[^>]*>/,
    )
    expect(technicalDetails.match(/^<details[^>]*>/)?.[0]).not.toContain('open')
    expect(technicalDetails).toContain('<strong>Technical details</strong>')
    expect(technicalDetails).toContain('How this decision was verified and can be reproduced')
    expect(technicalDetails).toContain('baseline--mom-balanced')
    expect(technicalDetails).toContain('candidate--mom-balanced')
    expect(technicalDetails).toContain('10000000-0000-4000-8000-000000000001')
    expect(technicalDetails).toContain('abba-interleaved.v1')
    expect(technicalDetails).toContain('arm-incumbent')
    expect(technicalDetails).toContain('arm-new')
    expect(technicalDetails).toContain(digest('d'))
    expect(technicalDetails).toContain('Run records')
    expect(technicalDetails).toContain('Release readiness · Candidate run')
    expect(technicalDetails).toContain('Manifest artifact')
    expect(technicalDetails).toContain('Private execution receipt')
    expect(technicalDetails).toContain('Server execution receipt')
    expect(technicalDetails).toContain('method:capacity.slo-envelope.v1')
    expect(technicalDetails).toContain('server_attested_campaign_decision.v99 :: deny_on_g7')
    expect(technicalDetails).toContain(
      'campaign_slot:g7 reducer capacity.slo-envelope.v1 signed fail_closed',
    )
    expect(technicalDetails).toContain(
      'rerun_executor=server_brokered_live with receipt://internal-secret',
    )

    const surface = productSurface(decisionSurface)
    expect(surface).not.toMatch(/\b(?:E[0-5]|G[0-9])\b/)
    expect(surface).not.toMatch(/\b[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.v\d+\b/i)
    expect(surface).not.toMatch(/\b(?:worst|all|paired|removed|added|frozen)-?arm\b/i)
  })
})
