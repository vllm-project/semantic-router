import type { EvaluationMixture } from '../../../src/types/evaluationPlane'

export function evaluationRunID(serial: number): string {
  if (!Number.isSafeInteger(serial) || serial < 0 || serial > 999_999_999_999) {
    throw new Error('Evaluation test run serial is outside the canonical UUID fixture range.')
  }
  return `00000000-0000-4000-8000-${String(serial).padStart(12, '0')}`
}

export const EVALUATION_RUN_IDS = {
  candidate: evaluationRunID(1),
  baseline: evaluationRunID(2),
  unpaired: evaluationRunID(3),
  live: evaluationRunID(4),
  failed: evaluationRunID(5),
  cancelled: evaluationRunID(6),
  olderBaseline: evaluationRunID(7),
  olderCandidate: evaluationRunID(8),
  secondBaseline: evaluationRunID(9),
  secondCandidate: evaluationRunID(10),
  campaign: evaluationRunID(11),
  candidateLive: evaluationRunID(12),
  baselineLive: evaluationRunID(13),
  candidateConfirmation: evaluationRunID(14),
  campaignG2: evaluationRunID(15),
  campaignG4: evaluationRunID(16),
  campaignG5Reference: evaluationRunID(17),
  campaignG5Live: evaluationRunID(18),
  campaignG7: evaluationRunID(19),
} as const

const EVALUATION_MOM_ID = 'mom-37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f'
export const EVALUATION_BASELINE_MOM_TARGET_ID = `baseline--${EVALUATION_MOM_ID}`
export const EVALUATION_MOM_TARGET_ID = `candidate--${EVALUATION_MOM_ID}`

const EVALUATION_MOM_BASE = {
  id: EVALUATION_MOM_ID,
  entrypoint_model: 'test-mom',
  aliases: ['test-mom'],
  recipe_name: 'default',
  recipe_description: 'Recipe-scoped Mixture-of-Models evaluation target.',
  recipe_digest: `sha256:${'1'.repeat(64)}`,
  pool_digest: `sha256:${'2'.repeat(64)}`,
  selector_policy_digest: `sha256:${'4'.repeat(64)}`,
  selector_digest: `sha256:${'5'.repeat(64)}`,
  adaptation_digest: `sha256:${'6'.repeat(64)}`,
  binding_digest: `sha256:${'3'.repeat(64)}`,
  model_arms: [
    {
      id: 'arm-fast',
      model: 'model-fast',
      provider_model_id_digest: `sha256:${'4'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 0.1,
      output_cost_per_million_tokens_usd: 0.2,
      capabilities: ['chat'],
      modalities: ['text'],
      config_digest: `sha256:${'6'.repeat(64)}`,
    },
    {
      id: 'arm-strong',
      model: 'model-strong',
      provider_model_id_digest: `sha256:${'5'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 0.4,
      output_cost_per_million_tokens_usd: 0.8,
      capabilities: ['chat', 'vision'],
      modalities: ['text', 'image'],
      config_digest: `sha256:${'7'.repeat(64)}`,
    },
  ],
  support_models: [],
  fallback_arm_id: 'arm-fast',
  decisions: [{ name: 'route', algorithm: 'static', arm_ids: ['arm-fast', 'arm-strong'] }],
}
export const EVALUATION_MOM: EvaluationMixture = {
  ...EVALUATION_MOM_BASE,
  // Frozen server wire fixture: keeping these hashes independent from the
  // browser verifier makes E2E fail if its canonicalization drifts.
  routing_recipe_plan: {
    contract_version: 'routing-recipe-plan.v1',
    plan_digest:
      'sha256:60785679d820196a5b9a5ce816b763a6f856f7823fbf0f4571290d82b0312a0a',
    target_snapshot_digest:
      'sha256:3947c7529f16542cddadbdc3a1dc98cae2303600bf75cfcc962a84abb84c0352',
    arm_ids: ['arm-fast', 'arm-strong'],
    fallback_arm_id: 'arm-fast',
    signals: [{ id: 'domain:reasoning', value_kind: 'numeric' }],
    projections: [
      {
        id: 'projection:oracle-probability',
        value_kind: 'probability',
        outcome_binding: 'selected_is_oracle',
      },
    ],
    top_k: [1, 2],
  },
}
