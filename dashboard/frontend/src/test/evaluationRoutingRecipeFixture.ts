import type {
  EvaluationMixture,
  EvaluationRoutingRecipeInputSpec,
  EvaluationRoutingRecipePlan,
  EvaluationRoutingRecipeProjectionSpec,
} from '../types/evaluationPlane'
import {
  evaluationRoutingRecipePlanDigest,
  evaluationRoutingRecipeTargetDigest,
} from '../utils/evaluationRoutingRecipeContract'

/** Test-only authoring helper. Runtime plans remain exclusively server-owned. */
export function buildEvaluationRoutingRecipePlan(
  mixture: Pick<
    EvaluationMixture,
    | 'adaptation_digest'
    | 'binding_digest'
    | 'pool_digest'
    | 'recipe_digest'
    | 'selector_digest'
    | 'selector_policy_digest'
    | 'model_arms'
    | 'fallback_arm_id'
  >,
  signals: EvaluationRoutingRecipeInputSpec[] = [],
  projections: EvaluationRoutingRecipeProjectionSpec[] = [],
): EvaluationRoutingRecipePlan {
  const armCount = mixture.model_arms.length
  const compareID = (left: { id: string }, right: { id: string }) =>
    left.id < right.id ? -1 : left.id > right.id ? 1 : 0
  const body: Omit<EvaluationRoutingRecipePlan, 'plan_digest'> = {
    contract_version: 'routing-recipe-plan.v1',
    target_snapshot_digest: evaluationRoutingRecipeTargetDigest(mixture),
    arm_ids: mixture.model_arms.map((arm) => arm.id).sort(),
    ...(mixture.fallback_arm_id ? { fallback_arm_id: mixture.fallback_arm_id } : {}),
    signals: [...signals].sort(compareID),
    projections: [...projections].sort(compareID),
    top_k:
      armCount > 0
        ? [...new Set([1, Math.min(3, armCount), Math.min(5, armCount)])].sort(
            (left, right) => left - right,
          )
        : [],
  }
  return { ...body, plan_digest: evaluationRoutingRecipePlanDigest(body) }
}
