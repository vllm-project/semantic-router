import type {
  EvaluationMixture,
  EvaluationMixtureDecision,
  EvaluationModelArm,
  EvaluationSupportModel,
} from '../types/evaluationPlane'
import {
  type EvaluationRecord,
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isFiniteNumber,
  isNonEmptyText,
  isNonNegativeInteger,
  isTextArray,
} from './evaluationContractValidation'
import {
  isEvaluationRoutingRecipePlan,
  isUnavailableEvaluationCatalogRoutingRecipePlan,
} from './evaluationRoutingRecipeContract'

const PORTABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const SHA256 = /^sha256:[0-9a-f]{64}$/
const MODALITIES = new Set(['text', 'image', 'document', 'audio', 'video'])

function exactUniqueTextArray(value: unknown): value is string[] {
  return isTextArray(value) && new Set(value).size === value.length
}

function isModelArm(value: unknown): value is EvaluationModelArm {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'id',
      'model',
      'provider_model_id_digest',
      'input_cost_per_million_tokens_usd',
      'output_cost_per_million_tokens_usd',
      'capabilities',
      'modalities',
      'context_window_tokens',
      'parameter_size',
      'runtime_revision',
      'config_digest',
    ]) ||
    !isNonEmptyText(value.id) ||
    !PORTABLE_ID.test(value.id) ||
    !isNonEmptyText(value.model) ||
    typeof value.provider_model_id_digest !== 'string' ||
    !SHA256.test(value.provider_model_id_digest) ||
    !isFiniteNumber(value.input_cost_per_million_tokens_usd) ||
    value.input_cost_per_million_tokens_usd < 0 ||
    !isFiniteNumber(value.output_cost_per_million_tokens_usd) ||
    value.output_cost_per_million_tokens_usd < 0 ||
    (value.capabilities !== undefined && !exactUniqueTextArray(value.capabilities)) ||
    (value.modalities !== undefined &&
      (!exactUniqueTextArray(value.modalities) ||
        value.modalities.some((modality) => !MODALITIES.has(modality)))) ||
    (value.context_window_tokens !== undefined &&
      (!isNonNegativeInteger(value.context_window_tokens) || value.context_window_tokens < 1)) ||
    (value.parameter_size !== undefined && !isNonEmptyText(value.parameter_size)) ||
    (value.runtime_revision !== undefined && !isNonEmptyText(value.runtime_revision)) ||
    (value.config_digest !== undefined &&
      (typeof value.config_digest !== 'string' || !SHA256.test(value.config_digest)))
  ) {
    return false
  }
  return true
}

function isDecision(value: unknown, armIDs: Set<string>): value is EvaluationMixtureDecision {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, ['name', 'algorithm', 'arm_ids']) &&
    isNonEmptyText(value.name) &&
    isNonEmptyText(value.algorithm) &&
    exactUniqueTextArray(value.arm_ids) &&
    value.arm_ids.length > 0 &&
    value.arm_ids.every((armID) => armIDs.has(armID))
  )
}

function isSupportModel(value: unknown): value is EvaluationSupportModel {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, [
      'model',
      'provider_model_id_digest',
      'config_digest',
      'runtime_revision',
      'backend_topology_digest',
    ]) &&
    isNonEmptyText(value.model) &&
    value.model.length <= 512 &&
    typeof value.provider_model_id_digest === 'string' &&
    SHA256.test(value.provider_model_id_digest) &&
    typeof value.config_digest === 'string' &&
    SHA256.test(value.config_digest) &&
    (value.runtime_revision === undefined || isNonEmptyText(value.runtime_revision)) &&
    typeof value.backend_topology_digest === 'string' &&
    SHA256.test(value.backend_topology_digest)
  )
}

function isEvaluationMixtureContract(
  value: unknown,
  allowEmptyPool: boolean,
): value is EvaluationMixture {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'id',
      'entrypoint_model',
      'aliases',
      'recipe_name',
      'recipe_description',
      'recipe_digest',
      'pool_digest',
      'selector_policy_digest',
      'selector_digest',
      'adaptation_digest',
      'binding_digest',
      'model_arms',
      'support_models',
      'fallback_arm_id',
      'decisions',
      'routing_recipe_plan',
    ]) ||
    !isNonEmptyText(value.id) ||
    !PORTABLE_ID.test(value.id) ||
    !isNonEmptyText(value.entrypoint_model) ||
    !exactUniqueTextArray(value.aliases) ||
    !value.aliases.includes(value.entrypoint_model) ||
    !isNonEmptyText(value.recipe_name) ||
    typeof value.recipe_description !== 'string' ||
    typeof value.recipe_digest !== 'string' ||
    !SHA256.test(value.recipe_digest) ||
    typeof value.pool_digest !== 'string' ||
    !SHA256.test(value.pool_digest) ||
    typeof value.selector_policy_digest !== 'string' ||
    !SHA256.test(value.selector_policy_digest) ||
    typeof value.selector_digest !== 'string' ||
    !SHA256.test(value.selector_digest) ||
    typeof value.adaptation_digest !== 'string' ||
    !SHA256.test(value.adaptation_digest) ||
    typeof value.binding_digest !== 'string' ||
    !SHA256.test(value.binding_digest) ||
    !Array.isArray(value.model_arms) ||
    (!allowEmptyPool && value.model_arms.length === 0) ||
    value.model_arms.some((arm) => !isModelArm(arm)) ||
    !Array.isArray(value.support_models) ||
    value.support_models.some((model) => !isSupportModel(model)) ||
    !Array.isArray(value.decisions)
  ) {
    return false
  }
  const arms = value.model_arms as EvaluationRecord[]
  const armIDs = new Set(arms.map((arm) => arm.id as string))
  const models = new Set(arms.map((arm) => arm.model as string))
  const supportModels = (value.support_models as EvaluationSupportModel[]).map(
    (model) => model.model,
  )
  if (
    armIDs.size !== arms.length ||
    models.size !== arms.length ||
    new Set(supportModels).size !== supportModels.length ||
    supportModels.some((model) => models.has(model)) ||
    supportModels.some((model, index) => index > 0 && model <= supportModels[index - 1]) ||
    (value.fallback_arm_id !== undefined &&
      (typeof value.fallback_arm_id !== 'string' || !armIDs.has(value.fallback_arm_id))) ||
    value.decisions.some((decision) => !isDecision(decision, armIDs))
  ) {
    return false
  }
  const decisionNames = (value.decisions as EvaluationRecord[]).map(
    (decision) => decision.name as string,
  )
  if (new Set(decisionNames).size !== decisionNames.length) return false
  const mixture = value as unknown as Omit<EvaluationMixture, 'routing_recipe_plan'>
  return allowEmptyPool
    ? isUnavailableEvaluationCatalogRoutingRecipePlan(value.routing_recipe_plan, mixture)
    : isEvaluationRoutingRecipePlan(value.routing_recipe_plan, mixture)
}

export function isEvaluationMixture(value: unknown): value is EvaluationMixture {
  return isEvaluationMixtureContract(value, false)
}

export function isUnavailableEvaluationCatalogMixture(value: unknown): value is EvaluationMixture {
  return (
    isEvaluationRecord(value) &&
    Array.isArray(value.model_arms) &&
    value.model_arms.length === 0 &&
    isEvaluationMixtureContract(value, true)
  )
}
