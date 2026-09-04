import type {
  EvaluationMixture,
  EvaluationRoutingRecipeInputSpec,
  EvaluationRoutingRecipePlan,
  EvaluationRoutingRecipeProjectionSpec,
  EvaluationRun,
} from '../types/evaluationPlane'
import type {
  EvaluationRoutingRecipeInputAvailabilityReport,
  EvaluationRoutingRecipeLatencyReport,
  EvaluationRoutingRecipeMetricAvailability,
  EvaluationRoutingRecipeProjectionOutcomeReport,
  EvaluationRoutingRecipeReliabilityBin,
  EvaluationRoutingRecipeReport,
  EvaluationRoutingRecipeTopKReport,
} from '../types/evaluationRoutingRecipeReport'
import {
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isFiniteNumber,
  isNonNegativeInteger,
} from './evaluationContractValidation'

const PLAN_VERSION = 'routing-recipe-plan.v1'
const REPORT_VERSION = 'routing-recipe-eval.v1'
const DIGEST = /^sha256:[0-9a-f]{64}$/
const PORTABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const MAX_ARMS = 64
const MAX_INPUTS = 128
const MAX_CASES = 100_000
const SIGNAL_TYPES = new Set([
  'authz',
  'classifier',
  'complexity',
  'context',
  'conversation',
  'domain',
  'embedding',
  'event',
  'fact_check',
  'jailbreak',
  'kb',
  'keyword',
  'language',
  'metadata',
  'modality',
  'pii',
  'preference',
  'reask',
  'structure',
  'user_feedback',
])

function rotateRight(value: number, distance: number): number {
  return (value >>> distance) | (value << (32 - distance))
}

/** Small synchronous SHA-256 used only to verify immutable browser contracts. */
function sha256(value: string): string {
  const source = new TextEncoder().encode(value)
  const bitLength = source.length * 8
  const paddedLength = Math.ceil((source.length + 9) / 64) * 64
  const padded = new Uint8Array(paddedLength)
  padded.set(source)
  padded[source.length] = 0x80
  const view = new DataView(padded.buffer)
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 2 ** 32), false)
  view.setUint32(paddedLength - 4, bitLength >>> 0, false)

  const constants = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]
  const hash = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]
  const words = new Uint32Array(64)
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4, false)
    }
    for (let index = 16; index < 64; index += 1) {
      const s0 =
        rotateRight(words[index - 15], 7) ^
        rotateRight(words[index - 15], 18) ^
        (words[index - 15] >>> 3)
      const s1 =
        rotateRight(words[index - 2], 17) ^
        rotateRight(words[index - 2], 19) ^
        (words[index - 2] >>> 10)
      words[index] = (words[index - 16] + s0 + words[index - 7] + s1) >>> 0
    }
    let [a, b, c, d, e, f, g, h] = hash
    for (let index = 0; index < 64; index += 1) {
      const sigma1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25)
      const choice = (e & f) ^ (~e & g)
      const first = (h + sigma1 + choice + constants[index] + words[index]) >>> 0
      const sigma0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22)
      const majority = (a & b) ^ (a & c) ^ (b & c)
      const second = (sigma0 + majority) >>> 0
      h = g
      g = f
      f = e
      e = (d + first) >>> 0
      d = c
      c = b
      b = a
      a = (first + second) >>> 0
    }
    hash[0] = (hash[0] + a) >>> 0
    hash[1] = (hash[1] + b) >>> 0
    hash[2] = (hash[2] + c) >>> 0
    hash[3] = (hash[3] + d) >>> 0
    hash[4] = (hash[4] + e) >>> 0
    hash[5] = (hash[5] + f) >>> 0
    hash[6] = (hash[6] + g) >>> 0
    hash[7] = (hash[7] + h) >>> 0
  }
  return hash.map((word) => word.toString(16).padStart(8, '0')).join('')
}

function digestCanonical(value: unknown): string {
  return `sha256:${sha256(JSON.stringify(value))}`
}

function unique(values: string[]): boolean {
  return new Set(values).size === values.length
}

function comparePortableID(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function validInputID(value: unknown, projection: boolean): value is string {
  if (typeof value !== 'string' || value.trim() !== value || value.length > 128) return false
  const parts = value.split(':')
  if ((parts.length !== 2 && parts.length !== 3) || parts.some((part) => !PORTABLE_ID.test(part))) {
    return false
  }
  if (parts[0] !== parts[0].toLowerCase()) return false
  if (projection) return parts[0] === 'projection' && parts.length === 2
  if (parts[0] === 'projection') return false
  if (parts[0] === 'kb_metric') return parts.length === 3
  return SIGNAL_TYPES.has(parts[0]) && (parts.length === 2 || parts[0] === 'classifier')
}

function isInputSpec(value: unknown): value is EvaluationRoutingRecipeInputSpec {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, ['id', 'value_kind']) &&
    validInputID(value.id, false) &&
    (value.value_kind === 'numeric' || value.value_kind === 'none')
  )
}

function isProjectionSpec(value: unknown): value is EvaluationRoutingRecipeProjectionSpec {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, ['id', 'value_kind', 'outcome_binding']) &&
    validInputID(value.id, true) &&
    (value.value_kind === 'numeric' || value.value_kind === 'probability') &&
    (value.outcome_binding === 'selected_pool_quality' ||
      value.outcome_binding === 'selected_is_oracle')
  )
}

function frozenTopK(armCount: number): number[] {
  return armCount > 0
    ? [...new Set([1, Math.min(3, armCount), Math.min(5, armCount)])].sort((a, b) => a - b)
    : []
}

export function evaluationRoutingRecipeTargetDigest(
  mixture: Pick<
    EvaluationMixture,
    | 'adaptation_digest'
    | 'binding_digest'
    | 'pool_digest'
    | 'recipe_digest'
    | 'selector_digest'
    | 'selector_policy_digest'
  >,
): string {
  return digestCanonical({
    adaptation_digest: mixture.adaptation_digest,
    binding_digest: mixture.binding_digest,
    pool_digest: mixture.pool_digest,
    recipe_digest: mixture.recipe_digest,
    selector_digest: mixture.selector_digest,
    selector_policy_digest: mixture.selector_policy_digest,
  })
}

export function evaluationRoutingRecipePlanDigest(
  plan: Omit<EvaluationRoutingRecipePlan, 'plan_digest'>,
): string {
  return digestCanonical({
    ContractVersion: plan.contract_version,
    TargetSnapshotDigest: plan.target_snapshot_digest,
    ArmIDs: [...plan.arm_ids].sort(),
    FallbackArmID: plan.fallback_arm_id || '',
    Signals: [...plan.signals].sort((left, right) => comparePortableID(left.id, right.id)),
    Projections: [...plan.projections].sort((left, right) => comparePortableID(left.id, right.id)),
    TopK: [...plan.top_k].sort((left, right) => left - right),
  })
}

function isEvaluationRoutingRecipePlanContract(
  value: unknown,
  mixture: Omit<EvaluationMixture, 'routing_recipe_plan'>,
  allowEmptyPool: boolean,
): value is EvaluationRoutingRecipePlan {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'contract_version',
      'plan_digest',
      'target_snapshot_digest',
      'arm_ids',
      'fallback_arm_id',
      'signals',
      'projections',
      'top_k',
    ]) ||
    value.contract_version !== PLAN_VERSION ||
    typeof value.plan_digest !== 'string' ||
    !DIGEST.test(value.plan_digest) ||
    typeof value.target_snapshot_digest !== 'string' ||
    !DIGEST.test(value.target_snapshot_digest) ||
    !Array.isArray(value.arm_ids) ||
    (!allowEmptyPool && value.arm_ids.length < 1) ||
    value.arm_ids.length > MAX_ARMS ||
    value.arm_ids.some((arm) => typeof arm !== 'string' || !PORTABLE_ID.test(arm)) ||
    !unique(value.arm_ids as string[]) ||
    !Array.isArray(value.signals) ||
    value.signals.length > MAX_INPUTS ||
    value.signals.some((item) => !isInputSpec(item)) ||
    !unique((value.signals as EvaluationRoutingRecipeInputSpec[]).map((item) => item.id)) ||
    !Array.isArray(value.projections) ||
    value.projections.length > MAX_INPUTS ||
    value.projections.some((item) => !isProjectionSpec(item)) ||
    !unique(
      (value.projections as EvaluationRoutingRecipeProjectionSpec[]).map((item) => item.id),
    ) ||
    !Array.isArray(value.top_k) ||
    value.top_k.length > MAX_ARMS ||
    value.top_k.some((item) => !Number.isInteger(item) || item < 1) ||
    JSON.stringify(value.top_k) !== JSON.stringify(frozenTopK(value.arm_ids.length)) ||
    (value.fallback_arm_id !== undefined &&
      (typeof value.fallback_arm_id !== 'string' || !value.arm_ids.includes(value.fallback_arm_id)))
  ) {
    return false
  }
  const armIDs = mixture.model_arms.map((arm) => arm.id).sort()
  const planArmIDs = [...(value.arm_ids as string[])].sort()
  if (
    JSON.stringify(planArmIDs) !== JSON.stringify(armIDs) ||
    (value.fallback_arm_id || '') !== (mixture.fallback_arm_id || '') ||
    value.target_snapshot_digest !== evaluationRoutingRecipeTargetDigest(mixture)
  ) {
    return false
  }
  const plan = value as unknown as EvaluationRoutingRecipePlan
  return (
    value.plan_digest ===
    evaluationRoutingRecipePlanDigest({
      contract_version: plan.contract_version,
      target_snapshot_digest: plan.target_snapshot_digest,
      arm_ids: plan.arm_ids,
      ...(plan.fallback_arm_id ? { fallback_arm_id: plan.fallback_arm_id } : {}),
      signals: plan.signals,
      projections: plan.projections,
      top_k: plan.top_k,
    })
  )
}

export function isEvaluationRoutingRecipePlan(
  value: unknown,
  mixture: Omit<EvaluationMixture, 'routing_recipe_plan'>,
): value is EvaluationRoutingRecipePlan {
  return isEvaluationRoutingRecipePlanContract(value, mixture, false)
}

export function isUnavailableEvaluationCatalogRoutingRecipePlan(
  value: unknown,
  mixture: Omit<EvaluationMixture, 'routing_recipe_plan'>,
): value is EvaluationRoutingRecipePlan {
  return (
    mixture.model_arms.length === 0 && isEvaluationRoutingRecipePlanContract(value, mixture, true)
  )
}

function isLatency(value: unknown, present: number): value is EvaluationRoutingRecipeLatencyReport {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, ['available', 'reason', 'sample_count', 'p50_ms', 'p95_ms']) ||
    typeof value.available !== 'boolean' ||
    !isNonNegativeInteger(value.sample_count) ||
    value.sample_count > present
  ) {
    return false
  }
  const p50 = value.p50_ms === undefined ? 0 : value.p50_ms
  const p95 = value.p95_ms === undefined ? 0 : value.p95_ms
  if (!isFiniteNumber(p50) || !isFiniteNumber(p95)) return false
  if (value.available) {
    return value.reason === undefined && value.sample_count >= 2 && p50 >= 0 && p95 >= p50
  }
  return (
    typeof value.reason === 'string' &&
    PORTABLE_ID.test(value.reason) &&
    value.sample_count <= 1 &&
    p50 === 0 &&
    p95 === 0
  )
}

function isInputAvailability(
  value: unknown,
  id: string,
  expected: number,
): value is EvaluationRoutingRecipeInputAvailabilityReport {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, [
      'id',
      'expected',
      'present',
      'missing',
      'error',
      'timeout',
      'latency',
    ]) &&
    value.id === id &&
    value.expected === expected &&
    isNonNegativeInteger(value.present) &&
    isNonNegativeInteger(value.missing) &&
    isNonNegativeInteger(value.error) &&
    isNonNegativeInteger(value.timeout) &&
    value.present + value.missing + value.error + value.timeout === expected &&
    isLatency(value.latency, value.present)
  )
}

function isMetricAvailability(
  value: unknown,
  expected: number,
  minimum: number,
  maximum: number,
): value is EvaluationRoutingRecipeMetricAvailability {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, ['available', 'reason', 'value', 'sample_count']) ||
    typeof value.available !== 'boolean' ||
    !isNonNegativeInteger(value.sample_count) ||
    value.sample_count > expected
  ) {
    return false
  }
  const metricValue = value.value === undefined ? 0 : value.value
  if (!isFiniteNumber(metricValue)) return false
  return value.available
    ? value.reason === undefined &&
        value.sample_count > 0 &&
        metricValue >= minimum &&
        metricValue <= maximum
    : typeof value.reason === 'string' && PORTABLE_ID.test(value.reason) && metricValue === 0
}

function isReliabilityBin(
  value: unknown,
  index: number,
): value is EvaluationRoutingRecipeReliabilityBin {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'lower',
      'upper',
      'count',
      'mean_prediction',
      'observed_frequency',
    ]) ||
    value.lower !== index / 10 ||
    value.upper !== (index + 1) / 10 ||
    !isNonNegativeInteger(value.count)
  ) {
    return false
  }
  const prediction = value.mean_prediction === undefined ? 0 : value.mean_prediction
  const observed = value.observed_frequency === undefined ? 0 : value.observed_frequency
  return (
    isFiniteNumber(prediction) &&
    prediction >= 0 &&
    prediction <= 1 &&
    isFiniteNumber(observed) &&
    observed >= 0 &&
    observed <= 1 &&
    (value.count > 0 || (prediction === 0 && observed === 0))
  )
}

function isProjectionOutcome(
  value: unknown,
  projectionID: string,
  expected: number,
): value is EvaluationRoutingRecipeProjectionOutcomeReport {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'projection_id',
      'spearman',
      'brier',
      'ece_10',
      'reliability_bins',
    ]) ||
    value.projection_id !== projectionID ||
    !isMetricAvailability(value.spearman, expected, -1, 1) ||
    !isMetricAvailability(value.brier, expected, 0, 1) ||
    !isMetricAvailability(value.ece_10, expected, 0, 1) ||
    !Array.isArray(value.reliability_bins)
  ) {
    return false
  }
  if (!value.brier.available || !value.ece_10.available) return value.reliability_bins.length === 0
  return (
    value.reliability_bins.length === 10 &&
    value.reliability_bins.every(isReliabilityBin) &&
    value.reliability_bins.reduce((total, bin) => total + bin.count, 0) ===
      value.ece_10.sample_count
  )
}

function isTopK(
  value: unknown,
  k: number,
  expected: number,
): value is EvaluationRoutingRecipeTopKReport {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, ['k', 'feasible_oracle_recall']) &&
    value.k === k &&
    isMetricAvailability(value.feasible_oracle_recall, expected, 0, 1)
  )
}

function isRoutingRecipeReport(
  value: unknown,
  plan: EvaluationRoutingRecipePlan,
): value is EvaluationRoutingRecipeReport {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, ['contract_version', 'plan_digest', 'e1', 'e2']) ||
    value.contract_version !== REPORT_VERSION ||
    value.plan_digest !== plan.plan_digest ||
    !isEvaluationRecord(value.e1) ||
    !hasOnlyEvaluationFields(value.e1, [
      'expected_decisions',
      'observed_decisions',
      'signals',
      'projections',
      'eligibility_complete',
      'selected_feasible',
    ]) ||
    !isNonNegativeInteger(value.e1.expected_decisions) ||
    value.e1.expected_decisions < 1 ||
    value.e1.expected_decisions > MAX_CASES ||
    value.e1.observed_decisions !== value.e1.expected_decisions ||
    !isNonNegativeInteger(value.e1.eligibility_complete) ||
    value.e1.eligibility_complete > value.e1.expected_decisions ||
    !isNonNegativeInteger(value.e1.selected_feasible) ||
    value.e1.selected_feasible > value.e1.expected_decisions ||
    !Array.isArray(value.e1.signals) ||
    value.e1.signals.length !== plan.signals.length ||
    !Array.isArray(value.e1.projections) ||
    value.e1.projections.length !== plan.projections.length
  ) {
    return false
  }
  const expected = value.e1.expected_decisions
  const signalIDs = plan.signals.map((item) => item.id).sort()
  const projectionIDs = plan.projections.map((item) => item.id).sort()
  if (
    !value.e1.signals.every((item, index) =>
      isInputAvailability(item, signalIDs[index], expected),
    ) ||
    !value.e1.projections.every((item, index) =>
      isInputAvailability(item, projectionIDs[index], expected),
    ) ||
    !isEvaluationRecord(value.e2) ||
    !hasOnlyEvaluationFields(value.e2, ['projection_outcomes', 'top_k', 'oracle_regret']) ||
    !Array.isArray(value.e2.projection_outcomes) ||
    value.e2.projection_outcomes.length !== plan.projections.length ||
    !value.e2.projection_outcomes.every((item, index) =>
      isProjectionOutcome(item, projectionIDs[index], expected),
    ) ||
    !Array.isArray(value.e2.top_k) ||
    value.e2.top_k.length !== plan.top_k.length ||
    !value.e2.top_k.every((item, index) => isTopK(item, plan.top_k[index], expected)) ||
    !isMetricAvailability(value.e2.oracle_regret, expected, 0, 1)
  ) {
    return false
  }
  return true
}

export function decodeEvaluationRoutingRecipeReport(
  value: unknown,
  run: EvaluationRun,
): EvaluationRoutingRecipeReport | null {
  const required =
    run.mode === 'live' && run.mixture !== undefined && run.track_ids.includes('routing')
  if (!required) {
    if (value !== null) {
      throw new Error(
        'Routing recipe report must be explicit null outside a live Mixture routing run.',
      )
    }
    return null
  }
  if (!run.mixture || !isRoutingRecipeReport(value, run.mixture.routing_recipe_plan)) {
    throw new Error(
      'Live Mixture routing report is missing its server-owned routing recipe evidence.',
    )
  }
  return value
}
