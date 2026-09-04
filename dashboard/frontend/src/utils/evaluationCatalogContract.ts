import type { EvaluationCatalog } from '../types/evaluationPlane'
import {
  EVALUATION_CAMPAIGN_GATE_IDS,
  EVALUATION_GATE_CONTRACT_VERSION,
  isEvaluationMethodEvidenceSource,
} from '../types/evaluationPlane'
import {
  assertCurrentEvaluationContract,
  EVALUATION_EVIDENCE_LEVEL_SET,
  EVALUATION_GATE_DISPOSITION_SET,
  EVALUATION_MODE_SET,
  EVALUATION_TRACK_ID_SET,
  type EvaluationRecord,
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isKnownValue,
  isKnownValueArray,
  isNonEmptyText,
  isNonNegativeInteger,
  isPortableEvaluationID,
  isStringRecord,
  isTextArray,
} from './evaluationContractValidation'
import {
  isEvaluationMixture,
  isUnavailableEvaluationCatalogMixture,
} from './evaluationMixtureContract'
import { hasValidCampaignProtocol } from './evaluationCampaignProtocol'

const METHOD_STATUSES = new Set(['configured', 'data_required'])
const CAMPAIGN_BINDING_KINDS = new Set(['run', 'controlled_pair', 'fidelity_pair'])
const CAMPAIGN_GATE_ID_SET = new Set<string>(EVALUATION_CAMPAIGN_GATE_IDS)

function isUnique(values: unknown[]): boolean {
  return new Set(values).size === values.length
}

function isUniqueTextArray(value: unknown, allowEmpty = true): value is string[] {
  return isTextArray(value, allowEmpty) && isUnique(value)
}

function isTargetExecutorMap(value: unknown, modes: unknown): boolean {
  if (
    !isEvaluationRecord(value) ||
    !Array.isArray(modes) ||
    !isKnownValueArray(modes, EVALUATION_MODE_SET, false)
  ) {
    return false
  }
  const declaredModes = modes as string[]
  const keys = Object.keys(value)
  return (
    keys.length === declaredModes.length &&
    keys.every((mode) => declaredModes.includes(mode)) &&
    declaredModes.every((mode) => {
      const executors = value[mode]
      return (
        Array.isArray(executors) &&
        executors.length > 0 &&
        executors.every(isPortableEvaluationID) &&
        isUnique(executors)
      )
    })
  )
}

function isSuiteExecutorMap(value: unknown, modes: unknown): boolean {
  if (!isEvaluationRecord(value) || !Array.isArray(modes)) return false
  const declaredModes = modes as string[]
  const keys = Object.keys(value)
  return (
    keys.length === declaredModes.length &&
    keys.every((mode) => declaredModes.includes(mode)) &&
    declaredModes.every((mode) => isPortableEvaluationID(value[mode]))
  )
}

function isCatalogTrack(value: unknown): value is EvaluationRecord {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, [
      'id',
      'name',
      'description',
      'modes',
      'metrics',
      'evidence_levels',
    ]) &&
    isKnownValue(value.id, EVALUATION_TRACK_ID_SET) &&
    isNonEmptyText(value.name) &&
    typeof value.description === 'string' &&
    isKnownValueArray(value.modes, EVALUATION_MODE_SET, false) &&
    isUnique(value.modes as unknown[]) &&
    isUniqueTextArray(value.metrics) &&
    isKnownValueArray(value.evidence_levels, EVALUATION_EVIDENCE_LEVEL_SET, false) &&
    isUnique(value.evidence_levels as unknown[])
  )
}

function isCatalogMethod(
  value: unknown,
  suiteTrackIDs: string[],
  gateIDs: ReadonlySet<string>,
): value is EvaluationRecord {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'id',
      'track_id',
      'qualified_gate_ids',
      'evidence_source',
      'status',
      'reason',
    ]) ||
    !isPortableEvaluationID(value.id) ||
    !isKnownValue(value.track_id, EVALUATION_TRACK_ID_SET) ||
    !suiteTrackIDs.includes(value.track_id) ||
    !isUniqueTextArray(value.qualified_gate_ids) ||
    value.qualified_gate_ids.some((gateID) => !gateIDs.has(gateID)) ||
    !isEvaluationMethodEvidenceSource(value.evidence_source) ||
    typeof value.status !== 'string' ||
    !METHOD_STATUSES.has(value.status)
  ) {
    return false
  }
  return value.status === 'data_required'
    ? isNonEmptyText(value.reason)
    : value.reason === undefined
}

function isCampaignSlot(value: unknown, trackIDs: ReadonlySet<string>): value is EvaluationRecord {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, [
      'gate_id',
      'name',
      'description',
      'disposition',
      'binding_kind',
      'track_id',
      'mode',
      'minimum_evidence_level',
      'accepted_executor_ids',
    ]) &&
    isKnownValue(value.gate_id, CAMPAIGN_GATE_ID_SET) &&
    isNonEmptyText(value.name) &&
    typeof value.description === 'string' &&
    isKnownValue(value.disposition, EVALUATION_GATE_DISPOSITION_SET) &&
    typeof value.binding_kind === 'string' &&
    CAMPAIGN_BINDING_KINDS.has(value.binding_kind) &&
    typeof value.track_id === 'string' &&
    trackIDs.has(value.track_id) &&
    isKnownValue(value.mode, EVALUATION_MODE_SET) &&
    isKnownValue(value.minimum_evidence_level, EVALUATION_EVIDENCE_LEVEL_SET) &&
    Array.isArray(value.accepted_executor_ids) &&
    value.accepted_executor_ids.length > 0 &&
    value.accepted_executor_ids.every(isPortableEvaluationID) &&
    isUnique(value.accepted_executor_ids)
  )
}

function isCatalogProfile(
  value: unknown,
  trackIDs: ReadonlySet<string>,
): value is EvaluationRecord {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, ['id', 'name', 'description', 'campaign_slots']) ||
    !isPortableEvaluationID(value.id) ||
    !isNonEmptyText(value.name) ||
    typeof value.description !== 'string' ||
    !Array.isArray(value.campaign_slots) ||
    value.campaign_slots.length === 0 ||
    value.campaign_slots.some((slot) => !isCampaignSlot(slot, trackIDs))
  ) {
    return false
  }
  const gateIDs = value.campaign_slots.map((slot) => (slot as EvaluationRecord).gate_id)
  return isUnique(gateIDs)
}

function isCatalogSuite(
  value: unknown,
  trackIDs: ReadonlySet<string>,
  gateIDs: ReadonlySet<string>,
): value is EvaluationRecord {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'id',
      'executors',
      'name',
      'description',
      'track_ids',
      'modes',
      'evidence_level',
      'case_count',
      'campaign_protocol',
      'revision',
      'tags',
      'methods',
    ]) ||
    !isPortableEvaluationID(value.id) ||
    !isNonEmptyText(value.name) ||
    typeof value.description !== 'string' ||
    !isKnownValueArray(value.track_ids, EVALUATION_TRACK_ID_SET, false) ||
    !isUnique(value.track_ids as unknown[]) ||
    (value.track_ids as string[]).some((trackID) => !trackIDs.has(trackID)) ||
    !isKnownValueArray(value.modes, EVALUATION_MODE_SET, false) ||
    !isUnique(value.modes as unknown[]) ||
    !isSuiteExecutorMap(value.executors, value.modes) ||
    !isKnownValue(value.evidence_level, EVALUATION_EVIDENCE_LEVEL_SET) ||
    (value.case_count !== undefined && !isNonNegativeInteger(value.case_count)) ||
    !hasValidCampaignProtocol(value) ||
    !isNonEmptyText(value.revision) ||
    !isUniqueTextArray(value.tags) ||
    !Array.isArray(value.methods) ||
    value.methods.length === 0 ||
    value.methods.some((method) =>
      !isCatalogMethod(method, value.track_ids as string[], gateIDs),
    )
  ) {
    return false
  }
  return isUnique(value.methods.map((method) => (method as EvaluationRecord).id))
}

function isCatalogTarget(value: unknown, trackIDs: ReadonlySet<string>): boolean {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'id',
      'name',
      'description',
      'kind',
      'track_ids',
      'modes',
      'accepted_executors',
      'evidence_level',
      'healthy',
      'labels',
      'mixture',
    ]) ||
    !isPortableEvaluationID(value.id) ||
    !isNonEmptyText(value.name) ||
    typeof value.description !== 'string' ||
    !isPortableEvaluationID(value.kind) ||
    !isKnownValueArray(value.track_ids, EVALUATION_TRACK_ID_SET) ||
    !isUnique(value.track_ids as unknown[]) ||
    (value.track_ids as string[]).some((trackID) => !trackIDs.has(trackID)) ||
    !isKnownValueArray(value.modes, EVALUATION_MODE_SET, false) ||
    !isUnique(value.modes as unknown[]) ||
    !isTargetExecutorMap(value.accepted_executors, value.modes) ||
    (value.evidence_level !== undefined &&
      !isKnownValue(value.evidence_level, EVALUATION_EVIDENCE_LEVEL_SET)) ||
    (value.healthy !== undefined && typeof value.healthy !== 'boolean') ||
    (value.labels !== undefined && !isStringRecord(value.labels))
  ) {
    return false
  }
  const mixtureTarget = value.kind === 'mixture-of-models'
  if (mixtureTarget !== (value.mixture !== undefined)) return false
  if ((value.track_ids as unknown[]).length > 0) {
    return value.mixture === undefined || isEvaluationMixture(value.mixture)
  }
  return (
    mixtureTarget &&
    value.healthy === false &&
    (isEvaluationMixture(value.mixture) || isUnavailableEvaluationCatalogMixture(value.mixture))
  )
}

export function decodeEvaluationCatalog(payload: unknown): EvaluationCatalog {
  assertCurrentEvaluationContract(payload, 'Evaluation catalog response')
  if (
    !hasOnlyEvaluationFields(payload, [
      'schema_version',
      'gate_contract_version',
      'generated_at',
      'change_profiles',
      'tracks',
      'suites',
      'targets',
    ]) ||
    payload.gate_contract_version !== EVALUATION_GATE_CONTRACT_VERSION ||
    !isNonEmptyText(payload.generated_at) ||
    !Array.isArray(payload.tracks) ||
    payload.tracks.length === 0 ||
    payload.tracks.some((track) => !isCatalogTrack(track))
  ) {
    throw new Error('Evaluation catalog response is incomplete.')
  }

  const tracks = payload.tracks as EvaluationRecord[]
  const trackIDs = tracks.map((track) => track.id as string)
  const trackIDSet = new Set(trackIDs)
  if (
    !isUnique(trackIDs) ||
    !Array.isArray(payload.change_profiles) ||
    payload.change_profiles.length === 0 ||
    payload.change_profiles.some((profile) => !isCatalogProfile(profile, trackIDSet))
  ) {
    throw new Error('Evaluation catalog response is incomplete.')
  }

  const profiles = payload.change_profiles as EvaluationRecord[]
  const profileIDs = profiles.map((profile) => profile.id)
  const gateIDs = new Set(
    profiles.flatMap((profile) =>
      (profile.campaign_slots as EvaluationRecord[]).map((slot) => slot.gate_id as string),
    ),
  )
  if (
    !isUnique(profileIDs) ||
    !Array.isArray(payload.suites) ||
    payload.suites.some((suite) => !isCatalogSuite(suite, trackIDSet, gateIDs))
  ) {
    throw new Error('Evaluation catalog response is incomplete.')
  }

  const suites = payload.suites as EvaluationRecord[]
  const suiteIDs = suites.map((suite) => suite.id)
  if (
    !isUnique(suiteIDs) ||
    !Array.isArray(payload.targets) ||
    payload.targets.some((target) => !isCatalogTarget(target, trackIDSet))
  ) {
    throw new Error('Evaluation catalog response is incomplete.')
  }
  const targetIDs = (payload.targets as EvaluationRecord[]).map((target) => target.id)
  if (!isUnique(targetIDs)) {
    throw new Error('Evaluation catalog response is incomplete.')
  }
  return payload as unknown as EvaluationCatalog
}
