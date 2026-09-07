import type { CreateEvaluationRunPayload, EvaluationCatalog } from '../types/evaluationPlane'
import {
  decodeEvaluationCapacityLoadProtocol,
  decodeEvaluationCapacitySLO,
  requiresCapacitySLO,
} from './evaluationCapacitySLOContract'
import {
  type EvaluationRecord,
  hasOnlyEvaluationFields,
} from './evaluationContractValidation'
import { requireCanonicalEvaluationRunID } from './evaluationRunContract'

const CREATE_RUN_FIELDS = [
  'client_request_id',
  'name',
  'description',
  'suite_ids',
  'track_ids',
  'mode',
  'target_id',
  'change_profile',
  'sample_limit',
  'concurrency',
  'capacity_slo',
  'capacity_load_protocol',
  'seed',
  'baseline_run_id',
] as const

function validateCreateRunFields(request: CreateEvaluationRunPayload) {
  if (!hasOnlyEvaluationFields(request as unknown as EvaluationRecord, CREATE_RUN_FIELDS)) {
    throw new Error('Evaluation create intent contains non-contract fields.')
  }
  requireCanonicalEvaluationRunID(request.client_request_id)
  if (request.baseline_run_id) requireCanonicalEvaluationRunID(request.baseline_run_id)
  const name = request.name.trim()
  const description = request.description.trim()
  if (!name || new TextEncoder().encode(name).length > 200) {
    throw new Error('Evaluation run name must contain 1–200 UTF-8 bytes.')
  }
  if (new TextEncoder().encode(description).length > 4_000) {
    throw new Error('Evaluation run description must contain at most 4000 UTF-8 bytes.')
  }
  if (
    !Number.isInteger(request.sample_limit) ||
    request.sample_limit < 1 ||
    request.sample_limit > 100_000
  ) {
    throw new Error('Evaluation sample limit must be an integer between 1 and 100000.')
  }
  if (
    !Number.isInteger(request.concurrency) ||
    request.concurrency < 1 ||
    request.concurrency > 128
  ) {
    throw new Error('Evaluation concurrency must be an integer between 1 and 128.')
  }
  if (!Number.isInteger(request.seed) || request.seed < 0 || request.seed > 4_294_967_295) {
    throw new Error('Evaluation seed must be an integer between 0 and 4294967295.')
  }
  return { name, description }
}

function createRunCapacity(request: CreateEvaluationRunPayload) {
  const capacityRequired = requiresCapacitySLO(request.mode, request.track_ids)
  if (!capacityRequired) {
    if (request.capacity_slo !== undefined || request.capacity_load_protocol !== undefined) {
      throw new Error('Performance settings are available only for live performance evaluation.')
    }
    return {}
  }
  if (request.concurrency < 2) {
    throw new Error('Live performance evaluation requires at least two parallel requests.')
  }
  if (request.capacity_slo === undefined || request.capacity_load_protocol === undefined) {
    throw new Error('Live performance evaluation requires performance goals and a load pattern.')
  }
  const capacitySLO = decodeEvaluationCapacitySLO(request.capacity_slo)
  if (capacitySLO.required_concurrency > request.concurrency) {
    throw new Error('Required parallel load cannot exceed the run limit.')
  }
  return {
    capacity_slo: capacitySLO,
    capacity_load_protocol: decodeEvaluationCapacityLoadProtocol(
      request.capacity_load_protocol,
      request.concurrency,
    ),
  }
}

function createRunCatalogSelection(
  request: CreateEvaluationRunPayload,
  catalog: EvaluationCatalog,
) {
  const changeProfile = catalog.change_profiles.find((item) => item.id === request.change_profile)
  if (!changeProfile) throw new Error('Select the type of change being evaluated.')
  const target = catalog.targets.find((item) => item.id === request.target_id)
  if (!target) throw new Error('Select an available evaluation source.')
  if (!target.modes.includes(request.mode) || target.healthy === false) {
    throw new Error('The selected evaluation source cannot run this evaluation.')
  }
  if (new Set(request.suite_ids).size !== request.suite_ids.length) {
    throw new Error('Selected benchmarks must not contain duplicates.')
  }
  if (new Set(request.track_ids).size !== request.track_ids.length) {
    throw new Error('Selected evaluation areas must not contain duplicates.')
  }
  const suitesByID = new Map(catalog.suites.map((suite) => [suite.id, suite]))
  const suites = request.suite_ids.map((id) => suitesByID.get(id))
  if (suites.some((suite) => !suite)) {
    throw new Error('One or more selected benchmarks are no longer available.')
  }
  if (
    suites.some(
      (suite) =>
        !suite?.modes.includes(request.mode) ||
        !target.accepted_executors[request.mode]?.includes(suite.executors[request.mode] || ''),
    )
  ) {
    throw new Error('Every selected benchmark must support this source and run type.')
  }
  const executorIDs = new Set(suites.map((suite) => suite?.executors[request.mode]))
  if (executorIDs.size !== 1 || executorIDs.has(undefined)) {
    throw new Error('The selected benchmarks cannot run together.')
  }
  const suiteTrackIDs = new Set(suites.flatMap((suite) => suite?.track_ids || []))
  if (
    request.track_ids.some(
      (trackID) => !target.track_ids.includes(trackID) || !suiteTrackIDs.has(trackID),
    )
  ) {
    throw new Error('Every evaluation area must be supported by the source and benchmarks.')
  }
  return {
    changeProfileID: changeProfile.id,
    suiteIDs: catalog.suites
      .map((suite) => suite.id)
      .filter((id) => request.suite_ids.includes(id)),
    trackIDs: catalog.tracks
      .map((track) => track.id)
      .filter((id) => request.track_ids.includes(id)),
  }
}

export function buildCreateRunPayload(
  request: CreateEvaluationRunPayload,
  catalog: EvaluationCatalog,
): CreateEvaluationRunPayload {
  const { name, description } = validateCreateRunFields(request)
  const capacity = createRunCapacity(request)
  const selection = createRunCatalogSelection(request, catalog)
  return {
    client_request_id: request.client_request_id,
    name,
    description,
    suite_ids: selection.suiteIDs,
    track_ids: selection.trackIDs,
    mode: request.mode,
    target_id: request.target_id,
    change_profile: selection.changeProfileID,
    sample_limit: request.sample_limit,
    concurrency: request.concurrency,
    ...capacity,
    seed: request.seed,
    ...(request.baseline_run_id ? { baseline_run_id: request.baseline_run_id } : {}),
  }
}
