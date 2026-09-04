import {
  EVALUATION_METHOD_EVIDENCE_SOURCE,
  type EvaluationCatalog,
  type EvaluationCatalogMethod,
  type EvaluationMode,
  type EvaluationTrackId,
} from '../../types/evaluationPlane'

export type EvaluationMethodReadinessStatus = 'ready' | 'setup_required'

export interface EvaluationMethodReadinessEntry {
  method: EvaluationCatalogMethod
  qualifiedGateNames: string[]
  readiness: EvaluationMethodReadinessStatus
  suiteID: string
  suiteName: string
  revision: string
  executors: EvaluationCatalog['suites'][number]['executors']
}

export interface EvaluationMethodReadinessFilter {
  query: string
  track: EvaluationTrackId | 'all'
  status: EvaluationMethodReadinessStatus | 'all'
}

export function requiredEvaluationMethodMode(method: EvaluationCatalogMethod): EvaluationMode {
  if (
    method.evidence_source === EVALUATION_METHOD_EVIDENCE_SOURCE.DIAGNOSTIC_FIXTURE ||
    method.evidence_source === EVALUATION_METHOD_EVIDENCE_SOURCE.NORMALIZED_IMPORT
  ) {
    return 'replay'
  }
  return 'live'
}

function methodHasExecutableTarget(
  catalog: EvaluationCatalog,
  suite: EvaluationCatalog['suites'][number],
  method: EvaluationCatalogMethod,
): boolean {
  const mode = requiredEvaluationMethodMode(method)
  const executorID = suite.executors[mode]
  if (!executorID || !suite.modes.includes(mode)) return false
  return catalog.targets.some((target) => {
    const healthy = mode === 'live' ? target.healthy === true : target.healthy !== false
    const validKind = mode !== 'live' || target.kind === 'mixture-of-models'
    return (
      healthy &&
      validKind &&
      target.modes.includes(mode) &&
      target.track_ids.includes(method.track_id) &&
      target.accepted_executors[mode]?.includes(executorID) === true
    )
  })
}

export function buildEvaluationMethodReadiness(
  catalog: EvaluationCatalog,
): EvaluationMethodReadinessEntry[] {
  const gateNames = new Map<string, string>(
    catalog.change_profiles.flatMap((profile) =>
      profile.campaign_slots.map((slot) => [slot.gate_id, slot.name] as const),
    ),
  )
  return catalog.suites.flatMap((suite) =>
    suite.methods.map((method) => ({
      method,
      qualifiedGateNames: method.qualified_gate_ids.map(
        (gateID) => gateNames.get(gateID) || 'Release readiness',
      ),
      readiness:
        method.status === 'configured' && methodHasExecutableTarget(catalog, suite, method)
          ? 'ready'
          : 'setup_required',
      suiteID: suite.id,
      suiteName: suite.name,
      revision: suite.revision,
      executors: suite.executors,
    })),
  )
}

export function filterEvaluationMethodReadiness(
  methods: EvaluationMethodReadinessEntry[],
  { query, track, status }: EvaluationMethodReadinessFilter,
): EvaluationMethodReadinessEntry[] {
  const normalizedQuery = query.trim().toLowerCase()
  return methods.filter(
    ({ method, qualifiedGateNames, readiness, suiteID, suiteName }) =>
      (track === 'all' || method.track_id === track) &&
      (status === 'all' || readiness === status) &&
      (!normalizedQuery ||
        [
          method.id,
          suiteID,
          suiteName,
          method.track_id,
          method.evidence_source,
          method.reason || '',
          ...method.qualified_gate_ids,
          ...qualifiedGateNames,
        ]
          .join(' ')
          .toLowerCase()
          .includes(normalizedQuery)),
  )
}

export function countEvaluationMethodReadiness(
  methods: EvaluationMethodReadinessEntry[],
): Record<EvaluationMethodReadinessStatus, number> {
  return methods.reduce(
    (result, { readiness }) => ({ ...result, [readiness]: result[readiness] + 1 }),
    { ready: 0, setup_required: 0 },
  )
}
