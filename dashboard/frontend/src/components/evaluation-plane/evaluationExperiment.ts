import type {
  EvaluationCatalog,
  EvaluationCatalogSuite,
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
  EvaluationChangeProfileId,
  EvaluationMode,
  EvaluationRun,
  EvaluationTrackId,
  EvidenceLevel,
} from '../../types/evaluationPlane'
import {
  equalEvaluationCapacityLoadProtocol,
  equalEvaluationCapacitySLO,
} from '../../utils/evaluationCapacitySLOContract'

export const EVALUATION_RUN_LIMITS = {
  name: 200,
  description: 4000,
  sampleLimit: 100000,
  concurrency: 128,
  seed: 4294967295,
} as const

export interface EvaluationExactCohort {
  mode: EvaluationMode
  targetID: string
  changeProfile: EvaluationChangeProfileId
  suiteIDs: string[]
  trackIDs: EvaluationTrackId[]
  sampleLimit: number
  concurrency: number
  capacitySLO?: EvaluationCapacitySLO
  capacityLoadProtocol?: EvaluationCapacityLoadProtocol
  seed: number
}

const EVIDENCE_LEVELS: EvidenceLevel[] = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']

function unique<T>(values: T[]): T[] {
  return [...new Set(values)]
}

function sameSet<T>(left: T[], right: T[]): boolean {
  const normalizedLeft = new Set(left)
  const normalizedRight = new Set(right)
  return (
    normalizedLeft.size === left.length &&
    normalizedRight.size === right.length &&
    normalizedLeft.size === normalizedRight.size &&
    [...normalizedLeft].every((value) => normalizedRight.has(value))
  )
}

export function supportedEvaluationTracks(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationMode,
): EvaluationTrackId[] {
  const target = catalog.targets.find((candidate) => candidate.id === targetID)
  if (!target || target.healthy === false || !target.modes.includes(mode)) return []
  return catalog.tracks
    .filter((track) => track.modes.includes(mode) && target.track_ids.includes(track.id))
    .map((track) => track.id)
}

export function compatibleEvaluationSuites(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationMode,
): EvaluationCatalogSuite[] {
  const availableTracks = supportedEvaluationTracks(catalog, targetID, mode)
  const target = catalog.targets.find((candidate) => candidate.id === targetID)
  const acceptedExecutors = new Set(target?.accepted_executors[mode] || [])
  return catalog.suites.filter(
    (suite) =>
      suite.modes.includes(mode) &&
      acceptedExecutors.has(suite.executors[mode] || '') &&
      suite.track_ids.length > 0 &&
      suite.track_ids.some((trackID) => availableTracks.includes(trackID)),
  )
}

export function reconcileEvaluationScope(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationMode,
  suiteIDs: string[],
  trackIDs: EvaluationTrackId[],
): { suiteIDs: string[]; trackIDs: EvaluationTrackId[] } {
  const compatibleSuiteIDs = new Set(
    compatibleEvaluationSuites(catalog, targetID, mode).map((suite) => suite.id),
  )
  const requestedSuiteIDs = new Set(unique(suiteIDs))
  const requestedSuites = catalog.suites.filter(
    (suite) => requestedSuiteIDs.has(suite.id) && compatibleSuiteIDs.has(suite.id),
  )
  const cohortExecutor = requestedSuites[0]?.executors[mode]
  const nextSuiteIDs = requestedSuites
    .filter((suite) => suite.executors[mode] === cohortExecutor)
    .map((suite) => suite.id)
  const selectedSuiteTracks = new Set(
    catalog.suites
      .filter((suite) => nextSuiteIDs.includes(suite.id))
      .flatMap((suite) => suite.track_ids),
  )
  const availableTracks = new Set(supportedEvaluationTracks(catalog, targetID, mode))
  return {
    suiteIDs: nextSuiteIDs,
    trackIDs: catalog.tracks
      .map((track) => track.id)
      .filter(
        (trackID) =>
          trackIDs.includes(trackID) &&
          selectedSuiteTracks.has(trackID) &&
          availableTracks.has(trackID),
      ),
  }
}

export function toggleEvaluationSuite(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationMode,
  suiteIDs: string[],
  trackIDs: EvaluationTrackId[],
  suiteID: string,
): { suiteIDs: string[]; trackIDs: EvaluationTrackId[] } {
  const current = reconcileEvaluationScope(catalog, targetID, mode, suiteIDs, trackIDs)
  if (current.suiteIDs.includes(suiteID)) {
    return reconcileEvaluationScope(
      catalog,
      targetID,
      mode,
      current.suiteIDs.filter((id) => id !== suiteID),
      current.trackIDs,
    )
  }

  const suite = compatibleEvaluationSuites(catalog, targetID, mode).find(
    (candidate) => candidate.id === suiteID,
  )
  if (!suite) return current
  const selectedExecutor = catalog.suites.find((candidate) =>
    current.suiteIDs.includes(candidate.id),
  )?.executors[mode]
  if (selectedExecutor && selectedExecutor !== suite.executors[mode]) {
    return reconcileEvaluationScope(catalog, targetID, mode, [suite.id], [...suite.track_ids])
  }
  return reconcileEvaluationScope(
    catalog,
    targetID,
    mode,
    [...current.suiteIDs, suite.id],
    [...current.trackIDs, ...suite.track_ids],
  )
}

export function selectedSuiteTracks(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationMode,
  suiteIDs: string[],
): EvaluationTrackId[] {
  const scope = reconcileEvaluationScope(
    catalog,
    targetID,
    mode,
    suiteIDs,
    catalog.tracks.map((track) => track.id),
  )
  return scope.trackIDs
}

export function minimumCatalogEvidenceClass(
  catalog: EvaluationCatalog,
  suiteIDs: string[],
): EvidenceLevel | null {
  if (suiteIDs.length === 0) return null
  const suites = suiteIDs.map((suiteID) =>
    catalog.suites.find((candidate) => candidate.id === suiteID),
  )
  if (suites.some((suite) => !suite)) return null
  return suites.reduce<EvidenceLevel>((minimum, suite) => {
    const level = suite?.evidence_level || 'E0'
    return EVIDENCE_LEVELS.indexOf(level) < EVIDENCE_LEVELS.indexOf(minimum) ? level : minimum
  }, 'E5')
}

export function exactCohortFromRun(run: EvaluationRun): EvaluationExactCohort {
  return {
    mode: run.mode,
    targetID: run.target_id,
    changeProfile: run.change_profile,
    suiteIDs: [...run.suite_ids],
    trackIDs: [...run.track_ids],
    sampleLimit: run.sample_limit,
    concurrency: run.concurrency,
    ...(run.capacity_slo ? { capacitySLO: { ...run.capacity_slo } } : {}),
    ...(run.capacity_load_protocol
      ? {
          capacityLoadProtocol: {
            ...run.capacity_load_protocol,
            concurrency_levels: [...run.capacity_load_protocol.concurrency_levels],
          },
        }
      : {}),
    seed: run.seed,
  }
}

export function exactCohortMatchesRun(cohort: EvaluationExactCohort, run: EvaluationRun): boolean {
  return (
    cohort.mode === run.mode &&
    cohort.targetID === run.target_id &&
    cohort.changeProfile === run.change_profile &&
    cohort.sampleLimit === run.sample_limit &&
    cohort.concurrency === run.concurrency &&
    equalEvaluationCapacitySLO(cohort.capacitySLO, run.capacity_slo) &&
    equalEvaluationCapacityLoadProtocol(cohort.capacityLoadProtocol, run.capacity_load_protocol) &&
    cohort.seed === run.seed &&
    sameSet(cohort.suiteIDs, run.suite_ids) &&
    sameSet(cohort.trackIDs, run.track_ids)
  )
}
