import { useMemo } from 'react'

import type {
  EvaluationCapacitySLO,
  EvaluationCatalog,
  EvaluationMode,
  EvaluationRun,
} from '../../types/evaluationPlane'
import { EVALUATION_SCHEMA_VERSION } from '../../types/evaluationPlane'
import {
  defaultEvaluationCapacityLoadProtocol,
  requiresCapacitySLO,
} from '../../utils/evaluationCapacitySLOContract'
import {
  compatibleEvaluationSuites,
  minimumCatalogEvidenceClass,
  reconcileEvaluationScope,
  selectedSuiteTracks,
  supportedEvaluationTracks,
} from './evaluationExperiment'
import type {
  EvaluationCapacitySLOInput,
  EvaluationExperimentFormState,
} from './useEvaluationExperimentFormState'

export interface EvaluationExperimentFormInitialSelection {
  requestedTarget: EvaluationCatalog['targets'][number] | undefined
  mode: EvaluationMode
  targetID: string
  suiteIDs: string[]
  trackIDs: ReturnType<typeof reconcileEvaluationScope>['trackIDs']
}

export function initialEvaluationExperimentFormSelection(
  catalog: EvaluationCatalog,
  requestedTargetID: string | undefined,
  preserveMissingLiveTarget: boolean,
): EvaluationExperimentFormInitialSelection {
  const requestedTarget = requestedTargetID
    ? catalog.targets.find(
        (target) =>
          target.id === requestedTargetID && target.modes.includes('live') && target.mixture,
      )
    : undefined
  const preferredLiveTarget = catalog.targets.find(
    (target) =>
      target.kind === 'mixture-of-models' &&
      Boolean(target.mixture) &&
      target.modes.includes('live') &&
      target.healthy !== false,
  )
  const replayFallbackTarget = catalog.targets.find(
    (target) => target.modes.includes('replay') && target.healthy !== false,
  )
  const mode: EvaluationMode =
    requestedTarget || preserveMissingLiveTarget || preferredLiveTarget ? 'live' : 'replay'
  const targetID =
    requestedTarget?.id ||
    (!preserveMissingLiveTarget
      ? preferredLiveTarget?.id || replayFallbackTarget?.id
      : undefined) ||
    ''
  const initialSuite = compatibleEvaluationSuites(catalog, targetID, mode)[0]
  const suiteIDs = initialSuite ? [initialSuite.id] : []
  const trackIDs = reconcileEvaluationScope(
    catalog,
    targetID,
    mode,
    suiteIDs,
    initialSuite?.track_ids || [],
  ).trackIDs
  return { requestedTarget, mode, targetID, suiteIDs, trackIDs }
}

function capacitySLOFromInput(input: EvaluationCapacitySLOInput): EvaluationCapacitySLO {
  const numberFromInput = (value: string) => (value.trim() ? Number(value) : Number.NaN)
  return {
    schema_version: EVALUATION_SCHEMA_VERSION,
    required_concurrency: numberFromInput(input.requiredConcurrency),
    max_latency_p95_ms: numberFromInput(input.maxLatencyP95MS),
    max_error_rate: numberFromInput(input.maxErrorRate),
    min_throughput_rps: numberFromInput(input.minThroughputRPS),
    min_throughput_scaling_efficiency: numberFromInput(input.minThroughputScalingEfficiency),
  }
}

export function inputFromCapacitySLO(slo: EvaluationCapacitySLO): EvaluationCapacitySLOInput {
  return {
    requiredConcurrency: String(slo.required_concurrency),
    maxLatencyP95MS: String(slo.max_latency_p95_ms),
    maxErrorRate: String(slo.max_error_rate),
    minThroughputRPS: String(slo.min_throughput_rps),
    minThroughputScalingEfficiency: String(slo.min_throughput_scaling_efficiency),
  }
}

interface EvaluationExperimentFormDerivationProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  state: EvaluationExperimentFormState
}

export function useEvaluationExperimentFormDerivation({
  catalog,
  runs,
  runLedgerAvailable,
  runLedgerComplete,
  state,
}: EvaluationExperimentFormDerivationProps) {
  const availableTrackIDs = useMemo(
    () => supportedEvaluationTracks(catalog, state.targetID, state.mode),
    [catalog, state.mode, state.targetID],
  )
  const compatibleSuites = useMemo(
    () => compatibleEvaluationSuites(catalog, state.targetID, state.mode),
    [catalog, state.mode, state.targetID],
  )
  const completedRuns = useMemo(
    () =>
      runLedgerAvailable && runLedgerComplete
        ? runs.filter((run) => run.status === 'completed')
        : [],
    [runLedgerAvailable, runLedgerComplete, runs],
  )
  const selectedBaseline = completedRuns.find((run) => run.id === state.baselineRunID) || null
  const capacitySLOActive = requiresCapacitySLO(state.mode, state.trackIDs)
  const capacitySLO = capacitySLOActive ? capacitySLOFromInput(state.capacitySLOInput) : undefined
  const capacityLoadProtocol = capacitySLOActive
    ? selectedBaseline?.capacity_load_protocol ||
      (Number.isSafeInteger(state.concurrency) && state.concurrency >= 2
        ? defaultEvaluationCapacityLoadProtocol(state.concurrency)
        : undefined)
    : undefined
  const selectableTrackIDs = useMemo(
    () => selectedSuiteTracks(catalog, state.targetID, state.mode, state.suiteIDs),
    [catalog, state.mode, state.suiteIDs, state.targetID],
  )
  const selectedChangeProfile = catalog.change_profiles.find(
    (profile) => profile.id === state.changeProfile,
  )
  const gateApplicability = selectedChangeProfile?.campaign_slots || []
  const catalogEvidenceClass = minimumCatalogEvidenceClass(
    catalog,
    state.suiteIDs.filter((suiteID) => compatibleSuites.some((suite) => suite.id === suiteID)),
  )
  return {
    availableTrackIDs,
    compatibleSuites,
    completedRuns,
    selectedBaseline,
    baselineLocked: selectedBaseline !== null,
    capacitySLOActive,
    capacitySLO,
    capacityLoadProtocol,
    selectableTrackIDs,
    selectedChangeProfile,
    gateApplicability,
    catalogEvidenceClass,
  }
}

export type EvaluationExperimentFormDerivation = ReturnType<
  typeof useEvaluationExperimentFormDerivation
>
