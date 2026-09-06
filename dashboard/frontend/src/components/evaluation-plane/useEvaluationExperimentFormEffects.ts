import { useEffect } from 'react'

import type { EvaluationCatalog } from '../../types/evaluationPlane'
import { compatibleEvaluationSuites, reconcileEvaluationScope } from './evaluationExperiment'
import { baselineCohortIssue } from './evaluationExperimentValidation'
import type {
  EvaluationExperimentFormDerivation,
  EvaluationExperimentFormInitialSelection,
} from './useEvaluationExperimentFormDerivation'
import type { EvaluationExperimentFormState } from './useEvaluationExperimentFormState'

interface EvaluationExperimentFormEffectsProps {
  catalog: EvaluationCatalog
  canAutoStart: boolean
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  pending: boolean
  preserveMissingLiveTarget: boolean
  initial: EvaluationExperimentFormInitialSelection
  state: EvaluationExperimentFormState
  derived: EvaluationExperimentFormDerivation
}

function useRequestedTargetSelection({
  catalog,
  pending,
  initial,
  state,
}: EvaluationExperimentFormEffectsProps): void {
  const requestedTarget = initial.requestedTarget
  const { baselineRunID, setMode, setSuiteIDs, setTargetID, setTrackIDs } = state
  useEffect(() => {
    if (!requestedTarget || baselineRunID || pending) return
    const suite = compatibleEvaluationSuites(catalog, requestedTarget.id, 'live')[0]
    const suiteIDs = suite ? [suite.id] : []
    setMode('live')
    setTargetID(requestedTarget.id)
    setSuiteIDs(suiteIDs)
    setTrackIDs(
      reconcileEvaluationScope(
        catalog,
        requestedTarget.id,
        'live',
        suiteIDs,
        suite?.track_ids || [],
      ).trackIDs,
    )
  }, [
    baselineRunID,
    catalog,
    pending,
    requestedTarget,
    setMode,
    setSuiteIDs,
    setTargetID,
    setTrackIDs,
  ])
}

function useEvaluationScopeReconciliation({
  catalog,
  preserveMissingLiveTarget,
  initial,
  state,
  derived,
}: EvaluationExperimentFormEffectsProps): void {
  const requestedTarget = initial.requestedTarget
  const {
    baselineRunID,
    changeProfile,
    mode,
    setChangeProfile,
    setSuiteIDs,
    setTargetID,
    setTrackIDs,
    suiteIDs,
    targetID,
  } = state
  useEffect(() => {
    if (baselineRunID) return
    if (requestedTarget && targetID === requestedTarget.id && mode === 'live') return
    if (preserveMissingLiveTarget && mode === 'live' && !targetID) return
    const compatibleTarget = catalog.targets.find(
      (target) => target.id === targetID && target.modes.includes(mode) && target.healthy !== false,
    )
    if (!compatibleTarget) {
      setTargetID(
        catalog.targets.find((target) => target.modes.includes(mode) && target.healthy !== false)
          ?.id || '',
      )
    }
  }, [
    baselineRunID,
    catalog.targets,
    mode,
    preserveMissingLiveTarget,
    requestedTarget,
    setTargetID,
    targetID,
  ])

  useEffect(() => {
    setSuiteIDs((current) =>
      current.filter((suiteID) => derived.compatibleSuites.some((suite) => suite.id === suiteID)),
    )
  }, [derived.compatibleSuites, setSuiteIDs])

  useEffect(() => {
    setTrackIDs(
      (current) => reconcileEvaluationScope(catalog, targetID, mode, suiteIDs, current).trackIDs,
    )
  }, [catalog, mode, setTrackIDs, suiteIDs, targetID])

  useEffect(() => {
    if (!catalog.change_profiles.some((profile) => profile.id === changeProfile)) {
      setChangeProfile(catalog.change_profiles[0]?.id || '')
    }
  }, [catalog.change_profiles, changeProfile, setChangeProfile])
}

function useBaselineSelectionValidation({
  catalog,
  runLedgerAvailable,
  runLedgerComplete,
  state,
  derived,
}: EvaluationExperimentFormEffectsProps): void {
  const { baselineRunID, setBaselineRunID, setValidationError } = state
  useEffect(() => {
    if ((!runLedgerAvailable || !runLedgerComplete) && baselineRunID) {
      setBaselineRunID('')
      setValidationError(
        runLedgerAvailable
          ? 'Baseline selection was cleared because run history is incomplete.'
          : 'Baseline selection was cleared because run history is unavailable.',
      )
      return
    }
    if (!baselineRunID) return
    const baseline = derived.completedRuns.find((run) => run.id === baselineRunID)
    const issue = baseline
      ? baselineCohortIssue(catalog, baseline)
      : 'The run is no longer available.'
    if (issue) {
      setBaselineRunID('')
      setValidationError(`Baseline selection was cleared. ${issue}`)
    }
  }, [
    baselineRunID,
    catalog,
    derived.completedRuns,
    runLedgerAvailable,
    runLedgerComplete,
    setBaselineRunID,
    setValidationError,
  ])
}

function useEvaluationFormAvailability({
  canAutoStart,
  state,
}: EvaluationExperimentFormEffectsProps): void {
  const { errorRef, setAutoStart, validationError } = state
  useEffect(() => {
    if (!canAutoStart) setAutoStart(false)
  }, [canAutoStart, setAutoStart])

  useEffect(() => {
    if (validationError) errorRef.current?.focus()
  }, [errorRef, validationError])
}

export function useEvaluationExperimentFormEffects(
  props: EvaluationExperimentFormEffectsProps,
): void {
  useRequestedTargetSelection(props)
  useEvaluationScopeReconciliation(props)
  useBaselineSelectionValidation(props)
  useEvaluationFormAvailability(props)
}
