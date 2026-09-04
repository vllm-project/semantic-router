import type { EvaluationCatalog, EvaluationTrackId } from '../../types/evaluationPlane'
import { exactCohortFromRun, toggleEvaluationSuite } from './evaluationExperiment'
import { baselineCohortIssue } from './evaluationExperimentValidation'
import {
  inputFromCapacitySLO,
  type EvaluationExperimentFormDerivation,
} from './useEvaluationExperimentFormDerivation'
import {
  EMPTY_CAPACITY_SLO_INPUT,
  type EvaluationCapacitySLOInput,
  type EvaluationExperimentFormState,
} from './useEvaluationExperimentFormState'

interface EvaluationExperimentFormActionsProps {
  catalog: EvaluationCatalog
  pending: boolean
  state: EvaluationExperimentFormState
  derived: EvaluationExperimentFormDerivation
}

export function buildEvaluationExperimentFormActions({
  catalog,
  pending,
  state,
  derived,
}: EvaluationExperimentFormActionsProps) {
  const setCapacitySLOField = (field: keyof EvaluationCapacitySLOInput, value: string) => {
    state.setCapacitySLOInput((current) => ({ ...current, [field]: value }))
  }

  const applyCapacitySLOPreset = (preset: EvaluationCapacitySLOInput) => {
    state.setCapacitySLOInput(preset)
  }

  const toggleSuite = (suiteID: string) => {
    if (derived.baselineLocked || pending) return
    const next = toggleEvaluationSuite(
      catalog,
      state.targetID,
      state.mode,
      state.suiteIDs,
      state.trackIDs,
      suiteID,
    )
    state.setSuiteIDs(next.suiteIDs)
    state.setTrackIDs(next.trackIDs)
  }

  const toggleTrack = (trackID: EvaluationTrackId) => {
    if (derived.baselineLocked || pending || !derived.selectableTrackIDs.includes(trackID)) return
    state.setTrackIDs((current) =>
      current.includes(trackID) ? current.filter((id) => id !== trackID) : [...current, trackID],
    )
  }

  const selectBaseline = (runID: string) => {
    if (pending) return
    if (!runID) {
      state.setBaselineRunID('')
      state.setValidationError('')
      return
    }
    const baseline = derived.completedRuns.find((run) => run.id === runID)
    const issue = baseline ? baselineCohortIssue(catalog, baseline) : 'The run is unavailable.'
    if (!baseline || issue) {
      state.setBaselineRunID('')
      state.setValidationError(`This run cannot be used as a baseline. ${issue}`)
      return
    }
    const cohort = exactCohortFromRun(baseline)
    state.setMode(cohort.mode)
    state.setTargetID(cohort.targetID)
    state.setChangeProfile(cohort.changeProfile)
    state.setSuiteIDs(cohort.suiteIDs)
    state.setTrackIDs(cohort.trackIDs)
    state.setSampleLimit(cohort.sampleLimit)
    state.setConcurrency(cohort.concurrency)
    state.setCapacitySLOInput(
      cohort.capacitySLO ? inputFromCapacitySLO(cohort.capacitySLO) : EMPTY_CAPACITY_SLO_INPUT,
    )
    state.setSeed(cohort.seed)
    state.setBaselineRunID(baseline.id)
    state.setValidationError('')
  }

  return {
    setCapacitySLOField,
    applyCapacitySLOPreset,
    toggleSuite,
    toggleTrack,
    selectBaseline,
  }
}

export type EvaluationExperimentFormActions = ReturnType<
  typeof buildEvaluationExperimentFormActions
>
