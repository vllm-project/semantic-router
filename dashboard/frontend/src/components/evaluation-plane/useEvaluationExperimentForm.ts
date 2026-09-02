import type {
  EvaluationCatalog,
  EvaluationExperimentIntent,
  EvaluationRun,
} from '../../types/evaluationPlane'
import {
  buildEvaluationExperimentFormActions,
  type EvaluationExperimentFormActions,
} from './useEvaluationExperimentFormActions'
import {
  initialEvaluationExperimentFormSelection,
  type EvaluationExperimentFormDerivation,
  useEvaluationExperimentFormDerivation,
} from './useEvaluationExperimentFormDerivation'
import { useEvaluationExperimentFormEffects } from './useEvaluationExperimentFormEffects'
import {
  type EvaluationExperimentFormState,
  useEvaluationExperimentFormState,
} from './useEvaluationExperimentFormState'
import { buildEvaluationExperimentSubmitWorkflow } from './useEvaluationExperimentFormWorkflow'

interface UseEvaluationExperimentFormProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  canAutoStart: boolean
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  pending: boolean
  initialTargetID?: string
  preserveMissingLiveTarget?: boolean
  onSubmit: (intent: EvaluationExperimentIntent) => Promise<boolean>
}

function buildEvaluationExperimentFormModel(
  state: EvaluationExperimentFormState,
  derived: EvaluationExperimentFormDerivation,
  actions: EvaluationExperimentFormActions,
  submit: ReturnType<typeof buildEvaluationExperimentSubmitWorkflow>,
) {
  return {
    name: state.name,
    description: state.description,
    mode: state.mode,
    changeProfile: state.changeProfile,
    targetID: state.targetID,
    suiteIDs: state.suiteIDs,
    trackIDs: state.trackIDs,
    sampleLimit: state.sampleLimit,
    concurrency: state.concurrency,
    capacitySLOActive: derived.capacitySLOActive,
    capacitySLOInput: state.capacitySLOInput,
    capacityLoadProtocol: derived.capacityLoadProtocol,
    seed: state.seed,
    baselineRunID: state.baselineRunID,
    autoStart: state.autoStart,
    validationError: state.validationError,
    errorRef: state.errorRef,
    availableTrackIDs: derived.availableTrackIDs,
    compatibleSuites: derived.compatibleSuites,
    completedRuns: derived.completedRuns,
    baselineLocked: derived.baselineLocked,
    selectableTrackIDs: derived.selectableTrackIDs,
    selectedChangeProfile: derived.selectedChangeProfile,
    gateApplicability: derived.gateApplicability,
    catalogEvidenceClass: derived.catalogEvidenceClass,
    setName: state.setName,
    setDescription: state.setDescription,
    setMode: state.setMode,
    setChangeProfile: state.setChangeProfile,
    setTargetID: state.setTargetID,
    setSampleLimit: state.setSampleLimit,
    setConcurrency: state.setConcurrency,
    setCapacitySLOField: actions.setCapacitySLOField,
    applyCapacitySLOPreset: actions.applyCapacitySLOPreset,
    setSeed: state.setSeed,
    setAutoStart: state.setAutoStart,
    toggleSuite: actions.toggleSuite,
    toggleTrack: actions.toggleTrack,
    selectBaseline: actions.selectBaseline,
    submit,
  }
}

export default function useEvaluationExperimentForm(props: UseEvaluationExperimentFormProps) {
  const preserveMissingLiveTarget = props.preserveMissingLiveTarget ?? false
  const initial = initialEvaluationExperimentFormSelection(
    props.catalog,
    props.initialTargetID,
    preserveMissingLiveTarget,
  )
  const state = useEvaluationExperimentFormState({
    mode: initial.mode,
    changeProfile: props.catalog.change_profiles[0]?.id || '',
    targetID: initial.targetID,
    suiteIDs: initial.suiteIDs,
    trackIDs: initial.trackIDs,
    canAutoStart: props.canAutoStart,
  })
  const derived = useEvaluationExperimentFormDerivation({
    catalog: props.catalog,
    runs: props.runs,
    runLedgerAvailable: props.runLedgerAvailable,
    runLedgerComplete: props.runLedgerComplete,
    state,
  })
  useEvaluationExperimentFormEffects({
    catalog: props.catalog,
    canAutoStart: props.canAutoStart,
    runLedgerAvailable: props.runLedgerAvailable,
    runLedgerComplete: props.runLedgerComplete,
    pending: props.pending,
    preserveMissingLiveTarget,
    initial,
    state,
    derived,
  })
  const actions = buildEvaluationExperimentFormActions({
    catalog: props.catalog,
    pending: props.pending,
    state,
    derived,
  })
  const submit = buildEvaluationExperimentSubmitWorkflow({
    catalog: props.catalog,
    runs: props.runs,
    pending: props.pending,
    onSubmit: props.onSubmit,
    state,
    derived,
  })
  return buildEvaluationExperimentFormModel(state, derived, actions, submit)
}

export type EvaluationExperimentFormModel = ReturnType<typeof useEvaluationExperimentForm>
