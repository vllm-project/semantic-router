import type {
  EvaluationCampaignGateID,
  EvaluationCatalogCampaignSlot,
  EvaluationRun,
} from '../../types/evaluationPlane'
import type { EvaluationCampaignBuilderModel } from './useEvaluationCampaignBuilder'
import { runOptionLabels } from './evaluationRunPresentation'
import commonStyles from './EvaluationCampaign.module.css'
import styles from './EvaluationCampaignBuilderMatrix.module.css'

const RUN_BINDING_KEYS = {
  G2: 'g2_run_id',
  G4: 'g4_run_id',
  G6: 'g6_run_id',
  G7: 'g7_run_id',
  G8: 'g8_run_id',
  G9: 'g9_run_id',
} as const

function ControlledPairSlotBinding({
  runs,
  model,
}: {
  runs: EvaluationRun[]
  model: EvaluationCampaignBuilderModel
}) {
  const pair = model.draft.gateBindings.g3_controlled_pair
  const baseline = runs.find((run) => run.id === pair?.baseline_run_id)
  const candidate = runs.find((run) => run.id === pair?.candidate_run_id)
  const labels = runOptionLabels([
    ...(baseline ? [baseline] : []),
    ...(candidate ? [candidate] : []),
  ])
  if (!pair) return <output aria-label="Controlled comparison runs">Launch below</output>
  return (
    <output aria-label="Controlled comparison runs">
      {baseline ? labels.get(baseline.id) : 'Saved baseline run'} →{' '}
      {candidate ? labels.get(candidate.id) : 'Saved candidate run'}
    </output>
  )
}

function FidelitySlotBinding({
  model,
  disabled,
}: {
  model: EvaluationCampaignBuilderModel
  disabled: boolean
}) {
  const binding = model.draft.gateBindings.g5_fidelity
  const referenceLabels = runOptionLabels(model.fidelityReferences)
  const liveLabels = runOptionLabels(model.fidelityLiveRuns)
  return (
    <div className={styles.pairInputs}>
      <label>
        <span className={commonStyles.srOnly}>Reference run</span>
        <select
          aria-label="Reference run"
          value={binding?.reference_run_id || ''}
          disabled={disabled || model.fidelityReferences.length === 0}
          onChange={(event) => model.changeFidelityReference(event.target.value)}
        >
          <option value="">
            {model.fidelityReferences.length
              ? 'Select a reference run'
              : 'No compatible reference run'}
          </option>
          {model.fidelityReferences.map((run) => (
            <option key={run.id} value={run.id}>
              {referenceLabels.get(run.id)}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span className={commonStyles.srOnly}>Candidate run</span>
        <select
          aria-label="Candidate run"
          value={binding?.live_run_id || ''}
          disabled={disabled || !binding?.reference_run_id || model.fidelityLiveRuns.length === 0}
          onChange={(event) => model.changeFidelityLive(event.target.value)}
        >
          <option value="">
            {binding?.reference_run_id && model.fidelityLiveRuns.length === 0
              ? 'No matching candidate run'
              : 'Select a matching candidate run'}
          </option>
          {model.fidelityLiveRuns.map((run) => (
            <option key={run.id} value={run.id}>
              {liveLabels.get(run.id)}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}

function RunSlotBinding({
  slot,
  model,
  disabled,
}: {
  slot: EvaluationCatalogCampaignSlot
  model: EvaluationCampaignBuilderModel
  disabled: boolean
}) {
  const gateID = slot.gate_id as keyof typeof RUN_BINDING_KEYS
  const value = model.draft.gateBindings[RUN_BINDING_KEYS[gateID]] || ''
  const options = model.options.get(slot.gate_id) || []
  const labels = runOptionLabels(options)
  return (
    <select
      aria-label={`${slot.name} run`}
      value={value}
      disabled={disabled || options.length === 0}
      onChange={(event) =>
        model.changeRunBinding(slot.gate_id as EvaluationCampaignGateID, event.target.value)
      }
    >
      <option value="">
        {options.length ? 'Select a completed run' : 'No compatible completed run'}
      </option>
      {options.map((run) => (
        <option key={run.id} value={run.id}>
          {labels.get(run.id)}
        </option>
      ))}
    </select>
  )
}

export default function EvaluationCampaignSlotBinding({
  slot,
  runs,
  model,
  disabled,
}: {
  slot: EvaluationCatalogCampaignSlot
  runs: EvaluationRun[]
  model: EvaluationCampaignBuilderModel
  disabled: boolean
}) {
  if (slot.disposition === 'not_applicable') {
    return <output aria-label={`${slot.name} selection`}>No run required</output>
  }
  if (slot.binding_kind === 'controlled_pair') {
    return <ControlledPairSlotBinding runs={runs} model={model} />
  }
  if (slot.binding_kind === 'fidelity_pair') {
    return <FidelitySlotBinding model={model} disabled={disabled} />
  }
  return <RunSlotBinding slot={slot} model={model} disabled={disabled} />
}
