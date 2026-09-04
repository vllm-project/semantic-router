import type { FormEvent } from 'react'

import type {
  EvaluationCatalog,
  EvaluationExperimentIntent,
  EvaluationRun,
} from '../../types/evaluationPlane'
import { newEvaluationClientRequestID } from '../../utils/evaluationIdentity'
import { validateEvaluationDraft } from './evaluationExperimentValidation'
import type { EvaluationExperimentFormDerivation } from './useEvaluationExperimentFormDerivation'
import {
  EMPTY_CAPACITY_SLO_INPUT,
  type EvaluationExperimentFormState,
} from './useEvaluationExperimentFormState'

interface EvaluationExperimentFormWorkflowProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  pending: boolean
  onSubmit: (intent: EvaluationExperimentIntent) => Promise<boolean>
  state: EvaluationExperimentFormState
  derived: EvaluationExperimentFormDerivation
}

export function buildEvaluationExperimentSubmitWorkflow({
  catalog,
  runs,
  pending,
  onSubmit,
  state,
  derived,
}: EvaluationExperimentFormWorkflowProps) {
  const { targetID, changeProfile } = state
  return async (event: FormEvent) => {
    event.preventDefault()
    if (pending) return
    if (!changeProfile) {
      state.setValidationError('Select the type of change being evaluated.')
      return
    }
    const error = validateEvaluationDraft(catalog, runs, {
      name: state.name,
      description: state.description,
      mode: state.mode,
      targetID,
      changeProfile,
      suiteIDs: state.suiteIDs,
      trackIDs: state.trackIDs,
      sampleLimit: state.sampleLimit,
      concurrency: state.concurrency,
      capacitySLO: derived.capacitySLO,
      capacityLoadProtocol: derived.capacityLoadProtocol,
      seed: state.seed,
      baselineRunID: state.baselineRunID,
    })
    if (error) {
      state.setValidationError(error)
      return
    }
    state.setValidationError('')
    const request: Omit<EvaluationExperimentIntent, 'client_request_id'> = {
      name: state.name.trim(),
      description: state.description.trim(),
      suite_ids: state.suiteIDs,
      track_ids: state.trackIDs,
      mode: state.mode,
      target_id: targetID,
      change_profile: changeProfile,
      sample_limit: state.sampleLimit,
      concurrency: state.concurrency,
      ...(derived.capacitySLO ? { capacity_slo: derived.capacitySLO } : {}),
      ...(derived.capacityLoadProtocol
        ? { capacity_load_protocol: derived.capacityLoadProtocol }
        : {}),
      seed: state.seed,
      ...(state.baselineRunID ? { baseline_run_id: state.baselineRunID } : {}),
      autoStart: state.autoStart,
    }
    const fingerprint = JSON.stringify(request)
    if (!state.createAttempt.current || state.createAttempt.current.fingerprint !== fingerprint) {
      state.createAttempt.current = { fingerprint, id: newEvaluationClientRequestID() }
    }
    const created = await onSubmit({
      ...request,
      client_request_id: state.createAttempt.current.id,
    })
    if (created) {
      state.createAttempt.current = null
      state.setName('')
      state.setDescription('')
      state.setCapacitySLOInput(EMPTY_CAPACITY_SLO_INPUT)
    }
  }
}
