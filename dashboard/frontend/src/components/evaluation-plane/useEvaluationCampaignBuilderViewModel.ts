import { useCallback, useMemo, useState } from 'react'

import type {
  EvaluationControlledPairReadyGuard,
} from '../../hooks/evaluationControlledPairHookSupport'
import type { EvaluationControlledPairExecution } from '../../types/evaluationControlledPair'
import type { EvaluationChangeProfileId, EvaluationRun } from '../../types/evaluationPlane'
import { campaignSlotRunIDs } from './evaluationCampaignSupport'
import type { EvaluationCampaignBuilderProps } from './evaluationCampaignBuilderTypes'

function findResumablePair(runs: EvaluationRun[], activePairID: string | null) {
  if (activePairID) return null
  const pairs = new Map<string, EvaluationChangeProfileId>()
  runs.forEach((run) => {
    if (run.controlled_pair && ['pending', 'running', 'sealing'].includes(run.status)) {
      pairs.set(run.controlled_pair.pair_id, run.change_profile)
    }
  })
  if (pairs.size !== 1) return null
  const [id, profileID] = [...pairs][0]
  return { id, profileID }
}

export default function useEvaluationCampaignBuilderViewModel(
  props: EvaluationCampaignBuilderProps,
) {
  const [controlledPairProfileLocked, setControlledPairProfileLocked] = useState(false)
  const { model } = props
  const resumablePair = useMemo(
    () => findResumablePair(props.runs, props.activeControlledPairID),
    [props.activeControlledPairID, props.runs],
  )
  const requiredSlots = model.slots.filter((slot) => slot.disposition === 'required')
  const readyRequiredSlots = requiredSlots.filter((slot) => {
    const ids = campaignSlotRunIDs(slot, model.draft.gateBindings)
    return ids.length === (slot.binding_kind === 'run' ? 1 : 2)
  }).length
  const onControlledPairReady = useCallback(
    async (
      execution: EvaluationControlledPairExecution,
      isCurrent: EvaluationControlledPairReadyGuard,
    ) => {
      if (
        execution.baseline_run.change_profile !== model.draft.changeProfile ||
        execution.candidate_run.change_profile !== model.draft.changeProfile
      ) {
        throw new Error('The recovered controlled comparison belongs to a different change type.')
      }
      const refreshed = await props.onRefreshRuns()
      if (!isCurrent()) return
      if (!refreshed) {
        throw new Error('Controlled comparison completed, but run history could not be refreshed.')
      }
      if (!isCurrent()) return
      model.applyControlledPair(execution.baseline_run.id, execution.candidate_run.id)
    },
    [model, props],
  )
  return {
    g3: model.slots.find((slot) => slot.gate_id === 'G3'),
    inputDisabled: props.createPending || !props.allRunsLoaded || !props.runLedgerComplete,
    profileLocked: Boolean(props.activeControlledPairID) || controlledPairProfileLocked,
    resumablePair,
    requiredSlots,
    readyRequiredSlots,
    setControlledPairProfileLocked,
    onControlledPairReady,
  }
}

export type EvaluationCampaignBuilderViewModel = ReturnType<
  typeof useEvaluationCampaignBuilderViewModel
>
