import { useEffect, useMemo, useState } from 'react'

import type { EvaluationControlledPairReadyGuard } from '../../hooks/evaluationControlledPairHookSupport'
import { useEvaluationControlledPair } from '../../hooks/useEvaluationControlledPair'
import useEvaluationCampaignReadiness from '../../hooks/useEvaluationCampaignReadiness'
import type { EvaluationControlledPairExecution } from '../../types/evaluationControlledPair'
import type { EvaluationCampaignReadiness } from '../../types/evaluationCampaign'
import type {
  EvaluationCatalogCampaignSlot,
  EvaluationCatalogChangeProfile,
  EvaluationChangeProfileId,
  EvaluationRun,
} from '../../types/evaluationPlane'
import {
  controlledPairBaselineSourceOptions,
  controlledPairCandidateSourceOptions,
} from './evaluationCampaignSupport'
import EvaluationCampaignControlledPairView from './EvaluationCampaignControlledPairView'

interface EvaluationCampaignControlledPairProps {
  runs: EvaluationRun[]
  profile: EvaluationCatalogChangeProfile
  slot: EvaluationCatalogCampaignSlot
  readiness: EvaluationCampaignReadiness | null
  canCreate: boolean
  disabled: boolean
  activePairID: string | null
  resumablePair: { id: string; profileID: EvaluationChangeProfileId } | null
  onProfileLockChange: (locked: boolean) => void
  onPairIdentityChange: (pairID: string | null, profileID: EvaluationChangeProfileId | null) => void
  onReady: (
    execution: EvaluationControlledPairExecution,
    isCurrent: EvaluationControlledPairReadyGuard,
  ) => void | Promise<void>
}

function selectionGuidance(
  baselineOptions: EvaluationRun[],
  candidateOptions: EvaluationRun[],
  baselineSourceID: string,
  candidateLoading: boolean,
  candidateError: string | null,
): string {
  if (!baselineOptions.length) {
    return 'No completed live Mixture run is available for controlled value comparison. Run a compatible live evaluation first.'
  }
  if (!baselineSourceID) {
    return 'Choose the completed live baseline for the fresh paired comparison.'
  }
  if (candidateLoading) return 'Checking compatible candidate runs for this baseline.'
  if (candidateError) return 'Compatible candidates could not be loaded. Choose the baseline again to retry.'
  if (!candidateOptions.length)
    return 'No completed live candidate matches this baseline evaluation setup.'
  return 'Choose the matching candidate, then launch a fresh order-balanced comparison.'
}

export default function EvaluationCampaignControlledPair(
  props: EvaluationCampaignControlledPairProps,
) {
  const { activePairID, onPairIdentityChange, onProfileLockChange, profile } = props
  const [baselineSourceID, setBaselineSourceID] = useState('')
  const [candidateSourceID, setCandidateSourceID] = useState('')
  const pair = useEvaluationControlledPair(props.onReady, {
    activePairID,
    onPairIdentity: (pairID) => onPairIdentityChange(pairID, pairID ? profile.id : null),
  })
  const baselineOptions = useMemo(
    () => controlledPairBaselineSourceOptions(props.runs, props.readiness, props.slot),
    [props.readiness, props.runs, props.slot],
  )
  const readinessLedgerRevision = useMemo(
    () => props.runs.map((run) => `${run.id}:${run.status}`).join('|'),
    [props.runs],
  )
  const candidateReadiness = useEvaluationCampaignReadiness(
    props.profile,
    Boolean(baselineSourceID && props.canCreate && !props.disabled),
    {
      controlledPairBaselineRunID: baselineSourceID || undefined,
    },
    readinessLedgerRevision,
  )
  const candidateOptions = useMemo(
    () =>
      controlledPairCandidateSourceOptions(
        props.runs,
        candidateReadiness.readiness,
        props.slot,
      ),
    [candidateReadiness.readiness, props.runs, props.slot],
  )
  const busy = ['creating', 'recovering', 'running', 'assigning'].includes(pair.status)
  const profileLocked =
    Boolean(activePairID) || busy || Boolean(pair.execution && pair.status !== 'ready')
  useEffect(() => {
    onProfileLockChange(profileLocked)
  }, [onProfileLockChange, profileLocked])
  useEffect(
    () => () => {
      onProfileLockChange(false)
    },
    [onProfileLockChange],
  )
  useEffect(() => {
    if (pair.status === 'ready' && activePairID) onPairIdentityChange(null, null)
  }, [activePairID, onPairIdentityChange, pair.status])
  return (
    <EvaluationCampaignControlledPairView
      slotGateID={props.slot.gate_id}
      baselineSourceID={baselineSourceID}
      candidateSourceID={candidateSourceID}
      baselineOptions={baselineOptions}
      candidateOptions={candidateOptions}
      canCreate={props.canCreate}
      disabled={props.disabled}
      busy={busy}
      activePairID={activePairID}
      resumablePair={props.resumablePair}
      sourceReady={Boolean(baselineSourceID && candidateSourceID)}
      selectionRationale={selectionGuidance(
        baselineOptions,
        candidateOptions,
        baselineSourceID,
        candidateReadiness.loading,
        candidateReadiness.error,
      )}
      pair={pair}
      onBaselineSourceChange={(runID) => {
        setBaselineSourceID(runID)
        setCandidateSourceID('')
        pair.reset()
      }}
      onCandidateSourceChange={(runID) => {
        setCandidateSourceID(runID)
        pair.reset()
      }}
      onClearSavedPair={() => {
        pair.reset()
        onPairIdentityChange(null, null)
      }}
      onResumePair={() => {
        if (props.resumablePair) {
          onPairIdentityChange(props.resumablePair.id, props.resumablePair.profileID)
        }
      }}
    />
  )
}
