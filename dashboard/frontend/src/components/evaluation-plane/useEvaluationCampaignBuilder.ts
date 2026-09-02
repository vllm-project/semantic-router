import { useMemo, useState } from 'react'

import type {
  EvaluationCatalog,
  EvaluationCatalogCampaignSlot,
  EvaluationCampaignGateID,
  EvaluationChangeProfileId,
  EvaluationRun,
} from '../../types/evaluationPlane'
import useEvaluationCampaignReadiness from '../../hooks/useEvaluationCampaignReadiness'
import {
  campaignRunOptions,
  fidelityLiveOptions,
  fidelityReferenceOptions,
  type EvaluationCampaignDraft,
  newEvaluationCampaignClientRequestID,
  validateEvaluationCampaignDraft,
} from './evaluationCampaignSupport'

const EMPTY_CAMPAIGN_SLOTS: EvaluationCatalogCampaignSlot[] = []

function initialDraft(
  catalog: EvaluationCatalog,
  lockedChangeProfile?: EvaluationChangeProfileId | null,
): EvaluationCampaignDraft {
  const profile =
    catalog.change_profiles.find((candidate) => candidate.id === lockedChangeProfile) ||
    catalog.change_profiles.find((candidate) => candidate.id === 'recipe') ||
    catalog.change_profiles[0]
  if (!profile) throw new Error('Evaluation catalog requires at least one change type.')
  return {
    clientRequestID: newEvaluationCampaignClientRequestID(),
    name: '',
    description: '',
    changeProfile: profile.id,
    gateBindings: {},
  }
}

interface UseEvaluationCampaignBuilderProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  allRunsLoaded: boolean
  lockedChangeProfile?: EvaluationChangeProfileId | null
  onClearCreateError: () => void
}

export default function useEvaluationCampaignBuilder({
  catalog,
  runs,
  runLedgerAvailable,
  runLedgerComplete,
  allRunsLoaded,
  lockedChangeProfile,
  onClearCreateError,
}: UseEvaluationCampaignBuilderProps) {
  const [storedDraft, setDraft] = useState<EvaluationCampaignDraft>(() =>
    initialDraft(catalog, lockedChangeProfile),
  )
  const alignDraft = (draft: EvaluationCampaignDraft): EvaluationCampaignDraft =>
    lockedChangeProfile && draft.changeProfile !== lockedChangeProfile
      ? { ...draft, changeProfile: lockedChangeProfile, gateBindings: {} }
      : draft
  const draft = alignDraft(storedDraft)
  const profile =
    catalog.change_profiles.find((candidate) => candidate.id === draft.changeProfile) ||
    catalog.change_profiles[0]
  const slots = profile?.campaign_slots || EMPTY_CAMPAIGN_SLOTS
  const readinessLedgerRevision = useMemo(
    () => runs.map((run) => `${run.id}:${run.status}`).join('|'),
    [runs],
  )
  const readinessState = useEvaluationCampaignReadiness(
    profile,
    runLedgerAvailable && runLedgerComplete && allRunsLoaded,
    {
      controlledPairBaselineRunID:
        draft.gateBindings.g3_controlled_pair?.baseline_run_id || undefined,
      fidelityReferenceRunID: draft.gateBindings.g5_fidelity?.reference_run_id || undefined,
    },
    readinessLedgerRevision,
  )
  const validation = validateEvaluationCampaignDraft(
    catalog,
    runs,
    draft,
    readinessState.readiness,
    readinessState.loading,
    readinessState.error,
    runLedgerAvailable,
    runLedgerComplete,
    allRunsLoaded,
  )
  const options = useMemo(
    () =>
      new Map(
        slots
          .filter((slot) => slot.binding_kind === 'run')
          .map((slot) => [slot.gate_id, campaignRunOptions(runs, readinessState.readiness, slot)]),
      ),
    [readinessState.readiness, runs, slots],
  )
  const fidelitySlot = slots.find((slot) => slot.binding_kind === 'fidelity_pair')
  const fidelityReferences = useMemo(
    () =>
      fidelitySlot ? fidelityReferenceOptions(runs, readinessState.readiness, fidelitySlot) : [],
    [fidelitySlot, readinessState.readiness, runs],
  )
  const fidelityLiveRuns = useMemo(
    () =>
      fidelitySlot
        ? fidelityLiveOptions(runs, readinessState.readiness, fidelitySlot)
        : [],
    [fidelitySlot, readinessState.readiness, runs],
  )

  const revise = (change: (current: EvaluationCampaignDraft) => EvaluationCampaignDraft) => {
    onClearCreateError()
    setDraft((current) => ({
      ...change(alignDraft(current)),
      clientRequestID: newEvaluationCampaignClientRequestID(),
    }))
  }

  const changeProfile = (changeProfile: EvaluationChangeProfileId) => {
    if (lockedChangeProfile) return
    revise((current) => ({ ...current, changeProfile, gateBindings: {} }))
  }

  const changeRunBinding = (gateID: EvaluationCampaignGateID, value: string) => {
    let key: 'g2_run_id' | 'g4_run_id' | 'g6_run_id' | 'g7_run_id' | 'g8_run_id' | 'g9_run_id'
    switch (gateID) {
      case 'G2':
        key = 'g2_run_id'
        break
      case 'G4':
        key = 'g4_run_id'
        break
      case 'G6':
        key = 'g6_run_id'
        break
      case 'G7':
        key = 'g7_run_id'
        break
      case 'G8':
        key = 'g8_run_id'
        break
      case 'G9':
        key = 'g9_run_id'
        break
      case 'G3':
      case 'G5':
        throw new Error(`Release check ${gateID} does not accept a single-run binding.`)
    }
    revise((current) => ({
      ...current,
      gateBindings: {
        ...current.gateBindings,
        [key]: value || undefined,
      },
    }))
  }

  const changeFidelityReference = (value: string) => {
    revise((current) => ({
      ...current,
      gateBindings: {
        ...current.gateBindings,
        g5_fidelity: value ? { reference_run_id: value, live_run_id: '' } : undefined,
      },
    }))
  }

  const changeFidelityLive = (value: string) => {
    revise((current) => {
      const referenceRunID = current.gateBindings.g5_fidelity?.reference_run_id
      return {
        ...current,
        gateBindings: {
          ...current.gateBindings,
          g5_fidelity:
            referenceRunID && value
              ? { reference_run_id: referenceRunID, live_run_id: value }
              : referenceRunID
                ? { reference_run_id: referenceRunID, live_run_id: '' }
                : undefined,
        },
      }
    })
  }

  const applyControlledPair = (baselineRunID: string, candidateRunID: string) => {
    revise((current) => ({
      ...current,
      gateBindings: {
        ...current.gateBindings,
        g3_controlled_pair: {
          baseline_run_id: baselineRunID,
          candidate_run_id: candidateRunID,
        },
      },
    }))
  }

  return {
    draft,
    profile,
    slots,
    requiredSlotCount: slots.filter((slot) => slot.disposition === 'required').length,
    advisorySlotCount: slots.filter((slot) => slot.disposition === 'advisory').length,
    validation,
    readiness: readinessState.readiness,
    readinessLoading: readinessState.loading,
    readinessError: readinessState.error,
    options,
    fidelityReferences,
    fidelityLiveRuns,
    revise,
    changeProfile,
    changeRunBinding,
    changeFidelityReference,
    changeFidelityLive,
    applyControlledPair,
    reset: () => setDraft(initialDraft(catalog, lockedChangeProfile)),
  }
}

export type EvaluationCampaignBuilderModel = ReturnType<typeof useEvaluationCampaignBuilder>
