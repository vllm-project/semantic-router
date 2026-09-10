import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaignGateBindings,
  EvaluationCampaignReadiness,
  EvaluationCampaignSlotReadiness,
} from '../../types/evaluationCampaign'
import type {
  EvaluationCatalog,
  EvaluationCatalogCampaignSlot,
  EvaluationChangeProfileId,
  EvaluationRun,
} from '../../types/evaluationPlane'
import { canonicalEvaluationCampaignGateBindings } from '../../utils/evaluationCampaignBindingContract'
import { newEvaluationClientRequestID } from '../../utils/evaluationIdentity'

const EVALUATION_CAMPAIGN_LIMITS = { name: 200, description: 4000 } as const

export interface EvaluationCampaignDraft {
  clientRequestID: string
  name: string
  description: string
  changeProfile: EvaluationChangeProfileId
  gateBindings: EvaluationCampaignGateBindings
}

export function newEvaluationCampaignClientRequestID(): string {
  return newEvaluationClientRequestID()
}

function readinessForSlot(
  readiness: EvaluationCampaignReadiness | null,
  slot: EvaluationCatalogCampaignSlot,
): EvaluationCampaignSlotReadiness | undefined {
  return readiness?.slots.find((candidate) => candidate.gate_id === slot.gate_id)
}

function runsWithIDs(runs: EvaluationRun[], runIDs: string[]): EvaluationRun[] {
  const byID = new Map(runs.map((run) => [run.id, run]))
  return runIDs.flatMap((runID) => {
    const run = byID.get(runID)
    return run ? [run] : []
  })
}

export function controlledPairBaselineSourceOptions(
  runs: EvaluationRun[],
  readiness: EvaluationCampaignReadiness | null,
  slot: EvaluationCatalogCampaignSlot,
): EvaluationRun[] {
  return runsWithIDs(
    runs,
    readinessForSlot(readiness, slot)?.controlled_pair_source_run_ids || [],
  )
}

export function controlledPairCandidateSourceOptions(
  runs: EvaluationRun[],
  readiness: EvaluationCampaignReadiness | null,
  slot: EvaluationCatalogCampaignSlot,
): EvaluationRun[] {
  return runsWithIDs(
    runs,
    readinessForSlot(readiness, slot)?.controlled_pair_candidate_run_ids || [],
  )
}

export function campaignRunOptions(
  runs: EvaluationRun[],
  readiness: EvaluationCampaignReadiness | null,
  slot: EvaluationCatalogCampaignSlot,
): EvaluationRun[] {
  return runsWithIDs(runs, readinessForSlot(readiness, slot)?.eligible_run_ids || [])
}

export function fidelityReferenceOptions(
  runs: EvaluationRun[],
  readiness: EvaluationCampaignReadiness | null,
  slot: EvaluationCatalogCampaignSlot,
): EvaluationRun[] {
  return runsWithIDs(runs, readinessForSlot(readiness, slot)?.fidelity_reference_run_ids || [])
}

export function fidelityLiveOptions(
  runs: EvaluationRun[],
  readiness: EvaluationCampaignReadiness | null,
  slot: EvaluationCatalogCampaignSlot,
): EvaluationRun[] {
  return runsWithIDs(runs, readinessForSlot(readiness, slot)?.fidelity_live_run_ids || [])
}

export function campaignSlotRunIDs(
  slot: EvaluationCatalogCampaignSlot,
  bindings: EvaluationCampaignGateBindings,
): string[] {
  switch (slot.gate_id) {
    case 'G2':
      return bindings.g2_run_id ? [bindings.g2_run_id] : []
    case 'G3':
      return bindings.g3_controlled_pair
        ? [
            bindings.g3_controlled_pair.baseline_run_id,
            bindings.g3_controlled_pair.candidate_run_id,
          ].filter(Boolean)
        : []
    case 'G4':
      return bindings.g4_run_id ? [bindings.g4_run_id] : []
    case 'G5':
      return bindings.g5_fidelity
        ? [bindings.g5_fidelity.reference_run_id, bindings.g5_fidelity.live_run_id].filter(Boolean)
        : []
    case 'G6':
      return bindings.g6_run_id ? [bindings.g6_run_id] : []
    case 'G7':
      return bindings.g7_run_id ? [bindings.g7_run_id] : []
    case 'G8':
      return bindings.g8_run_id ? [bindings.g8_run_id] : []
    case 'G9':
      return bindings.g9_run_id ? [bindings.g9_run_id] : []
  }
}

function utf8Length(value: string): number {
  return new TextEncoder().encode(value).length
}

function pairAvailable(
  slotReadiness: EvaluationCampaignSlotReadiness,
  bindingKind: EvaluationCatalogCampaignSlot['binding_kind'],
  firstRunID: string,
  secondRunID: string,
): boolean {
  if (bindingKind === 'controlled_pair') {
    return (
      slotReadiness.controlled_pair_source_run_ids.includes(firstRunID) &&
      slotReadiness.controlled_pair_candidate_run_ids.includes(secondRunID)
    )
  }
  if (bindingKind === 'fidelity_pair') {
    return (
      slotReadiness.fidelity_reference_run_ids.includes(firstRunID) &&
      slotReadiness.fidelity_live_run_ids.includes(secondRunID)
    )
  }
  return false
}

export function validateEvaluationCampaignDraft(
  catalog: EvaluationCatalog,
  runs: EvaluationRun[],
  draft: EvaluationCampaignDraft,
  readiness: EvaluationCampaignReadiness | null,
  readinessLoading: boolean,
  readinessError: string | null,
  ledgerAvailable: boolean,
  ledgerComplete: boolean,
  allRunsLoaded: boolean,
): string | null {
  const name = draft.name.trim()
  const description = draft.description.trim()
  if (!ledgerAvailable) return 'Run history is unavailable. Retry before creating a decision.'
  if (!ledgerComplete) {
    return 'Some saved runs could not be verified. Repair or remove them before creating a decision.'
  }
  if (!allRunsLoaded) return 'Load all runs before selecting evidence for this decision.'
  if (readinessLoading) return 'Verifying which runs can support this release decision.'
  if (readinessError) return 'Release-check readiness is unavailable. Retry before continuing.'
  if (!name) return 'Enter a decision name.'
  if (utf8Length(name) > EVALUATION_CAMPAIGN_LIMITS.name) {
    return 'The decision name is too long. Shorten it before continuing.'
  }
  if (utf8Length(description) > EVALUATION_CAMPAIGN_LIMITS.description) {
    return 'The decision notes are too long. Shorten them before continuing.'
  }
  const profile = catalog.change_profiles.find((candidate) => candidate.id === draft.changeProfile)
  if (!profile) return 'Select an available change type.'
  if (!readiness || readiness.change_profile !== profile.id) {
    return 'Release-check readiness has not been verified for this change type.'
  }
  if (readiness.total_runs !== runs.length) {
    return 'Run history changed while release checks were verified. Refresh runs before continuing.'
  }
  const byID = new Map(runs.map((run) => [run.id, run]))
  const allBoundRunIDs = profile.campaign_slots.flatMap((slot) =>
    campaignSlotRunIDs(slot, draft.gateBindings),
  )
  if (new Set(allBoundRunIDs).size !== allBoundRunIDs.length) {
    return 'Use a different completed run for each release check.'
  }
  for (const slot of profile.campaign_slots) {
    const ids = campaignSlotRunIDs(slot, draft.gateBindings)
    const slotReadiness = readinessForSlot(readiness, slot)
    if (!slotReadiness) return 'Release-check readiness is incomplete. Retry before continuing.'
    if (slot.disposition === 'required' && ids.length === 0) {
      return 'Select evidence for every required release check.'
    }
    if (slot.disposition === 'not_applicable' && ids.length > 0) {
      return 'Remove evidence from checks that are not required for this change.'
    }
    if (ids.some((id) => !byID.has(id))) {
      return 'One selected run is no longer available. Refresh run history and select it again.'
    }
    if (slot.binding_kind === 'run' && ids.some((id) => !slotReadiness.eligible_run_ids.includes(id))) {
      return 'One selected run no longer meets its release check. Choose a compatible completed run.'
    }
    if (slot.binding_kind !== 'run' && ids.length === 1) {
      return 'Select both runs for each comparison check.'
    }
    if (new Set(ids).size !== ids.length) {
      return 'Use two different runs for each comparison check.'
    }
    if (
      slot.binding_kind !== 'run' &&
      ids.length === 2 &&
      !pairAvailable(slotReadiness, slot.binding_kind, ids[0], ids[1])
    ) {
      return slot.binding_kind === 'controlled_pair'
        ? 'Launch a fresh controlled comparison for the selected baseline and candidate.'
        : 'Select a matching reference and fresh candidate run.'
    }
  }
  return null
}

export function buildEvaluationCampaignRequest(
  draft: EvaluationCampaignDraft,
): CreateEvaluationCampaignPayload {
  return {
    client_request_id: draft.clientRequestID,
    name: draft.name.trim(),
    description: draft.description.trim(),
    change_profile: draft.changeProfile,
    gate_bindings: canonicalEvaluationCampaignGateBindings(draft.gateBindings),
  }
}
