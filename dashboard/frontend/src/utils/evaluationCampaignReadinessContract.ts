import type {
  EvaluationCampaignReadiness,
  EvaluationCampaignSlotReadiness,
} from '../types/evaluationCampaign'
import type { EvaluationCatalogChangeProfile } from '../types/evaluationPlane'
import {
  assertCurrentEvaluationContract,
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isNonNegativeInteger,
} from './evaluationContractValidation'
import { isCanonicalEvaluationRunID } from './evaluationRunContract'

export interface EvaluationCampaignReadinessAnchors {
  controlledPairBaselineRunID?: string
  fidelityReferenceRunID?: string
}

function uniqueRunIDs(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.every(isCanonicalEvaluationRunID) &&
    new Set(value).size === value.length
  )
}

function slotReadiness(
  value: unknown,
  expected: EvaluationCatalogChangeProfile['campaign_slots'][number],
  anchors: EvaluationCampaignReadinessAnchors,
): value is EvaluationCampaignSlotReadiness {
  if (
    !isEvaluationRecord(value) ||
    !hasOnlyEvaluationFields(value, [
      'gate_id',
      'binding_kind',
      'eligible_run_ids',
      'controlled_pair_source_run_ids',
      'controlled_pair_candidate_run_ids',
      'fidelity_reference_run_ids',
      'fidelity_live_run_ids',
    ]) ||
    value.gate_id !== expected.gate_id ||
    value.binding_kind !== expected.binding_kind ||
    !uniqueRunIDs(value.eligible_run_ids) ||
    !uniqueRunIDs(value.controlled_pair_source_run_ids) ||
    !uniqueRunIDs(value.controlled_pair_candidate_run_ids) ||
    !uniqueRunIDs(value.fidelity_reference_run_ids) ||
    !uniqueRunIDs(value.fidelity_live_run_ids)
  ) {
    return false
  }
  if (
    (!anchors.controlledPairBaselineRunID &&
      value.controlled_pair_candidate_run_ids.length > 0) ||
    (!anchors.fidelityReferenceRunID && value.fidelity_live_run_ids.length > 0) ||
    (anchors.controlledPairBaselineRunID !== undefined &&
      value.controlled_pair_candidate_run_ids.includes(
        anchors.controlledPairBaselineRunID,
      )) ||
    (anchors.fidelityReferenceRunID !== undefined &&
      value.fidelity_live_run_ids.includes(anchors.fidelityReferenceRunID))
  ) {
    return false
  }
  if (expected.binding_kind === 'run') {
    return (
      value.controlled_pair_source_run_ids.length === 0 &&
      value.controlled_pair_candidate_run_ids.length === 0 &&
      value.fidelity_reference_run_ids.length === 0 &&
      value.fidelity_live_run_ids.length === 0
    )
  }
  if (expected.binding_kind === 'controlled_pair') {
    return (
      value.eligible_run_ids.length === 0 &&
      value.fidelity_reference_run_ids.length === 0 &&
      value.fidelity_live_run_ids.length === 0
    )
  }
  return (
    value.eligible_run_ids.length === 0 &&
    value.controlled_pair_source_run_ids.length === 0 &&
    value.controlled_pair_candidate_run_ids.length === 0
  )
}

export function decodeEvaluationCampaignReadiness(
  payload: unknown,
  profile: EvaluationCatalogChangeProfile,
  anchors: EvaluationCampaignReadinessAnchors,
): EvaluationCampaignReadiness {
  assertCurrentEvaluationContract(payload, 'Evaluation campaign readiness response')
  if (
    !hasOnlyEvaluationFields(payload, [
      'schema_version',
      'change_profile',
      'next_cursor',
      'total_runs',
      'slots',
    ]) ||
    payload.change_profile !== profile.id ||
    (payload.next_cursor !== undefined &&
      (typeof payload.next_cursor !== 'string' || payload.next_cursor.length === 0)) ||
    !isNonNegativeInteger(payload.total_runs) ||
    !Array.isArray(payload.slots) ||
    payload.slots.length !== profile.campaign_slots.length ||
    payload.slots.some(
      (slot, index) => !slotReadiness(slot, profile.campaign_slots[index], anchors),
    )
  ) {
    throw new Error('Evaluation campaign readiness response is incomplete.')
  }
  return payload as unknown as EvaluationCampaignReadiness
}

function appendPageRunIDs(target: string[], additions: string[]): void {
  const existing = new Set(target)
  for (const runID of additions) {
    if (existing.has(runID)) {
      throw new Error('Evaluation campaign readiness pages overlap.')
    }
    target.push(runID)
    existing.add(runID)
  }
}

export function mergeEvaluationCampaignReadinessPages(
  pages: EvaluationCampaignReadiness[],
): EvaluationCampaignReadiness {
  const first = pages[0]
  if (!first) throw new Error('Evaluation campaign readiness response is incomplete.')
  const merged: EvaluationCampaignReadiness = {
    schema_version: first.schema_version,
    change_profile: first.change_profile,
    total_runs: first.total_runs,
    slots: first.slots.map((slot) => ({
      gate_id: slot.gate_id,
      binding_kind: slot.binding_kind,
      eligible_run_ids: [],
      controlled_pair_source_run_ids: [],
      controlled_pair_candidate_run_ids: [],
      fidelity_reference_run_ids: [],
      fidelity_live_run_ids: [],
    })),
  }
  for (const page of pages) {
    if (
      page.schema_version !== merged.schema_version ||
      page.change_profile !== merged.change_profile ||
      page.total_runs !== merged.total_runs ||
      page.slots.length !== merged.slots.length
    ) {
      throw new Error('Evaluation campaign readiness pages are inconsistent.')
    }
    page.slots.forEach((slot, index) => {
      const target = merged.slots[index]
      if (slot.gate_id !== target.gate_id || slot.binding_kind !== target.binding_kind) {
        throw new Error('Evaluation campaign readiness pages are inconsistent.')
      }
      appendPageRunIDs(target.eligible_run_ids, slot.eligible_run_ids)
      appendPageRunIDs(
        target.controlled_pair_source_run_ids,
        slot.controlled_pair_source_run_ids,
      )
      appendPageRunIDs(
        target.controlled_pair_candidate_run_ids,
        slot.controlled_pair_candidate_run_ids,
      )
      appendPageRunIDs(target.fidelity_reference_run_ids, slot.fidelity_reference_run_ids)
      appendPageRunIDs(target.fidelity_live_run_ids, slot.fidelity_live_run_ids)
    })
  }
  return merged
}
