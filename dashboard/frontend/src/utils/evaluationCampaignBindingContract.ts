import type {
  EvaluationCampaignEvidenceBindingRole,
  EvaluationCampaignGateBindings,
} from '../types/evaluationCampaign'
import type { EvaluationCampaignGateID, EvaluationCampaignSlotID } from '../types/evaluationPlane'
import {
  hasOnlyEvaluationFields as exact,
  isEvaluationRecord as record,
} from './evaluationContractValidation'
import { isCanonicalEvaluationRunID } from './evaluationRunContract'

export interface EvaluationCampaignExpectedAnchor {
  slot_id: EvaluationCampaignSlotID
  gate_id: EvaluationCampaignGateID
  binding_role: EvaluationCampaignEvidenceBindingRole
  run_id: string
}

const BINDING_FIELDS = [
  'g2_run_id',
  'g3_controlled_pair',
  'g4_run_id',
  'g5_fidelity',
  'g6_run_id',
  'g7_run_id',
  'g8_run_id',
  'g9_run_id',
] as const

function canonicalOptionalRunID(value: unknown): boolean {
  return value === undefined || isCanonicalEvaluationRunID(value)
}

function controlledPair(value: unknown): boolean {
  return (
    value === undefined ||
    (record(value) &&
      exact(value, ['baseline_run_id', 'candidate_run_id']) &&
      isCanonicalEvaluationRunID(value.baseline_run_id) &&
      isCanonicalEvaluationRunID(value.candidate_run_id) &&
      value.baseline_run_id !== value.candidate_run_id)
  )
}

function fidelityPair(value: unknown): boolean {
  return (
    value === undefined ||
    (record(value) &&
      exact(value, ['reference_run_id', 'live_run_id']) &&
      isCanonicalEvaluationRunID(value.reference_run_id) &&
      isCanonicalEvaluationRunID(value.live_run_id) &&
      value.reference_run_id !== value.live_run_id)
  )
}

export function isEvaluationCampaignGateBindings(
  value: unknown,
): value is EvaluationCampaignGateBindings {
  if (
    !record(value) ||
    !exact(value, BINDING_FIELDS) ||
    !canonicalOptionalRunID(value.g2_run_id) ||
    !controlledPair(value.g3_controlled_pair) ||
    !canonicalOptionalRunID(value.g4_run_id) ||
    !fidelityPair(value.g5_fidelity) ||
    !canonicalOptionalRunID(value.g6_run_id) ||
    !canonicalOptionalRunID(value.g7_run_id) ||
    !canonicalOptionalRunID(value.g8_run_id) ||
    !canonicalOptionalRunID(value.g9_run_id)
  ) {
    return false
  }
  const runIDs = evaluationCampaignExpectedAnchors(
    value as unknown as EvaluationCampaignGateBindings,
  ).map((anchor) => anchor.run_id)
  return new Set(runIDs).size === runIDs.length
}

export function canonicalEvaluationCampaignGateBindings(
  bindings: EvaluationCampaignGateBindings,
): EvaluationCampaignGateBindings {
  if (!isEvaluationCampaignGateBindings(bindings)) {
    throw new Error('Campaign gate bindings are malformed.')
  }
  return {
    ...(bindings.g2_run_id ? { g2_run_id: bindings.g2_run_id } : {}),
    ...(bindings.g3_controlled_pair
      ? {
          g3_controlled_pair: {
            baseline_run_id: bindings.g3_controlled_pair.baseline_run_id,
            candidate_run_id: bindings.g3_controlled_pair.candidate_run_id,
          },
        }
      : {}),
    ...(bindings.g4_run_id ? { g4_run_id: bindings.g4_run_id } : {}),
    ...(bindings.g5_fidelity
      ? {
          g5_fidelity: {
            reference_run_id: bindings.g5_fidelity.reference_run_id,
            live_run_id: bindings.g5_fidelity.live_run_id,
          },
        }
      : {}),
    ...(bindings.g6_run_id ? { g6_run_id: bindings.g6_run_id } : {}),
    ...(bindings.g7_run_id ? { g7_run_id: bindings.g7_run_id } : {}),
    ...(bindings.g8_run_id ? { g8_run_id: bindings.g8_run_id } : {}),
    ...(bindings.g9_run_id ? { g9_run_id: bindings.g9_run_id } : {}),
  }
}

export function evaluationCampaignExpectedAnchors(
  bindings: EvaluationCampaignGateBindings,
): EvaluationCampaignExpectedAnchor[] {
  const anchors: EvaluationCampaignExpectedAnchor[] = []
  const run = (
    slot_id: EvaluationCampaignSlotID,
    gate_id: EvaluationCampaignGateID,
    run_id?: string,
  ) => {
    if (run_id) anchors.push({ slot_id, gate_id, binding_role: 'evidence', run_id })
  }
  run('g2', 'G2', bindings.g2_run_id)
  if (bindings.g3_controlled_pair) {
    anchors.push({
      slot_id: 'g3',
      gate_id: 'G3',
      binding_role: 'baseline',
      run_id: bindings.g3_controlled_pair.baseline_run_id,
    })
    anchors.push({
      slot_id: 'g3',
      gate_id: 'G3',
      binding_role: 'candidate',
      run_id: bindings.g3_controlled_pair.candidate_run_id,
    })
  }
  run('g4', 'G4', bindings.g4_run_id)
  if (bindings.g5_fidelity) {
    anchors.push({
      slot_id: 'g5',
      gate_id: 'G5',
      binding_role: 'reference',
      run_id: bindings.g5_fidelity.reference_run_id,
    })
    anchors.push({
      slot_id: 'g5',
      gate_id: 'G5',
      binding_role: 'live',
      run_id: bindings.g5_fidelity.live_run_id,
    })
  }
  run('g6', 'G6', bindings.g6_run_id)
  run('g7', 'G7', bindings.g7_run_id)
  run('g8', 'G8', bindings.g8_run_id)
  run('g9', 'G9', bindings.g9_run_id)
  return anchors
}
