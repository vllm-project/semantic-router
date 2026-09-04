import type { EvaluationRun } from '../../../src/types/evaluationPlane'
import { EVALUATION_ATTESTATION_REVISION } from '../../../src/types/evaluationPlane'
import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign,
  EvaluationCampaignEvidenceAnchor,
  EvaluationCampaignGateBindings,
} from '../../../src/types/evaluationCampaign'
import type { EvaluationFailureSummary } from '../../../src/types/evaluationReport'
import { evaluationFidelityEvidence, evaluationPairedLiveEvidence } from './campaignEvidence'
import { evaluationCatalog } from './catalog'

type CampaignEvidenceIdentity = Pick<
  EvaluationCampaignEvidenceAnchor,
  'slot_id' | 'gate_id' | 'binding_role' | 'run_id'
>

// This is the mock server's independent wire projection. Do not reuse the
// browser decoder/helper here: the E2E boundary must catch ordering or role
// drift between the client contract and the server response.
function campaignEvidenceIdentities(
  bindings: EvaluationCampaignGateBindings,
): CampaignEvidenceIdentity[] {
  const identities: CampaignEvidenceIdentity[] = []
  const add = (identity: CampaignEvidenceIdentity | undefined) => {
    if (identity?.run_id) identities.push(identity)
  }
  add(
    bindings.g2_run_id
      ? {
          slot_id: 'g2',
          gate_id: 'G2',
          binding_role: 'evidence',
          run_id: bindings.g2_run_id,
        }
      : undefined,
  )
  add(
    bindings.g3_controlled_pair
      ? {
          slot_id: 'g3',
          gate_id: 'G3',
          binding_role: 'baseline',
          run_id: bindings.g3_controlled_pair.baseline_run_id,
        }
      : undefined,
  )
  add(
    bindings.g3_controlled_pair
      ? {
          slot_id: 'g3',
          gate_id: 'G3',
          binding_role: 'candidate',
          run_id: bindings.g3_controlled_pair.candidate_run_id,
        }
      : undefined,
  )
  add(
    bindings.g4_run_id
      ? {
          slot_id: 'g4',
          gate_id: 'G4',
          binding_role: 'evidence',
          run_id: bindings.g4_run_id,
        }
      : undefined,
  )
  add(
    bindings.g5_fidelity
      ? {
          slot_id: 'g5',
          gate_id: 'G5',
          binding_role: 'reference',
          run_id: bindings.g5_fidelity.reference_run_id,
        }
      : undefined,
  )
  add(
    bindings.g5_fidelity
      ? {
          slot_id: 'g5',
          gate_id: 'G5',
          binding_role: 'live',
          run_id: bindings.g5_fidelity.live_run_id,
        }
      : undefined,
  )
  for (const [slotID, gateID, runID] of [
    ['g6', 'G6', bindings.g6_run_id],
    ['g7', 'G7', bindings.g7_run_id],
    ['g8', 'G8', bindings.g8_run_id],
    ['g9', 'G9', bindings.g9_run_id],
  ] as const) {
    add(
      runID
        ? { slot_id: slotID, gate_id: gateID, binding_role: 'evidence', run_id: runID }
        : undefined,
    )
  }
  return identities
}

export function evaluationCampaign(request: CreateEvaluationCampaignPayload): EvaluationCampaign {
  const campaignDigest = `sha256:${'c'.repeat(64)}`
  const evidence = campaignEvidenceIdentities(request.gate_bindings).map((anchor, index) => {
    const digit = ((index + 1) % 15).toString(16)
    return {
      ...anchor,
      ...(anchor.slot_id === 'g3' && anchor.binding_role === 'baseline'
        ? {}
        : { candidate_subject_digest: `sha256:${'e'.repeat(64)}` }),
      manifest_semantic_digest: `sha256:${digit.repeat(64)}`,
      manifest_artifact_digest: `sha256:${((index + 2) % 15).toString(16).repeat(64)}`,
      report_digest: `sha256:${((index + 4) % 15).toString(16).repeat(64)}`,
      private_receipt_digest: `sha256:${((index + 7) % 15).toString(16).repeat(64)}`,
      execution_attestation_digest: `sha256:${((index + 10) % 15).toString(16).repeat(64)}`,
    }
  })
  const baselineLive = evidence.find(
    (anchor) => anchor.slot_id === 'g3' && anchor.binding_role === 'baseline',
  )
  const candidateLive = evidence.find(
    (anchor) => anchor.slot_id === 'g3' && anchor.binding_role === 'candidate',
  )
  const fidelityReference = evidence.find(
    (anchor) => anchor.slot_id === 'g5' && anchor.binding_role === 'reference',
  )
  const fidelityLive = evidence.find(
    (anchor) => anchor.slot_id === 'g5' && anchor.binding_role === 'live',
  )
  const profile = evaluationCatalog.change_profiles.find(
    (candidate) => candidate.id === request.change_profile,
  )!
  const gateDefinitions = [
    { id: 'G0', name: 'Reproducibility', disposition: 'required' as const },
    { id: 'G1', name: 'Static correctness', disposition: 'required' as const },
    ...profile.campaign_slots.map((slot) => ({
      id: slot.gate_id,
      name: slot.name,
      disposition: slot.disposition,
    })),
  ]
  const gates = gateDefinitions.map((gate) => {
    const disposition = gate.disposition
    if (disposition === 'not_applicable') {
      return {
        id: gate.id,
        name: gate.name,
        disposition,
        verdict: 'not_applicable' as const,
        evidence_level: 'E5' as const,
        source: 'campaign_contract',
        evidence_refs: [],
        rationale: 'The gate is not applicable to this change profile.',
      }
    }
    return {
      id: gate.id,
      name: gate.name,
      disposition,
      verdict: 'pass' as const,
      evidence_level: 'E5' as const,
      source: gate.id === 'G0' || gate.id === 'G1' ? 'server_anchors' : 'gate_binding',
      evidence_refs: [],
      rationale: `${gate.name} is supported by sealed campaign evidence.`,
    }
  })
  const requiredGates = gates.filter((gate) => gate.disposition === 'required')
  const verdict = requiredGates.some((gate) => gate.verdict === 'fail')
    ? ('fail' as const)
    : requiredGates.every((gate) => gate.verdict === 'pass')
      ? ('pass' as const)
      : ('unavailable' as const)
  return {
    schema_version: 'evaluation.v1',
    contract_version: 'evaluation-campaign.v2',
    id: request.client_request_id,
    name: request.name,
    description: request.description,
    change_profile: request.change_profile,
    status: 'decided',
    gate_bindings: request.gate_bindings,
    manifest_digest: campaignDigest,
    created_at: '2026-08-30T02:00:00Z',
    decision: {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-campaign.v2',
      attestation_revision: EVALUATION_ATTESTATION_REVISION,
      campaign_id: request.client_request_id,
      campaign_digest: campaignDigest,
      decision_digest: `sha256:${'d'.repeat(64)}`,
      verdict,
      summary:
        verdict === 'pass'
          ? 'All required promotion campaign gates passed.'
          : 'One or more required promotion campaign gates remain unavailable.',
      gates,
      evidence,
      ...(baselineLive && candidateLive
        ? {
            paired_live_evidence: evaluationPairedLiveEvidence(baselineLive, candidateLive),
          }
        : {}),
      ...(fidelityReference && fidelityLive
        ? { fidelity_evidence: evaluationFidelityEvidence(fidelityReference, fidelityLive) }
        : {}),
      recommendations: ['Advance through the guarded rollout defined for this change profile.'],
      created_at: '2026-08-30T02:00:00Z',
    },
  }
}

export function failureSummary(run: EvaluationRun): EvaluationFailureSummary {
  return {
    schema_version: 'evaluation.v1',
    total_records: run.track_ids.length * 4,
    failed: 0,
    unavailable: 0,
    by_track: [...run.track_ids].sort().map((track_id) => ({
      track_id,
      succeeded: 4,
      failed: 0,
      unavailable: 0,
    })),
  }
}
