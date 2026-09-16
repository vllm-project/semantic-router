import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign,
  EvaluationCampaignDecision,
  EvaluationCampaignEvidenceAnchor,
  EvaluationCampaignGate,
  EvaluationCampaignGateBindings,
} from '../types/evaluationCampaign'
import type { EvaluationChangeProfileId } from '../types/evaluationPlane'
import {
  EVALUATION_ATTESTATION_REVISION,
  EVALUATION_CAMPAIGN_CONTRACT_VERSION,
  EVALUATION_RELEASE_GATE_IDS,
  EVALUATION_SCHEMA_VERSION,
} from '../types/evaluationPlane'
import {
  canonicalEvaluationCampaignGateBindings,
  evaluationCampaignExpectedAnchors,
  isEvaluationCampaignGateBindings,
  type EvaluationCampaignExpectedAnchor,
} from './evaluationCampaignBindingContract'
import { decodeEvaluationCampaignPairedLiveEvidence } from './evaluationCampaignPairedContract'
import { decodeEvaluationCampaignFidelityEvidence } from './evaluationCampaignFidelityContract'
import {
  hasOnlyEvaluationFields as exact,
  isEvaluationRecord as record,
  isFiniteNumber as finite,
  isKnownValue as known,
  isNonNegativeInteger as integer,
  isPortableEvaluationID,
  EVALUATION_GATE_DISPOSITION_SET,
  EVALUATION_GATE_VERDICT_SET,
  EVALUATION_SUMMARY_VERDICT_SET,
} from './evaluationContractValidation'
import { isCanonicalEvaluationRunID } from './evaluationRunContract'

const EVIDENCE_LEVELS = new Set(['E0', 'E1', 'E2', 'E3', 'E4', 'E5'])
const DIGEST = /^sha256:[0-9a-f]{64}$/
const NAME_LIMIT = 200
const DESCRIPTION_LIMIT = 4000

function text(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.trim() === value
}

function boundedText(value: unknown, maximum: number, allowEmpty: boolean): value is string {
  return (
    typeof value === 'string' &&
    value.trim() === value &&
    (allowEmpty || value.length > 0) &&
    new TextEncoder().encode(value).length <= maximum
  )
}

function texts(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(text)
}

function timestamp(value: unknown): value is string {
  return text(value) && Number.isFinite(Date.parse(value))
}

function digest(value: unknown): value is string {
  return typeof value === 'string' && DIGEST.test(value)
}

function validThreshold(value: unknown): boolean {
  return (
    record(value) &&
    exact(value, ['operator', 'value', 'unit']) &&
    (value.operator === '>=' || value.operator === '<=') &&
    finite(value.value) &&
    text(value.unit)
  )
}

function decodeGate(value: unknown, index: number): EvaluationCampaignGate {
  const gateID = EVALUATION_RELEASE_GATE_IDS[index]
  if (
    !gateID ||
    !record(value) ||
    !exact(value, [
      'id',
      'name',
      'disposition',
      'verdict',
      'evidence_level',
      'source',
      'evidence_refs',
      'observed',
      'threshold',
      'sample_count',
      'rationale',
    ]) ||
    value.id !== gateID ||
    !text(value.name) ||
    !known(value.disposition, EVALUATION_GATE_DISPOSITION_SET) ||
    !known(value.verdict, EVALUATION_GATE_VERDICT_SET) ||
    !known(value.evidence_level, EVIDENCE_LEVELS) ||
    !text(value.source) ||
    !texts(value.evidence_refs) ||
    new Set(value.evidence_refs).size !== value.evidence_refs.length ||
    (value.observed !== undefined && !finite(value.observed)) ||
    (value.threshold !== undefined && !validThreshold(value.threshold)) ||
    (value.sample_count !== undefined && !integer(value.sample_count)) ||
    !text(value.rationale)
  ) {
    throw new Error(`Evaluation campaign gate ${gateID || index} is invalid.`)
  }
  if (
    (value.disposition === 'not_applicable' && value.verdict !== 'not_applicable') ||
    ((value.disposition === 'required' || value.disposition === 'advisory') &&
      value.verdict === 'not_applicable') ||
    (value.observed === undefined) !== (value.threshold === undefined)
  ) {
    throw new Error(`Evaluation campaign gate ${gateID} has an invalid verdict or observation.`)
  }
  return value as unknown as EvaluationCampaignGate
}

function decodeEvidenceAnchor(
  value: unknown,
  expected: EvaluationCampaignExpectedAnchor,
): EvaluationCampaignEvidenceAnchor {
  if (
    !record(value) ||
    !exact(value, [
      'slot_id',
      'gate_id',
      'binding_role',
      'run_id',
      'candidate_subject_digest',
      'manifest_semantic_digest',
      'manifest_artifact_digest',
      'report_digest',
      'private_receipt_digest',
      'execution_attestation_digest',
    ]) ||
    value.slot_id !== expected.slot_id ||
    value.gate_id !== expected.gate_id ||
    value.binding_role !== expected.binding_role ||
    value.run_id !== expected.run_id ||
    !isCanonicalEvaluationRunID(value.run_id) ||
    (value.candidate_subject_digest !== undefined && !digest(value.candidate_subject_digest)) ||
    !digest(value.manifest_semantic_digest) ||
    !digest(value.manifest_artifact_digest) ||
    !digest(value.report_digest) ||
    !digest(value.private_receipt_digest) ||
    (value.execution_attestation_digest !== undefined &&
      !digest(value.execution_attestation_digest))
  ) {
    throw new Error(`Evaluation campaign evidence anchor ${expected.slot_id} is invalid.`)
  }
  return value as unknown as EvaluationCampaignEvidenceAnchor
}

function decodeDecision(
  value: unknown,
  campaignID: string,
  campaignDigest: string,
  changeProfile: EvaluationChangeProfileId,
  bindings: EvaluationCampaignGateBindings,
): EvaluationCampaignDecision {
  if (
    !record(value) ||
    !exact(value, [
      'schema_version',
      'contract_version',
      'attestation_revision',
      'campaign_id',
      'campaign_digest',
      'decision_digest',
      'verdict',
      'summary',
      'gates',
      'evidence',
      'paired_live_evidence',
      'fidelity_evidence',
      'recommendations',
      'created_at',
    ]) ||
    value.schema_version !== EVALUATION_SCHEMA_VERSION ||
    value.contract_version !== EVALUATION_CAMPAIGN_CONTRACT_VERSION ||
    value.attestation_revision !== EVALUATION_ATTESTATION_REVISION ||
    value.campaign_id !== campaignID ||
    value.campaign_digest !== campaignDigest ||
    !digest(value.decision_digest) ||
    !known(value.verdict, EVALUATION_SUMMARY_VERDICT_SET) ||
    !text(value.summary) ||
    !Array.isArray(value.gates) ||
    value.gates.length !== EVALUATION_RELEASE_GATE_IDS.length ||
    !Array.isArray(value.evidence) ||
    !texts(value.recommendations) ||
    !timestamp(value.created_at)
  ) {
    throw new Error('Evaluation campaign decision is invalid.')
  }
  const gates = value.gates.map((gate, index) => decodeGate(gate, index))
  const expectedAnchors = evaluationCampaignExpectedAnchors(bindings)
  if (value.evidence.length !== expectedAnchors.length) {
    throw new Error('Evaluation campaign decision evidence is incomplete.')
  }
  const evidence = value.evidence.map((anchor, index) =>
    decodeEvidenceAnchor(anchor, expectedAnchors[index]),
  )
  const controlledPair = bindings.g3_controlled_pair
  if (Boolean(controlledPair) !== (value.paired_live_evidence !== undefined)) {
    throw new Error('Evaluation campaign paired-live evidence presence is invalid.')
  }
  const pairedLiveEvidence = controlledPair
    ? decodeEvaluationCampaignPairedLiveEvidence(
        value.paired_live_evidence,
        controlledPair,
        evidence,
      )
    : undefined
  const fidelityBinding = bindings.g5_fidelity
  if (Boolean(fidelityBinding) !== (value.fidelity_evidence !== undefined)) {
    throw new Error('Evaluation campaign fidelity evidence presence is invalid.')
  }
  const fidelityEvidence = fidelityBinding
    ? decodeEvaluationCampaignFidelityEvidence(
        value.fidelity_evidence,
        changeProfile,
        fidelityBinding,
        evidence,
      )
    : undefined
  const requiredGates = gates.filter((gate) => gate.disposition === 'required')
  const expectedVerdict = requiredGates.some((gate) => gate.verdict === 'fail')
    ? 'fail'
    : requiredGates.every((gate) => gate.verdict === 'pass')
      ? 'pass'
      : 'unavailable'
  if (value.verdict !== expectedVerdict) {
    throw new Error('Evaluation campaign decision verdict does not match its required gates.')
  }
  return {
    ...(value as unknown as EvaluationCampaignDecision),
    gates,
    evidence,
    ...(pairedLiveEvidence ? { paired_live_evidence: pairedLiveEvidence } : {}),
    ...(fidelityEvidence ? { fidelity_evidence: fidelityEvidence } : {}),
  }
}

function decodeEvaluationCampaignDecision(
  payload: unknown,
  campaign: Pick<EvaluationCampaign, 'id' | 'manifest_digest' | 'change_profile' | 'gate_bindings'>,
): EvaluationCampaignDecision {
  return decodeDecision(
    payload,
    campaign.id,
    campaign.manifest_digest,
    campaign.change_profile,
    campaign.gate_bindings,
  )
}

export function decodeEvaluationCampaign(
  payload: unknown,
  expectedID?: string,
): EvaluationCampaign {
  if (
    !record(payload) ||
    !exact(payload, [
      'schema_version',
      'contract_version',
      'id',
      'name',
      'description',
      'change_profile',
      'status',
      'gate_bindings',
      'manifest_digest',
      'created_at',
      'decision',
    ]) ||
    payload.schema_version !== EVALUATION_SCHEMA_VERSION ||
    payload.contract_version !== EVALUATION_CAMPAIGN_CONTRACT_VERSION ||
    !isCanonicalEvaluationRunID(payload.id) ||
    (expectedID !== undefined && payload.id !== expectedID) ||
    !boundedText(payload.name, NAME_LIMIT, false) ||
    !boundedText(payload.description, DESCRIPTION_LIMIT, true) ||
    !isPortableEvaluationID(payload.change_profile) ||
    payload.status !== 'decided' ||
    !isEvaluationCampaignGateBindings(payload.gate_bindings) ||
    !digest(payload.manifest_digest) ||
    !timestamp(payload.created_at)
  ) {
    throw new Error('Evaluation campaign response is incomplete.')
  }
  const bindings = canonicalEvaluationCampaignGateBindings(payload.gate_bindings)
  const decision = decodeEvaluationCampaignDecision(payload.decision, {
    id: payload.id,
    manifest_digest: payload.manifest_digest,
    change_profile: payload.change_profile as EvaluationChangeProfileId,
    gate_bindings: bindings,
  })
  if (decision.created_at !== payload.created_at) {
    throw new Error('Evaluation campaign decision timestamp does not match its campaign.')
  }
  return {
    ...(payload as unknown as EvaluationCampaign),
    gate_bindings: bindings,
    decision,
  }
}

export function buildCreateEvaluationCampaignPayload(
  request: CreateEvaluationCampaignPayload,
): CreateEvaluationCampaignPayload {
  if (
    !record(request) ||
    !exact(request, ['client_request_id', 'name', 'description', 'change_profile', 'gate_bindings'])
  ) {
    throw new Error('Campaign request contains non-contract fields.')
  }
  const name = request.name.trim()
  const description = request.description.trim()
  if (!isCanonicalEvaluationRunID(request.client_request_id)) {
    throw new Error('Campaign request identity must be a canonical UUID.')
  }
  if (!boundedText(name, NAME_LIMIT, false)) {
    throw new Error(`Campaign name must be at most ${NAME_LIMIT} bytes.`)
  }
  if (!boundedText(description, DESCRIPTION_LIMIT, true)) {
    throw new Error(`Campaign description must be at most ${DESCRIPTION_LIMIT} bytes.`)
  }
  if (!isPortableEvaluationID(request.change_profile)) {
    throw new Error('Campaign change profile identity is invalid.')
  }
  return {
    client_request_id: request.client_request_id,
    name,
    description,
    change_profile: request.change_profile,
    gate_bindings: canonicalEvaluationCampaignGateBindings(request.gate_bindings),
  }
}
