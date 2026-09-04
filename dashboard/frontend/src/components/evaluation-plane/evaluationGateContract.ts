import type { EvaluationChangeProfileId, GateDisposition } from '../../types/evaluationPlane'

export const SUPPORTED_GATE_CONTRACT_VERSION = 'evaluation-release-gates.v1'

export interface GateContractDefinition {
  id: `G${number}`
  name: string
  description: string
}

export const GATE_CONTRACT_DEFINITIONS: readonly GateContractDefinition[] = [
  {
    id: 'G0',
    name: 'Reproducibility',
    description: 'Frozen manifests, snapshots, seeds, failures, and artifact lineage.',
  },
  {
    id: 'G1',
    name: 'Static correctness',
    description: 'Strict schemas, conformance, references, coverage, and deterministic replay.',
  },
  {
    id: 'G2',
    name: 'Hard policy',
    description: 'Privacy, security, locality, authorization, modality, context, and tool rules.',
  },
  {
    id: 'G3',
    name: 'Offline value',
    description: 'Qualified lift over the baseline and no-information frontier.',
  },
  {
    id: 'G4',
    name: 'Robustness / OOD',
    description: 'Invariant, temporal, source, language, domain, and modality slices.',
  },
  {
    id: 'G5',
    name: 'Live fidelity',
    description: 'Replay-to-live agreement, fresh outputs, and complete failure accounting.',
  },
  {
    id: 'G6',
    name: 'Reliability / trajectory',
    description: 'Terminal success, continuity, recovery, isolation, and tool idempotency.',
  },
  {
    id: 'G7',
    name: 'Cost / latency / capacity',
    description: 'Three ledgers, latency decomposition, saturation, SLOs, and headroom.',
  },
  {
    id: 'G8',
    name: 'Shadow / canary',
    description: 'Qualified assignment, divergence, guardrails, risk budget, and rollback.',
  },
  {
    id: 'G9',
    name: 'Online preference',
    description: 'Exposure, propensity, effective sample size, confidence, and segments.',
  },
]

const required = 'required'
const advisory = 'advisory'
const notApplicable = 'not_applicable'

const PROFILE_APPLICABILITY: Record<EvaluationChangeProfileId, readonly GateDisposition[]> = {
  schema_adapter: [
    required,
    required,
    advisory,
    advisory,
    required,
    advisory,
    notApplicable,
    advisory,
    notApplicable,
    notApplicable,
  ],
  recipe: [
    required,
    required,
    required,
    required,
    required,
    required,
    notApplicable,
    required,
    advisory,
    notApplicable,
  ],
  selector: [
    required,
    required,
    required,
    required,
    required,
    required,
    advisory,
    required,
    required,
    notApplicable,
  ],
  model_pool: [
    required,
    required,
    required,
    required,
    required,
    required,
    advisory,
    required,
    required,
    notApplicable,
  ],
  runtime_capacity: [
    required,
    required,
    required,
    advisory,
    advisory,
    required,
    advisory,
    required,
    required,
    notApplicable,
  ],
  agent_multimodal: [
    required,
    required,
    required,
    required,
    required,
    required,
    required,
    required,
    required,
    advisory,
  ],
  online_adaptation: [
    required,
    required,
    required,
    required,
    required,
    required,
    required,
    required,
    required,
    required,
  ],
}

export interface ApplicableGate extends GateContractDefinition {
  disposition: GateDisposition
}

export function gateApplicabilityForProfile(
  profile: EvaluationChangeProfileId,
): readonly ApplicableGate[] {
  return GATE_CONTRACT_DEFINITIONS.map((gate, index) => ({
    ...gate,
    disposition: PROFILE_APPLICABILITY[profile][index],
  }))
}
