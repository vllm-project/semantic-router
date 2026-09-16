import {
  EVALUATION_METHOD_EVIDENCE_SOURCE,
  type EvaluationCatalog,
  type EvaluationCatalogMethod,
  type EvaluationMethodEvidenceSource,
  type EvaluationTrackId,
} from '../../types/evaluationPlane'
import type { EvaluationMethodReadinessStatus } from './evaluationMethodReadinessModel'
import { requiredEvaluationMethodMode } from './evaluationMethodReadinessModel'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'

interface EvaluationMethodTechnicalDetail {
  label: string
  message: string
}

export const EVALUATION_METHOD_STATUS_LABELS = {
  ready: 'Ready',
  setup_required: 'Setup required',
} as const satisfies Record<EvaluationMethodReadinessStatus, string>

export const EVALUATION_METHOD_EVIDENCE_SOURCE_LABELS = {
  [EVALUATION_METHOD_EVIDENCE_SOURCE.DIAGNOSTIC_FIXTURE]: 'Built-in diagnostic',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.LIVE_RUNTIME]: 'Live system',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.NORMALIZED_IMPORT]: 'Imported benchmark',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.SERVER_BROKERED_LIVE]: 'Managed live run',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.LIVE_PRODUCTION]: 'Sealed production results',
} as const satisfies Record<EvaluationMethodEvidenceSource, string>

const READY_GUIDANCE: Record<EvaluationMethodEvidenceSource, string> = {
  [EVALUATION_METHOD_EVIDENCE_SOURCE.DIAGNOSTIC_FIXTURE]:
    'Ready to verify the evaluation setup and report path.',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.LIVE_RUNTIME]:
    'Ready to run against the selected Mixture.',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.NORMALIZED_IMPORT]:
    'Ready for exploratory analysis. Use a managed run before making a release decision.',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.SERVER_BROKERED_LIVE]:
    'Ready for a managed workload-shift evaluation.',
  [EVALUATION_METHOD_EVIDENCE_SOURCE.LIVE_PRODUCTION]:
    'Configured sealed production-results source is ready for evaluation.',
}

const LIVE_SETUP_GUIDANCE: Record<EvaluationTrackId, string> = {
  routing: 'Connect complete routing-decision results for the selected Mixture, then refresh.',
  model_pool: 'Connect complete per-model results for the selected pool, then refresh.',
  joint: 'Connect complete routed-system outcomes for the selected Mixture, then refresh.',
  agentic: 'Connect complete repeated agent-task results for the selected Mixture, then refresh.',
  multimodal: 'Connect complete supported-input results for the selected Mixture, then refresh.',
  preference: 'Connect complete assigned preference outcomes, then refresh.',
  safety: 'Connect complete policy-test outcomes for the selected configuration, then refresh.',
  capacity: 'Connect repeated live-load results for the selected service objective, then refresh.',
}

export function evaluationMethodSetupGuidance(
  method: EvaluationCatalogMethod,
  readiness: EvaluationMethodReadinessStatus,
): string {
  if (readiness === 'ready') return READY_GUIDANCE[method.evidence_source]
  if (method.status === 'configured') {
    return requiredEvaluationMethodMode(method) === 'live'
      ? 'Connect a healthy Mixture that supports this method, then refresh.'
      : 'Connect an available replay target that supports this method, then refresh.'
  }
  if (method.evidence_source === EVALUATION_METHOD_EVIDENCE_SOURCE.LIVE_PRODUCTION) {
    return 'Connect complete sealed assignments, outcomes, and safety controls, then refresh.'
  }
  if (method.evidence_source === EVALUATION_METHOD_EVIDENCE_SOURCE.NORMALIZED_IMPORT) {
    return 'Import a verified benchmark suite with complete results, then refresh.'
  }
  if (method.evidence_source === EVALUATION_METHOD_EVIDENCE_SOURCE.SERVER_BROKERED_LIVE) {
    return 'Connect the managed workload-shift source, then refresh.'
  }
  if (method.evidence_source === EVALUATION_METHOD_EVIDENCE_SOURCE.DIAGNOSTIC_FIXTURE) {
    return 'Restore the built-in diagnostic data, then refresh.'
  }
  return LIVE_SETUP_GUIDANCE[method.track_id]
}

export function evaluationMethodCapabilityLabel(
  method: EvaluationCatalogMethod,
  qualifiedGateNames: string[],
): string {
  return qualifiedGateNames.length
    ? `${qualifiedGateNames.join(' and ')} method`
    : `${TRACK_PRESENTATION[method.track_id].label} measurement`
}

export function evaluationMethodTechnicalDetails({
  method,
  suiteID,
  revision,
  executors,
}: {
  method: EvaluationCatalogMethod
  suiteID: string
  revision: string
  executors: EvaluationCatalog['suites'][number]['executors']
}): EvaluationMethodTechnicalDetail[] {
  const executorIDs = Object.entries(executors).flatMap(([mode, executorID]) =>
    executorID ? [`${mode}: ${executorID}`] : [],
  )
  return [
    { label: 'Method ID', message: method.id },
    { label: 'Suite ID', message: suiteID },
    { label: 'Track ID', message: method.track_id },
    { label: 'Evidence source ID', message: method.evidence_source },
    { label: 'Suite revision', message: revision },
    { label: 'Executor IDs', message: executorIDs.join(' · ') || 'None recorded' },
    {
      label: 'Release check IDs',
      message: method.qualified_gate_ids.join(' · ') || 'None recorded',
    },
    ...(method.reason ? [{ label: 'Recorded setup response', message: method.reason }] : []),
  ]
}
