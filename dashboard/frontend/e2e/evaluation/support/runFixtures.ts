import type {
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
  EvaluationChangeProfileId,
  EvaluationRun,
  EvaluationTrackId,
} from '../../../src/types/evaluationPlane'
import { EVALUATION_TRACK_IDS } from '../../../src/types/evaluationPlane'
import { EVALUATION_MOM, EVALUATION_MOM_TARGET_ID, EVALUATION_RUN_IDS } from './mixtureFixture'

const DEFAULT_CAPACITY_SLO: EvaluationCapacitySLO = {
  schema_version: 'evaluation.v1',
  required_concurrency: 4,
  max_latency_p95_ms: 750,
  max_error_rate: 0.02,
  min_throughput_rps: 10,
  min_throughput_scaling_efficiency: 0.7,
}

function e2eCapacityLoadProtocol(concurrency: number): EvaluationCapacityLoadProtocol {
  if (!Number.isSafeInteger(concurrency) || concurrency < 2 || concurrency > 128) {
    throw new Error('E2E capacity fixtures require concurrency between 2 and 128.')
  }
  const concurrencyLevels = [1]
  for (let level = 2; level < concurrency; level *= 2) concurrencyLevels.push(level)
  concurrencyLevels.push(concurrency)
  return {
    schema_version: 'evaluation.v1',
    kind: 'closed-loop',
    concurrency_levels: concurrencyLevels,
    warmup_request_multiplier: 2,
    measurement_requests_per_repetition: 100,
    repetitions_per_level: 3,
    minimum_measurement_clusters_per_level: 3,
    confidence_level: 0.95,
    max_error_rate_cluster_range: 0.05,
    max_throughput_cv: 0.2,
    max_latency_p95_cv: 0.2,
  }
}

export function evaluationRun(
  id: string,
  name: string,
  status: EvaluationRun['status'],
  createdAt: string,
  changeProfile: EvaluationChangeProfileId = 'recipe',
  overrides: Partial<EvaluationRun> = {},
): EvaluationRun {
  const active = status === 'running' || status === 'sealing'
  const mode = overrides.mode || (active ? 'live' : 'replay')
  const live = mode === 'live'
  const trackIDs: EvaluationTrackId[] = live
    ? ['routing', 'multimodal', 'capacity']
    : [...EVALUATION_TRACK_IDS]
  const suiteIDs = live
    ? ['live-mom-core', 'live-multimodal', 'live-capacity']
    : ['evaluation-smoke']
  const terminal = ['completed', 'failed', 'cancelled'].includes(status)
  const progress = {
    percent: status === 'completed' ? 100 : active ? 45 : terminal ? 55 : 0,
    completed: status === 'completed' ? trackIDs.length : active || terminal ? 3 : 0,
    total: trackIDs.length,
    message:
      status === 'running'
        ? 'Executing capacity track'
        : status === 'sealing'
          ? 'Sealing evaluation evidence'
          : status === 'failed'
            ? 'Worker exited before report publication'
            : status === 'cancelled'
              ? 'Execution cancelled'
              : status === 'completed'
                ? 'Evidence complete'
                : 'Awaiting execution',
  }
  const run: EvaluationRun = {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name,
    description: `${name} description`,
    status,
    mode,
    evidence_level: 'E0',
    target_id: live ? EVALUATION_MOM_TARGET_ID : 'fixture',
    change_profile: changeProfile,
    suite_ids: suiteIDs,
    track_ids: trackIDs,
    sample_limit: 4,
    concurrency: 4,
    seed: 42,
    progress,
    created_at: createdAt,
    started_at: status === 'pending' ? undefined : createdAt,
    completed_at: terminal ? '2026-08-29T00:10:00Z' : undefined,
    error: status === 'failed' ? 'Evaluation worker exited before a report was sealed.' : undefined,
    mixture: live ? EVALUATION_MOM : undefined,
  }
  const merged = {
    ...run,
    ...overrides,
    client_request_id: overrides.id || id,
    progress: { ...progress, ...overrides.progress },
  }
  const trackEvidenceLevels =
    overrides.track_evidence_levels ||
    Object.fromEntries(merged.track_ids.map((trackID) => [trackID, merged.evidence_level]))
  const capacitySLORequired = merged.mode === 'live' && merged.track_ids.includes('capacity')
  const capacityLoadProtocol = capacitySLORequired
    ? overrides.capacity_load_protocol || e2eCapacityLoadProtocol(merged.concurrency)
    : undefined
  return {
    ...merged,
    track_evidence_levels: trackEvidenceLevels,
    ...(capacitySLORequired
      ? {
          capacity_slo: overrides.capacity_slo || DEFAULT_CAPACITY_SLO,
          capacity_load_protocol: capacityLoadProtocol,
        }
      : { capacity_slo: undefined, capacity_load_protocol: undefined }),
  }
}

export const defaultEvaluationRuns = [
  evaluationRun(
    EVALUATION_RUN_IDS.candidate,
    'Candidate recipe',
    'completed',
    '2026-08-29T00:00:00Z',
    'recipe',
    {
      baseline_run_id: EVALUATION_RUN_IDS.baseline,
    },
  ),
  evaluationRun(
    EVALUATION_RUN_IDS.baseline,
    'Production baseline',
    'completed',
    '2026-08-28T00:00:00Z',
  ),
  evaluationRun(
    EVALUATION_RUN_IDS.unpaired,
    'Unpaired diagnostic',
    'completed',
    '2026-08-27T12:00:00Z',
  ),
  evaluationRun(
    EVALUATION_RUN_IDS.live,
    'Live AMD validation',
    'running',
    '2026-08-27T00:00:00Z',
    'runtime_capacity',
  ),
  evaluationRun(EVALUATION_RUN_IDS.failed, 'Failed diagnostic', 'failed', '2026-08-26T00:00:00Z'),
  evaluationRun(
    EVALUATION_RUN_IDS.cancelled,
    'Cancelled diagnostic',
    'cancelled',
    '2026-08-25T00:00:00Z',
  ),
]
