import {
  EVALUATION_MOM,
  EVALUATION_MOM_TARGET_ID,
  EVALUATION_RUN_IDS,
  evaluationRunID,
} from './mixtureFixture'
import { controlledPairSourceRuns } from './campaignActions'
import { evaluationRun } from './runFixtures'

export function releaseDecisionRuns() {
  const { baseline: baselineLive, candidate: candidateLive } = controlledPairSourceRuns()
  const hardPolicy = evaluationRun(
    EVALUATION_RUN_IDS.campaignG2,
    'Hard-policy qualification',
    'completed',
    '2026-08-29T03:00:00Z',
    'recipe',
    {
      mode: 'live',
      suite_ids: ['live-hard-policy'],
      track_ids: ['safety'],
      sample_limit: 64,
      target_id: EVALUATION_MOM_TARGET_ID,
      mixture: EVALUATION_MOM,
      evidence_level: 'E4',
      completed_at: '2026-08-29T03:10:00Z',
    },
  )
  const declaredShift = evaluationRun(
    EVALUATION_RUN_IDS.campaignG4,
    'Declared-shift qualification',
    'completed',
    '2026-08-29T04:00:00Z',
    'recipe',
    {
      mode: 'live',
      suite_ids: ['normalized-promotion-cohort'],
      track_ids: ['routing'],
      sample_limit: 64,
      target_id: EVALUATION_MOM_TARGET_ID,
      mixture: EVALUATION_MOM,
      evidence_level: 'E4',
      completed_at: '2026-08-29T04:10:00Z',
    },
  )
  const fidelityReference = evaluationRun(
    EVALUATION_RUN_IDS.campaignG5Reference,
    'Live fidelity reference',
    'completed',
    '2026-08-29T05:00:00Z',
    'recipe',
    {
      mode: 'live',
      suite_ids: ['live-mom-core'],
      track_ids: ['joint'],
      sample_limit: 64,
      target_id: EVALUATION_MOM_TARGET_ID,
      mixture: EVALUATION_MOM,
      evidence_level: 'E4',
      completed_at: '2026-08-29T05:10:00Z',
    },
  )
  const fidelityLive = evaluationRun(
    EVALUATION_RUN_IDS.campaignG5Live,
    'Fresh live fidelity confirmation',
    'completed',
    '2026-08-29T06:00:00Z',
    'recipe',
    {
      mode: 'live',
      suite_ids: ['live-mom-core'],
      track_ids: ['joint'],
      sample_limit: 64,
      target_id: EVALUATION_MOM_TARGET_ID,
      mixture: EVALUATION_MOM,
      evidence_level: 'E5',
      completed_at: '2026-08-29T06:10:00Z',
    },
  )
  const capacity = evaluationRun(
    EVALUATION_RUN_IDS.campaignG7,
    'Capacity envelope qualification',
    'completed',
    '2026-08-29T07:00:00Z',
    'recipe',
    {
      mode: 'live',
      suite_ids: ['live-capacity'],
      track_ids: ['capacity'],
      sample_limit: 64,
      target_id: EVALUATION_MOM_TARGET_ID,
      mixture: EVALUATION_MOM,
      evidence_level: 'E5',
      completed_at: '2026-08-29T07:10:00Z',
    },
  )
  const capacityDuplicate = evaluationRun(
    evaluationRunID(20),
    'Capacity envelope qualification',
    'completed',
    '2026-08-29T07:20:00Z',
    'recipe',
    {
      ...capacity,
      id: evaluationRunID(20),
      client_request_id: evaluationRunID(20),
      created_at: '2026-08-29T07:20:00Z',
      completed_at: '2026-08-29T07:30:00Z',
    },
  )
  return [
    capacityDuplicate,
    capacity,
    fidelityLive,
    fidelityReference,
    declaredShift,
    hardPolicy,
    candidateLive,
    baselineLive,
  ]
}
