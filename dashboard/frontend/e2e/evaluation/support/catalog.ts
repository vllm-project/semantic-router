import canonicalCatalogDocument from '../../../../../src/vllm-sr/tests/fixtures/evaluation/catalog.json' with { type: 'json' }
import type {
  EvaluationCatalog,
  EvaluationCatalogSuite,
  EvaluationCatalogTarget,
} from '../../../src/types/evaluationPlane'
import {
  EVALUATION_BASELINE_MOM_TARGET_ID,
  EVALUATION_MOM,
  EVALUATION_MOM_TARGET_ID,
} from './mixtureFixture'

const canonicalCatalog = canonicalCatalogDocument as unknown as Omit<
  EvaluationCatalog,
  'generated_at'
>

// The real server appends installed normalized suites and discovered Mixture
// targets to the built-in catalog. E2E adds only that scenario-owned inventory;
// release profiles, campaign slots, tracks, and built-in suites stay sourced
// from the cross-language canonical fixture.
const installedNormalizedSuite: EvaluationCatalogSuite = {
  id: 'normalized-promotion-cohort',
  executors: {
    replay: 'normalized-suite-replay.v1',
    live: 'normalized-suite-live.v1',
  },
  name: 'Normalized promotion cohort',
  description: 'Server-declared declared-shift and reference-to-live collection capability.',
  track_ids: ['routing', 'joint'],
  modes: ['replay', 'live'],
  evidence_level: 'E0',
  revision: 'normalized-promotion.v1',
  tags: ['normalized'],
  methods: [
    {
      id: 'normalized-promotion.routing.live.v1',
      track_id: 'routing',
      qualified_gate_ids: ['G4'],
      evidence_source: 'server_brokered_live',
      status: 'configured',
    },
    {
      id: 'normalized-promotion.joint.v1',
      track_id: 'joint',
      qualified_gate_ids: [],
      evidence_source: 'normalized_import',
      status: 'configured',
    },
  ],
}

const trackIDs = canonicalCatalog.tracks.map((track) => track.id)

function mixtureTarget(
  id: string,
  deployment: 'Baseline' | 'Candidate',
): EvaluationCatalogTarget {
  return {
    id,
    name: `test-mom · ${deployment}`,
    description: 'Recipe-scoped Mixture-of-Models evaluation target.',
    kind: 'mixture-of-models',
    track_ids: trackIDs,
    modes: ['replay', 'live'],
    accepted_executors: {
      replay: ['mom-cohort-replay.v1', 'normalized-suite-replay.v1'],
      live: ['live-runtime.v1', 'normalized-suite-live.v1'],
    },
    healthy: true,
    labels: { deployment },
    mixture: EVALUATION_MOM,
  }
}

export const evaluationCatalog: EvaluationCatalog = {
  ...canonicalCatalog,
  generated_at: '2026-08-29T00:00:00Z',
  suites: [...canonicalCatalog.suites, installedNormalizedSuite],
  targets: [
    ...canonicalCatalog.targets,
    mixtureTarget(EVALUATION_BASELINE_MOM_TARGET_ID, 'Baseline'),
    mixtureTarget(EVALUATION_MOM_TARGET_ID, 'Candidate'),
  ],
}
