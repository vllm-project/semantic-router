import type {
  CreateEvaluationRunPayload,
  EvaluationCatalog,
  EvaluationCatalogCampaignSlot,
  EvaluationRun,
} from '../types/evaluationPlane'
import { EVALUATION_TRACK_IDS } from '../types/evaluationPlane'
import { buildEvaluationRoutingRecipePlan } from './evaluationRoutingRecipeFixture'

export const RUN_ID = '11111111-1111-4111-8111-111111111111'
export const CREATE_RUN_ID = '4d0b4f2c-1fc5-40b0-b04e-876ad9d4d8e2'
export const BASELINE_RUN_ID = '22222222-2222-4222-8222-222222222222'
export const CANDIDATE_RUN_ID = '33333333-3333-4333-8333-333333333333'
export const QUARANTINED_EVIDENCE_ID = 'bundle-entry-7f9d2a'
export const CAMPAIGN_ID = '44444444-4444-4444-8444-444444444444'
export const CAMPAIGN_LIVE_ID = '55555555-5555-4555-8555-555555555555'
export const CAMPAIGN_LIVE_BASELINE_ID = '66666666-6666-4666-8666-666666666666'
export const CAMPAIGN_CONFIRMATION_ID = '77777777-7777-4777-8777-777777777777'

export const canonicalCampaignSlots = [
  ['G2', 'run', 'Policy enforcement'],
  ['G3', 'controlled_pair', 'Controlled value comparison'],
  ['G4', 'run', 'Workload-shift robustness'],
  ['G5', 'fidelity_pair', 'Live consistency'],
  ['G6', 'run', 'Fault recovery'],
  ['G7', 'run', 'Cost, latency, and capacity'],
  ['G8', 'run', 'Canary safety'],
  ['G9', 'run', 'Online preference'],
].map(([gate_id, binding_kind, name]) => ({
  gate_id,
  name,
  description: 'Collect the results required for this release check.',
  disposition: 'not_applicable',
  binding_kind,
  track_id: 'joint',
  mode: 'live',
  minimum_evidence_level: 'E0',
  accepted_executor_ids: ['server-executor.v1'],
})) as EvaluationCatalogCampaignSlot[]

const canonicalBuiltinSuiteIDs = [
  'evaluation-smoke',
  'live-mom-core',
  'live-agent-tasks',
  'live-fault-recovery',
  'live-multimodal',
  'live-hard-policy',
  'live-production-experiment',
  'live-capacity',
] as const

const canonicalCatalogTracks: EvaluationCatalog['tracks'] = EVALUATION_TRACK_IDS.map((trackID) => ({
  id: trackID,
  name: trackID,
  description: `${trackID} evaluation area`,
  modes: ['replay'],
  metrics: [],
  evidence_levels: ['E2'],
}))

export const canonicalBuiltinSuites: EvaluationCatalog['suites'] = canonicalBuiltinSuiteIDs.map(
  (suiteID, index) => {
    const trackID = EVALUATION_TRACK_IDS[index]
    return {
      id: suiteID,
      executors: { replay: `fixture-${trackID}.v1` },
      name: suiteID,
      description: `${trackID} built-in benchmark`,
      track_ids: [trackID],
      modes: ['replay'],
      evidence_level: 'E2',
      revision: `${suiteID}.v1`,
      tags: ['fixture'],
      methods: [
        {
          id: `fixture.${trackID}.builtin.v1`,
          track_id: trackID,
          qualified_gate_ids: [],
          evidence_source: 'diagnostic_fixture',
          status: 'configured',
        },
      ],
    }
  },
)

export function evaluationCatalogFixture(
  overrides: Partial<EvaluationCatalog> = {},
): EvaluationCatalog {
  return {
    schema_version: 'evaluation.v1',
    gate_contract_version: 'evaluation-release-gates.v2',
    generated_at: '2026-08-29T00:00:00Z',
    change_profiles: [
      {
        id: 'recipe',
        name: 'Routing recipe',
        description: 'Recipe signal, decision, algorithm, and policy changes.',
        campaign_slots: canonicalCampaignSlots,
      },
    ],
    tracks: canonicalCatalogTracks,
    suites: canonicalBuiltinSuites,
    targets: [],
    ...overrides,
  }
}

export const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: RUN_ID,
  client_request_id: RUN_ID,
  name: 'Candidate',
  description: 'Compare recipe',
  status: 'pending',
  mode: 'replay',
  evidence_level: 'E2',
  track_evidence_levels: { routing: 'E2' },
  target_id: 'target-approved',
  change_profile: 'recipe',
  suite_ids: ['suite-routing'],
  track_ids: ['routing'],
  sample_limit: 25,
  concurrency: 2,
  seed: 42,
  progress: { percent: 0, completed: 0, total: 1 },
  created_at: '2026-08-29T00:00:00Z',
}

export const catalog: EvaluationCatalog = evaluationCatalogFixture({
  change_profiles: [
    {
      id: 'recipe',
      name: 'Routing recipe',
      description: 'Recipe signal, decision, algorithm, and policy changes.',
      campaign_slots: canonicalCampaignSlots,
    },
  ],
  suites: [
    ...canonicalBuiltinSuites,
    {
      id: 'suite-routing',
      executors: { replay: 'fixture-replay.v1' },
      name: 'Routing suite',
      description: 'Replay suite',
      track_ids: ['routing'],
      modes: ['replay'],
      evidence_level: 'E2',
      revision: 'suite-routing.v1',
      tags: ['fixture'],
      methods: [
        {
          id: 'fixture.routing.v1',
          track_id: 'routing',
          qualified_gate_ids: [],
          evidence_source: 'diagnostic_fixture',
          status: 'configured',
        },
      ],
    },
  ],
  targets: [
    {
      id: 'target-approved',
      name: 'Approved target',
      description: 'Server target',
      kind: 'replay',
      track_ids: ['routing'],
      modes: ['replay'],
      accepted_executors: { replay: ['fixture-replay.v1'] },
    },
  ],
})

export const request: CreateEvaluationRunPayload = {
  client_request_id: CREATE_RUN_ID,
  name: ' Candidate ',
  description: ' Compare recipe ',
  suite_ids: ['suite-routing'],
  track_ids: ['routing'],
  mode: 'replay',
  target_id: 'target-approved',
  change_profile: 'recipe',
  sample_limit: 25,
  concurrency: 2,
  seed: 42,
}

export const completedRun: EvaluationRun = {
  ...run,
  status: 'completed',
  progress: { percent: 100, completed: 1, total: 1 },
  started_at: '2026-08-29T00:00:01Z',
  completed_at: '2026-08-29T00:00:02Z',
}

export function reportFor(reportRun: EvaluationRun) {
  const gates = Array.from({ length: 10 }, (_, index) => ({
    id: `G${index}`,
    name: `Gate ${index}`,
    description: `Published release check ${index}.`,
    ...(index === 4 ? { track_id: 'routing' } : {}),
    disposition: 'required',
    verdict: 'unavailable',
    change_profile: reportRun.change_profile,
    contract_version: 'evaluation-release-gates.v2',
    evidence_refs: [`gate:G${index}`],
    evidence_level: 'E0',
    sample_count: 0,
    coverage: { evaluated: 0, total: 0, fraction: 0 },
    owner: 'evaluation-service',
    evaluated_at: '2026-08-29T00:00:02Z',
    rationale: 'No qualified release evidence was produced.',
  }))
  return {
    schema_version: 'evaluation.v1',
    attestation_revision: 'evaluation-server-attestation.v2',
    run: reportRun,
    summary: {
      verdict: 'unavailable',
      quality_score: null,
      latency_p95_ms: null,
      runtime_cost: null,
      capacity_tco: null,
      coverage: { evaluated: 0, total: 0, fraction: 0 },
      passed_gates: 0,
      failed_gates: 0,
      unavailable_gates: 10,
    },
    tracks: [
      {
        track_id: 'routing',
        status: 'unavailable',
        evidence_level: reportRun.track_evidence_levels.routing,
        summary: 'No qualified evidence was produced.',
        coverage: { evaluated: 0, total: 0, fraction: 0 },
        metrics: [],
        gates: [gates[4]],
      },
    ],
    metrics: [],
    gates,
    costs: {
      runtime: { amount: null, currency: 'USD' },
      evaluation_overhead: { amount: null, currency: 'USD' },
      capacity_tco: { amount: null, currency: 'USD' },
    },
    recommendations: [],
    provenance: {
      schema_version: 'evaluation.v1',
      generated_at: '2026-08-29T00:00:02Z',
      target_id: reportRun.target_id,
      seed: reportRun.seed,
    },
    artifacts: [],
    method_reports: [],
    routing_recipe_report: null,
  }
}

export function methodReportFixture() {
  const slice = { schema_version: 'evaluation-method.v2', id: 'all' } as const
  const plan = {
    schema_version: 'evaluation-method.v2',
    id: 'r2-compound-case-action-budget',
    analysis_unit: 'case_action_budget',
    cluster_unit: 'case',
    slices: [slice],
    curve_domain: 'shared_budget' as const,
    missingness: 'fail_closed' as const,
  }
  return {
    method: {
      schema_version: 'evaluation-method.v2',
      id: 'r2.compound-model-budget.v2',
      version: 'evaluation-method.v2',
      status: 'exploratory-import',
      execution_owner: 'server',
      input_schema: 'r2-compound-input',
      export_schema: 'r2-compound-report',
      live_input_complete: false,
      live_grader: false,
      applicable_tracks: ['model_pool'],
      live_tracks: [],
      produced_metric_ids: ['r2.compound_model_budget.audc'],
      evidence_ceiling: 'E5',
      native_parity: 'source_qualified',
      required_artifact_ids: ['curves'],
      analysis_plan: plan,
    },
    analysis_plan: plan,
    action_refs: [{ schema_version: 'evaluation-method.v2', id: 'small' }],
    slice_refs: [slice],
    raw_shared_domain_curve: [
      {
        action: { schema_version: 'evaluation-method.v2', id: 'small' },
        budget: 100,
        mean_score: 0.5,
        case_count: 2,
      },
    ],
    audc: 0,
    nauc: 0.5,
    peak: 0.5,
    qnc: 0.5,
    missing_case_action_budget_cells: 0,
  }
}

export const strictContractRun: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: RUN_ID,
  client_request_id: RUN_ID,
  name: 'Strict contract run',
  description: '',
  status: 'completed',
  mode: 'replay',
  evidence_level: 'E0',
  track_evidence_levels: { routing: 'E0' },
  target_id: 'fixture',
  change_profile: 'recipe',
  suite_ids: ['evaluation-smoke'],
  track_ids: ['routing'],
  sample_limit: 4,
  concurrency: 1,
  seed: 42,
  progress: { percent: 100, completed: 1, total: 1 },
  created_at: '2026-08-30T00:00:00Z',
  completed_at: '2026-08-30T00:01:00Z',
}

const mixtureBase = {
  id: 'mom-live',
  entrypoint_model: 'quality-router',
  aliases: ['quality-router'],
  recipe_name: 'quality',
  recipe_description: 'Quality routing',
  recipe_digest: `sha256:${'1'.repeat(64)}`,
  pool_digest: `sha256:${'2'.repeat(64)}`,
  selector_policy_digest: `sha256:${'3'.repeat(64)}`,
  selector_digest: `sha256:${'4'.repeat(64)}`,
  adaptation_digest: `sha256:${'5'.repeat(64)}`,
  binding_digest: `sha256:${'6'.repeat(64)}`,
  model_arms: [
    {
      id: 'arm-a',
      model: 'model-a',
      provider_model_id_digest: `sha256:${'7'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 1,
      output_cost_per_million_tokens_usd: 2,
    },
  ],
  support_models: [],
  decisions: [{ name: 'route', algorithm: 'semantic', arm_ids: ['arm-a'] }],
}

export const mixture = {
  ...mixtureBase,
  routing_recipe_plan: buildEvaluationRoutingRecipePlan(mixtureBase),
}

const unavailableMixtureBase = {
  ...mixtureBase,
  id: 'mom-unavailable',
  entrypoint_model: 'unavailable-router',
  aliases: ['unavailable-router'],
  model_arms: [],
  support_models: [],
  decisions: [],
}

export const unavailableMixture = {
  ...unavailableMixtureBase,
  routing_recipe_plan: buildEvaluationRoutingRecipePlan(unavailableMixtureBase),
}

export const catalogWithUnavailableMixture = evaluationCatalogFixture({
  targets: [
    {
      id: 'mom-unavailable',
      name: 'Unavailable Mixture',
      description: 'Inspectable, but not executable',
      kind: 'mixture-of-models',
      track_ids: [],
      modes: ['replay', 'live'],
      accepted_executors: {
        replay: ['mom-cohort-replay.v1'],
        live: ['live-runtime.v1'],
      },
      healthy: false,
      mixture: unavailableMixture,
    },
  ],
})
