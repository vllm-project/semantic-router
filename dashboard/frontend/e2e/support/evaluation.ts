import type { Page } from '@playwright/test'

import type {
  CreateEvaluationRunRequest,
  EvaluationCatalog,
  EvaluationChangeProfileId,
  EvaluationComparison,
  EvaluationGate,
  EvaluationReport,
  EvaluationRun,
} from '../../src/types/evaluationPlane'
import { EVALUATION_TRACK_IDS, TRACK_PRESENTATION } from '../../src/types/evaluationPlane'
import {
  gateApplicabilityForProfile,
  SUPPORTED_GATE_CONTRACT_VERSION,
} from '../../src/components/evaluation-plane/evaluationGateContract'

export const evaluationCatalog: EvaluationCatalog = {
  schema_version: 'evaluation.v1',
  gate_contract_version: SUPPORTED_GATE_CONTRACT_VERSION,
  generated_at: '2026-08-29T00:00:00Z',
  change_profiles: [
    {
      id: 'schema_adapter',
      name: 'Schema / adapter',
      description: 'Strict schema and adapter parity changes.',
    },
    {
      id: 'recipe',
      name: 'Routing recipe',
      description: 'Recipe signal, decision, algorithm, and policy changes.',
    },
    {
      id: 'selector',
      name: 'Selector / binding',
      description: 'Selector, projection, classifier, and binding changes.',
    },
    {
      id: 'model_pool',
      name: 'Model pool',
      description: 'Logical arm composition, capability, quality, and price changes.',
    },
    {
      id: 'runtime_capacity',
      name: 'Runtime / capacity',
      description: 'Serving runtime, placement, capacity, and transport changes.',
    },
    {
      id: 'agent_multimodal',
      name: 'Agent / multimodal',
      description: 'Agent trajectory, tool, state, and multimodal changes.',
    },
    {
      id: 'online_adaptation',
      name: 'Online adaptation',
      description: 'Online assignment, preference, feedback, and adaptive policy changes.',
    },
  ],
  tracks: EVALUATION_TRACK_IDS.map((id) => ({
    id,
    name: TRACK_PRESENTATION[id].label,
    description: TRACK_PRESENTATION[id].description,
    modes: ['agentic', 'preference', 'safety'].includes(id) ? ['replay'] : ['replay', 'live'],
    metrics: [`${id}.quality`, `${id}.latency`],
    evidence_levels: ['E0'],
  })),
  suites: [
    {
      id: 'evaluation-smoke',
      name: 'Evaluation harness smoke',
      description: 'Deterministic plumbing evidence; it is not a live model-quality claim.',
      track_ids: [...EVALUATION_TRACK_IDS],
      modes: ['replay'],
      evidence_level: 'E0',
      case_count: 4,
      revision: 'builtin-v1',
    },
    {
      id: 'live-routing-core',
      name: 'Live routing core',
      description: 'Runtime route-decision diagnostics.',
      track_ids: ['routing'],
      modes: ['live'],
      evidence_level: 'E3',
      revision: 'executor-v1',
    },
    {
      id: 'live-model-pool',
      name: 'Live model pool',
      description: 'Dense direct-arm matrix over the server-owned pool.',
      track_ids: ['model_pool'],
      modes: ['live'],
      evidence_level: 'E4',
      revision: 'executor-v1',
    },
    {
      id: 'live-joint',
      name: 'Live routing + pool',
      description: 'Correlated routed execution and dense pool evidence.',
      track_ids: ['routing', 'model_pool', 'joint'],
      modes: ['live'],
      evidence_level: 'E5',
      revision: 'executor-v1',
    },
    {
      id: 'live-multimodal',
      name: 'Live multimodal',
      description: 'Non-text requests on an explicitly capable pool.',
      track_ids: ['multimodal'],
      modes: ['live'],
      evidence_level: 'E5',
      revision: 'executor-v1',
    },
    {
      id: 'live-capacity',
      name: 'Live capacity',
      description: 'Bounded live concurrency sweep.',
      track_ids: ['capacity'],
      modes: ['live'],
      evidence_level: 'E5',
      revision: 'executor-v1',
    },
  ],
  targets: [
    {
      id: 'fixture',
      name: 'Built-in replay fixture',
      description: 'Local deterministic harness validation with no production-quality claim.',
      kind: 'builtin-fixture',
      track_ids: [...EVALUATION_TRACK_IDS],
      modes: ['replay'],
      evidence_level: 'E0',
      healthy: true,
    },
    {
      id: 'runtime',
      name: 'Active vLLM-SR runtime',
      description: 'Server-managed endpoints; the catalog advertises only qualified capabilities.',
      kind: 'runtime',
      track_ids: ['routing', 'model_pool', 'joint', 'multimodal', 'capacity'],
      modes: ['live'],
      healthy: true,
    },
  ],
}

export function evaluationRun(
  id: string,
  name: string,
  status: EvaluationRun['status'],
  createdAt: string,
  changeProfile: EvaluationChangeProfileId = 'recipe',
): EvaluationRun {
  const live = status === 'running'
  const trackIDs = live
    ? (['routing', 'model_pool', 'joint', 'multimodal', 'capacity'] as const)
    : EVALUATION_TRACK_IDS
  return {
    schema_version: 'evaluation.v1',
    id,
    name,
    description: `${name} description`,
    status,
    mode: live ? 'live' : 'replay',
    evidence_level: live ? 'E3' : 'E0',
    target_id: live ? 'runtime' : 'fixture',
    change_profile: changeProfile,
    suite_ids: live
      ? ['live-routing-core', 'live-model-pool', 'live-joint', 'live-multimodal', 'live-capacity']
      : ['evaluation-smoke'],
    track_ids: [...trackIDs],
    sample_limit: 4,
    concurrency: 4,
    seed: 42,
    progress: {
      percent: status === 'completed' ? 100 : status === 'running' ? 45 : 0,
      completed: status === 'completed' ? trackIDs.length : status === 'running' ? 3 : 0,
      total: trackIDs.length,
      message: status === 'running' ? 'Executing capacity track' : 'Evidence complete',
    },
    created_at: createdAt,
    started_at: status === 'pending' ? undefined : createdAt,
    completed_at: status === 'completed' ? '2026-08-29T00:10:00Z' : undefined,
  }
}

export const defaultEvaluationRuns = [
  evaluationRun('candidate-run', 'Candidate recipe', 'completed', '2026-08-29T00:00:00Z'),
  evaluationRun('baseline-run', 'Production baseline', 'completed', '2026-08-28T00:00:00Z'),
  evaluationRun(
    'live-run',
    'Live AMD validation',
    'running',
    '2026-08-27T00:00:00Z',
    'runtime_capacity',
  ),
]

const gateTrackIDs: Partial<Record<`G${number}`, EvaluationGate['track_id']>> = {
  G2: 'safety',
  G3: 'joint',
  G4: 'routing',
  G5: 'joint',
  G6: 'agentic',
  G7: 'capacity',
  G9: 'preference',
}

function evaluationGates(run: EvaluationRun): EvaluationGate[] {
  const coverage = {
    evaluated: 4,
    total: 4,
    fraction: 1,
    unavailable: 0,
    confidence_level: 0.95,
    confidence_interval: [0.51, 1] as [number, number],
  }
  return gateApplicabilityForProfile(run.change_profile).map((gate) => {
    const isNotApplicable = gate.disposition === 'not_applicable'
    const isUnavailable = gate.id === 'G4' || gate.id === 'G8'
    return {
      ...gate,
      track_id: gateTrackIDs[gate.id],
      verdict: isNotApplicable
        ? ('not_applicable' as const)
        : isUnavailable
          ? ('unavailable' as const)
          : ('pass' as const),
      change_profile: run.change_profile,
      contract_version: SUPPORTED_GATE_CONTRACT_VERSION,
      evidence_refs: ['records.jsonl'],
      evidence_level: isUnavailable ? ('E5' as const) : run.evidence_level,
      observed: isNotApplicable || isUnavailable ? null : 1,
      threshold:
        isNotApplicable || isUnavailable
          ? undefined
          : { operator: '>=', value: 1, unit: 'fraction' },
      sample_count: isNotApplicable ? undefined : 4,
      coverage: isNotApplicable
        ? undefined
        : isUnavailable
          ? { ...coverage, evaluated: 0, fraction: 0, unavailable: 4 }
          : coverage,
      owner: 'Evaluation Platform',
      evaluated_at: '2026-08-29T00:10:00Z',
      rationale: isUnavailable
        ? 'Qualified assignment evidence was not available; this gate does not pass.'
        : undefined,
    }
  })
}

export function evaluationReport(run = defaultEvaluationRuns[0]): EvaluationReport {
  const coverage = { evaluated: 4, total: 4, fraction: 1, unavailable: 0 }
  const gates = evaluationGates(run)
  return {
    schema_version: 'evaluation.v1',
    run,
    summary: {
      verdict: gates.some(
        (gate) => gate.disposition === 'required' && gate.verdict === 'unavailable',
      )
        ? 'unavailable'
        : 'pass',
      quality_score: run.evidence_level === 'E0' ? null : 0.91,
      latency_p95_ms: run.evidence_level === 'E0' ? null : 342,
      runtime_cost: run.evidence_level === 'E0' ? null : 4.2,
      capacity_tco: run.evidence_level === 'E0' ? null : 120000,
      coverage,
      passed_gates: gates.filter((gate) => gate.verdict === 'pass').length,
      failed_gates: gates.filter((gate) => gate.verdict === 'fail').length,
      unavailable_gates: gates.filter((gate) => gate.verdict === 'unavailable').length,
    },
    tracks: EVALUATION_TRACK_IDS.map((trackID) => ({
      track_id: trackID,
      status: run.track_ids.includes(trackID) ? ('completed' as const) : ('unavailable' as const),
      evidence_level: run.evidence_level,
      summary: run.track_ids.includes(trackID)
        ? `${TRACK_PRESENTATION[trackID].label} evidence completed.`
        : `${TRACK_PRESENTATION[trackID].label} is not advertised by this target.`,
      coverage: run.track_ids.includes(trackID)
        ? coverage
        : { evaluated: 0, total: 4, fraction: 0, unavailable: 4 },
      metrics: run.track_ids.includes(trackID)
        ? [
            {
              id: `${trackID}.quality`,
              name: `${TRACK_PRESENTATION[trackID].label} quality`,
              track_id: trackID,
              value: 0.91,
              unit: 'ratio',
              sample_count: 4,
            },
          ]
        : [],
      gates: run.track_ids.includes(trackID)
        ? gates.filter((gate) => gate.track_id === trackID)
        : [],
    })),
    metrics: [
      {
        id: 'quality',
        name: 'System quality',
        value: 0.91,
        unit: 'ratio',
        baseline_value: 0.88,
        delta: 0.03,
      },
      {
        id: 'latency',
        name: 'P95 latency',
        value: 342,
        unit: 'ms',
        baseline_value: 370,
        delta: -28,
      },
    ],
    gates,
    costs: {
      runtime: {
        amount: 4.2,
        currency: 'USD',
        input_tokens: 12000,
        output_tokens: 6000,
        gpu_seconds: 420,
      },
      evaluation_overhead: { amount: 0.8, currency: 'USD', input_tokens: 2000, output_tokens: 800 },
      capacity_tco: { amount: 120000, currency: 'USD', energy_kwh: 1800 },
    },
    recommendations: [
      'Collect qualified robustness evidence before promoting the routing recipe.',
      'Retain the capacity guardrail at the current concurrency limit.',
    ],
    provenance: {
      schema_version: 'evaluation.v1',
      generated_at: '2026-08-29T00:10:00Z',
      code_revision: '8dedee7',
      benchmark_revisions: Object.fromEntries(run.suite_ids.map((id) => [id, 'builtin-v1'])),
      workload_snapshot_digest: 'sha256:workload',
      policy_snapshot_digest: 'sha256:policy',
      pool_snapshot_digest: 'sha256:pool',
      environment_snapshot_digest: 'sha256:environment',
      target_id: run.target_id,
      seed: 42,
      redaction_policy: 'private-prompts-v1',
    },
    artifacts: [
      {
        id: 'report-html',
        name: 'report.html',
        kind: 'html',
        digest: 'sha256:report-html',
        media_type: 'text/html',
        size_bytes: 4096,
      },
      {
        id: 'failure-summary-json',
        name: 'failure-summary.json',
        kind: 'json',
        digest: 'sha256:failure-summary',
        media_type: 'application/json',
      },
      {
        id: 'run-manifest-json',
        name: 'run-manifest.json',
        kind: 'manifest',
        digest: 'sha256:manifest',
        media_type: 'application/json',
      },
    ],
  }
}

export const evaluationComparison: EvaluationComparison = {
  schema_version: 'evaluation.v1',
  baseline_run_id: 'baseline-run',
  candidate_run_id: 'candidate-run',
  verdict: 'unavailable',
  summary: 'Candidate deltas are favorable, but required robustness evidence is unavailable.',
  metrics: [
    {
      id: 'quality',
      name: 'System quality',
      value: 0.91,
      unit: 'ratio',
      baseline_value: 0.88,
      delta: 0.03,
    },
    { id: 'latency', name: 'P95 latency', value: 342, unit: 'ms', baseline_value: 370, delta: -28 },
  ],
  gates: evaluationReport().gates.filter((gate) => gate.id === 'G4'),
  recommendations: ['Collect qualified robustness evidence before a guarded live trial.'],
}

export async function mockEvaluationPlane(page: Page, initialRuns = defaultEvaluationRuns) {
  let runs = [...initialRuns]
  const createdRequests: CreateEvaluationRunRequest[] = []
  let cancelCount = 0
  let startCount = 0
  let eventStreamCount = 0

  await page.route('**/api/evaluation/v1/catalog', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(evaluationCatalog),
    })
  })
  await page.route('**/api/evaluation/v1/compare?*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(evaluationComparison),
    })
  })
  await page.route('**/api/evaluation/v1/runs/*/events', async (route) => {
    eventStreamCount += 1
    const event = {
      id: 'sse-event-1',
      run_id: 'live-run',
      type: 'progress',
      timestamp: '2026-08-29T00:05:00Z',
      message: 'Executing routing track from SSE',
    }
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: { 'Cache-Control': 'no-cache' },
      body: `id: sse-event-1\nevent: progress\ndata: ${JSON.stringify(event)}\n\n`,
    })
  })
  await page.route('**/api/evaluation/v1/runs/*/report', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const run = runs.find((candidate) => candidate.id === id) || runs[0]
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(evaluationReport(run)),
    })
  })
  await page.route('**/api/evaluation/v1/runs/*/cancel', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const current = runs.find((run) => run.id === id) || runs[0]
    const cancelled = {
      ...current,
      status: 'cancelled' as const,
      completed_at: '2026-08-29T00:11:00Z',
    }
    runs = runs.map((run) => (run.id === id ? cancelled : run))
    cancelCount += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ run: cancelled }),
    })
  })
  await page.route('**/api/evaluation/v1/runs/*/start', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const current = runs.find((run) => run.id === id) || runs[0]
    const started = { ...current, status: 'running' as const }
    runs = runs.map((run) => (run.id === id ? started : run))
    startCount += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ run: started }),
    })
  })
  await page.route('**/api/evaluation/v1/runs', async (route) => {
    if (route.request().method() === 'POST') {
      const request = route.request().postDataJSON() as CreateEvaluationRunRequest
      createdRequests.push(request)
      const created = {
        ...evaluationRun('created-run', request.name, 'pending', '2026-08-29T01:00:00Z'),
        ...request,
        evidence_level: 'E3' as const,
      }
      runs = [created, ...runs]
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({ run: created }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ runs }),
    })
  })

  return {
    createdRequests,
    getCancelCount: () => cancelCount,
    getStartCount: () => startCount,
    getEventStreamCount: () => eventStreamCount,
  }
}
