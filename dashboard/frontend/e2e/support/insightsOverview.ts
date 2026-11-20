import type { Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './auth'

// A day of traffic from every agent harness the router knows, and one long
// Claude Code session: a main agent that started nine subagents, switched
// models, compacted, read a picture, failed once, was blocked once by the
// guard and sat idle for twenty-five minutes.
export const LONG_SESSION = 'key_3fa9c2:0d6c7a52-8a43-4f0e-9a4f-2b7d1f0c9e11'

const REPLAYS = '**/api/router/api/v1/observability/replays'
const FLASH = 'deepseek/deepseek-v4.1-flash'
const QWEN = 'qwen/qwen3.8-27b'
const START = Date.parse('2026-11-19T09:00:00Z')
const SUBAGENTS = 9
const IDLE_AFTER = 70
const IDLE_MS = 25 * 60_000

export const HARNESS_COUNTS = {
  claude_code: 412,
  codex: 268,
  opencode: 141,
  cursor: 87,
  openclaw: 36,
  api_client: 520,
}

const cost = (total: number) => ({
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 3_100_000,
  completion_tokens: 41_000,
  by_currency: [{ currency: 'USD', total, router_call_spend: 0 }],
})

// Each subagent runs six calls in its own stretch of ten; main drives the rest.
function agentOf(index: number): string {
  if (index < 8) return 'main'
  const slot = Math.floor((index - 8) / 10)
  return slot < SUBAGENTS && (index - 8) % 10 < 6 ? `subagent ${slot + 1}` : 'main'
}

function markersOf(index: number, agent: string, first: boolean): string[] | undefined {
  const markers: string[] = []
  if (agent !== 'main' && first) markers.push('subagent_start')
  if (index === 30) markers.push('compaction')
  if (index === 45) markers.push('image')
  if (index % 9 === 4) markers.push('tool_loop')
  if (index === 66) markers.push('failure')
  if (index === 120) markers.push('guard', 'block')
  if (index % 7 === 3 && index !== 120) markers.push('switch')
  return markers.length ? markers : undefined
}

function longRunCalls() {
  const seen = new Set<string>()
  return Array.from({ length: 160 }, (_, index) => {
    const agent = agentOf(index)
    const first = !seen.has(agent)
    seen.add(agent)
    const model = index % 7 === 3 ? QWEN : FLASH
    const failed = index === 66
    return {
      record_id: `long-${index + 1}`,
      agent,
      agent_type: agent === 'main' ? undefined : index % 3 ? 'Explore' : 'general-purpose',
      request_class: index === 30 ? 'compaction' : agent === 'main' ? 'main' : 'subagent',
      timestamp: new Date(
        START + index * 12_000 + (index >= IDLE_AFTER ? IDLE_MS : 0),
      ).toISOString(),
      duration_ms: index === 120 ? 15 : 6_000 + (index % 5) * 900,
      decision:
        index === 120
          ? 'guard'
          : index === 30
            ? 'housekeeping'
            : agent === 'main'
              ? 'agentic'
              : 'explore',
      signals:
        index === 120
          ? undefined
          : ['keyword:code_request', 'context:long_context', 'agent:claude_code'],
      model: index === 120 ? undefined : model,
      previous_model: model === QWEN ? FLASH : undefined,
      effort: model === QWEN ? 'high' : 'medium',
      selection_method: 'session_aware',
      session_action: model === QWEN ? 'switch' : first ? 'establish_baseline' : 'stay',
      reason:
        model === QWEN ? 'switch_allowed' : first ? 'new_session' : 'stay_has_best_adjusted_score',
      image_count: index === 45 ? 1 : undefined,
      lifecycle_state: failed ? 'failed' : 'completed',
      response_status: failed ? 502 : 200,
      prompt_tokens: index === 120 ? undefined : 18_000 + index * 40,
      completion_tokens: index === 120 ? undefined : 240 + (index % 4) * 60,
      cost: index === 120 ? undefined : 0.0021,
      currency: 'USD',
      tool_results: index % 9 === 4 ? 3 : undefined,
      tool_failures: index % 9 === 4 ? 1 : undefined,
      router_calls: [
        { kind: 'decision_model', model: 'primary', calls: 1, cost: 0.0001, currency: 'USD' },
      ],
      markers: markersOf(index, agent, first),
    }
  })
}

/** The long session's run as the runs list and the run read answer it. */
function longRun(calls: ReturnType<typeof longRunCalls>, mixedCurrency: boolean) {
  const ended = Date.parse(calls[calls.length - 1].timestamp) + 6_000
  const agents = [
    'main',
    ...Array.from({ length: SUBAGENTS }, (_, index) => `subagent ${index + 1}`),
  ]
  const run = {
    session_id: LONG_SESSION,
    session_name: 'Port the billing service to the new queue',
    harness: 'claude_code',
    started_at: new Date(START).toISOString(),
    ended_at: new Date(ended).toISOString(),
    duration_ms: ended - START,
    calls: calls.length,
    agents: agents.map((label, index) => {
      const own = calls.filter((call) => call.agent === label)
      return {
        label,
        calls: own.length,
        first_call_at: own[0].timestamp,
        last_call_at: own[own.length - 1].timestamp,
        models: [FLASH, QWEN],
        cost: cost(own.length * 0.0021),
        ...(index
          ? {
              id: `5e0c1a${index}f2b77`,
              parent: 'main',
              type: index % 3 ? 'Explore' : 'general-purpose',
            }
          : {}),
      }
    }),
    models: [
      { name: FLASH, value: calls.filter((call) => call.model === FLASH).length },
      { name: QWEN, value: calls.filter((call) => call.model === QWEN).length },
    ],
    outcome: {
      status: 'answered',
      final_answer_record_id: 'long-160',
      tool_results: 54,
      tool_failures: 18,
    },
    cost: mixedCurrency
      ? {
          ...cost(0.334),
          unpriced_records: 4,
          // the session's older calls were priced in plain units, before USD list prices
          by_currency: [
            { currency: 'USD', total: 0.334, router_call_spend: 0 },
            { currency: 'XXX', total: 0.2112, router_call_spend: 0 },
          ],
        }
      : cost(0.334),
    events: {
      subagent_start: 9,
      compaction: 1,
      image: 1,
      tool_loop: 18,
      failure: 1,
      guard: 1,
      block: 1,
      switch: calls.filter((call) => call.markers?.includes('switch')).length,
    },
  }
  return run
}

export async function mockLongRun(page: Page, { mixedCurrency = false } = {}) {
  await mockAuthenticatedAppShell(page)
  const calls = longRunCalls()
  const run = longRun(calls, mixedCurrency)
  await page.route(`${REPLAYS}/models`, (route) =>
    route.fulfill({ json: { object: 'list', data: [] } }),
  )
  await page.route(`${REPLAYS}/run?*`, (route) =>
    route.fulfill({ json: { object: 'router_replay.run', run, calls } }),
  )
}

/**
 * The Sessions view listing the long session, its run read held until the
 * returned release is called; counts the run reads the page makes.
 */
export async function mockLongSession(page: Page) {
  await mockOverview(page, { mixedCurrency: true })
  const calls = longRunCalls()
  const run = longRun(calls, true)
  let release: () => void = () => undefined
  const released = new Promise<void>((resolve) => {
    release = resolve
  })
  const reads = { count: 0 }
  await page.route(`${REPLAYS}/runs*`, (route) =>
    route.fulfill({
      json: {
        object: 'router_replay.runs',
        count: 1,
        total: 1,
        limit: 100,
        offset: 0,
        has_more: false,
        data: [run],
      },
    }),
  )
  await page.route(`${REPLAYS}/run?*`, async (route) => {
    reads.count += 1
    await released
    await route.fulfill({ json: { object: 'router_replay.run', run, calls } })
  })
  return { release, reads }
}

const record = (id: string, harness: string | undefined, minutes: number, model: string) => ({
  id,
  timestamp: new Date(Date.now() - minutes * 60_000).toISOString(),
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 300,
  recipe: 'default',
  original_model: 'vllm-sr/auto',
  decision: 'agentic',
  selected_model: model,
  lifecycle_state: 'completed',
  response_status: 200,
  duration_ms: 900,
  ...(harness ? { agent: { harness } } : {}),
})

const records = [
  record('claude-1', 'claude_code', 2, FLASH),
  record('codex-1', 'codex', 4, QWEN),
  record('opencode-1', 'opencode', 6, FLASH),
  record('cursor-1', 'cursor', 8, QWEN),
  record('openclaw-1', 'openclaw', 11, FLASH),
  record('api-1', undefined, 13, QWEN),
]

const aggregate = {
  object: 'router_replay.aggregate',
  record_count: 1464,
  lifecycle: { completed: 1449, failed: 9, aborted: 2, in_progress: 4, unknown: 0 },
  summary: {
    total_saved: 18.42,
    gross_saved: 19.03,
    netted_router_call_spend: 0.61,
    baseline_spend: 31.9,
    actual_spend: 13.48,
    router_call_spend: 0.61,
    total_spend: 14.09,
    currency: 'USD',
    cost_record_count: 1322,
    excluded_record_count: 142,
    coverage: {
      priced_records: 1322,
      partly_priced_records: 0,
      unpriced_records: 142,
      unpriced_reasons: { no_price_configured: 138, not_completed: 4 },
      looper_hop_records: 0,
      list_price_records: 1322,
      caller_login_records: 0,
      router_calls: 1464,
      unpriced_router_calls: 0,
      baseline_sources: { configured_model: 1322 },
    },
  },
  provider_cache: {
    no_cache_spend: 19.6,
    actual_spend: 13.48,
    net_savings: 6.12,
    currency: 'USD',
    priced_records: 980,
    excluded_records: 484,
    by_decision: [],
    by_selected_model: [],
  },
  model_selection: [
    { name: FLASH, value: 902 },
    { name: QWEN, value: 371 },
    { name: 'qwen/qwen3.6-35b-a3b', value: 191 },
  ],
  decision_distribution: [
    { name: 'agentic', value: 811 },
    { name: 'fast', value: 402 },
    { name: 'housekeeping', value: 166 },
    { name: 'guard', value: 85 },
  ],
  signal_distribution: [],
  token_volume: {
    input_tokens: 48_200_000,
    output_tokens: 1_310_000,
    total_tokens: 49_510_000,
    excluded_record_count: 12,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: ['default'],
  available_decisions: ['agentic', 'fast', 'housekeeping', 'guard'],
  available_models: [FLASH, QWEN, 'qwen/qwen3.6-35b-a3b'],
}

const config = {
  version: 'v0.5',
  listeners: [{ name: 'http-8899', port: 8899 }],
  providers: {
    defaults: { model: QWEN },
    models: [
      { name: QWEN, backend_refs: [{ name: 'local', provider: 'vllm' }] },
      { name: FLASH, backend_refs: [{ name: 'api', provider: 'deepseek' }] },
    ],
  },
  entrypoints: [{ model_names: ['vllm-sr/auto'], recipe: 'default', default: true }],
  recipes: [
    {
      name: 'default',
      signals: { keywords: [{ name: 'code_request' }] },
      decisions: [
        { name: 'agentic', priority: 300, modelRefs: [{ model: FLASH }] },
        { name: 'fast', priority: 200, modelRefs: [{ model: QWEN }] },
      ],
    },
  ],
}

/** Home on the same day of traffic, with the router's configuration and services. */
export async function mockHomeOverview(page: Page) {
  await mockOverview(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({ json: { deployments: [] } }),
  )
  await page.route('**/api/status', (route) =>
    route.fulfill({
      json: {
        overall: 'healthy',
        serving_mode: 'router',
        deployment_type: 'docker',
        version: 'v0.5.0',
        services: [
          { name: 'Router', status: 'running', healthy: true, message: 'ExtProc routing engine' },
          { name: 'Dashboard', status: 'running', healthy: true },
        ],
        models: { models: [] },
      },
    }),
  )
}

// The same day, part of it priced in plain units before USD list prices were set.
const mixedAggregate = {
  ...aggregate,
  summary: {
    ...aggregate.summary,
    by_currency: [
      {
        currency: 'USD',
        total_saved: 17.81,
        gross_saved: 18.42,
        netted_router_call_spend: 0.61,
        baseline_spend: 31.9,
        actual_spend: 13.48,
        router_call_spend: 0.61,
        total_spend: 14.09,
        cost_record_count: 1322,
      },
      {
        currency: 'XXX',
        total_saved: 40.1,
        baseline_spend: 120.4,
        actual_spend: 80.3,
        total_spend: 80.3,
        cost_record_count: 2210,
      },
    ],
  },
}

export async function mockOverview(page: Page, { mixedCurrency = false } = {}) {
  await mockAuthenticatedAppShell(page)
  await page.route(/\/api\/router\/api\/v1\/observability\/replays(\?|$)/, (route) =>
    route.fulfill({
      json: {
        object: 'router_replay.list',
        count: records.length,
        total: 1464,
        has_more: true,
        harness_counts: HARNESS_COUNTS,
        data: records,
      },
    }),
  )
  await page.route(`${REPLAYS}/aggregate*`, (route) =>
    route.fulfill({ json: mixedCurrency ? mixedAggregate : aggregate }),
  )
  await page.route(`${REPLAYS}/models`, (route) =>
    route.fulfill({ json: { object: 'list', data: [] } }),
  )
  await page.route(`${REPLAYS}/runs*`, (route) =>
    route.fulfill({
      json: {
        object: 'list',
        count: 0,
        total: 0,
        limit: 100,
        offset: 0,
        has_more: false,
        data: [],
      },
    }),
  )
}
