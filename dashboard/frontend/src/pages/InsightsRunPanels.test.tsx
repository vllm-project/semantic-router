import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { RunAgentsPanel, RunCallsPanel, RunOutcomesPanel } from './InsightsRunPanels'
import type { RunCall, RunSummary } from './insightsRunTypes'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}

const START = Date.parse('2026-11-16T09:00:00Z')
const calls: RunCall[] = Array.from({ length: 60 }, (_, index) => ({
  record_id: `rec-${index + 1}`,
  agent: index % 3 === 2 ? 'subagent 1' : 'main',
  model: 'qwen-chat',
  timestamp: new Date(START + index * 1000).toISOString(),
  duration_ms: 800,
  lifecycle_state: 'completed',
  markers: index === 41 ? ['failure'] : undefined,
}))

const run: RunSummary = {
  session_id: 'run-1',
  started_at: new Date(START).toISOString(),
  ended_at: new Date(START + 60_000).toISOString(),
  duration_ms: 60_000,
  calls: calls.length,
  agents: [
    {
      label: 'main',
      calls: 40,
      first_call_at: '2026-11-16T09:00:00Z',
      last_call_at: '2026-11-16T09:01:00Z',
      models: ['qwen-chat'],
      cost,
    },
    {
      label: 'subagent 1',
      id: '3fa9c2d81e07',
      parent: 'main',
      kind: 'collab_spawn',
      calls: 20,
      first_call_at: '2026-11-16T09:00:02Z',
      last_call_at: '2026-11-16T09:00:59Z',
      cost,
    },
  ],
  models: [{ name: 'qwen-chat', value: 60 }],
  outcome: {
    status: 'answered',
    final_answer_record_id: 'rec-60',
    tool_results: 12,
    tool_failures: 1,
    reported: [
      {
        record_id: 'rec-3',
        outcome: { source: 'eval', target: 'model', verdict: 'good_fit', reason: 'Tests pass' },
      },
      {
        record_id: 'rec-42',
        outcome: { source: 'user', target: 'route', verdict: 'underpowered' },
      },
    ],
  },
  cost,
  events: { failure: 1 },
}

const numbers = new Map(calls.map((entry, index) => [entry.record_id, index + 1]))
const render = (node: React.ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>)

describe('run panels', () => {
  it('pages the calls in the browser and keeps the agent and event filters', () => {
    const html = render(
      <RunCallsPanel
        run={run}
        calls={calls}
        numbers={numbers}
        view={{ agents: [], markers: [], page: 2, size: 25 }}
        onView={() => undefined}
        onShow={() => undefined}
      />,
    )
    expect(html).toContain('26–50 of 60 calls')
    expect(html).toContain('Page 2 of 3')
    expect(html).toContain('>Agent<')
    expect(html).toContain('>Events<')
    const followed = render(
      <RunCallsPanel
        run={run}
        calls={calls}
        numbers={numbers}
        view={{ agents: ['subagent 1'], markers: ['failure'], page: 1, size: 25 }}
        onView={() => undefined}
        onShow={() => undefined}
      />,
    )
    expect(followed).toContain('1–1 of 1 call')
    expect(followed).toContain('data-marker="failure"')
  })

  it('lists the agents as a tree with who spawned each', () => {
    const html = render(<RunAgentsPanel run={run} onFollow={() => undefined} />)
    expect(html).toContain('Main agent')
    expect(html).toContain('Spawned by main · #3fa9c2d8')
    expect(html).toContain('spawned agent')
    expect(html).toContain('1m 0s')
  })

  it('states how the run ended and each reported outcome with its verdict', () => {
    const html = render(<RunOutcomesPanel run={run} calls={calls} numbers={numbers} />)
    expect(html).toContain('Final answer on call #60')
    expect(html).toContain('1 failed')
    expect(html).toContain('1 good fit · 1 underpowered')
    expect(html).toContain('Evaluation')
    expect(html).toContain('Tests pass')
    expect(html).toContain('#42')
    expect(html).toContain('bg-success-soft')
    expect(html).toContain('bg-warning-soft')
  })

  it('says when no outcome was reported, and why the table is empty', () => {
    const html = render(
      <RunOutcomesPanel
        run={{ ...run, outcome: { ...run.outcome, reported: undefined } }}
        calls={calls}
        numbers={numbers}
      />,
    )
    expect(html).toContain('No outcomes reported for this run')
    expect(html).toContain('None reported')
    expect(html).not.toContain('Reported by')
  })
})
