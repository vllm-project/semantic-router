import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { RunCallDetails, RunDiagram } from './InsightsRunDiagram'
import { buildRunLanes } from './insightsRunSupport'
import type { RunCall, RunResponse } from './insightsRunTypes'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}
const failed: RunCall = {
  record_id: 'rec-2',
  agent: 'subagent 1',
  model: 'qwen-chat',
  timestamp: '2026-11-16T09:00:05Z',
  duration_ms: 1000,
  lifecycle_state: 'failed',
  response_status: 502,
  decision: 'remember',
  signals: ['keyword:email'],
  effort: 'high',
  reason: '',
  markers: ['failure'],
}
const response: RunResponse = {
  object: 'router_replay.run',
  run: {
    session_id: 'run-1',
    started_at: '2026-11-16T09:00:00Z',
    ended_at: '2026-11-16T09:00:10Z',
    duration_ms: 10_000,
    calls: 2,
    agents: [
      { label: 'main', calls: 1, first_call_at: '', last_call_at: '', cost },
      {
        label: 'subagent 1',
        parent: 'main',
        kind: 'collab_spawn',
        calls: 1,
        first_call_at: '',
        last_call_at: '',
        cost,
      },
    ],
    models: [
      { name: 'qwen-msg', value: 1 },
      { name: 'qwen-chat', value: 1 },
    ],
    outcome: { status: 'failed', tool_results: 0, tool_failures: 0 },
    cost,
  },
  calls: [
    {
      record_id: 'rec-1',
      agent: 'main',
      model: 'qwen-msg',
      timestamp: '2026-11-16T09:00:00Z',
      duration_ms: 4000,
      lifecycle_state: 'completed',
    },
    failed,
  ],
}

describe('run diagram', () => {
  it('draws a lane per agent with the spawn and the failure marked', () => {
    const html = renderToStaticMarkup(
      <RunDiagram run={response.run} lanes={buildRunLanes(response)} onSelect={() => undefined} />,
    )
    expect(html).toContain('↳ subagent 1')
    expect(html).toContain('collab_spawn')
    expect(html).toContain('title="Spawned by main"')
    expect(html).toContain('left:50%')
    expect(html).toContain('data-marker="failure"')
  })

  it('details a step with its route, a hidden reason, and its record', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <RunCallDetails call={failed} />
      </MemoryRouter>,
    )
    expect(html).toContain('keyword:email')
    expect(html).toContain('high')
    expect(html).toContain('failed 502')
    expect(html).toContain('Hidden from this viewer')
    expect(html).toContain('href="/insights/rec-2"')
  })
})
