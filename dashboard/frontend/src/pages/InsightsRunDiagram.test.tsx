import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'

import { ModelLabelsContext } from './insightsModelLabelSupport'
import { RunCallDetails, RunDiagram, RunLegend, RunSegmentPanel } from './InsightsRunDiagram'
import { buildRunTimeline } from './insightsRunTimeline'
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
  duration_ms: 5000,
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

const timeline = buildRunTimeline(response, 0, 2)

describe('run diagram', () => {
  it('draws a labelled lane per agent with the spawn, and the failure on the events strip', () => {
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <RunDiagram run={response.run} timeline={timeline} onSelect={() => undefined} />
      </TooltipProvider>,
    )
    expect(html).toContain('Main agent')
    expect(html).toContain('Subagent 1')
    expect(html).toContain('collab_spawn')
    expect(html).toContain('title="Started by the main agent"')
    expect(html).toContain('left:50%')
    const strip = html.slice(
      html.indexOf('aria-label="Run events"'),
      html.indexOf('aria-label="Run timeline"'),
    )
    expect(strip).toContain('data-event="failure"')
    expect(strip).toContain('aria-label="Failed"')
    expect(html).toContain('>Events<')
  })

  it('rules a compaction through every lane and labels its marker in words where there is room', () => {
    const compacted: RunResponse = {
      ...response,
      calls: [
        response.calls[0],
        { ...failed, agent: 'main', markers: ['compaction'], lifecycle_state: 'completed' },
      ],
    }
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <RunDiagram
          run={response.run}
          timeline={buildRunTimeline(compacted, 0, 2)}
          numbers={new Map([['rec-2', 2]])}
          onSelect={() => undefined}
        />
      </TooltipProvider>,
    )
    expect(html).toContain('data-compaction-rule')
    expect(html).toContain('aria-label="Compaction, call 2"')
    expect(html).toMatch(/data-event="compaction"[^>]*>.*Compaction<\/span>/)
  })

  it("lets a lane's agent be followed, and dims the agents not followed", () => {
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <RunDiagram
          run={response.run}
          timeline={timeline}
          agents={['subagent 1']}
          onSelect={() => undefined}
          onAgent={() => undefined}
        />
      </TooltipProvider>,
    )
    expect(html).toContain('aria-pressed="true"')
    expect(html).toContain('aria-pressed="false"')
    expect(html).toContain('title="Follow subagent 1&#x27;s calls"')
    // the dimmed agent's label and its track
    expect(html.match(/opacity-45/g)).toHaveLength(2)
  })

  it('folds a long idle stretch and says how long it lasted', () => {
    const idle: RunResponse = {
      ...response,
      calls: [
        { ...response.calls[0], duration_ms: 1000 },
        { ...response.calls[0], record_id: 'rec-3', timestamp: '2026-11-16T09:10:00Z' },
      ],
    }
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <RunDiagram
          run={response.run}
          timeline={buildRunTimeline(idle, 0, 2)}
          onSelect={() => undefined}
        />
      </TooltipProvider>,
    )
    expect(html).toContain('9m 59s idle')
    expect(html).toContain('repeating-linear-gradient')
  })

  it('details a step with its route, a hidden reason, and its record', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <TooltipProvider>
          <RunCallDetails call={failed} />
        </TooltipProvider>
      </MemoryRouter>,
    )
    expect(html).toContain('data-event="failure"')
    expect(html).toContain('title="keyword:email"')
    expect(html).toContain('high')
    expect(html).toContain('failed 502')
    expect(html).toContain('Hidden from this viewer')
    expect(html).toContain('href="/observe/requests/rec-2"')
  })

  it('shows each model by its label and keeps the configured name beside it', () => {
    const labels = { 'qwen-msg': 'Qwen3.6 35B A3B', 'qwen-chat': 'qwen36' }
    const [, subagent] = timeline.lanes
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <TooltipProvider>
          <ModelLabelsContext.Provider value={labels}>
            <RunLegend run={{ ...response.run, events: { failure: 1 } }} timeline={timeline} />
            <RunDiagram run={response.run} timeline={timeline} onSelect={() => undefined} />
            <RunSegmentPanel
              segment={subagent.blocks[0]}
              call={failed}
              onSelectCall={() => undefined}
            />
          </ModelLabelsContext.Provider>
        </TooltipProvider>
      </MemoryRouter>,
    )
    expect(html).toContain('Qwen3.6 35B A3B<small>qwen-msg</small>')
    expect(html).toContain('aria-label="Qwen3.6 35B A3B (qwen-msg), 1 call"')
    expect(html).toContain('aria-label="Step details"')
    expect(html).toContain('qwen36<small>qwen-chat</small>')
    // the legend explains each marker in words, with the run's count
    expect(html).toContain('data-legend="failure"')
    expect(html).toMatch(/data-legend="failure"[^]*Failed<\/span>[^]*>1<\/span>/)
  })

  it("lists a block's tool calls with their time and result, and prices a list price", () => {
    const toolCall: RunCall = {
      ...failed,
      request_kind: 'tool turn',
      cost: 0.002,
      currency: 'USD',
      pricing_source: 'catalog_list_price',
      tool_results: 2,
      tool_calls: [
        { name: 'Bash', arguments: 'pytest -q', status: 'succeeded', client_ms: 13_000 },
        {
          name: 'Read',
          summary: 'README.md',
          status: 'failed',
          timing_source: 'plugin',
          started_at: '2026-11-16T09:00:06Z',
          ended_at: '2026-11-16T09:00:08Z',
        },
      ],
    }
    const [, subagent] = timeline.lanes
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <TooltipProvider>
          <RunSegmentPanel
            segment={{ ...subagent.blocks[0], calls: [toolCall, toolCall] }}
            call={toolCall}
            onSelectCall={() => undefined}
          />
        </TooltipProvider>
      </MemoryRouter>,
    )
    expect(html).toContain('aria-label="Tool calls"')
    expect(html).toContain('pytest -q')
    expect(html).toContain('13.0s')
    expect(html).toContain('title="Run time the harness reported">2.0s')
    expect(html).toContain('data-status="failed"')
    expect(html).toContain('tool turn')
    expect(html).toContain('list price')
    expect(html).toContain('2 results sent')
  })
})
