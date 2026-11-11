import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import InsightsRecordSection from './InsightsRecordSection'
import InsightsSessionRoutes from './InsightsSessionRoutes'
import { fetchInsightsTrajectory } from './insightsPageApi'
import { buildInsightsRecordSections, createInsightsTableColumns } from './insightsPageSupport'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'

const record: InsightsRecord = {
  id: 'synthetic-record',
  timestamp: '2026-09-14T00:00:00Z',
  session_id: 'session-one',
  turn_index: 1,
  decision_tier: 0,
  decision_priority: 0,
  signals: { domain: ['technical'] },
  selected_model: 'beta',
  selection_method: 'multi_factor',
  route_diagnostics: {
    selection_reasoning: 'beta has the highest qualified score',
    previous_model: 'alpha',
    selected_model: 'beta',
    session_policy_applied: false,
    session_action: 'switch',
    session_reason: 'observe_only',
    request_demand_snapshots: [
      {
        stage: 'dispatch',
        model: 'beta',
        prompt_tokens: 80,
        reserved_output_tokens: 200,
        total_demand_tokens: 280,
        total_demand_known: true,
        counting_source: 'character_estimate',
        output_reserve_source: 'request',
      },
    ],
  },
  learning: {
    protection: {
      mode: 'observe',
      scope: 'conversation',
      current_model: 'alpha',
      selected_model: 'alpha',
      active_tool_loop: true,
      hard_locked: true,
      hard_lock_reason: 'hard_lock=tool_loop',
      base_scores: { alpha: 0, beta: 0.9 },
      final_scores: { alpha: 1.1, beta: 0.9 },
      candidate_traces: { alpha: { tool_loop_penalty: 0 }, beta: { handoff_penalty: 0.2 } },
    },
  },
}

function renderRecord(value: InsightsRecord) {
  return renderToStaticMarkup(
    <>
      {buildInsightsRecordSections(value, { isReadonly: true }).map((section, index) => (
        <InsightsRecordSection key={section.title} section={section} sectionIndex={index} />
      ))}
    </>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('recorded routing explanation', () => {
  it('renders actual selection separately from an observed hold and preserves zero scores', () => {
    const html = renderRecord(record)
    expect(html).toContain('Actual selected model</span><div>beta')
    expect(html).toContain('Observed model suggestion</span><div>alpha')
    expect(html).toContain('Protection applied</span><div>No')
    expect(html).toContain('Observed hold reason')
    expect(html).toContain('hard_lock=tool_loop')
    expect(html).toContain('Observed Candidate Scores')
    expect(html).toContain('0.0000')
    expect(html).toContain('1.1000')
    expect(html).toContain('beta has the highest qualified score')
    expect(html).toContain('Request Capacity')
    const capacity = buildInsightsRecordSections(record, { isReadonly: true }).find(
      (section) => section.title === 'Request Capacity',
    )
    expect(renderToStaticMarkup(<>{capacity?.fields[0].value}</>)).toContain('character_estimate')
    expect(html).not.toContain('NaN')
  })

  it('shows actual holds and missing evidence without inventing candidate scores', () => {
    const html = renderRecord({
      ...record,
      selected_model: 'alpha',
      learning: undefined,
      session_policy: {
        mode: 'apply',
        selected_model: 'alpha',
        hard_lock_reason: 'tool_loop',
        missing_signals: ['cache_warmth'],
      },
      route_diagnostics: {
        session_policy_applied: true,
        session_action: 'hard_lock',
        signal_errors: { 'domain:test': 'classifier unavailable' },
      },
    })
    expect(html).toContain('Protection applied</span><div>Yes')
    expect(html).toContain('cache_warmth')
    expect(html).toContain('classifier unavailable')
    expect(html).not.toContain('Candidate Scores')
    expect(html).not.toContain('Observed model suggestion')
  })

  it('keeps each same-turn tool hop visible and distinguishes failures from completed routes', () => {
    const trajectory: InsightsTrajectory = {
      object: 'router_replay.trajectory',
      session_id: 'session-one',
      recipe: 'first',
      record_count: 2,
      turn_count: 1,
      messages: [],
      routes: [
        {
          record_id: 'one',
          timestamp: record.timestamp,
          turn_index: 1,
          decision: 'tools',
          selected_model: 'alpha',
          session_policy_applied: true,
          session_action: 'hard_lock',
          session_reason: 'tool_loop',
          lifecycle_state: 'completed',
          response_status: 200,
          duration_ms: 17,
        },
        {
          record_id: 'two',
          timestamp: record.timestamp,
          agent: '9f2c1a',
          turn_index: 1,
          decision: 'review',
          previous_model: 'alpha',
          selected_model: 'beta',
          session_policy_applied: false,
          session_action: 'switch',
          session_reason: 'observe_only',
          lifecycle_state: 'failed',
          response_status: 503,
          duration_ms: 23,
        },
      ],
    }
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <InsightsSessionRoutes trajectory={trajectory} />
      </MemoryRouter>,
    )
    expect(html).toContain('Session routing history')
    expect(html).toContain('completed')
    expect(html).toContain('failed')
    expect(html).toContain('HTTP 503')
    expect(html).toContain('0.017s')
    expect(html).toContain('0.023s')
    expect(html).toContain('observe_only')
    expect(html).toContain('/insights/one')
    expect(html).toContain('/insights/two')
    // a subagent's call names its agent, the main conversation's says main
    expect(html).toContain('9f2c1a')
    expect(html).toContain('>main<')
  })

  it('lists each request with its session, next to the request id', () => {
    const columns = createInsightsTableColumns()
    expect(columns.map((column) => column.key).slice(0, 2)).toEqual(['request_id', 'session_id'])
    const session = columns[1]
    const cell = renderToStaticMarkup(
      <>{session.render!({ ...record, session_id: '21f9e946-5ddf-4b90-a897-7db814a78317' })}</>,
    )
    expect(cell).toContain('title="21f9e946-5ddf-4b90-a897-7db814a78317"')
    expect(renderToStaticMarkup(<>{session.render!({ ...record, session_id: undefined })}</>)).toBe(
      '-',
    )
  })

  it('says in words why a session kept or switched its model, with the code in the title', () => {
    const route = {
      timestamp: record.timestamp,
      turn_index: 0,
      decision: 'think_fast',
      selected_model: 'qwen36',
      lifecycle_state: 'completed',
      response_status: 200,
      duration_ms: 5,
    }
    const trajectory: InsightsTrajectory = {
      object: 'router_replay.trajectory',
      session_id: 'session-one',
      recipe: 'default',
      record_count: 3,
      turn_count: 1,
      messages: [],
      routes: [
        {
          ...route,
          record_id: 'a',
          session_action: 'stay',
          session_reason: 'session_scope_protect',
          session_policy_applied: true,
        },
        {
          ...route,
          record_id: 'b',
          agent: '10ee5571',
          session_action: 'stay',
          session_reason: 'session_scope_protect',
          session_policy_applied: true,
        },
        {
          ...route,
          record_id: 'c',
          agent: '990eaa8c',
          session_action: 'switch',
          session_reason: 'previous_model_not_in_candidates',
          session_policy_applied: true,
        },
      ],
    }
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <InsightsSessionRoutes trajectory={trajectory} />
      </MemoryRouter>,
    )
    expect(html).toContain('Kept the model · by session protection')
    expect(html).toContain('Kept this session&#x27;s model')
    expect(html).toContain('Kept this agent&#x27;s model')
    expect(html).toContain('Switched model')
    expect(html).toContain('The previous model is not a candidate here')
    expect(html).toContain('title="session_scope_protect"')
    expect(html).toContain('title="previous_model_not_in_candidates"')
  })

  it.each(['first', ''])('requests the exact record recipe scope %j', async (recipe) => {
    const fetch = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ messages: [], routes: [] }), { status: 200 }),
      )
    vi.stubGlobal('fetch', fetch)
    await fetchInsightsTrajectory('same/id', recipe)
    const request = new URL(fetch.mock.calls[0][0], 'http://localhost')
    expect(request.searchParams.get('session_id')).toBe('same/id')
    expect(request.searchParams.has('recipe')).toBe(true)
    expect(request.searchParams.get('recipe')).toBe(recipe)
  })
})
