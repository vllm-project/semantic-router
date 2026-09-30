import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import InsightsRecordSection from './InsightsRecordSection'
import InsightsSessionRoutes from './InsightsSessionRoutes'
import { fetchInsightsTrajectory } from './insightsPageApi'
import { buildInsightsRecordSections } from './insightsPageSupport'
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
    expect(html).toContain('17 ms')
    expect(html).toContain('23 ms')
    expect(html).toContain('observe_only')
    expect(html).toContain('/insights/one')
    expect(html).toContain('/insights/two')
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
