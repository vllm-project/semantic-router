import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import InsightsRecordTrace from './InsightsRecordTrace'
import InsightsSessionRoutes from './InsightsSessionRoutes'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'
const record: InsightsRecord = {
  id: 'record-a',
  timestamp: '2026-09-15T00:00:00Z',
  session_id: 'shared-session',
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
}
const trajectory: InsightsTrajectory = {
  object: 'router_replay.trajectory',
  session_id: 'shared-session',
  recipe: 'example',
  record_count: 3,
  turn_count: 2,
  routes: ['conversation-a', 'conversation-a', 'conversation-b'].map((conversation_id, index) => ({
    record_id: `record-${index}`,
    timestamp: record.timestamp,
    conversation_id,
    turn_index: 0,
    session_policy_applied: false,
    lifecycle_state: 'completed',
    duration_ms: 5,
  })),
  messages: [
    { role: 'user', content: 'First task', conversation_id: 'conversation-a', turn_index: 0 },
    {
      role: 'assistant',
      conversation_id: 'conversation-a',
      turn_index: 0,
      tool_calls: [
        {
          id: 'call-a',
          type: 'function',
          function: { name: 'read_state', arguments: '{}' },
        },
      ],
    },
    {
      role: 'tool',
      content: 'ready',
      tool_call_id: 'call-a',
      conversation_id: 'conversation-a',
      turn_index: 0,
    },
    { role: 'user', content: 'Separate task', conversation_id: 'conversation-b', turn_index: 0 },
    {
      role: 'assistant',
      content: 'Separate complete',
      conversation_id: 'conversation-b',
      turn_index: 0,
    },
  ],
}
describe('Replay conversation identity', () => {
  it('shows boundaries without hiding any same-turn route and exposes full IDs', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <InsightsSessionRoutes trajectory={trajectory} />
      </MemoryRouter>,
    )
    expect(html).toContain('Conversation 1')
    expect(html).toContain('Conversation 2')
    expect(html).toContain('conversation-a')
    expect(html).toContain('conversation-b')
    expect(html.match(/href=/g)).toHaveLength(3)
  })
  it('keeps reset turn numbers in different conversations separate, including tools', () => {
    const html = renderToStaticMarkup(
      <InsightsRecordTrace record={record} trajectory={trajectory} />,
    )
    expect(html).toContain('2 turns · 1 tool call')
    expect(html).toContain('Conversation 1 · Turn 1')
    expect(html).toContain('Conversation 2 · Turn 1')
    expect(html).toContain('conversation-a')
    expect(html).toContain('conversation-b')
    expect(html).toContain('read_state')
    expect(html).toContain('Separate complete')
  })
  it('does not invent conversation identities for legacy records', () => {
    const legacy: InsightsTrajectory = {
      ...trajectory,
      routes: trajectory.routes?.map((route) => ({ ...route, conversation_id: undefined })),
      messages: trajectory.messages.map((message) => ({ ...message, conversation_id: undefined })),
    }
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <InsightsSessionRoutes trajectory={legacy} />
        <InsightsRecordTrace record={record} trajectory={legacy} />
      </MemoryRouter>,
    )
    expect(html).not.toContain('Conversation 1')
    expect(html).not.toContain('Conversation 2')
    expect(html).toContain('1 turn · 1 tool call')
  })
})
