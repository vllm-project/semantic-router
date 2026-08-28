import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import InsightsRecordTrace from './InsightsRecordTrace'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'

const record: InsightsRecord = {
  id: 'record-1',
  timestamp: '2026-08-28T00:00:00Z',
  session_id: 'session-1',
  turn_index: 1,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
}

describe('InsightsRecordTrace', () => {
  it('renders every turn, tool call, result, and final answer in one record trace', () => {
    const trajectory: InsightsTrajectory = {
      object: 'router_replay.trajectory',
      session_id: 'session-1',
      record_count: 3,
      turn_count: 2,
      messages: [
        { role: 'user', content: 'how are you?', turn_index: 0 },
        { role: 'assistant', content: 'Doing well.', turn_index: 0 },
        { role: 'user', content: 'what do you know?', turn_index: 1 },
        {
          role: 'assistant',
          turn_index: 1,
          tool_calls: [
            {
              id: 'call-1',
              type: 'function',
              function: { name: 'web_search', arguments: '{"q":"vllm-sr"}' },
            },
          ],
        },
        {
          role: 'tool',
          content: 'search result',
          tool_call_id: 'call-1',
          tool_name: 'web_search',
          turn_index: 1,
        },
        { role: 'assistant', content: 'Final answer.', turn_index: 1 },
      ],
    }

    const markup = renderToStaticMarkup(
      <InsightsRecordTrace record={record} trajectory={trajectory} />,
    )
    expect(markup).toContain('Record trace')
    expect(markup).toContain('2 turns · 1 tool call')
    expect(markup).toContain('how are you?')
    expect(markup).toContain('web_search')
    expect(markup).toContain('search result')
    expect(markup).toContain('Final answer.')
  })
})
