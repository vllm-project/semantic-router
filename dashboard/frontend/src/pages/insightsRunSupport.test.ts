import { describe, expect, it } from 'vitest'

import { formatRunCost, getRunPath, runAgentTreeText, runTitle } from './insightsRunSupport'
import { compactID } from './insightsReviewSupport'
import type { RunResponse } from './insightsRunTypes'

const START = Date.parse('2026-11-16T09:00:00Z')

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}

function run(calls: RunResponse['calls']): RunResponse {
  const agent = (label: string, parent?: string) => ({
    label,
    parent,
    calls: calls.filter((entry) => entry.agent === label).length,
    first_call_at: '',
    last_call_at: '',
    cost,
  })
  return {
    object: 'router_replay.run',
    run: {
      session_id: 'run-1',
      started_at: new Date(START).toISOString(),
      ended_at: new Date(START + 100_000).toISOString(),
      duration_ms: 100_000,
      calls: calls.length,
      // The response lists the nested subagent before its sibling; lanes follow the tree.
      agents: [agent('main'), agent('subagent 2', 'subagent 1'), agent('subagent 1', 'main')],
      models: [],
      outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
      cost,
    },
    calls,
  }
}

describe('run labels', () => {
  it('formats cost per currency and names an unpriced run', () => {
    expect(
      formatRunCost({
        ...cost,
        by_currency: [{ currency: 'USD', total: 0.0123, router_call_spend: 0 }],
      }),
    ).toBe('$0.0123')
    expect(formatRunCost({ ...cost, unpriced_records: 2 })).toBe('no price')
    // One call whose classifier call has no price counts once.
    expect(
      formatRunCost({
        ...cost,
        by_currency: [{ currency: 'USD', total: 0.1031, router_call_spend: 0 }],
        unpriced_records: 1,
        unpriced_router_calls: 1,
      }),
    ).toBe('$0.1031 (1 unpriced)')
  })

  it('names a run by its session name, else its task, else its session', () => {
    const summary = run([]).run
    expect(runTitle({ ...summary, session_name: 'Average helper', prompt: 'fix average' })).toBe(
      'Average helper',
    )
    expect(runTitle({ ...summary, prompt: ' fix average ' })).toBe('fix average')
    expect(runTitle({ ...summary, session_id: '0123456789abcdef0123' })).toBe(
      `Session ${compactID('0123456789abcdef0123')}`,
    )
  })

  it('reads the agent tree at a glance and links a run with its step', () => {
    expect(runAgentTreeText(run([]).run)).toBe('main, 2 subagents')
    expect(getRunPath('tenant:a/b', 'rec-1')).toBe(
      '/insights/run?session=tenant%3Aa%2Fb&step=rec-1',
    )
  })
})
