import { describe, expect, it } from 'vitest'

import {
  buildRunLanes,
  formatRunCost,
  getRunPath,
  runAgentTreeText,
  runTitle,
} from './insightsRunSupport'
import { compactID } from './insightsReviewSupport'
import type { RunCall, RunResponse } from './insightsRunTypes'

const START = Date.parse('2026-11-16T09:00:00Z')

function call(
  agent: string,
  model: string,
  second: number,
  fields: Partial<RunCall> = {},
): RunCall {
  return {
    record_id: `${agent}-${second}`,
    agent,
    model,
    timestamp: new Date(START + second * 1000).toISOString(),
    duration_ms: 1000,
    lifecycle_state: 'completed',
    ...fields,
  }
}

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}

function run(calls: RunCall[]): RunResponse {
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

describe('buildRunLanes', () => {
  it("groups an agent's calls on one model until a switch or another agent's call", () => {
    const lanes = buildRunLanes(
      run([
        call('main', 'small', 0),
        call('main', 'small', 10, { markers: ['tool_failure'] }),
        call('subagent 1', 'small', 20),
        call('main', 'small', 30),
        call('main', 'large', 40, { markers: ['switch'] }),
        call('subagent 2', 'small', 50, { markers: ['failure'] }),
      ]),
    )
    expect(lanes.map((lane) => [lane.agent.label, lane.depth])).toEqual([
      ['main', 0],
      ['subagent 1', 1],
      ['subagent 2', 2],
    ])
    const main = lanes[0].segments
    expect(main.map((segment) => [segment.model, segment.calls.length])).toEqual([
      ['small', 2],
      ['small', 1],
      ['large', 1],
    ])
    expect(main[0].markers).toEqual(['tool_failure'])
    expect(main[2].markers).toEqual(['switch'])
    expect(main[0].left).toBe(0)
    expect(main[2].left).toBeCloseTo(40)
    expect(lanes[2].segments[0].markers).toEqual(['failure'])
  })

  it('keeps a few hundred calls to a handful of segments', () => {
    const calls = Array.from({ length: 300 }, (_, index) =>
      call('main', index < 200 ? 'small' : 'large', index / 4),
    )
    expect(buildRunLanes(run(calls))[0].segments).toHaveLength(2)
  })
})

describe('run labels', () => {
  it('formats cost per currency and names an unpriced run', () => {
    expect(
      formatRunCost({
        ...cost,
        by_currency: [{ currency: 'USD', total: 0.0123, router_call_spend: 0 }],
      }),
    ).toBe('$0.0123')
    expect(formatRunCost({ ...cost, unpriced_records: 2 })).toBe('no price')
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
      '/observe/runs?session=tenant%3Aa%2Fb&step=rec-1',
    )
  })
})

describe('a Looper call', () => {
  it('spans the hops its record started after', () => {
    const looper = call('main', 'large', 60, {
      duration_ms: 0,
      hops: [call('main', 'small', 50), call('main', 'large', 55)],
    })
    const [segment] = buildRunLanes(run([looper]))[0].segments
    expect(segment.left).toBeCloseTo(50)
    expect(segment.width).toBeCloseTo(10)
  })
})
