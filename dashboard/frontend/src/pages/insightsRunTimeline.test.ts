import { describe, expect, it } from 'vitest'

import { buildRunTimeline, idleGaps } from './insightsRunTimeline'
import type { RunAgent, RunCall, RunResponse } from './insightsRunTypes'

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

function agent(label: string, parent?: string): RunAgent {
  return { label, parent, calls: 0, first_call_at: '', last_call_at: '', cost }
}

function run(calls: RunCall[], agents?: RunAgent[]): RunResponse {
  return {
    object: 'router_replay.run',
    run: {
      session_id: 'run-1',
      started_at: calls[0]?.timestamp ?? new Date(START).toISOString(),
      ended_at: new Date(START + 100_000).toISOString(),
      duration_ms: 100_000,
      calls: calls.length,
      // The response lists the nested subagent before its sibling; lanes follow the tree.
      agents: agents ?? [
        agent('main'),
        agent('subagent 2', 'subagent 1'),
        agent('subagent 1', 'main'),
      ],
      models: [],
      outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
      cost,
    },
    calls,
  }
}

const shape = (response: RunResponse, from = 0, to = response.calls.length) =>
  buildRunTimeline(response, from, to).lanes.map((lane) => [
    lane.agent.label,
    lane.blocks.map((block) => [block.model, block.calls.length]),
  ])

describe('the run timeline', () => {
  it("draws a lane per agent in tree order, an agent's calls on one model as one block", () => {
    const response = run([
      call('main', 'small', 0),
      call('main', 'small', 1, { markers: ['tool_failure'] }),
      call('subagent 1', 'small', 20),
      call('main', 'small', 30),
      call('main', 'large', 31, { markers: ['switch'] }),
      call('subagent 2', 'small', 50, { markers: ['failure'] }),
    ])
    const timeline = buildRunTimeline(response, 0, response.calls.length)
    expect(timeline.lanes.map((lane) => [lane.agent.label, lane.depth])).toEqual([
      ['main', 0],
      ['subagent 1', 1],
      ['subagent 2', 2],
    ])
    const [main, first, nested] = timeline.lanes
    // the main agent waits on its subagent: the wait stays a gap between two blocks
    expect(main.blocks.map((block) => [block.model, block.calls.length])).toEqual([
      ['small', 2],
      ['small', 1],
      ['large', 1],
    ])
    expect(main.blocks[0].markers).toEqual(['tool_failure'])
    expect(main.blocks[2].markers).toEqual(['switch'])
    expect(main.blocks[0].left).toBe(0)
    expect(main.blocks[1].left).toBeCloseTo((30 / 51) * 100)
    expect(first.spawnAt).toBeCloseTo((20 / 51) * 100)
    expect(nested.blocks[0].markers).toEqual(['failure'])
    expect(timeline.breaks).toEqual([])
  })

  it('keeps a few hundred calls to a handful of blocks', () => {
    const calls = Array.from({ length: 300 }, (_, index) =>
      call('main', index < 200 ? 'small' : 'large', index / 4),
    )
    expect(shape(run(calls))).toEqual([
      [
        'main',
        [
          ['small', 200],
          ['large', 100],
        ],
      ],
    ])
  })

  it('reads subagents working side by side as one block each', () => {
    const calls = [call('main', 'small', 0)]
    for (let round = 0; round < 30; round += 1) {
      calls.push(call('subagent 1', 'small', 2 + round))
      calls.push(call('subagent 2', 'large', 2.3 + round))
    }
    calls.push(call('main', 'small', 33))
    const agents = [agent('main'), agent('subagent 1', 'main'), agent('subagent 2', 'main')]
    expect(shape(run(calls, agents))).toEqual([
      [
        'main',
        [
          ['small', 1],
          ['small', 1],
        ],
      ],
      ['subagent 1', [['small', 30]]],
      ['subagent 2', [['large', 30]]],
    ])
  })

  it('folds an idle stretch to a narrow mark, and no block spans it', () => {
    const response = run([
      call('main', 'small', 0),
      call('main', 'small', 5),
      call('main', 'small', 7205),
      call('main', 'small', 7210),
    ])
    const timeline = buildRunTimeline(response, 0, 4)
    expect(timeline.activeMs).toBe(12_000)
    expect(timeline.breaks).toHaveLength(1)
    expect(timeline.breaks[0].ms).toBe(7_199_000)
    expect(timeline.breaks[0].left).toBeCloseTo(48.5)
    expect(timeline.breaks[0].width).toBe(3)
    const blocks = timeline.lanes[0].blocks
    expect(blocks.map((block) => block.calls.length)).toEqual([2, 2])
    expect(blocks[0].width).toBeCloseTo(48.5)
    expect(blocks[1].left).toBeCloseTo(51.5)
    expect(blocks[1].left + blocks[1].width).toBeCloseTo(100)
  })

  it('folds only the gaps that are long against the window and at least a minute', () => {
    const spans: [number, number][] = [
      [0, 1_000],
      [500, 2_000],
      [62_000, 63_000],
      [100_000, 101_000],
    ]
    expect(idleGaps(spans, 60_000)).toEqual([[2_000, 62_000]])
    expect(idleGaps([], 60_000)).toEqual([])
  })

  it('pages through a long run, each window on its own axis', () => {
    const calls = Array.from({ length: 250 }, (_, index) =>
      call(index >= 120 && index < 140 ? 'subagent 1' : 'main', 'small', index * 2),
    )
    const agents = [agent('main'), agent('subagent 1', 'main')]
    const response = run(calls, agents)
    const middle = buildRunTimeline(response, 100, 200)
    expect([middle.from, middle.to]).toEqual([100, 200])
    expect(middle.begin).toBe(START + 200_000)
    expect(middle.lanes.map((lane) => lane.agent.label)).toEqual(['main', 'subagent 1'])
    expect(middle.lanes[1].spawnAt).toBeCloseTo((40 / 199) * 100)
    const late = buildRunTimeline(response, 200, 300)
    expect([late.from, late.to]).toEqual([200, 250])
    expect(late.lanes.map((lane) => lane.agent.label)).toEqual(['main'])
    expect(buildRunTimeline(response, 300, 400).lanes).toEqual([])
  })

  it('spans the hops a Looper call made before its record started', () => {
    const looper = call('main', 'large', 60, {
      duration_ms: 0,
      hops: [call('main', 'small', 50), call('main', 'large', 55)],
    })
    const timeline = buildRunTimeline(run([looper]), 0, 1)
    expect(timeline.begin).toBe(START + 50_000)
    expect(timeline.end).toBe(START + 60_000)
    const [block] = timeline.lanes[0].blocks
    expect(block.left).toBe(0)
    expect(block.width).toBeCloseTo(100)
  })
})
