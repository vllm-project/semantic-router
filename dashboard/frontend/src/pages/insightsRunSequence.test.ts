import { describe, expect, it } from 'vitest'

import { buildRunSequence, formatRowCost, sideKinds, type RunRow } from './insightsRunSequence'
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
    decision: 'default',
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
  const agent = (label: string, parent?: string, after?: string) => ({
    label,
    parent,
    spawned_after_record_id: after,
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
      ended_at: new Date(START + 300_000).toISOString(),
      duration_ms: 300_000,
      calls: calls.length,
      // The response lists the nested subagent before its sibling; columns follow the tree.
      agents: [
        agent('main'),
        agent('subagent 2', 'subagent 1'),
        agent('subagent 1', 'main', 'main-1'),
      ],
      models: [],
      outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
      cost,
    },
    calls,
  }
}

// Each row as [kind, where, how many calls], for reading a layout at a glance.
function shape(rows: RunRow[]) {
  return rows.map((row) => {
    switch (row.kind) {
      case 'turns':
        return ['turns', `${row.agent} ${row.model} ${row.decision}`, row.calls.length]
      case 'side':
        return ['side', sideKinds(row.calls), row.calls.length]
      case 'spawn':
      case 'return':
        return [row.kind, `${row.from}->${row.to}`, 0]
      default:
        return ['idle', `${(row.end - row.begin) / 1000}s`, 0]
    }
  })
}

describe('buildRunSequence', () => {
  it("collapses an agent's consecutive turns on one model and decision", () => {
    const sequence = buildRunSequence(
      run([
        call('main', 'fable', 0),
        call('main', 'fable', 2, { markers: ['tool_failure'] }),
        call('main', 'fable', 4),
        call('main', 'fable', 6, { decision: 'review' }),
        call('main', 'sonnet', 8, { markers: ['switch'] }),
        call('subagent 1', 'haiku', 10),
        call('main', 'sonnet', 12),
      ]),
    )
    expect(sequence.columns.map((column) => [column.agent.label, column.depth])).toEqual([
      ['main', 0],
      ['subagent 1', 1],
      ['subagent 2', 2],
    ])
    expect(shape(sequence.rows)).toEqual([
      ['turns', 'main fable default', 3],
      ['turns', 'main fable review', 1],
      ['turns', 'main sonnet default', 1],
      ['spawn', '0->1', 0],
      ['turns', 'subagent 1 haiku default', 1],
      ['turns', 'main sonnet default', 1],
    ])
    const [first] = sequence.rows
    expect(first.kind === 'turns' && first.markers).toEqual(['tool_failure'])
    // The row spans its first call's start to its last call's end.
    expect(first.end - first.begin).toBe(5000)
  })

  it('keeps side requests off the lifelines without splitting the turns around them', () => {
    const sequence = buildRunSequence(
      run([
        call('main', 'fable', 0),
        call('main', 'haiku', 1, { request_kind: 'session_title' }),
        call('main', 'haiku', 1.5, { request_kind: 'auxiliary', markers: ['failure'] }),
        call('main', 'fable', 2),
        call('main', 'haiku', 3, { request_kind: 'compaction' }),
        call('main', 'fable', 4, { decision: 'other' }),
        call('main', 'haiku', 5, { request_kind: 'auxiliary' }),
      ]),
    )
    expect(shape(sequence.rows)).toEqual([
      ['turns', 'main fable default', 2],
      ['side', 'session_title, auxiliary, compaction', 3],
      ['turns', 'main fable other', 1],
      ['side', 'auxiliary', 1],
    ])
    expect(sequence.rows[1].kind === 'side' && sequence.rows[1].markers).toEqual(['failure'])
  })

  it('draws an arrow from the agent that spawned a subagent to its first call', () => {
    const sequence = buildRunSequence(
      run([
        call('main', 'fable', 0, { record_id: 'main-1' }),
        call('subagent 1', 'haiku', 2),
        call('subagent 2', 'haiku', 3),
        call('subagent 1', 'haiku', 4),
      ]),
    )
    const spawns = sequence.rows.filter((row) => row.kind === 'spawn')
    expect(spawns).toEqual([
      expect.objectContaining({
        from: 0,
        to: 1,
        parent: 'main',
        agent: 'subagent 1',
        call: 'main-1',
      }),
      expect.objectContaining({ from: 1, to: 2, parent: 'subagent 1', agent: 'subagent 2' }),
    ])
    // Only an agent's first call is spawned.
    expect(shape(sequence.rows).filter(([kind]) => kind === 'spawn')).toHaveLength(2)
  })

  it('returns each subagent to the call that took its result, also in parallel', () => {
    const response = run([
      call('main', 'fable', 0, { record_id: 'main-1' }),
      call('subagent 1', 'haiku', 2),
      call('subagent 2', 'haiku', 3),
      call('subagent 1', 'haiku', 4),
      call('main', 'fable', 5, { record_id: 'main-2' }),
      call('main', 'fable', 7, { record_id: 'main-3' }),
    ])
    // Two parallel subagents of main: the second started before the first
    // returned, both returned to main's next call, and the second failed.
    const [main, second, first] = response.run.agents
    response.run.agents = [
      main,
      {
        ...first,
        task: 'Explore · Find callers',
        returned_to_record_id: 'main-2',
        return_status: 'succeeded',
      },
      // no task kept: the arrow reads the kind its harness sent
      {
        ...second,
        parent: 'main',
        kind: 'collab_spawn',
        returned_to_record_id: 'main-2',
        return_status: 'failed',
      },
    ]
    const sequence = buildRunSequence(response)
    expect(shape(sequence.rows)).toEqual([
      ['turns', 'main fable default', 1],
      ['spawn', '0->1', 0],
      ['turns', 'subagent 1 haiku default', 1],
      ['spawn', '0->2', 0],
      ['turns', 'subagent 2 haiku default', 1],
      ['turns', 'subagent 1 haiku default', 1],
      ['return', '1->0', 0],
      ['return', '2->0', 0],
      // The call that took the results starts a row of its own.
      ['turns', 'main fable default', 2],
    ])
    const arrows = sequence.rows.filter((row) => row.kind === 'spawn' || row.kind === 'return')
    expect(arrows.map((row) => row.kind !== 'idle' && 'label' in row && row.label)).toEqual([
      'Explore · Find callers',
      'collab_spawn',
      'returned · 2 calls · no price',
      'failed · 1 call · no price',
    ])
  })

  it('marks a gap with no call in flight over 30 seconds as idle, and splits the turns there', () => {
    const sequence = buildRunSequence(
      run([
        call('main', 'fable', 0),
        call('main', 'fable', 30),
        call('main', 'haiku', 40, { request_kind: 'auxiliary' }),
        call('main', 'fable', 160),
      ]),
    )
    expect(shape(sequence.rows)).toEqual([
      ['turns', 'main fable default', 2],
      ['side', 'auxiliary', 1],
      ['idle', '119s', 0],
      ['turns', 'main fable default', 1],
    ])
  })

  it('places a Looper call where its hops began', () => {
    const looper = call('main', 'large', 60, {
      duration_ms: 0,
      hops: [call('main', 'small', 50), call('main', 'large', 55)],
    })
    const [row] = buildRunSequence(run([looper])).rows
    expect(row.begin).toBe(START + 50_000)
    expect(row.end).toBe(START + 60_000)
  })

  it("prices a row and says when the price is the catalog's estimate", () => {
    expect(
      formatRowCost([
        call('main', 'fable', 0, { cost: 0.01, currency: 'USD' }),
        call('main', 'fable', 1, {
          cost: 0.02,
          currency: 'USD',
          pricing_source: 'catalog_list_price',
        }),
        call('main', 'fable', 2, { lifecycle_state: 'failed', markers: ['block'] }),
      ]),
    ).toBe('$0.03 list price')
    expect(formatRowCost([call('main', 'fable', 0, { cost: 0.01, currency: 'USD' })])).toBe('$0.01')
    expect(formatRowCost([call('main', 'fable', 0)])).toBe('')
  })
})
