import { describe, expect, it } from 'vitest'

import { callEvents, eventCountText, recordEvents, runEventCounts } from './insightsAgentEvents'
import { harnessInfo, PRIMARY_HARNESSES } from './insightsHarness'
import type { RunAgent, RunCall, RunSummary } from './insightsRunTypes'
import {
  agentTypeText,
  buildAgentTree,
  harnessSignalRows,
  isDerivedSession,
} from './insightsSessionTree'

const cost = {
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
}
const agent = (label: string, fields: Partial<RunAgent> = {}): RunAgent => ({
  label,
  calls: 1,
  first_call_at: '2026-11-19T10:00:00Z',
  last_call_at: '2026-11-19T10:00:00Z',
  cost,
  ...fields,
})
const call = (
  record_id: string,
  agentLabel: string,
  timestamp: string,
  fields: Partial<RunCall> = {},
): RunCall => ({
  record_id,
  agent: agentLabel,
  timestamp,
  duration_ms: 100,
  lifecycle_state: 'completed',
  ...fields,
})
const run = (fields: Partial<RunSummary>): RunSummary => ({
  session_id: 'key_1:session-1',
  started_at: '2026-11-19T10:00:00Z',
  ended_at: '2026-11-19T10:01:00Z',
  duration_ms: 60000,
  calls: 1,
  agents: [agent('main')],
  models: [],
  outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
  cost,
  ...fields,
})

describe('session tree', () => {
  it('nests each subagent under the agent that started it and keeps every agent’s turns in time order', () => {
    const summary = run({
      agents: [
        agent('subagent 2', { parent: 'subagent 1', type: 'Explore' }),
        agent('main'),
        agent('subagent 1', { parent: 'main', kind: 'collab_spawn' }),
      ],
    })
    const tree = buildAgentTree({
      run: summary,
      calls: [
        call('b', 'main', '2026-11-19T10:00:20Z'),
        call('a', 'main', '2026-11-19T10:00:10Z'),
        call('c', 'subagent 1', '2026-11-19T10:00:15Z'),
        call('d', 'subagent 2', '2026-11-19T10:00:16Z'),
      ],
    })
    expect(tree).toHaveLength(1)
    expect(tree[0].agent.label).toBe('main')
    expect(tree[0].calls.map((entry) => entry.record_id)).toEqual(['a', 'b'])
    expect(tree[0].children[0].agent.label).toBe('subagent 1')
    expect(tree[0].children[0].children[0].agent.label).toBe('subagent 2')
    expect(tree[0].children[0].children[0].depth).toBe(2)
    expect(agentTypeText(tree[0].children[0].agent)).toBe('collab_spawn')
    expect(agentTypeText(tree[0].children[0].children[0].agent)).toBe('Explore')
  })

  it('keeps an agent whose parent the run does not hold at the top', () => {
    const tree = buildAgentTree({
      run: run({ agents: [agent('main'), agent('subagent 1', { parent: 'gone' })] }),
      calls: [],
    })
    expect(tree.map((node) => node.agent.label)).toEqual(['main', 'subagent 1'])
  })

  it('knows a session the router keyed from how the conversation starts', () => {
    expect(isDerivedSession('key_3fa9:cc-start-e939a0c78d77f163')).toBe(true)
    expect(isDerivedSession('cc-start-41e06260fa700b3b')).toBe(true)
    expect(isDerivedSession('key_3fa9:cursor-start-41e06260fa700b3b')).toBe(true)
    expect(isDerivedSession('openclaw-start-41e06260fa700b3b')).toBe(true)
    expect(isDerivedSession('key_3fa9:openclaw-3e87f001-0e05-4131-be78-0eb121f53ba4')).toBe(false)
    expect(isDerivedSession('key_3fa9:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483')).toBe(false)
  })
})

describe('agent events', () => {
  it('lists a call’s events in the page’s order and drops the guard flag when the guard refused', () => {
    expect(callEvents({ markers: ['tool_loop', 'image', 'block', 'guard', 'unknown'] })).toEqual([
      'tool_loop',
      'image',
      'block',
    ])
  })

  it('counts a run’s events without zero counts', () => {
    expect(runEventCounts({ events: { compaction: 2, image: 0, subagent_start: 1 } })).toEqual([
      ['subagent_start', 1],
      ['compaction', 2],
    ])
    expect(eventCountText('compaction', 2)).toBe('2 compactions')
    expect(eventCountText('block', 1)).toBe('1 guard refusal')
  })

  it('reads a record’s own events from its class and its images', () => {
    expect(recordEvents({ agent: { request_class: 'compaction' }, image_count: 1 })).toEqual([
      'compaction',
      'image',
    ])
    expect(recordEvents({ request_kind: 'session_title' })).toEqual(['housekeeping'])
    expect(recordEvents({ agent: { harness: 'codex' }, image_count: 0 })).toEqual([])
  })

  it('reads a guard refusal, a guard flag and a failure from a record as the Router marks a call', () => {
    // the guard's fast response refuses with a 200 and no model
    expect(
      recordEvents({
        jailbreak_detected: true,
        selection_method: 'fast_response',
        response_status: 200,
      }),
    ).toEqual(['block'])
    expect(recordEvents({ pii_blocked: true, response_status: 403 })).toEqual(['block'])
    expect(
      recordEvents({ hallucination_detected: true, selected_model: 'qwen', response_status: 200 }),
    ).toEqual(['guard'])
    expect(
      recordEvents({ lifecycle_state: 'failed', selected_model: 'qwen', response_status: 502 }),
    ).toEqual(['failure'])
    expect(
      recordEvents({
        agent: { request_class: 'compaction' },
        lifecycle_state: 'completed',
        response_status: 200,
        selected_model: 'qwen',
      }),
    ).toEqual(['compaction'])
  })
})

describe('signals by harness', () => {
  it('says which signals each harness gave, which it never sends and which the router derived', () => {
    const rows = harnessSignalRows([
      run({
        harness: 'claude_code',
        agents: [agent('main'), agent('subagent 1', { parent: 'main', type: 'Explore' })],
        events: { compaction: 1, image: 1, tool_loop: 3 },
      }),
      run({
        harness: 'cursor',
        session_id: 'key_1:cc-start-e939a0c78d77f163',
        events: { tool_loop: 1 },
      }),
      run({ harness: undefined, session_id: 'cc-start-0123456789abcdef' }),
    ])
    expect(rows.map((row) => row.harness)).toEqual(['claude_code', 'cursor', ''])
    const [claude, cursor] = rows
    expect(claude.cells.session).toEqual({ state: 'seen', count: 1 })
    expect(claude.cells.subagent_type).toEqual({ state: 'seen', count: 1 })
    expect(claude.cells.compaction).toEqual({ state: 'seen', count: 1 })
    expect(claude.cells.housekeeping).toEqual({ state: 'unseen' })
    expect(cursor.cells.session).toEqual({ state: 'derived', count: 1 })
    expect(cursor.cells.subagents).toEqual({ state: 'never' })
    expect(Object.keys(cursor.cells)).not.toContain('tool_loops')
  })

  it('names Cursor and OpenClaw among the identified harnesses', () => {
    expect(PRIMARY_HARNESSES).toEqual([
      'claude_code',
      'codex',
      'opencode',
      'cursor',
      'openclaw',
      'deepseek_harness',
    ])
    expect(harnessInfo('openclaw')).toMatchObject({ label: 'OpenClaw', known: true })
    expect(harnessInfo('cursor')).toMatchObject({ label: 'Cursor', known: true })
  })
})
