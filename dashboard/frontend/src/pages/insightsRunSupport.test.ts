import { describe, expect, it } from 'vitest'

import {
  callSpan,
  countVerdicts,
  filterRunCalls,
  formatRunCost,
  formatRunDuration,
  getRunPath,
  oneLine,
  orderRunAgents,
  outcomeWord,
  runAgentTreeText,
  runHeadline,
  runTitle,
  verdictLabel,
} from './insightsRunSupport'
import { compactID } from './insightsReviewSupport'
import type { RunAgent, RunCall, RunSummary } from './insightsRunTypes'

const START = Date.parse('2026-11-16T09:00:00Z')

function call(agent: string, second: number, fields: Partial<RunCall> = {}): RunCall {
  return {
    record_id: `${agent}-${second}`,
    agent,
    model: 'small',
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

const agent = (label: string, parent?: string): RunAgent => ({
  label,
  parent,
  calls: 1,
  first_call_at: '',
  last_call_at: '',
  cost,
})

const summary: RunSummary = {
  session_id: 'run-1',
  started_at: new Date(START).toISOString(),
  ended_at: new Date(START + 100_000).toISOString(),
  duration_ms: 100_000,
  calls: 3,
  agents: [agent('main'), agent('subagent 2', 'subagent 1'), agent('subagent 1', 'main')],
  models: [],
  outcome: { status: 'answered', tool_results: 0, tool_failures: 0 },
  cost,
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
  })

  it('leaves a currency the run spent nothing in out of a mixed total', () => {
    const mixed = (usd: number, units: number) =>
      formatRunCost({
        ...cost,
        by_currency: [
          { currency: 'USD', total: usd, router_call_spend: 0 },
          { currency: 'XXX', total: units, router_call_spend: 0 },
        ],
      })
    expect(mixed(0, 0.2882)).toBe('0.2882')
    expect(mixed(0.5, 0.2882)).toBe('$0.50 + 0.2882')
    expect(mixed(0, 0)).toBe('$0.0000')
  })

  it('names a run by its session name, else its task, else its session', () => {
    expect(runTitle({ ...summary, session_name: 'Average helper', prompt: 'fix average' })).toBe(
      'Average helper',
    )
    expect(runTitle({ ...summary, prompt: ' fix average ' })).toBe('fix average')
    expect(runTitle({ ...summary, session_id: '0123456789abcdef0123' })).toBe(
      `Session ${compactID('0123456789abcdef0123')}`,
    )
  })

  it('keeps a long task to one line for headers, rows and chips', () => {
    const listing = `<listing_a> Title: AMD Ryzen 9 9950X3D\n\nBrand: AMD   Model: (unknown) ${'Specifications '.repeat(40)}`
    const headline = runHeadline({ ...summary, prompt: listing })
    expect(headline.length).toBeLessThanOrEqual(120)
    expect(headline.endsWith('…')).toBe(true)
    expect(headline).not.toContain('\n')
    expect(headline.startsWith('<listing_a> Title: AMD Ryzen 9 9950X3D Brand: AMD Model:')).toBe(
      true,
    )
    expect(oneLine('  short   title ')).toBe('short title')
  })

  it('reads durations from milliseconds to hours', () => {
    expect(formatRunDuration(420)).toBe('420ms')
    expect(formatRunDuration(4_200)).toBe('4.2s')
    expect(formatRunDuration(125_000)).toBe('2m 5s')
    expect(formatRunDuration(7_260_000)).toBe('2h 1m')
  })

  it('reads the agent tree at a glance and links a run with its step', () => {
    expect(runAgentTreeText(summary)).toBe('main, 2 subagents')
    expect(getRunPath('tenant:a/b', 'rec-1')).toBe(
      '/observe/runs?session=tenant%3Aa%2Fb&step=rec-1',
    )
  })
})

describe('run agents and calls', () => {
  it('orders the agents as a tree, each after the agent that spawned it', () => {
    expect(
      orderRunAgents(summary.agents).map(({ agent: entry, depth }) => [entry.label, depth]),
    ).toEqual([
      ['main', 0],
      ['subagent 1', 1],
      ['subagent 2', 2],
    ])
  })

  it('keeps the calls of the agents and events the table follows', () => {
    const calls = [
      call('main', 0),
      call('subagent 1', 1, { markers: ['failure'] }),
      call('main', 2, { markers: ['switch', 'tool_failure'] }),
    ]
    const ids = (agents: string[], markers: string[]) =>
      filterRunCalls(calls, { agents, markers }).map((entry) => entry.record_id)
    expect(ids([], [])).toEqual(['main-0', 'subagent 1-1', 'main-2'])
    expect(ids(['main'], [])).toEqual(['main-0', 'main-2'])
    expect(ids([], ['failure', 'tool_failure'])).toEqual(['subagent 1-1', 'main-2'])
    expect(ids(['main'], ['failure'])).toEqual([])
  })

  it('spans the hops a Looper call made before its record started', () => {
    const looper = call('main', 60, {
      duration_ms: 0,
      hops: [call('main', 50), call('main', 55)],
    })
    expect(callSpan(looper)).toEqual([START + 50_000, START + 60_000])
  })
})

describe('reported outcomes', () => {
  it('counts verdicts in the order the outcome API lists them, unknown ones last', () => {
    const reported = ['failed', 'good_fit', 'retry_later', 'good_fit', 'underpowered'].map(
      (verdict) => ({ outcome: { verdict } }),
    )
    expect(countVerdicts(reported)).toEqual([
      { verdict: 'good_fit', count: 2 },
      { verdict: 'underpowered', count: 1 },
      { verdict: 'failed', count: 1 },
      { verdict: 'retry_later', count: 1 },
    ])
    expect(verdictLabel('overprovisioned')).toBe('Overprovisioned')
    expect(verdictLabel('retry_later')).toBe('retry later')
    expect(outcomeWord('eval')).toBe('Evaluation')
    expect(outcomeWord('ci_bot')).toBe('ci bot')
  })
})
