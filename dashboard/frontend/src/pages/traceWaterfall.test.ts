import { describe, expect, it } from 'vitest'

import type { TraceSpan } from './jaegerTraces'
import { layoutWaterfall, timelineTicks, visibleRows } from './traceWaterfall'

function testSpan(
  spanID: string,
  startTime: number,
  duration: number,
  parent?: string,
  overrides: Partial<TraceSpan> = {},
): TraceSpan {
  return {
    spanID,
    references: parent ? [{ refType: 'CHILD_OF', spanID: parent }] : [],
    operationName: `${spanID} operation`,
    serviceName: 'vllm-sr',
    startTime,
    duration,
    tags: [],
    logs: [],
    error: false,
    kind: null,
    ...overrides,
  }
}

const trace = {
  traceID: 'f00d',
  spans: [
    testSpan('a', 100, 300, 'root'),
    testSpan('root', 0, 1_000),
    testSpan('cycle-1', 600, 10, 'cycle-2'),
    testSpan('a1', 150, 100, 'a'),
    testSpan('orphan', 500, 100, 'missing'),
    testSpan('b', 50, 200, 'root'),
    testSpan('cycle-2', 610, 10, 'cycle-1'),
  ],
}

describe('trace waterfall layout', () => {
  it('orders spans depth first with children by start time', () => {
    const { rows, start, duration } = layoutWaterfall(trace)
    expect(start).toBe(0)
    expect(duration).toBe(1_000)
    expect(rows.map((row) => [row.span.spanID, row.depth])).toEqual([
      ['root', 0],
      ['b', 1],
      ['a', 1],
      ['a1', 2],
      ['orphan', 0],
      ['cycle-1', 0],
      ['cycle-2', 1],
    ])
  })

  it('positions each span on the trace timeline', () => {
    const { rows } = layoutWaterfall(trace)
    const timing = Object.fromEntries(rows.map((row) => [row.span.spanID, [row.offset, row.width]]))
    expect(timing.root).toEqual([0, 1])
    expect(timing.b).toEqual([0.05, 0.2])
    expect(timing.a1).toEqual([0.15, 0.1])
    expect(timing.orphan).toEqual([0.5, 0.1])
  })

  it('counts children and descendants and records ancestors for collapsing', () => {
    const { rows } = layoutWaterfall(trace)
    const byID = Object.fromEntries(rows.map((row) => [row.span.spanID, row]))
    expect(byID.root).toMatchObject({ childCount: 2, descendants: 3, ancestors: [] })
    expect(byID.a).toMatchObject({ childCount: 1, descendants: 1, ancestors: ['root'] })
    expect(byID.a1).toMatchObject({ childCount: 0, descendants: 0, ancestors: ['root', 'a'] })
    // A reference cycle has no root; its spans are still listed once.
    expect(byID['cycle-1']).toMatchObject({ childCount: 1, descendants: 1, ancestors: [] })
    expect(byID['cycle-2']).toMatchObject({ childCount: 0, ancestors: ['cycle-1'] })
  })

  it('hides the descendants of collapsed spans', () => {
    const { rows } = layoutWaterfall(trace)
    const ids = (collapsed: string[]) =>
      visibleRows(rows, new Set(collapsed)).map((row) => row.span.spanID)
    expect(ids([])).toHaveLength(7)
    expect(ids(['a'])).toEqual(['root', 'b', 'a', 'orphan', 'cycle-1', 'cycle-2'])
    expect(ids(['root', 'cycle-1'])).toEqual(['root', 'orphan', 'cycle-1'])
  })

  it('keeps an instant trace and an empty one drawable', () => {
    const instant = layoutWaterfall({ traceID: 'x', spans: [testSpan('only', 5, 0)] })
    expect(instant.duration).toBe(1)
    expect(instant.rows[0]).toMatchObject({ offset: 0, width: 0 })
    expect(layoutWaterfall({ traceID: 'y', spans: [] })).toEqual({
      rows: [],
      start: 0,
      duration: 1,
    })
  })

  it('marks the timeline in quarters', () => {
    expect(timelineTicks(1_000)).toEqual([
      { fraction: 0, micros: 0 },
      { fraction: 0.25, micros: 250 },
      { fraction: 0.5, micros: 500 },
      { fraction: 0.75, micros: 750 },
      { fraction: 1, micros: 1_000 },
    ])
  })
})
