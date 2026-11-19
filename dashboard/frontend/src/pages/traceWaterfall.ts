import { parentSpanID, type Trace, type TraceSpan } from './jaegerTraces'

export interface WaterfallRow {
  span: TraceSpan
  depth: number
  /** Where the span starts and how long it runs, as fractions of the trace. */
  offset: number
  width: number
  childCount: number
  /** Spans below this one, at any depth. */
  descendants: number
  /** Span IDs from the root down to the parent. */
  ancestors: string[]
}

export interface Waterfall {
  rows: WaterfallRow[]
  /** Microseconds since the epoch. */
  start: number
  /** Microseconds, at least one. */
  duration: number
}

const byStart = (left: TraceSpan, right: TraceSpan) => left.startTime - right.startTime

/**
 * Lays a trace out as a tree in depth-first order, children by start time,
 * with each span's timing relative to the whole trace. Spans whose parent the
 * trace lacks become roots, and spans in a reference cycle are listed at the
 * top level instead of being lost.
 */
export function layoutWaterfall(trace: Trace): Waterfall {
  const spanIDs = new Set(trace.spans.map((span) => span.spanID))
  const children = new Map<string, TraceSpan[]>()
  const roots: TraceSpan[] = []
  let start = Infinity
  let end = -Infinity
  for (const span of trace.spans) {
    start = Math.min(start, span.startTime)
    end = Math.max(end, span.startTime + span.duration)
    const parent = parentSpanID(span, spanIDs)
    if (parent === null) roots.push(span)
    else if (children.has(parent)) children.get(parent)!.push(span)
    else children.set(parent, [span])
  }
  if (!trace.spans.length) return { rows: [], start: 0, duration: 1 }
  const duration = Math.max(1, end - start)
  const rows: WaterfallRow[] = []
  const placed = new Set<TraceSpan>()

  const place = (top: TraceSpan) => {
    const stack: { span: TraceSpan; depth: number; ancestors: string[] }[] = [
      { span: top, depth: 0, ancestors: [] },
    ]
    while (stack.length) {
      const { span, depth, ancestors } = stack.pop()!
      if (placed.has(span)) continue
      placed.add(span)
      const kids = (children.get(span.spanID) ?? []).filter((kid) => !placed.has(kid)).sort(byStart)
      rows.push({
        span,
        depth,
        offset: Math.min(1, Math.max(0, (span.startTime - start) / duration)),
        width: Math.min(1, span.duration / duration),
        childCount: kids.length,
        descendants: 0,
        ancestors,
      })
      for (let index = kids.length - 1; index >= 0; index -= 1) {
        stack.push({ span: kids[index], depth: depth + 1, ancestors: [...ancestors, span.spanID] })
      }
    }
  }

  for (const root of [...roots].sort(byStart)) place(root)
  for (const span of [...trace.spans].sort(byStart)) place(span)

  // In depth-first order a span's descendants are the rows after it that sit deeper.
  const open: number[] = []
  rows.forEach((row, index) => {
    while (open.length && rows[open[open.length - 1]].depth >= row.depth) open.pop()
    for (const ancestor of open) rows[ancestor].descendants += 1
    open.push(index)
  })
  return { rows, start, duration }
}

/** The rows left after hiding every descendant of a collapsed span. */
export function visibleRows(
  rows: readonly WaterfallRow[],
  collapsed: ReadonlySet<string>,
): WaterfallRow[] {
  if (!collapsed.size) return [...rows]
  return rows.filter((row) => !row.ancestors.some((id) => collapsed.has(id)))
}

/** Evenly spaced points of the trace's timeline, as fractions with their offsets in microseconds. */
export function timelineTicks(duration: number, count = 4): { fraction: number; micros: number }[] {
  return Array.from({ length: count + 1 }, (_, index) => ({
    fraction: index / count,
    micros: (duration * index) / count,
  }))
}
