export interface TextDiffLine {
  kind: 'context' | 'add' | 'remove'
  text: string
  /** Line number in the side the line belongs to (the new text for context and additions). */
  number: number
}

export interface TextDiff {
  lines: TextDiffLine[]
  added: number
  removed: number
}

/**
 * The changed region of two texts with a little context: the common head and
 * tail are trimmed, and what remains is shown as removed then added. It is the
 * plain-text fallback for the deploy review when the diff editor is offline.
 */
export function changedRegion(before: string, after: string, context = 3): TextDiff {
  const old = before.split('\n')
  const next = after.split('\n')
  let head = 0
  while (head < old.length && head < next.length && old[head] === next[head]) head += 1
  let tail = 0
  while (
    tail < old.length - head &&
    tail < next.length - head &&
    old[old.length - 1 - tail] === next[next.length - 1 - tail]
  )
    tail += 1
  const removed = old.slice(head, old.length - tail)
  const added = next.slice(head, next.length - tail)
  if (removed.length === 0 && added.length === 0) return { lines: [], added: 0, removed: 0 }
  const start = Math.max(0, head - context)
  const lines: TextDiffLine[] = [
    ...next.slice(start, head).map((text, index) => ({
      kind: 'context' as const,
      text,
      number: start + index + 1,
    })),
    ...removed.map((text, index) => ({ kind: 'remove' as const, text, number: head + index + 1 })),
    ...added.map((text, index) => ({ kind: 'add' as const, text, number: head + index + 1 })),
    ...next
      .slice(next.length - tail, Math.min(next.length, next.length - tail + context))
      .map((text, index) => ({
        kind: 'context' as const,
        text,
        number: next.length - tail + index + 1,
      })),
  ]
  return { lines, added: added.length, removed: removed.length }
}
