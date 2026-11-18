/** Adds trimmed, non-empty candidates that are not already present; stops at the first one `validate` refuses. */
export function mergeChips(
  value: readonly string[],
  candidates: readonly string[],
  {
    caseSensitive = false,
    validate,
  }: { caseSensitive?: boolean; validate?: (candidate: string) => string | null } = {},
): { next: string[]; error: string | null } {
  const key = (item: string) => (caseSensitive ? item : item.toLowerCase())
  const existing = new Set(value.map(key))
  const next = [...value]
  for (const candidate of candidates.map((item) => item.trim()).filter(Boolean)) {
    if (existing.has(key(candidate))) continue
    const problem = validate?.(candidate) ?? null
    if (problem) return { next: [...value], error: problem }
    existing.add(key(candidate))
    next.push(candidate)
  }
  return { next, error: null }
}
