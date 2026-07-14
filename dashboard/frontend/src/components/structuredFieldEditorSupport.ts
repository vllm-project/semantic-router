export function normalizeStringList(value: unknown): string[] {
  const items = Array.isArray(value) ? value : typeof value === 'string' ? value.split(/[\n,]/) : []
  const seen = new Set<string>()

  return items
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.trim())
    .filter((item) => {
      if (!item || seen.has(item)) return false
      seen.add(item)
      return true
    })
}
