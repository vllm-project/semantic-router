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

export function updateStructuredObjectField<TItem extends object>(
  item: TItem,
  key: Extract<keyof TItem, string>,
  nextValue: unknown,
  preserveEmpty = false,
): TItem {
  const nextItem = { ...item } as TItem
  const record = nextItem as Record<string, unknown>
  if (nextValue === undefined || (nextValue === '' && !preserveEmpty)) {
    delete record[key]
  } else {
    record[key] = nextValue
  }
  return nextItem
}
