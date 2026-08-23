import type { AccessListParams } from '../utils/inferenceAccessApi'

export const ACCESS_PICKER_PAGE_SIZE = 20
export const ACCESS_PICKER_HYDRATION_CONCURRENCY = 6

export function accessPickerRequest(search: string, cursor?: string): AccessListParams {
  return {
    q: search.trim() || undefined,
    cursor,
    limit: ACCESS_PICKER_PAGE_SIZE,
    status: 'active',
  }
}

export function mergeAccessPickerPage<T>(
  current: T[],
  incoming: T[],
  append: boolean,
  id: (item: T) => string,
): T[] {
  const seen = new Set<string>()
  return (append ? [...current, ...incoming] : incoming).filter((item) => {
    const value = id(item)
    if (seen.has(value)) return false
    seen.add(value)
    return true
  })
}

export function missingSelectedPickerIds<T>(
  selectedIds: string[],
  visibleItems: T[],
  hydrated: Record<string, T>,
  failed: Record<string, true>,
  id: (item: T) => string,
): string[] {
  const visible = new Set(visibleItems.map(id))
  return selectedIds.filter(
    (selectedId) =>
      selectedId && !visible.has(selectedId) && !hydrated[selectedId] && !failed[selectedId],
  )
}
