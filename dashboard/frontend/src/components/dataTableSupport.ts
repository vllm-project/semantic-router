export interface PageWindow {
  page: number
  pageSize: number
  totalPages: number
  start: number
  end: number
}

export function getPageWindow(totalItems: number, requestedPage: number, requestedPageSize: number): PageWindow {
  const total = Math.max(0, Math.floor(totalItems))
  const pageSize = Math.max(1, Math.floor(requestedPageSize))
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const page = Math.min(totalPages, Math.max(1, Math.floor(requestedPage)))
  const start = (page - 1) * pageSize

  return {
    page,
    pageSize,
    totalPages,
    start,
    end: Math.min(total, start + pageSize),
  }
}

export function paginateRows<T>(rows: T[], window: PageWindow): T[] {
  return rows.slice(window.start, window.end)
}

export function updatePageSelection(
  selectedKeys: ReadonlySet<string>,
  pageKeys: string[],
  shouldSelect: boolean,
): Set<string> {
  const next = new Set(selectedKeys)
  for (const key of pageKeys) {
    if (shouldSelect) {
      next.add(key)
    } else {
      next.delete(key)
    }
  }
  return next
}
