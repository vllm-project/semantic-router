export interface PageWindow {
  page: number
  pageSize: number
  totalPages: number
  start: number
  end: number
}

export function getPageWindow(
  totalItems: number,
  requestedPage: number,
  requestedPageSize: number,
): PageWindow {
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

const INTERACTIVE_ROW_CONTENT = [
  'a',
  'button',
  'input',
  'select',
  'textarea',
  'label',
  'summary',
  '[contenteditable="true"]',
  '[role="button"]',
  '[role="checkbox"]',
  '[role="link"]',
  '[role="menuitem"]',
  '[role="switch"]',
  '[data-row-click-ignore]',
].join(',')

/**
 * A row click opens the row unless it lands on a control inside the row.
 * Clicks from portalled content (menus, dialogs) bubble through React to the
 * row as well, so a target outside the row element never counts.
 */
export function isInteractiveRowTarget(
  target: EventTarget | null,
  row: Pick<Element, 'contains'>,
): boolean {
  const element = target as Element | null
  if (!element || typeof element.closest !== 'function' || !row.contains(element)) return true
  const control = element.closest(INTERACTIVE_ROW_CONTENT)
  return Boolean(control && row.contains(control))
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
