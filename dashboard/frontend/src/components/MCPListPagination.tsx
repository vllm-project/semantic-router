import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'

import { Button } from '../ui'
import { getMCPPageCount, getMCPVisibleRange } from './mcpConfigPanelUtils'

interface MCPListPaginationProps {
  itemCount: number
  itemLabel: string
  page: number
  pageSize: number
  onPageChange: (page: number) => void
}

export function MCPListPagination({
  itemCount,
  itemLabel,
  page,
  pageSize,
  onPageChange,
}: MCPListPaginationProps) {
  const pageCount = getMCPPageCount(itemCount, pageSize)
  const safePage = Math.min(Math.max(1, page), pageCount)
  const range = getMCPVisibleRange(itemCount, safePage, pageSize)

  return (
    <nav
      className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground"
      aria-label={`${itemLabel} pagination`}
    >
      <span aria-live="polite">
        <span className="tabular">
          {range.start}–{range.end}
        </span>{' '}
        of <span className="tabular">{itemCount}</span> {itemLabel}
        {pageCount > 1 ? (
          <span className="ml-2 text-subtle-foreground">· {pageSize} per page</span>
        ) : null}
      </span>
      {pageCount > 1 ? (
        <span className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(1)}
            disabled={safePage === 1}
            aria-label={`First ${itemLabel} page`}
          >
            <ChevronsLeft />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(safePage - 1)}
            disabled={safePage === 1}
            aria-label={`Previous ${itemLabel} page`}
          >
            <ChevronLeft />
          </Button>
          <span className="px-1 tabular">
            Page {safePage} of {pageCount}
          </span>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(safePage + 1)}
            disabled={safePage === pageCount}
            aria-label={`Next ${itemLabel} page`}
          >
            <ChevronRight />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(pageCount)}
            disabled={safePage === pageCount}
            aria-label={`Last ${itemLabel} page`}
          >
            <ChevronsRight />
          </Button>
        </span>
      ) : null}
    </nav>
  )
}
