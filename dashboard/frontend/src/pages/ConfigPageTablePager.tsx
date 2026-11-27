import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'

import NativeSelect from '../components/NativeSelect'
import { Button } from '../ui'

interface ConfigPageTablePagerProps {
  /** Plural noun for the rows, such as `models`; it also names the control group. */
  itemLabel: string
  page: number
  totalPages: number
  total: number
  start: number
  end: number
  onPageChange: (page: number) => void
  pageSize?: number
  pageSizeOptions?: number[]
  onPageSizeChange?: (pageSize: number) => void
  disabled?: boolean
}

/** Pagination under a table: the visible range, an optional page size and page steps. */
export default function ConfigPageTablePager({
  itemLabel,
  page,
  totalPages,
  total,
  start,
  end,
  onPageChange,
  pageSize,
  pageSizeOptions,
  onPageSizeChange,
  disabled = false,
}: ConfigPageTablePagerProps) {
  if (total === 0) return null
  return (
    <div
      role="group"
      aria-label={`${itemLabel} pagination`}
      className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-1 text-xs text-muted-foreground"
    >
      <span className="tabular">
        {start + 1}–{end} of {total.toLocaleString()} {itemLabel}
      </span>
      <div className="flex items-center gap-5">
        {pageSize && pageSizeOptions && onPageSizeChange ? (
          <label className="flex items-center gap-2">
            Rows
            <NativeSelect
              size="sm"
              containerClassName="w-[4.75rem]"
              className="h-7 tabular"
              value={pageSize}
              disabled={disabled}
              onChange={(event) => onPageSizeChange(Number(event.target.value))}
            >
              {pageSizeOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </NativeSelect>
          </label>
        ) : null}
        <div className="flex items-center gap-0.5">
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(1)}
            disabled={disabled || page <= 1}
            aria-label="First page"
          >
            <ChevronsLeft />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(page - 1)}
            disabled={disabled || page <= 1}
            aria-label="Previous page"
          >
            <ChevronLeft />
          </Button>
          <span className="min-w-[5.5rem] text-center tabular text-foreground/80">
            Page {page} of {totalPages}
          </span>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(page + 1)}
            disabled={disabled || page >= totalPages}
            aria-label="Next page"
          >
            <ChevronRight />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={() => onPageChange(totalPages)}
            disabled={disabled || page >= totalPages}
            aria-label="Last page"
          >
            <ChevronsRight />
          </Button>
        </div>
      </div>
    </div>
  )
}
