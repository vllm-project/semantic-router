import React, { useEffect, useMemo, useState } from 'react'
import {
  ArrowDown,
  ArrowUp,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  ChevronsUpDown,
  Eye,
  MoreHorizontal,
  Pencil,
  Trash2,
} from 'lucide-react'

import {
  Button,
  Checkbox,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  cn,
} from '../ui'
import NativeSelect from './NativeSelect'
import {
  getPageWindow,
  isInteractiveRowTarget,
  paginateRows,
  updatePageSelection,
} from './dataTableSupport'

export interface Column<T> {
  key: string
  header: string
  width?: string
  minWidth?: string
  align?: 'left' | 'center' | 'right'
  render?: (row: T) => React.ReactNode
  sortable?: boolean
}

export interface DataTableSelection<T> {
  selectedKeys: ReadonlySet<string>
  onChange: (selectedKeys: Set<string>) => void
  isRowDisabled?: (row: T) => boolean
  label?: string
}

export interface DataTablePagination {
  pageSize?: number
  pageSizeOptions?: number[]
  itemLabel?: string
  resetKey?: string | number
}

export interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  keyExtractor: (row: T) => string
  onView?: (row: T) => void
  onEdit?: (row: T) => void
  onDelete?: (row: T) => void
  expandable?: boolean
  renderExpandedRow?: (row: T) => React.ReactNode
  isRowExpanded?: (row: T) => boolean
  onToggleExpand?: (row: T) => void
  emptyMessage?: string
  className?: string
  stickyActions?: boolean
  readonly?: boolean
  pagination?: DataTablePagination
  selection?: DataTableSelection<T>
}

const alignClass = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
} as const

function getColumnValue<T>(row: T, key: string): unknown {
  if (row !== null && typeof row === 'object') {
    return (row as Record<string, unknown>)[key]
  }

  return undefined
}

function compareColumnValues(aValue: unknown, bValue: unknown): number {
  if (aValue === bValue) return 0
  if (aValue === null || typeof aValue === 'undefined') return -1
  if (bValue === null || typeof bValue === 'undefined') return 1

  if (typeof aValue === 'number' && typeof bValue === 'number') {
    return aValue > bValue ? 1 : -1
  }

  return String(aValue).localeCompare(String(bValue), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function renderColumnValue(value: unknown): React.ReactNode {
  if (value === null || typeof value === 'undefined') return ''
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')
    return value
  if (React.isValidElement(value)) return value
  return String(value)
}

interface RowActionsProps<T> {
  row: T
  rowKey: string
  onView?: (row: T) => void
  onEdit?: (row: T) => void
  onDelete?: (row: T) => void
}

function RowActions<T>({ row, rowKey, onView, onEdit, onDelete }: RowActionsProps<T>) {
  if (onView && !onEdit && !onDelete) {
    return (
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        aria-label={`View ${rowKey}`}
        className="text-subtle-foreground group-hover:text-foreground"
        onClick={() => onView(row)}
      >
        <ChevronRight />
      </Button>
    )
  }

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label={`Actions for ${rowKey}`}
          className="text-subtle-foreground group-hover:text-foreground data-[state=open]:bg-accent data-[state=open]:text-foreground"
        >
          <MoreHorizontal />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40">
        {onView ? (
          <DropdownMenuItem onSelect={() => onView(row)}>
            <Eye />
            View
          </DropdownMenuItem>
        ) : null}
        {onEdit ? (
          <DropdownMenuItem onSelect={() => onEdit(row)}>
            <Pencil />
            Edit
          </DropdownMenuItem>
        ) : null}
        {onDelete ? (
          <>
            {onView || onEdit ? <DropdownMenuSeparator /> : null}
            <DropdownMenuItem variant="destructive" onSelect={() => onDelete(row)}>
              <Trash2 />
              Delete
            </DropdownMenuItem>
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  onView,
  onEdit,
  onDelete,
  expandable = false,
  renderExpandedRow,
  isRowExpanded,
  onToggleExpand,
  emptyMessage = 'No data available',
  className = '',
  stickyActions = true,
  readonly = false,
  pagination,
  selection,
}: DataTableProps<T>) {
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
  const pageSizeOptions = useMemo(() => {
    const requested = pagination?.pageSizeOptions ?? [25, 50, 100]
    return [
      ...new Set(requested.filter((value) => Number.isFinite(value) && value > 0).map(Math.floor)),
    ].sort((left, right) => left - right)
  }, [pagination?.pageSizeOptions])
  const [pageSize, setPageSize] = useState(pagination?.pageSize ?? pageSizeOptions[0] ?? 25)
  const [currentPage, setCurrentPage] = useState(1)
  const paginationEnabled = Boolean(pagination)
  const paginationResetKey = pagination?.resetKey

  const effectiveOnEdit = readonly ? undefined : onEdit
  const effectiveOnDelete = readonly ? undefined : onDelete
  const hasActions = Boolean(onView || effectiveOnEdit || effectiveOnDelete)

  const handleSort = (columnKey: string) => {
    if (sortColumn === columnKey) {
      setSortDirection((direction) => (direction === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortColumn(columnKey)
      setSortDirection('asc')
    }
  }

  const sortedData = useMemo(() => {
    if (!sortColumn) return data

    return [...data].sort((a, b) => {
      const comparison = compareColumnValues(
        getColumnValue(a, sortColumn),
        getColumnValue(b, sortColumn),
      )
      return sortDirection === 'asc' ? comparison : -comparison
    })
  }, [data, sortColumn, sortDirection])

  const pageWindow = getPageWindow(sortedData.length, currentPage, pageSize)
  const visibleData = paginationEnabled ? paginateRows(sortedData, pageWindow) : sortedData

  useEffect(() => {
    if (currentPage !== pageWindow.page) {
      setCurrentPage(pageWindow.page)
    }
  }, [currentPage, pageWindow.page])

  useEffect(() => {
    if (paginationEnabled) {
      setCurrentPage(1)
    }
  }, [paginationEnabled, paginationResetKey, sortColumn, sortDirection])

  const selectablePageKeys = selection
    ? visibleData.filter((row) => !selection.isRowDisabled?.(row)).map(keyExtractor)
    : []
  const selectedPageCount = selection
    ? selectablePageKeys.filter((key) => selection.selectedKeys.has(key)).length
    : 0
  const allPageRowsSelected =
    selectablePageKeys.length > 0 && selectedPageCount === selectablePageKeys.length
  const somePageRowsSelected = selectedPageCount > 0 && !allPageRowsSelected
  const auxiliaryColumnCount = (expandable ? 1 : 0) + (selection ? 1 : 0) + (hasActions ? 1 : 0)
  const stickyCell = stickyActions ? 'sticky right-0 z-[1]' : undefined

  const openRow = (event: React.MouseEvent<HTMLTableRowElement>, row: T) => {
    if (!onView || isInteractiveRowTarget(event.target, event.currentTarget)) return
    if (window.getSelection()?.toString()) return
    onView(row)
  }

  return (
    <div
      className={cn(
        'overflow-hidden rounded-xl border border-border bg-card text-card-foreground shadow-xs',
        className,
      )}
    >
      <Table className="min-w-[640px]">
        <colgroup>
          {selection ? <col className="w-11" /> : null}
          {expandable ? <col className="w-10" /> : null}
          {columns.map((column) => (
            <col key={column.key} style={{ width: column.width }} />
          ))}
          {hasActions ? <col className="w-14" /> : null}
        </colgroup>
        <TableHeader className="bg-surface-subtle/80">
          <TableRow>
            {selection ? (
              <TableHead className="w-11">
                <Checkbox
                  aria-label={`Select all ${selection.label ?? 'rows'} on this page`}
                  checked={
                    allPageRowsSelected ? true : somePageRowsSelected ? 'indeterminate' : false
                  }
                  disabled={selectablePageKeys.length === 0}
                  onCheckedChange={(checked) =>
                    selection.onChange(
                      updatePageSelection(
                        selection.selectedKeys,
                        selectablePageKeys,
                        checked === true,
                      ),
                    )
                  }
                />
              </TableHead>
            ) : null}
            {expandable ? (
              <TableHead className="w-10">
                <span className="sr-only">Expand row</span>
              </TableHead>
            ) : null}
            {columns.map((column) => {
              const activeSort = sortColumn === column.key
              return (
                <TableHead
                  key={column.key}
                  className={alignClass[column.align ?? 'left']}
                  style={{ width: column.width, minWidth: column.minWidth }}
                  aria-sort={
                    activeSort ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined
                  }
                >
                  {column.sortable ? (
                    <button
                      type="button"
                      onClick={() => handleSort(column.key)}
                      className={cn(
                        '-mx-1 inline-flex items-center gap-1 rounded px-1 py-0.5 outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40',
                        activeSort && 'text-foreground',
                        column.align === 'right' && 'flex-row-reverse',
                      )}
                    >
                      {column.header}
                      {activeSort ? (
                        sortDirection === 'asc' ? (
                          <ArrowUp className="size-3" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="size-3" aria-hidden="true" />
                        )
                      ) : (
                        <ChevronsUpDown className="size-3 opacity-40" aria-hidden="true" />
                      )}
                    </button>
                  ) : (
                    column.header
                  )}
                </TableHead>
              )
            })}
            {hasActions ? (
              <TableHead className={cn('w-14 bg-surface-subtle', stickyCell)}>
                <span className="sr-only">Actions</span>
              </TableHead>
            ) : null}
          </TableRow>
        </TableHeader>
        <TableBody>
          {visibleData.length === 0 ? (
            <TableRow className="hover:bg-transparent">
              <TableCell
                colSpan={columns.length + auxiliaryColumnCount}
                className="h-28 whitespace-normal text-center text-[13px] text-muted-foreground"
              >
                {emptyMessage}
              </TableCell>
            </TableRow>
          ) : (
            visibleData.map((row) => {
              const key = keyExtractor(row)
              const expanded = isRowExpanded?.(row) || false
              const selected = selection?.selectedKeys.has(key) ?? false
              const selectionDisabled = selection?.isRowDisabled?.(row) ?? false

              return (
                <React.Fragment key={key}>
                  <TableRow
                    data-state={selected ? 'selected' : undefined}
                    onClick={onView ? (event) => openRow(event, row) : undefined}
                    className={cn('group', onView && 'cursor-pointer', expanded && 'border-b-0')}
                  >
                    {selection ? (
                      <TableCell className="w-11">
                        <Checkbox
                          aria-label={`Select ${selection.label ?? 'row'} ${key}`}
                          checked={selected}
                          disabled={selectionDisabled}
                          onCheckedChange={(checked) =>
                            selection.onChange(
                              updatePageSelection(selection.selectedKeys, [key], checked === true),
                            )
                          }
                        />
                      </TableCell>
                    ) : null}
                    {expandable ? (
                      <TableCell className="w-10 px-2 first:pl-2.5">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => onToggleExpand?.(row)}
                          aria-label={`${expanded ? 'Collapse' : 'Expand'} ${key}`}
                          aria-expanded={expanded}
                        >
                          <ChevronRight
                            className={cn(
                              'transition-transform duration-200 ease-[cubic-bezier(0.16,1,0.3,1)]',
                              expanded && 'rotate-90',
                            )}
                          />
                        </Button>
                      </TableCell>
                    ) : null}
                    {columns.map((column) => (
                      <TableCell
                        key={column.key}
                        className={cn(
                          'whitespace-normal py-2.5',
                          alignClass[column.align ?? 'left'],
                        )}
                        style={{ width: column.width, minWidth: column.minWidth }}
                      >
                        {column.render
                          ? column.render(row)
                          : renderColumnValue(getColumnValue(row, column.key))}
                      </TableCell>
                    ))}
                    {hasActions ? (
                      <TableCell
                        className={cn(
                          'w-14 pr-3 text-right',
                          stickyCell &&
                            cn(
                              stickyCell,
                              'bg-card transition-colors group-hover:bg-surface-subtle group-data-[state=selected]:bg-primary-soft',
                            ),
                        )}
                      >
                        <RowActions
                          row={row}
                          rowKey={key}
                          onView={onView}
                          onEdit={effectiveOnEdit}
                          onDelete={effectiveOnDelete}
                        />
                      </TableCell>
                    ) : null}
                  </TableRow>
                  {expandable && expanded && renderExpandedRow ? (
                    <TableRow className="bg-surface-subtle/60 hover:bg-surface-subtle/60">
                      <TableCell
                        colSpan={columns.length + auxiliaryColumnCount}
                        className="h-auto whitespace-normal p-0 first:pl-0 last:pr-0"
                      >
                        {renderExpandedRow(row)}
                      </TableCell>
                    </TableRow>
                  ) : null}
                </React.Fragment>
              )
            })
          )}
        </TableBody>
      </Table>

      {pagination && sortedData.length > 0 ? (
        <div
          className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-t border-border px-4 py-2 text-xs text-muted-foreground"
          role="group"
          aria-label={`${pagination.itemLabel ?? 'items'} pagination`}
        >
          <span className="tabular">
            {pageWindow.start + 1}–{pageWindow.end} of {sortedData.length}{' '}
            {pagination.itemLabel ?? 'items'}
          </span>
          <div className="flex items-center gap-5">
            <label className="flex items-center gap-2">
              Rows
              <NativeSelect
                size="sm"
                containerClassName="w-[4.5rem]"
                className="h-7 tabular"
                value={pageSize}
                onChange={(event) => {
                  setPageSize(Number(event.target.value))
                  setCurrentPage(1)
                }}
              >
                {pageSizeOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </NativeSelect>
            </label>
            <div className="flex items-center gap-0.5">
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                onClick={() => setCurrentPage(1)}
                disabled={pageWindow.page === 1}
                aria-label="First page"
              >
                <ChevronsLeft />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                disabled={pageWindow.page === 1}
                aria-label="Previous page"
              >
                <ChevronLeft />
              </Button>
              <span className="min-w-[5.5rem] text-center tabular text-foreground/80">
                Page {pageWindow.page} of {pageWindow.totalPages}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                onClick={() => setCurrentPage((page) => Math.min(pageWindow.totalPages, page + 1))}
                disabled={pageWindow.page === pageWindow.totalPages}
                aria-label="Next page"
              >
                <ChevronRight />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                onClick={() => setCurrentPage(pageWindow.totalPages)}
                disabled={pageWindow.page === pageWindow.totalPages}
                aria-label="Last page"
              >
                <ChevronsRight />
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
