import React, { useEffect, useMemo, useRef, useState } from 'react'

import styles from './DataTable.module.css'
import { getPageWindow, paginateRows, updatePageSelection } from './dataTableSupport'

export interface Column<T> {
  key: string
  header: string
  width?: string
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
  readonly?: boolean
  pagination?: DataTablePagination
  selection?: DataTableSelection<T>
}

interface SelectionCheckboxProps {
  checked: boolean
  disabled?: boolean
  indeterminate?: boolean
  label: string
  onChange: (checked: boolean) => void
}

function SelectionCheckbox({
  checked,
  disabled = false,
  indeterminate = false,
  label,
  onChange,
}: SelectionCheckboxProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.indeterminate = indeterminate
    }
  }, [indeterminate])

  return (
    <input
      ref={inputRef}
      type="checkbox"
      className={styles.selectionCheckbox}
      aria-label={label}
      checked={checked}
      disabled={disabled}
      onChange={(event) => onChange(event.target.checked)}
    />
  )
}

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
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value
  if (React.isValidElement(value)) return value
  return String(value)
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
  readonly = false,
  pagination,
  selection,
}: DataTableProps<T>) {
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
  const pageSizeOptions = useMemo(() => {
    const requested = pagination?.pageSizeOptions ?? [25, 50, 100]
    return [...new Set(requested.filter((value) => Number.isFinite(value) && value > 0).map(Math.floor))]
      .sort((left, right) => left - right)
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
    ? visibleData
        .filter((row) => !selection.isRowDisabled?.(row))
        .map(keyExtractor)
    : []
  const selectedPageCount = selection
    ? selectablePageKeys.filter((key) => selection.selectedKeys.has(key)).length
    : 0
  const allPageRowsSelected = selectablePageKeys.length > 0 && selectedPageCount === selectablePageKeys.length
  const somePageRowsSelected = selectedPageCount > 0 && !allPageRowsSelected
  const auxiliaryColumnCount = (expandable ? 1 : 0) + (selection ? 1 : 0) + (hasActions ? 1 : 0)

  return (
    <div className={`${styles.tableContainer} ${className}`}>
      <div className={styles.tableViewport}>
        <table className={styles.table}>
          <thead className={styles.thead}>
            <tr>
              {selection ? (
                <th className={styles.selectionColumn}>
                  <SelectionCheckbox
                    checked={allPageRowsSelected}
                    indeterminate={somePageRowsSelected}
                    disabled={selectablePageKeys.length === 0}
                    label={`Select all ${selection.label ?? 'rows'} on this page`}
                    onChange={(checked) => selection.onChange(
                      updatePageSelection(selection.selectedKeys, selectablePageKeys, checked),
                    )}
                  />
                </th>
              ) : null}
              {expandable ? <th className={styles.expandColumn} aria-label="Expand row" /> : null}
              {columns.map((column) => {
                const activeSort = sortColumn === column.key
                return (
                  <th
                    key={column.key}
                    className={`${styles.th} ${column.sortable ? styles.sortable : ''}`}
                    style={{ width: column.width, textAlign: column.align || 'left' }}
                    aria-sort={activeSort ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined}
                  >
                    {column.sortable ? (
                      <button
                        type="button"
                        className={styles.sortButton}
                        onClick={() => handleSort(column.key)}
                      >
                        {column.header}
                        {activeSort ? (
                          <span className={styles.sortIcon} aria-hidden="true">
                            {sortDirection === 'asc' ? '↑' : '↓'}
                          </span>
                        ) : null}
                      </button>
                    ) : column.header}
                  </th>
                )
              })}
              {hasActions ? <th className={`${styles.th} ${styles.actionsColumn}`}>Actions</th> : null}
            </tr>
          </thead>
          <tbody className={styles.tbody}>
            {visibleData.length === 0 ? (
              <tr>
                <td colSpan={columns.length + auxiliaryColumnCount} className={styles.emptyState}>
                  {emptyMessage}
                </td>
              </tr>
            ) : visibleData.map((row) => {
              const key = keyExtractor(row)
              const expanded = isRowExpanded?.(row) || false
              const selectionDisabled = selection?.isRowDisabled?.(row) ?? false

              return (
                <React.Fragment key={key}>
                  <tr className={`${styles.tr} ${selection?.selectedKeys.has(key) ? styles.trSelected : ''}`}>
                    {selection ? (
                      <td className={styles.selectionCell}>
                        <SelectionCheckbox
                          checked={selection.selectedKeys.has(key)}
                          disabled={selectionDisabled}
                          label={`Select ${selection.label ?? 'row'} ${key}`}
                          onChange={(checked) => selection.onChange(
                            updatePageSelection(selection.selectedKeys, [key], checked),
                          )}
                        />
                      </td>
                    ) : null}
                    {expandable ? (
                      <td className={styles.expandCell}>
                        <button
                          type="button"
                          className={styles.expandButton}
                          onClick={() => onToggleExpand?.(row)}
                          aria-label={`${expanded ? 'Collapse' : 'Expand'} ${key}`}
                          aria-expanded={expanded}
                        >
                          <span className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`} aria-hidden="true">
                            ▶
                          </span>
                        </button>
                      </td>
                    ) : null}
                    {columns.map((column) => (
                      <td key={column.key} className={styles.td} style={{ textAlign: column.align || 'left' }}>
                        {column.render
                          ? column.render(row)
                          : renderColumnValue(getColumnValue(row, column.key))}
                      </td>
                    ))}
                    {hasActions ? (
                      <td className={`${styles.td} ${styles.actionsCell}`}>
                        <div className={styles.actionButtons}>
                          {onView ? (
                            <button
                              type="button"
                              className={`${styles.actionButton} ${styles.viewButton}`}
                              onClick={() => onView(row)}
                              aria-label={`View ${key}`}
                            >
                              View
                            </button>
                          ) : null}
                          {effectiveOnEdit ? (
                            <button
                              type="button"
                              className={`${styles.actionButton} ${styles.editButton}`}
                              onClick={() => effectiveOnEdit(row)}
                              aria-label={`Edit ${key}`}
                            >
                              Edit
                            </button>
                          ) : null}
                          {effectiveOnDelete ? (
                            <button
                              type="button"
                              className={`${styles.actionButton} ${styles.deleteButton}`}
                              onClick={() => effectiveOnDelete(row)}
                              aria-label={`Delete ${key}`}
                            >
                              Delete
                            </button>
                          ) : null}
                        </div>
                      </td>
                    ) : null}
                  </tr>
                  {expandable && expanded && renderExpandedRow ? (
                    <tr className={styles.expandedRow}>
                      <td colSpan={columns.length + auxiliaryColumnCount}>
                        {renderExpandedRow(row)}
                      </td>
                    </tr>
                  ) : null}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>

      {pagination && sortedData.length > 0 ? (
        <div className={styles.pagination} aria-label="Table pagination">
          <span className={styles.paginationSummary}>
            {pageWindow.start + 1}–{pageWindow.end} of {sortedData.length} {pagination.itemLabel ?? 'items'}
          </span>
          <label className={styles.pageSizeControl}>
            Rows
            <select
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number(event.target.value))
                setCurrentPage(1)
              }}
            >
              {pageSizeOptions.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
          <div className={styles.paginationControls}>
            <button type="button" onClick={() => setCurrentPage(1)} disabled={pageWindow.page === 1} aria-label="First page">«</button>
            <button type="button" onClick={() => setCurrentPage((page) => Math.max(1, page - 1))} disabled={pageWindow.page === 1} aria-label="Previous page">‹</button>
            <span>Page {pageWindow.page} of {pageWindow.totalPages}</span>
            <button type="button" onClick={() => setCurrentPage((page) => Math.min(pageWindow.totalPages, page + 1))} disabled={pageWindow.page === pageWindow.totalPages} aria-label="Next page">›</button>
            <button type="button" onClick={() => setCurrentPage(pageWindow.totalPages)} disabled={pageWindow.page === pageWindow.totalPages} aria-label="Last page">»</button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
