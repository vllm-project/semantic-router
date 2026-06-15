import React, { useState } from 'react'
import styles from './DataTable.module.css'

export interface Column<T> {
  key: string
  header: string
  width?: string
  align?: 'left' | 'center' | 'right'
  render?: (row: T) => React.ReactNode
  sortable?: boolean
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

  const aText = String(aValue)
  const bText = String(bValue)
  if (aText === bText) return 0
  return aText > bText ? 1 : -1
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
  readonly = false
}: DataTableProps<T>) {
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  // In readonly mode, disable edit and delete actions
  const effectiveOnEdit = readonly ? undefined : onEdit
  const effectiveOnDelete = readonly ? undefined : onDelete

  const handleSort = (columnKey: string) => {
    if (sortColumn === columnKey) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(columnKey)
      setSortDirection('asc')
    }
  }

  const sortedData = React.useMemo(() => {
    if (!sortColumn) return data

    return [...data].sort((a, b) => {
      const aValue = getColumnValue(a, sortColumn)
      const bValue = getColumnValue(b, sortColumn)

      const comparison = compareColumnValues(aValue, bValue)
      return sortDirection === 'asc' ? comparison : -comparison
    })
  }, [data, sortColumn, sortDirection])

  return (
    <div className={`${styles.tableContainer} ${className}`}>
      <table className={styles.table}>
        <thead className={styles.thead}>
          <tr>
            {expandable && <th className={styles.expandColumn}></th>}
            {columns.map((column) => (
              <th
                key={column.key}
                className={`${styles.th} ${column.sortable ? styles.sortable : ''}`}
                style={{ 
                  width: column.width,
                  textAlign: column.align || 'left'
                }}
                onClick={() => column.sortable && handleSort(column.key)}
              >
                {column.header}
                {column.sortable && sortColumn === column.key && (
                  <span className={styles.sortIcon}>
                    {sortDirection === 'asc' ? ' ↑' : ' ↓'}
                  </span>
                )}
              </th>
            ))}
            {(onView || onEdit || onDelete) && (
              <th className={`${styles.th} ${styles.actionsColumn}`}>Actions</th>
            )}
          </tr>
        </thead>
        <tbody className={styles.tbody}>
          {sortedData.length === 0 ? (
            <tr>
              <td 
                colSpan={columns.length + (expandable ? 1 : 0) + (onView || onEdit || onDelete ? 1 : 0)}
                className={styles.emptyState}
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sortedData.map((row) => {
              const key = keyExtractor(row)
              const isExpanded = isRowExpanded?.(row) || false

              return (
                <React.Fragment key={key}>
                  <tr className={styles.tr}>
                    {expandable && (
                      <td className={styles.expandCell}>
                        <button
                          className={styles.expandButton}
                          onClick={() => onToggleExpand?.(row)}
                        >
                          <span className={`${styles.expandIcon} ${isExpanded ? styles.expanded : ''}`}>
                            ▶
                          </span>
                        </button>
                      </td>
                    )}
                    {columns.map((column) => (
                      <td
                        key={column.key}
                        className={styles.td}
                        style={{ textAlign: column.align || 'left' }}
                      >
                        {column.render ? column.render(row) : renderColumnValue(getColumnValue(row, column.key))}
                      </td>
                    ))}
                    {(onView || onEdit || onDelete) && (
                      <td className={`${styles.td} ${styles.actionsCell}`}>
                        <div className={styles.actionButtons}>
                          {onView && (
                            <button
                              className={`${styles.actionButton} ${styles.viewButton}`}
                              onClick={() => onView(row)}
                            >
                              View
                            </button>
                          )}
                          {effectiveOnEdit && (
                            <button
                              className={`${styles.actionButton} ${styles.editButton}`}
                              onClick={() => effectiveOnEdit(row)}
                            >
                              Edit
                            </button>
                          )}
                          {effectiveOnDelete && (
                            <button
                              className={`${styles.actionButton} ${styles.deleteButton}`}
                              onClick={() => effectiveOnDelete(row)}
                            >
                              Delete
                            </button>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                  {expandable && isExpanded && renderExpandedRow && (
                    <tr className={styles.expandedRow}>
                      <td colSpan={columns.length + 2}>
                        {renderExpandedRow(row)}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })
          )}
        </tbody>
      </table>
    </div>
  )
}

