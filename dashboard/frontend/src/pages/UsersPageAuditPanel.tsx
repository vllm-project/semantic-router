import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { DataTable, type Column } from '../components/DataTable'
import styles from './UsersPage.module.css'
import {
  buildAuditLogQuery,
  createLatestAuditRequest,
  isAbortError,
  normalizeAuditLogPage,
  type AuditLog,
  type AuditSortOrder,
} from './usersAuditLogSupport'

const AUDIT_PAGE_SIZE_OPTIONS = [20, 50, 100] as const
const AUDIT_SORT_OPTIONS = [
  { value: 'createdAt', label: 'Time' },
  { value: 'id', label: 'ID' },
  { value: 'action', label: 'Action' },
  { value: 'resource', label: 'Resource' },
  { value: 'method', label: 'Method' },
  { value: 'statusCode', label: 'Status code' },
  { value: 'userId', label: 'User ID' },
] as const

const formatTs = (value?: number) => {
  if (!value) {
    return '-'
  }
  return new Date(value * 1000).toLocaleString()
}

const getResponseError = async (response: Response) => {
  const text = await response.text()
  return text || `Request failed: ${response.status}`
}

const UsersPageAuditPanel: React.FC = () => {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [userFilter, setUserFilter] = useState('')
  const [debouncedUserFilter, setDebouncedUserFilter] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [debouncedActionFilter, setDebouncedActionFilter] = useState('')
  const [resourceFilter, setResourceFilter] = useState('')
  const [debouncedResourceFilter, setDebouncedResourceFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [sortField, setSortField] = useState('createdAt')
  const [sortOrder, setSortOrder] = useState<AuditSortOrder>('desc')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(20)
  const [requests] = useState(createLatestAuditRequest)

  const fetchLogs = useCallback(async () => {
    if (fromDate && toDate && fromDate > toDate) {
      requests.abort()
      setLoading(false)
      setError('The start date must be on or before the end date.')
      return
    }

    const request = requests.start()
    setLoading(true)
    setError(null)
    try {
      const requestQuery = buildAuditLogQuery({
        query: debouncedQuery,
        user: debouncedUserFilter,
        action: debouncedActionFilter,
        resource: debouncedResourceFilter,
        status: statusFilter,
        from: fromDate,
        to: toDate,
        sort: sortField,
        order: sortOrder,
        page,
        limit: pageSize,
      })
      const response = await fetch(`/api/admin/audit-logs?${requestQuery}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: request.signal,
      })
      if (!response.ok) {
        throw new Error(await getResponseError(response))
      }
      const payload = normalizeAuditLogPage(
        (await response.json()) as Parameters<typeof normalizeAuditLogPage>[0],
        page,
        pageSize,
      )
      if (!request.isCurrent()) {
        return
      }

      const lastPage = Math.max(1, Math.ceil(payload.total / pageSize))
      if (page > lastPage) {
        setPage(lastPage)
        return
      }
      setLogs(payload.logs)
      setTotal(payload.total)
    } catch (requestError) {
      if (!request.isCurrent() || isAbortError(requestError)) {
        return
      }
      setError((requestError as Error).message)
    } finally {
      if (request.isCurrent()) {
        setLoading(false)
      }
      request.finish()
    }
  }, [
    debouncedActionFilter,
    debouncedQuery,
    debouncedResourceFilter,
    debouncedUserFilter,
    fromDate,
    page,
    pageSize,
    requests,
    sortField,
    sortOrder,
    statusFilter,
    toDate,
  ])

  useEffect(() => {
    void fetchLogs()
    return () => requests.abort()
  }, [fetchLogs, requests])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(query.trim())
      setDebouncedUserFilter(userFilter.trim())
      setDebouncedActionFilter(actionFilter.trim())
      setDebouncedResourceFilter(resourceFilter.trim())
    }, 300)
    return () => window.clearTimeout(timer)
  }, [actionFilter, query, resourceFilter, userFilter])

  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const currentPage = Math.min(page, totalPages)
  const hasFilters = Boolean(
    query.trim() ||
      userFilter.trim() ||
      actionFilter.trim() ||
      resourceFilter.trim() ||
      statusFilter !== 'all' ||
      fromDate ||
      toDate,
  )

  const resetFilters = () => {
    setQuery('')
    setDebouncedQuery('')
    setUserFilter('')
    setDebouncedUserFilter('')
    setActionFilter('')
    setDebouncedActionFilter('')
    setResourceFilter('')
    setDebouncedResourceFilter('')
    setStatusFilter('all')
    setFromDate('')
    setToDate('')
    setSortField('createdAt')
    setSortOrder('desc')
    setPage(1)
    setError(null)
  }

  const columns: Column<AuditLog>[] = useMemo(
    () => [
      { key: 'id', header: 'ID', width: '80px', render: (row) => `#${row.id}` },
      {
        key: 'createdAt',
        header: 'Time',
        width: '180px',
        render: (row) => formatTs(row.createdAt),
      },
      {
        key: 'action',
        header: 'Action',
        width: '160px',
        render: (row) => <code className={styles.code}>{row.action}</code>,
      },
      { key: 'resource', header: 'Resource', width: '210px' },
      { key: 'method', header: 'Method', width: '90px' },
      {
        key: 'statusCode',
        header: 'Code',
        width: '90px',
        render: (row) =>
          row.statusCode ? (
            <span
              className={`${styles.auditStatusCode} ${row.statusCode >= 400 ? styles.auditStatusCodeError : ''}`}
            >
              {row.statusCode}
            </span>
          ) : (
            '-'
          ),
      },
      {
        key: 'path',
        header: 'Path',
        width: '220px',
        render: (row) => <code className={styles.code}>{row.path}</code>,
      },
      { key: 'ip', header: 'IP', width: '150px', render: (row) => row.ip || '-' },
      { key: 'userId', header: 'User ID', width: '180px', render: (row) => row.userId || '-' },
    ],
    [],
  )

  return (
    <section className={styles.card}>
      <div className={styles.sectionHeader}>
        <div>
          <h2 className={styles.sectionTitle}>Audit logs</h2>
          <p className={styles.sectionDescription}>
            Search and inspect privileged activity across the dashboard without loading the full
            audit history into the browser.
          </p>
        </div>
        <span className={styles.auditCount} aria-live="polite">
          {total.toLocaleString()} records
        </span>
      </div>

      <div className={styles.auditFilterPanel}>
        <div className={styles.auditFilterGrid}>
          <label className={`${styles.auditFilterGroup} ${styles.auditSearchGroup}`}>
            <span>Search</span>
            <input
              className={styles.search}
              type="search"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value)
                setPage(1)
              }}
              placeholder="Action, path, IP, user agent, or status code"
            />
          </label>

          <label className={styles.auditFilterGroup}>
            <span>Status</span>
            <select
              className={styles.filter}
              value={statusFilter}
              onChange={(event) => {
                setStatusFilter(event.target.value)
                setPage(1)
              }}
            >
              <option value="all">All responses</option>
              <option value="success">Successful (2xx–3xx)</option>
              <option value="client_error">Client error (4xx)</option>
              <option value="server_error">Server error (5xx)</option>
            </select>
          </label>

          <label className={styles.auditFilterGroup}>
            <span>User ID</span>
            <input
              className={styles.filterInput}
              value={userFilter}
              onChange={(event) => {
                setUserFilter(event.target.value)
                setPage(1)
              }}
              placeholder="Exact user ID"
            />
          </label>

          <label className={styles.auditFilterGroup}>
            <span>Action</span>
            <input
              className={styles.filterInput}
              value={actionFilter}
              onChange={(event) => {
                setActionFilter(event.target.value)
                setPage(1)
              }}
              placeholder="e.g. user.update"
            />
          </label>

          <label className={styles.auditFilterGroup}>
            <span>Resource</span>
            <input
              className={styles.filterInput}
              value={resourceFilter}
              onChange={(event) => {
                setResourceFilter(event.target.value)
                setPage(1)
              }}
              placeholder="Exact resource"
            />
          </label>

          <label className={styles.auditFilterGroup}>
            <span>From</span>
            <input
              className={styles.filterInput}
              type="date"
              value={fromDate}
              max={toDate || undefined}
              onChange={(event) => {
                setFromDate(event.target.value)
                setPage(1)
              }}
            />
          </label>

          <label className={styles.auditFilterGroup}>
            <span>To</span>
            <input
              className={styles.filterInput}
              type="date"
              value={toDate}
              min={fromDate || undefined}
              onChange={(event) => {
                setToDate(event.target.value)
                setPage(1)
              }}
            />
          </label>

          <label className={styles.auditFilterGroup}>
            <span>Sort</span>
            <select
              className={styles.filter}
              value={sortField}
              onChange={(event) => {
                setSortField(event.target.value)
                setPage(1)
              }}
            >
              {AUDIT_SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className={styles.auditFilterGroup}>
            <span>Order</span>
            <select
              className={styles.filter}
              value={sortOrder}
              onChange={(event) => {
                setSortOrder(event.target.value === 'asc' ? 'asc' : 'desc')
                setPage(1)
              }}
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </label>

          <label className={styles.auditFilterGroup}>
            <span>Page size</span>
            <select
              className={styles.filter}
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number.parseInt(event.target.value, 10))
                setPage(1)
              }}
            >
              {AUDIT_PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className={styles.auditToolbarActions}>
          <button
            className={styles.secondaryButton}
            type="button"
            onClick={resetFilters}
            disabled={!hasFilters && sortField === 'createdAt' && sortOrder === 'desc'}
          >
            Reset filters
          </button>
          <button
            className={styles.secondaryButton}
            type="button"
            onClick={() => void fetchLogs()}
            disabled={loading}
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error ? (
        <div className={styles.auditError} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => void fetchLogs()}>
            Retry
          </button>
        </div>
      ) : null}

      {loading && logs.length === 0 ? (
        <div className={styles.loading} role="status">
          Loading audit logs...
        </div>
      ) : (
        <DataTable
          columns={columns}
          data={logs}
          keyExtractor={(row) => `${row.id}`}
          onEdit={undefined}
          onDelete={undefined}
          className={styles.auditTableContainer}
          emptyMessage={
            hasFilters
              ? 'No audit entries match the current filters.'
              : 'No audit log entries have been recorded.'
          }
        />
      )}

      <div className={styles.pagination}>
        <span aria-live="polite">
          Page {currentPage} / {totalPages} · {total.toLocaleString()} records
          {loading && logs.length > 0 ? ' · Refreshing…' : ''}
        </span>

        <div className={styles.paginationActions}>
          <button
            type="button"
            onClick={() => setPage(Math.max(1, currentPage - 1))}
            disabled={currentPage <= 1 || loading}
          >
            Previous
          </button>
          <button
            type="button"
            onClick={() => setPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage >= totalPages || loading}
          >
            Next
          </button>
        </div>
      </div>
    </section>
  )
}

export default UsersPageAuditPanel
