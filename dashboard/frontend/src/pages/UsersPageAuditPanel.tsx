import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, RefreshCw, ScrollText, SlidersHorizontal } from 'lucide-react'

import {
  Alert,
  Avatar,
  Badge,
  Button,
  Card,
  CodeView,
  DataToolbar,
  EmptyState,
  Field,
  Input,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
  cn,
} from '../ui'
import {
  buildAuditLogQuery,
  createLatestAuditRequest,
  isAbortError,
  normalizeAuditLogPage,
  type AuditLog,
  type AuditSortOrder,
} from './usersAuditLogSupport'
import { responseText } from './usersPageSupport'

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
const STATUS_OPTIONS = [
  { value: 'all', label: 'All responses' },
  { value: 'success', label: 'Successful' },
  { value: 'client_error', label: 'Client error' },
  { value: 'server_error', label: 'Server error' },
] as const

export interface AuditActor {
  name: string
  email: string
}

const dayFormat = new Intl.DateTimeFormat('en-US', {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  year: 'numeric',
})
const timeFormat = new Intl.DateTimeFormat('en-US', {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
})

function statusTone(code: number): 'success' | 'warning' | 'danger' | 'neutral' {
  if (!code) return 'neutral'
  if (code >= 500) return 'danger'
  if (code >= 400) return 'warning'
  return 'success'
}

function details(log: AuditLog): string {
  let extra: unknown = undefined
  if (log.extraJson) {
    try {
      extra = JSON.parse(log.extraJson)
    } catch {
      extra = log.extraJson
    }
  }
  return JSON.stringify(
    {
      id: log.id,
      action: log.action,
      resource: log.resource,
      method: log.method,
      path: log.path,
      statusCode: log.statusCode,
      userId: log.userId,
      ip: log.ip,
      userAgent: log.userAgent,
      ...(extra === undefined ? {} : { extra }),
    },
    null,
    2,
  )
}

function AuditRow({
  log,
  actor,
  open,
  onToggle,
}: {
  log: AuditLog
  actor?: AuditActor
  open: boolean
  onToggle: () => void
}) {
  const who =
    actor?.name || actor?.email || (log.userId ? `User ${log.userId.slice(0, 8)}` : 'System')
  const tone = statusTone(log.statusCode)
  return (
    <li>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="group flex w-full items-start gap-3 px-4 py-3 text-left outline-none transition-colors hover:bg-surface-subtle/70 focus-visible:bg-surface-subtle"
      >
        <Avatar name={who} size={26} className="relative mt-px ring-2 ring-card" />
        <span className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="text-[13px] leading-5">
            <span className="font-medium">{who}</span>{' '}
            <span className="text-muted-foreground">·</span>{' '}
            <code className="font-mono text-[12.5px]">{log.action}</code>
            {log.resource ? (
              <>
                {' '}
                <span className="text-muted-foreground">on</span>{' '}
                <span className="font-mono text-[12.5px]">{log.resource}</span>
              </>
            ) : null}
          </span>
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-subtle-foreground">
            <span className="font-mono text-[11px]">
              {log.method} {log.path}
            </span>
            {log.ip ? (
              <>
                <span aria-hidden>·</span>
                <span className="font-mono text-[11px]">{log.ip}</span>
              </>
            ) : null}
            {log.statusCode ? (
              <Badge size="sm" tone={tone === 'neutral' ? 'neutral' : tone} className="font-mono">
                {log.statusCode}
              </Badge>
            ) : null}
          </span>
        </span>
        <span className="shrink-0 pt-0.5 font-mono text-xs tabular text-subtle-foreground">
          {log.createdAt ? timeFormat.format(new Date(log.createdAt * 1000)) : '—'}
        </span>
        <ChevronDown
          className={cn(
            'mt-0.5 size-4 shrink-0 text-subtle-foreground opacity-0 transition-[transform,opacity] group-hover:opacity-100',
            open && 'rotate-180 opacity-100',
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="pb-3 pl-[54px] pr-4">
              <div className="overflow-hidden rounded-lg border border-border bg-surface-sunken/60 dark:bg-surface-sunken">
                <CodeView code={details(log)} language="json" className="py-2.5 text-[12px]" />
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </li>
  )
}

const UsersPageAuditPanel: React.FC<{ actors?: Record<string, AuditActor> }> = ({
  actors = {},
}) => {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)

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
      if (!response.ok) throw new Error(await responseText(response))
      const payload = normalizeAuditLogPage(
        (await response.json()) as Parameters<typeof normalizeAuditLogPage>[0],
        page,
        pageSize,
      )
      if (!request.isCurrent()) return

      const lastPage = Math.max(1, Math.ceil(payload.total / pageSize))
      if (page > lastPage) {
        setPage(lastPage)
        return
      }
      setLogs(payload.logs)
      setTotal(payload.total)
    } catch (requestError) {
      if (!request.isCurrent() || isAbortError(requestError)) return
      setError((requestError as Error).message)
    } finally {
      if (request.isCurrent()) setLoading(false)
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
  const advancedCount = [userFilter, actionFilter, resourceFilter].filter((value) =>
    value.trim(),
  ).length
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

  const days = useMemo(() => {
    const groups: Array<{ day: string; logs: AuditLog[] }> = []
    for (const log of logs) {
      const day = log.createdAt ? dayFormat.format(new Date(log.createdAt * 1000)) : 'Unknown time'
      const last = groups[groups.length - 1]
      if (last && last.day === day) last.logs.push(log)
      else groups.push({ day, logs: [log] })
    }
    return groups
  }, [logs])

  const filterChange =
    <T,>(setter: (value: T) => void) =>
    (value: T) => {
      setter(value)
      setPage(1)
    }

  return (
    <div className="flex flex-col gap-3">
      <DataToolbar
        search={query}
        onSearchChange={filterChange(setQuery)}
        placeholder="Search activity…"
        filters={
          <>
            <Select value={statusFilter} onValueChange={filterChange(setStatusFilter)}>
              <SelectTrigger size="sm" aria-label="Response status" className="min-w-36">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {STATUS_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <span className="flex items-center gap-1.5">
              <Input
                type="date"
                value={fromDate}
                max={toDate || undefined}
                onChange={(event) => filterChange(setFromDate)(event.target.value)}
                aria-label="From date"
                className="h-8 w-[9.5rem] text-xs"
              />
              <span className="text-xs text-subtle-foreground">to</span>
              <Input
                type="date"
                value={toDate}
                min={fromDate || undefined}
                onChange={(event) => filterChange(setToDate)(event.target.value)}
                aria-label="To date"
                className="h-8 w-[9.5rem] text-xs"
              />
            </span>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm">
                  <SlidersHorizontal />
                  More filters
                  {advancedCount ? (
                    <Badge size="sm" tone="brand">
                      {advancedCount}
                    </Badge>
                  ) : null}
                </Button>
              </PopoverTrigger>
              <PopoverContent align="start" className="w-80">
                <div className="flex flex-col gap-3">
                  <Field label="User ID" htmlFor="audit-user">
                    <Input
                      id="audit-user"
                      value={userFilter}
                      onChange={(event) => filterChange(setUserFilter)(event.target.value)}
                      placeholder="Exact user ID"
                      className="h-8 font-mono text-[12.5px]"
                    />
                  </Field>
                  <Field label="Action" htmlFor="audit-action">
                    <Input
                      id="audit-action"
                      value={actionFilter}
                      onChange={(event) => filterChange(setActionFilter)(event.target.value)}
                      placeholder="user.update"
                      className="h-8 font-mono text-[12.5px]"
                    />
                  </Field>
                  <Field label="Resource" htmlFor="audit-resource">
                    <Input
                      id="audit-resource"
                      value={resourceFilter}
                      onChange={(event) => filterChange(setResourceFilter)(event.target.value)}
                      placeholder="Exact resource"
                      className="h-8 font-mono text-[12.5px]"
                    />
                  </Field>
                  <div className="grid grid-cols-2 gap-2">
                    <Field label="Sort by" htmlFor="audit-sort">
                      <Select value={sortField} onValueChange={filterChange(setSortField)}>
                        <SelectTrigger id="audit-sort" size="sm" className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {AUDIT_SORT_OPTIONS.map((option) => (
                            <SelectItem key={option.value} value={option.value}>
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Field>
                    <Field label="Order" htmlFor="audit-order">
                      <Select
                        value={sortOrder}
                        onValueChange={(value) =>
                          filterChange(setSortOrder)(value === 'asc' ? 'asc' : 'desc')
                        }
                      >
                        <SelectTrigger id="audit-order" size="sm" className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="desc">Newest first</SelectItem>
                          <SelectItem value="asc">Oldest first</SelectItem>
                        </SelectContent>
                      </Select>
                    </Field>
                  </div>
                  <Field label="Rows per page" htmlFor="audit-rows">
                    <Select
                      value={String(pageSize)}
                      onValueChange={(value) =>
                        filterChange(setPageSize)(Number.parseInt(value, 10))
                      }
                    >
                      <SelectTrigger id="audit-rows" size="sm" className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {AUDIT_PAGE_SIZE_OPTIONS.map((size) => (
                          <SelectItem key={size} value={String(size)}>
                            {size}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={resetFilters}
                    disabled={!hasFilters && sortField === 'createdAt' && sortOrder === 'desc'}
                  >
                    Reset filters
                  </Button>
                </div>
              </PopoverContent>
            </Popover>
          </>
        }
        actions={
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => void fetchLogs()}
            disabled={loading}
            aria-label="Refresh audit logs"
          >
            <RefreshCw className={cn(loading && 'animate-spin')} />
          </Button>
        }
      />

      {error ? (
        <Alert
          tone="danger"
          title="The audit log could not be loaded"
          actions={
            <Button size="sm" variant="outline" onClick={() => void fetchLogs()}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      ) : null}

      <Card className="overflow-hidden">
        {loading && logs.length === 0 ? (
          <div className="flex flex-col divide-y divide-border">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="flex items-center gap-3 px-4 py-3.5">
                <Skeleton className="size-[26px] rounded-full" />
                <div className="flex flex-1 flex-col gap-1.5">
                  <Skeleton className="h-3.5 w-1/2" />
                  <Skeleton className="h-3 w-1/3" />
                </div>
                <Skeleton className="h-3 w-14" />
              </div>
            ))}
          </div>
        ) : logs.length ? (
          days.map((group) => (
            <section key={group.day}>
              <div className="flex items-center justify-between border-b border-border bg-surface-subtle/70 px-4 py-2 text-xs">
                <span className="font-medium text-muted-foreground">{group.day}</span>
                <span className="tabular text-subtle-foreground">{group.logs.length} events</span>
              </div>
              <ol className="relative border-b border-border last:border-0">
                <span aria-hidden className="absolute bottom-5 left-[29px] top-5 w-px bg-border" />
                {group.logs.map((log) => (
                  <AuditRow
                    key={log.id}
                    log={log}
                    actor={log.userId ? actors[log.userId] : undefined}
                    open={expanded === log.id}
                    onToggle={() => setExpanded((current) => (current === log.id ? null : log.id))}
                  />
                ))}
              </ol>
            </section>
          ))
        ) : (
          <EmptyState
            compact
            icon={ScrollText}
            tone="slate"
            title={hasFilters ? 'No activity matches' : 'No activity yet'}
            description={
              hasFilters
                ? 'No audit entries match the current filters.'
                : 'No audit log entries have been recorded.'
            }
            actions={
              hasFilters ? (
                <Button variant="outline" size="sm" onClick={resetFilters}>
                  Reset filters
                </Button>
              ) : null
            }
          />
        )}
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <span aria-live="polite">
          Page {currentPage} / {totalPages} · {total.toLocaleString()} records
          {loading && logs.length > 0 ? ' · Refreshing…' : ''}
        </span>
        <span className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="xs"
            onClick={() => setPage(Math.max(1, currentPage - 1))}
            disabled={currentPage <= 1 || loading}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="xs"
            onClick={() => setPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage >= totalPages || loading}
          >
            Next
          </Button>
        </span>
      </div>
    </div>
  )
}

export default UsersPageAuditPanel
