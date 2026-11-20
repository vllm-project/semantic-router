import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowDownWideNarrow,
  ArrowUpNarrowWide,
  CalendarRange,
  Check,
  ChevronDown,
  Download,
  ScrollText,
} from 'lucide-react'

import {
  Alert,
  Avatar,
  Badge,
  Button,
  Card,
  CodeView,
  DataToolbar,
  EmptyState,
  FilterPill,
  Input,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Separator,
  Skeleton,
  cn,
  toast,
  type FilterOption,
} from '../ui'
import {
  AUDIT_RESULT_OPTIONS,
  AUDIT_TIME_PRESETS,
  auditLogsCsv,
  auditTimeBounds,
  auditTimeLabel,
  buildAuditLogQuery,
  createLatestAuditRequest,
  isAbortError,
  normalizeAuditLogPage,
  type AuditLog,
  type AuditLogFacets,
  type AuditSortOrder,
  type AuditTimeRange,
} from './usersAuditLogSupport'
import { responseText } from './usersPageSupport'
import { useQuietRefresh } from '../hooks/useQuietRefresh'

const AUDIT_PAGE_SIZE_OPTIONS = [20, 50, 100] as const
const EXPORT_PAGE_SIZE = 200
const EXPORT_LIMIT = 10_000
const ANY_TIME: AuditTimeRange = { preset: 'any', from: '', to: '' }

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

function actorLabel(userId: string | undefined, actor?: AuditActor): string {
  return actor?.name || actor?.email || (userId ? `User ${userId.slice(0, 8)}` : 'System')
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
  const who = actorLabel(log.userId, actor)
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

/** A filter chip for one time window, styled like FilterPill. */
function TimeRangePill({
  value,
  onChange,
}: {
  value: AuditTimeRange
  onChange: (next: AuditTimeRange) => void
}) {
  const label = auditTimeLabel(value)
  const custom = value.preset === 'custom'
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            'inline-flex h-8 items-center gap-1.5 rounded-md border border-dashed border-border-strong bg-transparent px-2.5 text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground',
            label && 'border-solid bg-surface text-foreground shadow-xs',
          )}
        >
          <CalendarRange className="size-3.5" />
          Time
          {label ? (
            <>
              <Separator orientation="vertical" className="mx-0.5 h-4" />
              <span className="rounded bg-secondary px-1.5 text-xs">{label}</span>
            </>
          ) : null}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-1">
        {AUDIT_TIME_PRESETS.map((preset) => {
          const checked = value.preset === preset.value
          return (
            <button
              key={preset.value}
              type="button"
              role="menuitemradio"
              aria-checked={checked}
              onClick={() =>
                onChange(
                  preset.value === 'custom'
                    ? { ...value, preset: 'custom' }
                    : { preset: preset.value, from: '', to: '' },
                )
              }
              className="flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[13px] hover:bg-accent"
            >
              <Check className={cn('size-3.5 text-primary', !checked && 'invisible')} />
              {preset.label}
            </button>
          )
        })}
        {custom ? (
          <div className="mt-1 grid grid-cols-2 gap-2 border-t border-border px-1.5 pb-1.5 pt-2.5">
            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
              From
              <Input
                type="date"
                value={value.from}
                max={value.to || undefined}
                onChange={(event) => onChange({ ...value, from: event.target.value })}
                className="h-8 px-2 text-xs"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
              To
              <Input
                type="date"
                value={value.to}
                min={value.from || undefined}
                onChange={(event) => onChange({ ...value, to: event.target.value })}
                className="h-8 px-2 text-xs"
              />
            </label>
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}

function downloadText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

const UsersPageAuditPanel: React.FC<{ actors?: Record<string, AuditActor> }> = ({
  actors = {},
}) => {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [facets, setFacets] = useState<AuditLogFacets | null>(null)
  const [directory, setDirectory] = useState<Record<string, AuditActor>>({})
  const [exporting, setExporting] = useState(false)

  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [people, setPeople] = useState<string[]>([])
  const [actions, setActions] = useState<string[]>([])
  const [results, setResults] = useState<string[]>([])
  const [range, setRange] = useState<AuditTimeRange>(ANY_TIME)
  const [order, setOrder] = useState<AuditSortOrder>('desc')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(20)
  const [requests] = useState(createLatestAuditRequest)
  const facetsRequested = useRef(false)

  const known = useMemo(() => ({ ...directory, ...actors }), [actors, directory])

  const queryFor = useCallback(
    (pageNumber: number, limit: number) =>
      buildAuditLogQuery({
        query: debouncedQuery,
        user: people.join(','),
        action: actions.join(','),
        resource: '',
        status: results.length === AUDIT_RESULT_OPTIONS.length ? '' : results.join(','),
        ...auditTimeBounds(range),
        sort: 'createdAt',
        order,
        page: pageNumber,
        limit,
      }),
    [actions, debouncedQuery, order, people, range, results],
  )

  const fetchLogs = useCallback(async () => {
    if (range.preset === 'custom' && range.from && range.to && range.from > range.to) {
      requests.abort()
      setLoading(false)
      setError('The start date must be on or before the end date.')
      return
    }

    const request = requests.start()
    setLoading(true)
    setError(null)
    const withFacets = !facetsRequested.current
    try {
      const response = await fetch(
        `/api/admin/audit-logs?${queryFor(page, pageSize)}${withFacets ? '&facets=1' : ''}`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: request.signal,
        },
      )
      if (!response.ok) throw new Error(await responseText(response))
      const payload = normalizeAuditLogPage(
        (await response.json()) as Parameters<typeof normalizeAuditLogPage>[0],
        page,
        pageSize,
      )
      if (!request.isCurrent()) return
      if (withFacets) {
        facetsRequested.current = true
        if (payload.facets) setFacets(payload.facets)
      }

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
  }, [page, pageSize, queryFor, range, requests])

  useEffect(() => {
    void fetchLogs()
    return () => requests.abort()
  }, [fetchLogs, requests])

  // Reading the audit log is itself audited, so it refreshes when the tab
  // comes back rather than on a timer.
  useQuietRefresh(() => void fetchLogs(), { afterAwayMs: 60_000, enabled: page === 1 })

  useEffect(() => {
    const controller = new AbortController()
    void fetch('/api/admin/users?limit=200&sort=name&order=asc', { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload: { users?: Array<{ id: string; name: string; email: string }> } | null) => {
        if (!payload?.users) return
        setDirectory(
          Object.fromEntries(
            payload.users.map((user) => [user.id, { name: user.name, email: user.email }]),
          ),
        )
      })
      .catch(() => undefined)
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 300)
    return () => window.clearTimeout(timer)
  }, [query])

  const personOptions = useMemo<FilterOption[]>(() => {
    const ids = facets
      ? facets.users.map((facet) => facet.value)
      : Object.keys(known).filter((id) => logs.some((log) => log.userId === id))
    const counts = new Map(facets?.users.map((facet) => [facet.value, facet.count]))
    return [...new Set([...ids, ...people])].map((id) => {
      const label = actorLabel(id, known[id])
      return {
        value: id,
        label,
        count: counts.get(id),
        icon: <Avatar name={label} size={16} />,
      }
    })
  }, [facets, known, logs, people])

  const actionOptions = useMemo<FilterOption[]>(() => {
    const counts = new Map(facets?.actions.map((facet) => [facet.value, facet.count]))
    const values = facets
      ? facets.actions.map((facet) => facet.value)
      : [...new Set(logs.map((log) => log.action))].sort()
    return [...new Set([...values, ...actions])].map((value) => ({
      value,
      label: value,
      count: counts.get(value),
    }))
  }, [actions, facets, logs])

  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const currentPage = Math.min(page, totalPages)
  const hasFilters = Boolean(
    query.trim() ||
      people.length ||
      actions.length ||
      results.length ||
      auditTimeLabel(range) !== null,
  )

  const resetFilters = () => {
    setQuery('')
    setDebouncedQuery('')
    setPeople([])
    setActions([])
    setResults([])
    setRange(ANY_TIME)
    setPage(1)
    setError(null)
  }

  const exportCsv = async () => {
    setExporting(true)
    try {
      const collected: AuditLog[] = []
      for (let pageNumber = 1; collected.length < EXPORT_LIMIT; pageNumber += 1) {
        const response = await fetch(
          `/api/admin/audit-logs?${queryFor(pageNumber, EXPORT_PAGE_SIZE)}`,
        )
        if (!response.ok) throw new Error(await responseText(response))
        const payload = normalizeAuditLogPage(
          (await response.json()) as Parameters<typeof normalizeAuditLogPage>[0],
          pageNumber,
          EXPORT_PAGE_SIZE,
        )
        collected.push(...payload.logs)
        if (payload.logs.length < EXPORT_PAGE_SIZE || collected.length >= payload.total) break
      }
      const rows = collected.slice(0, EXPORT_LIMIT)
      const stamp = new Date().toISOString().slice(0, 10)
      downloadText(
        `audit-log-${stamp}.csv`,
        auditLogsCsv(rows, (userId) => (userId ? (known[userId]?.name ?? '') : '')),
      )
      toast.success('Audit log exported', {
        description: `${rows.length.toLocaleString()} ${rows.length === 1 ? 'event' : 'events'} · audit-log-${stamp}.csv`,
      })
    } catch (cause) {
      toast.error('The audit log was not exported', {
        description: cause instanceof Error ? cause.message : String(cause),
      })
    } finally {
      setExporting(false)
    }
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
        className="mb-0"
        filters={
          <>
            <FilterPill
              label="Person"
              selected={people}
              onChange={filterChange(setPeople)}
              options={personOptions}
            />
            <FilterPill
              label="Action"
              selected={actions}
              onChange={filterChange(setActions)}
              options={actionOptions}
            />
            <FilterPill
              label="Result"
              selected={results}
              onChange={filterChange(setResults)}
              options={AUDIT_RESULT_OPTIONS.map((option) => ({ ...option }))}
            />
            <TimeRangePill value={range} onChange={filterChange(setRange)} />
          </>
        }
        actions={
          <>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => filterChange(setOrder)(order === 'desc' ? 'asc' : 'desc')}
              aria-label={order === 'desc' ? 'Show oldest first' : 'Show newest first'}
            >
              {order === 'desc' ? <ArrowDownWideNarrow /> : <ArrowUpNarrowWide />}
              {order === 'desc' ? 'Newest first' : 'Oldest first'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void exportCsv()}
              loading={exporting}
              disabled={total === 0}
            >
              <Download />
              Export CSV
            </Button>
          </>
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
                    actor={log.userId ? known[log.userId] : undefined}
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
                ? 'Clear the search or filters to see every recorded event.'
                : 'Sign-ins, role changes and config saves show up here.'
            }
            actions={
              hasFilters ? (
                <Button variant="outline" size="sm" onClick={resetFilters}>
                  Clear filters
                </Button>
              ) : null
            }
          />
        )}
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-2" aria-live="polite">
          Page {currentPage} / {totalPages} · {total.toLocaleString()} events
          <Select
            value={String(pageSize)}
            onValueChange={(value) => filterChange(setPageSize)(Number.parseInt(value, 10))}
          >
            <SelectTrigger size="sm" aria-label="Rows per page" className="h-7 w-[4.5rem] text-xs">
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
