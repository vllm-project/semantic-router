import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDown,
  ChevronRight,
  Copy,
  Download,
  ListFilter,
  RefreshCw,
  ScrollText,
  Search,
  SearchX,
  WrapText,
} from 'lucide-react'

import {
  Alert,
  Button,
  Card,
  CopyButton,
  DataToolbar,
  EmptyState,
  FilterPill,
  IconTile,
  InputGroup,
  Label,
  MetaItem,
  PageContainer,
  PageHeader,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
  StatusDot,
  Switch,
  Tooltip,
  cn,
  toast,
} from '../ui'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import {
  LOG_LEVELS,
  describeLogsAvailability,
  filterLogs,
  getLogLevel,
  logsDownloadName,
  matchRanges,
  parseStructuredLine,
  splitServicePrefix,
  type LogEntry,
  type LogLevel,
} from './logsPageSupport'

interface LogsResponse {
  deployment_type: string
  service: string
  supported: boolean
  logs: LogEntry[]
  count: number
  error?: string
  message?: string
}

type Component = 'all' | 'router' | 'envoy' | 'dashboard'

const COMPONENTS: { value: Component; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'router', label: 'Router' },
  { value: 'envoy', label: 'Envoy' },
  { value: 'dashboard', label: 'Dashboard' },
]

const LINE_OPTIONS = [100, 200, 500, 1000]
const FOLLOW_MS = 5000

const levelBadge: Record<LogLevel, string> = {
  error: 'bg-destructive-soft text-destructive-foreground',
  warn: 'bg-warning-soft text-warning-foreground',
  info: 'bg-info-soft text-info-foreground',
  debug: 'bg-secondary text-subtle-foreground',
  other: 'bg-transparent text-subtle-foreground',
}

const levelDot: Record<LogLevel, string> = {
  error: 'bg-destructive',
  warn: 'bg-warning',
  info: 'bg-info',
  debug: 'bg-subtle-foreground',
  other: 'bg-border-strong',
}

const levelLabel: Record<LogLevel, string> = {
  error: 'Error',
  warn: 'Warning',
  info: 'Info',
  debug: 'Debug',
  other: 'Other',
}

const serviceTone: Record<string, string> = {
  router: 'text-chart-1',
  envoy: 'text-chart-4',
  dashboard: 'text-chart-3',
}

function formatDeploymentType(type: string) {
  if (type === 'none') return 'Not detected'
  if (type === 'detecting...') return 'Detecting'
  return type
    .split(/[-_\s]+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function Highlighted({ text, query }: { text: string; query: string }) {
  const ranges = matchRanges(text, query)
  if (!ranges.length) return <>{text}</>
  const parts: React.ReactNode[] = []
  let cursor = 0
  for (const range of ranges) {
    if (range.start > cursor) parts.push(text.slice(cursor, range.start))
    parts.push(
      <mark key={range.start} className="rounded-[3px] bg-chart-2/30 text-foreground">
        {text.slice(range.start, range.end)}
      </mark>,
    )
    cursor = range.end
  }
  if (cursor < text.length) parts.push(text.slice(cursor))
  return <>{parts}</>
}

const LogRow = React.memo(function LogRow({
  entry,
  index,
  query,
  wrap,
  expanded,
  onToggle,
  onFilter,
}: {
  entry: LogEntry
  index: number
  query: string
  wrap: boolean
  expanded: boolean
  onToggle: (index: number) => void
  onFilter: (value: string) => void
}) {
  const level = getLogLevel(entry.line)
  const { service, rest } = splitServicePrefix(entry.line)
  const fields = expanded ? parseStructuredLine(entry.line) : null
  const requestId = fields
    ? String(fields.request_id ?? fields.requestId ?? fields.x_request_id ?? '')
    : ''
  return (
    <div
      className={cn(
        level === 'error' && 'bg-destructive-soft/40',
        level === 'warn' && 'bg-warning-soft/40',
      )}
    >
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => onToggle(index)}
        className={cn(
          'group flex w-full items-start gap-2.5 px-3 py-[3px] text-left outline-none transition-colors hover:bg-accent/70 focus-visible:bg-accent/70 focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring',
          expanded && 'bg-accent/70',
        )}
      >
        <ChevronRight
          aria-hidden="true"
          className={cn(
            'mt-[3px] size-3 shrink-0 text-subtle-foreground transition-transform duration-200',
            expanded && 'rotate-90',
          )}
        />
        <span className="w-10 shrink-0 select-none text-right tabular-nums text-subtle-foreground/70">
          {index + 1}
        </span>
        <span
          className={cn(
            'mt-px inline-flex h-[17px] w-[52px] shrink-0 items-center justify-center rounded text-[10px] font-semibold uppercase tracking-[0.04em]',
            levelBadge[level],
          )}
        >
          {level === 'other' ? '—' : level}
        </span>
        <span
          className={cn(
            'min-w-0 flex-1',
            wrap ? 'whitespace-pre-wrap break-all' : 'truncate',
            level === 'error'
              ? 'text-destructive-foreground'
              : level === 'debug'
                ? 'text-muted-foreground'
                : 'text-foreground/90',
          )}
        >
          {service ? (
            <span className={cn('font-medium', serviceTone[service])}>[{service}]</span>
          ) : null}
          {service ? ' ' : null}
          <Highlighted text={rest} query={query} />
        </span>
      </button>
      {expanded ? (
        <div className="mx-3 mb-2 ml-[4.75rem] mt-1 overflow-hidden rounded-lg border border-border bg-card font-sans">
          <div className="flex items-center gap-1 border-b border-border py-1 pl-3 pr-1 text-[11px] text-subtle-foreground">
            <span className="mr-auto">
              {fields ? 'Structured fields' : 'Full line'}
              {entry.service ? ` · ${entry.service}` : ''}
            </span>
            {requestId ? (
              <Button variant="ghost" size="xs" onClick={() => onFilter(requestId)}>
                <ListFilter aria-hidden="true" />
                Only this request
              </Button>
            ) : null}
            <CopyButton value={entry.line} label="Copy line" />
          </div>
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-all px-3 py-2 font-mono text-[12px] leading-[1.6] text-foreground/90">
            {fields ? JSON.stringify(fields, null, 2) : entry.line}
          </pre>
        </div>
      ) : null}
    </div>
  )
})

const LogsPage: React.FC = () => {
  const [component, setComponent] = useState<Component>('all')
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [deploymentType, setDeploymentType] = useState<string>('detecting...')
  const [supported, setSupported] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(true)
  const [fetching, setFetching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [follow, setFollow] = useState(true)
  const [lines, setLines] = useState(200)
  const [query, setQuery] = useState('')
  const [levels, setLevels] = useState<LogLevel[]>([])
  const [wrap, setWrap] = useState(false)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const [atBottom, setAtBottom] = useState(true)
  const [unseen, setUnseen] = useState(0)
  const bodyRef = useRef<HTMLDivElement>(null)
  const atBottomRef = useRef(true)
  const requestControllerRef = useRef<AbortController | null>(null)
  const logsRef = useRef<LogEntry[]>([])

  const fetchLogs = useCallback(async () => {
    requestControllerRef.current?.abort()
    const controller = new AbortController()
    requestControllerRef.current = controller
    setFetching(true)
    try {
      const response = await fetch(`/api/logs?component=${component}&lines=${lines}`, {
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.statusText || `HTTP ${response.status}`}`)
      }
      const data: LogsResponse = await response.json()
      if (controller.signal.aborted) return
      const next = data.logs || []
      const previous = logsRef.current
      if (!atBottomRef.current && previous.length && next.length) {
        const last = previous[previous.length - 1].line
        let position = -1
        for (let index = next.length - 1; index >= 0; index -= 1) {
          if (next[index].line === last) {
            position = index
            break
          }
        }
        setUnseen((count) => count + (position === -1 ? next.length : next.length - 1 - position))
      }
      logsRef.current = next
      setLogs(next)
      setDeploymentType(data.deployment_type)
      setSupported(data.supported)
      setError(data.error || null)
      setMessage(data.message || null)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      if (requestControllerRef.current === controller) {
        setLoading(false)
        setFetching(false)
      }
    }
  }, [component, lines])

  const logsRequest = useMemo(() => createVisibilityAwareRequest(fetchLogs), [fetchLogs])

  useEffect(() => {
    setLoading(true)
    setExpanded(new Set())
    void logsRequest.run({ allowHidden: true })

    const refreshWhenVisible = () => {
      if (!document.hidden) void logsRequest.run()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      document.removeEventListener('visibilitychange', refreshWhenVisible)
      requestControllerRef.current?.abort()
    }
  }, [logsRequest])

  const polling = follow && supported !== false
  useEffect(() => {
    if (!polling) return
    const interval = window.setInterval(() => {
      if (!document.hidden) void logsRequest.run()
    }, FOLLOW_MS)
    return () => window.clearInterval(interval)
  }, [polling, logsRequest])

  const visible = useMemo(() => filterLogs(logs, query, levels), [levels, logs, query])
  const counts = useMemo(() => {
    const result: Record<LogLevel, number> = { error: 0, warn: 0, info: 0, debug: 0, other: 0 }
    for (const entry of logs) result[getLogLevel(entry.line)] += 1
    return result
  }, [logs])
  const visibleErrors = visible.filter((entry) => getLogLevel(entry.line) === 'error').length
  const visibleWarnings = visible.filter((entry) => getLogLevel(entry.line) === 'warn').length

  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const element = bodyRef.current
    if (element) element.scrollTo({ top: element.scrollHeight, behavior })
  }, [])

  useLayoutEffect(() => {
    if (atBottomRef.current) scrollToBottom('auto')
  }, [visible, scrollToBottom])

  const onScroll = () => {
    const element = bodyRef.current
    if (!element) return
    const bottom = element.scrollHeight - element.scrollTop - element.clientHeight < 28
    atBottomRef.current = bottom
    setAtBottom(bottom)
    if (bottom) setUnseen(0)
  }

  const toggle = useCallback((index: number) => {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }, [])

  const clearFilters = () => {
    setQuery('')
    setLevels([])
  }

  const copyVisible = () => {
    void navigator.clipboard
      ?.writeText(visible.map((entry) => entry.line).join('\n'))
      .then(() => toast.success(`Copied ${visible.length.toLocaleString()} lines`))
      .catch(() => toast.error('The clipboard is not available in this browser.'))
  }

  const download = () => {
    const name = logsDownloadName(component, new Date())
    const url = URL.createObjectURL(
      new Blob([visible.map((entry) => entry.line).join('\n')], { type: 'text/plain' }),
    )
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = name
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    toast.success('Logs downloaded', {
      description: `${visible.length.toLocaleString()} lines · ${name}`,
    })
  }

  const unsupported = supported === false
  const filtering = Boolean(query.trim()) || levels.length > 0

  return (
    <PageContainer width="wide">
      <PageHeader
        icon={<IconTile icon={ScrollText} tone="slate" size="lg" />}
        title="Logs"
        description="Live output from the Router, Envoy and Dashboard, read from the stack's bounded log spool."
        meta={
          <>
            <MetaItem label="Deployment">{formatDeploymentType(deploymentType)}</MetaItem>
            <MetaItem label="Access">{describeLogsAvailability(supported)}</MetaItem>
            <MetaItem label="Buffered">{logs.length.toLocaleString()} lines</MetaItem>
          </>
        }
        actions={
          <>
            <Button variant="ghost" onClick={copyVisible} disabled={!visible.length}>
              <Copy aria-hidden="true" />
              <span className="max-xl:sr-only">Copy</span>
            </Button>
            <Button variant="outline" onClick={download} disabled={!visible.length}>
              <Download aria-hidden="true" />
              <span className="max-xl:sr-only">Download</span>
            </Button>
            <Button variant="outline" onClick={() => void logsRequest.run({ allowHidden: true })}>
              <RefreshCw aria-hidden="true" className={cn(fetching && 'animate-spin')} />
              Refresh
            </Button>
          </>
        }
      />

      {error ? (
        <div role="alert" className="mb-4">
          <Alert
            tone="danger"
            title={logs.length ? 'Showing the last lines received' : 'Logs could not be read'}
          >
            {error}
          </Alert>
        </div>
      ) : null}
      {message && !error ? (
        <Alert tone={unsupported ? 'warning' : 'info'} className="mb-4">
          {message}
        </Alert>
      ) : null}

      <Card className="min-w-0 overflow-hidden">
        <section className="border-b border-border px-3 pt-3" aria-label="Log filters">
          <DataToolbar
            filters={
              <>
                <Segmented
                  aria-label="Log source"
                  value={component}
                  onValueChange={setComponent}
                  options={COMPONENTS}
                />
                <InputGroup
                  type="search"
                  aria-label="Search logs"
                  value={query}
                  placeholder="Search lines…"
                  leading={<Search aria-hidden="true" />}
                  containerClassName="w-full sm:w-64"
                  className="h-8"
                  onChange={(event) => setQuery(event.target.value)}
                />
                <FilterPill
                  label="Level"
                  selected={levels}
                  onChange={(next) => setLevels(next as LogLevel[])}
                  options={LOG_LEVELS.map((level) => ({
                    value: level,
                    label: levelLabel[level],
                    count: counts[level],
                    icon: <span className={cn('size-2 rounded-full', levelDot[level])} />,
                  }))}
                />
                <Select value={String(lines)} onValueChange={(value) => setLines(Number(value))}>
                  <SelectTrigger className="h-8 w-[9.5rem]" aria-label="Lines to read">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LINE_OPTIONS.map((option) => (
                      <SelectItem key={option} value={String(option)}>
                        Last {option.toLocaleString()} lines
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </>
            }
            actions={
              <>
                <span className="hidden text-xs text-muted-foreground tabular-nums xl:inline">
                  {visible.length.toLocaleString()}{' '}
                  {visible.length === logs.length ? 'lines' : `of ${logs.length.toLocaleString()}`}
                  {visibleErrors ? (
                    <span className="text-destructive-foreground"> · {visibleErrors} errors</span>
                  ) : null}
                  {visibleWarnings ? (
                    <span className="text-warning-foreground"> · {visibleWarnings} warnings</span>
                  ) : null}
                </span>
                <Tooltip content={wrap ? 'Truncate long lines' : 'Wrap long lines'}>
                  <Button
                    variant={wrap ? 'secondary' : 'ghost'}
                    size="icon-sm"
                    aria-pressed={wrap}
                    aria-label="Wrap lines"
                    onClick={() => setWrap((value) => !value)}
                  >
                    <WrapText />
                  </Button>
                </Tooltip>
                <span className="flex items-center gap-2">
                  <Switch id="logs-follow" checked={follow} onCheckedChange={setFollow} />
                  <Label htmlFor="logs-follow" className="text-foreground/85">
                    Follow
                  </Label>
                </span>
              </>
            }
          />
        </section>

        <div className="relative">
          <div
            ref={bodyRef}
            onScroll={onScroll}
            className={cn(
              'min-h-[360px] overflow-y-auto bg-surface-sunken py-1.5 font-mono text-[12px] leading-5',
              error || message ? 'h-[calc(100dvh-25rem)]' : 'h-[calc(100dvh-21rem)]',
            )}
            aria-label="Log lines"
            role="region"
            tabIndex={0}
          >
            {loading && logs.length === 0 ? (
              <div className="flex flex-col gap-2 px-4 py-3 font-sans" role="status">
                <span className="sr-only">Loading logs</span>
                {Array.from({ length: 12 }, (_, index) => (
                  <Skeleton
                    key={index}
                    aria-hidden="true"
                    className={cn('h-3.5', ['w-3/4', 'w-1/2', 'w-5/6', 'w-2/3'][index % 4])}
                  />
                ))}
              </div>
            ) : unsupported ? (
              <EmptyState
                compact
                icon={ScrollText}
                tone="slate"
                className="font-sans"
                title="Runtime logs unavailable"
                description="Start the local split runtime with vllm-sr serve to provision its bounded shared log spool."
              />
            ) : logs.length === 0 ? (
              <EmptyState
                compact
                icon={ScrollText}
                tone="slate"
                className="font-sans"
                title="No logs available"
                description={
                  deploymentType === 'none'
                    ? 'No running deployment detected. Start the router first.'
                    : 'The spool has no recent lines for this source. New output appears here automatically while Follow is on.'
                }
              />
            ) : visible.length === 0 ? (
              <EmptyState
                compact
                icon={SearchX}
                tone="slate"
                className="font-sans"
                title="No matching logs"
                description="Change the search or level filter."
                actions={
                  filtering ? (
                    <Button size="sm" variant="outline" onClick={clearFilters}>
                      Clear filters
                    </Button>
                  ) : undefined
                }
              />
            ) : (
              visible.map((entry, index) => (
                <LogRow
                  key={`${entry.service ?? 'service'}-${index}-${entry.line.slice(0, 32)}`}
                  entry={entry}
                  index={index}
                  query={query}
                  wrap={wrap}
                  expanded={expanded.has(index)}
                  onToggle={toggle}
                  onFilter={setQuery}
                />
              ))
            )}
          </div>

          {!atBottom && visible.length > 0 ? (
            <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
              <Button
                size="sm"
                className="pointer-events-auto rounded-full pl-3 pr-3.5 shadow-lg"
                onClick={() => {
                  atBottomRef.current = true
                  setAtBottom(true)
                  setUnseen(0)
                  scrollToBottom('smooth')
                }}
              >
                <ArrowDown aria-hidden="true" />
                Jump to latest
                {unseen ? (
                  <span className="rounded-full bg-primary-foreground/20 px-1.5 font-mono text-[11px] tabular-nums">
                    {unseen} new
                  </span>
                ) : null}
              </Button>
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border bg-surface-subtle/70 px-3 py-1.5 text-[11px] text-subtle-foreground">
          <span className="tabular-nums" aria-live="polite">
            {visible.length === logs.length
              ? `${logs.length.toLocaleString()} entries`
              : `${visible.length.toLocaleString()} of ${logs.length.toLocaleString()} entries`}
          </span>
          <span aria-hidden="true">·</span>
          <span>Last {lines.toLocaleString()} lines from the bounded spool</span>
          <span aria-hidden="true">·</span>
          <span className="inline-flex items-center gap-1.5">
            <StatusDot tone={polling ? 'success' : 'neutral'} pulse={polling && !error} />
            {polling
              ? 'Following every 5 s'
              : follow
                ? 'Follow paused until a log spool is mounted'
                : 'Follow is off'}
          </span>
          <span className="ml-auto hidden sm:inline">
            Select a line for its full text and fields
          </span>
        </div>
      </Card>
    </PageContainer>
  )
}

export default LogsPage
