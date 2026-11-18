import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowDown,
  ChevronRight,
  Copy,
  Download,
  Info,
  ListFilter,
  RotateCw,
  ScrollText,
  Search,
  SearchX,
  ShieldAlert,
  WrapText,
} from 'lucide-react'

import {
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
  LOG_SOURCE_LABEL,
  componentLabel,
  componentOptions,
  countNewLines,
  describeEmptyLogs,
  entryKey,
  filterLogs,
  formatLogTime,
  getLogLevel,
  lineSource,
  logLineText,
  logSourceOf,
  logsDownloadName,
  logsRequestFailure,
  matchRanges,
  parseStructuredLine,
  shortPodName,
  type LogEntry,
  type LogLevel,
  type LogsNotice,
  type LogsResponse,
} from './logsPageSupport'

const LINE_OPTIONS = [100, 200, 500, 1000]
const FOLLOW_MS = 5000
const MAX_SOURCE_WIDTH = 24

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

const noticeIcon: Record<LogsNotice['kind'], typeof ScrollText> = {
  unavailable: ShieldAlert,
  error: AlertTriangle,
  empty: ScrollText,
}

const noticeTone = { unavailable: 'gold', error: 'rose', empty: 'slate' } as const

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

/** The source column: a pod without its ReplicaSet hash, or a component. */
function sourceText(entry: LogEntry): string {
  return entry.pod ? shortPodName(entry.pod) : componentLabel(entry.service ?? '')
}

const LogRow = React.memo(function LogRow({
  entry,
  index,
  query,
  wrap,
  expanded,
  showTime,
  sourceWidth,
  onToggle,
  onFilter,
}: {
  entry: LogEntry
  index: number
  query: string
  wrap: boolean
  expanded: boolean
  showTime: boolean
  /** The source column's width in characters; 0 hides it. */
  sourceWidth: number
  onToggle: (index: number) => void
  onFilter: (value: string) => void
}) {
  const level = getLogLevel(entry.line)
  const fields = expanded ? parseStructuredLine(entry.line) : null
  const requestId = fields
    ? String(fields.request_id ?? fields.requestId ?? fields.x_request_id ?? '')
    : ''
  const source = lineSource(entry)
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
        {showTime ? (
          <span className="w-[8ch] shrink-0 select-none tabular-nums text-subtle-foreground/80">
            {formatLogTime(entry.timestamp) ?? '—'}
          </span>
        ) : (
          <span className="w-10 shrink-0 select-none text-right tabular-nums text-subtle-foreground/70">
            {index + 1}
          </span>
        )}
        <span
          className={cn(
            'mt-px inline-flex h-[17px] w-[52px] shrink-0 items-center justify-center rounded text-[10px] font-semibold uppercase tracking-[0.04em]',
            levelBadge[level],
          )}
        >
          {level === 'other' ? '—' : level}
        </span>
        {sourceWidth ? (
          <span
            title={source}
            style={{ width: `${sourceWidth}ch` }}
            className={cn(
              'shrink-0 truncate font-medium',
              serviceTone[entry.service ?? ''] ?? 'text-muted-foreground',
            )}
          >
            {sourceText(entry)}
          </span>
        ) : null}
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
          <Highlighted text={entry.line} query={query} />
        </span>
      </button>
      {expanded ? (
        <div className="mx-3 mb-2 ml-[4.75rem] mt-1 overflow-hidden rounded-lg border border-border bg-card font-sans">
          <div className="flex items-center gap-1 border-b border-border py-1 pl-3 pr-1 text-[11px] text-subtle-foreground">
            <span className="mr-auto min-w-0 truncate">
              {[fields ? 'Structured fields' : 'Full line', source, entry.timestamp]
                .filter(Boolean)
                .join(' · ')}
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
  const [component, setComponent] = useState('all')
  const [response, setResponse] = useState<LogsResponse | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
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
    try {
      const result = await fetch(
        `/api/logs?component=${encodeURIComponent(component)}&lines=${lines}`,
        { signal: controller.signal },
      )
      if (!result.ok) {
        throw new Error(logsRequestFailure(result.status, await result.json().catch(() => null)))
      }
      const data = (await result.json()) as LogsResponse
      if (controller.signal.aborted) return
      const next = data.logs ?? []
      if (!atBottomRef.current) setUnseen((count) => count + countNewLines(logsRef.current, next))
      logsRef.current = next
      setResponse({ ...data, logs: next })
      setRequestError(null)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setRequestError(err instanceof Error ? err.message : 'The log request failed.')
    } finally {
      if (requestControllerRef.current === controller) setLoading(false)
    }
  }, [component, lines])

  const logsRequest = useMemo(() => createVisibilityAwareRequest(fetchLogs), [fetchLogs])

  useEffect(() => {
    logsRef.current = []
    setResponse((current) => (current ? { ...current, logs: [], count: 0 } : current))
    setLoading(true)
    setUnseen(0)
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

  const unsupported = response?.supported === false
  const polling = follow && !unsupported
  useEffect(() => {
    if (!polling) return
    const interval = window.setInterval(() => {
      if (!document.hidden) void logsRequest.run()
    }, FOLLOW_MS)
    return () => window.clearInterval(interval)
  }, [polling, logsRequest])

  const components = response?.components
  useEffect(() => {
    if (components && component !== 'all' && !components.includes(component)) setComponent('all')
  }, [component, components])

  const logs = useMemo(() => response?.logs ?? [], [response])
  const source = logSourceOf(response?.deployment_type)
  const visible = useMemo(() => filterLogs(logs, query, levels), [levels, logs, query])
  const counts = useMemo(() => {
    const result: Record<LogLevel, number> = { error: 0, warn: 0, info: 0, debug: 0, other: 0 }
    for (const entry of logs) result[getLogLevel(entry.line)] += 1
    return result
  }, [logs])
  const visibleErrors = visible.filter((entry) => getLogLevel(entry.line) === 'error').length
  const visibleWarnings = visible.filter((entry) => getLogLevel(entry.line) === 'warn').length
  const showTime = logs.some((entry) => entry.timestamp)
  const mixed = source === 'kubernetes' || component === 'all'
  const sourceWidth = useMemo(() => {
    if (!mixed || !logs.length) return 0
    const longest = Math.max(...logs.map((entry) => sourceText(entry).length))
    return Math.min(Math.max(longest, 6), MAX_SOURCE_WIDTH)
  }, [logs, mixed])

  const emptyNotice = describeEmptyLogs({ response, requestError, following: polling })
  const inlineNotice: { tone: 'danger' | 'info'; text: string } | null = logs.length
    ? requestError
      ? {
          tone: 'danger',
          text: `Showing the last lines received. ${requestError}${polling ? ' Follow retries every 5 seconds.' : ''}`,
        }
      : response?.error
        ? { tone: 'danger', text: response.error }
        : response?.message
          ? { tone: 'info', text: response.message }
          : null
    : null

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

  const visibleText = () => visible.map((entry) => logLineText(entry, mixed)).join('\n')

  const copyVisible = () => {
    void navigator.clipboard
      ?.writeText(visibleText())
      .then(() => toast.success(`Copied ${visible.length.toLocaleString()} lines`))
      .catch(() => toast.error('The clipboard is not available in this browser.'))
  }

  const download = () => {
    const name = logsDownloadName(component, new Date())
    const url = URL.createObjectURL(new Blob([visibleText()], { type: 'text/plain' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = name
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    toast.success('Logs downloaded', {
      description: `${visible.length.toLocaleString()} lines · ${name}`,
    })
  }

  const filtering = Boolean(query.trim()) || levels.length > 0
  const NoticeIcon = emptyNotice ? noticeIcon[emptyNotice.kind] : ScrollText

  return (
    <PageContainer width="wide">
      <PageHeader
        icon={<IconTile icon={ScrollText} tone="slate" size="lg" />}
        title="Logs"
        description="Recent output of the Router and its services, redacted before it reaches the browser."
        meta={
          <>
            <MetaItem label="Source">
              {source ? LOG_SOURCE_LABEL[source] : 'Detecting'}
              {source === 'kubernetes' && response?.namespace ? ` · ${response.namespace}` : ''}
            </MetaItem>
            {source === 'kubernetes' && !unsupported ? (
              <MetaItem label="Pods">{(response?.pods?.length ?? 0).toLocaleString()}</MetaItem>
            ) : null}
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
          </>
        }
      />

      <Card className="h-[calc(100dvh-15.5rem)] min-h-[460px] min-w-0 overflow-hidden">
        <section className="border-b border-border px-3 pt-3" aria-label="Log filters">
          <DataToolbar
            filters={
              <>
                {components ? (
                  <Segmented
                    aria-label="Component"
                    value={component}
                    onValueChange={setComponent}
                    options={componentOptions(components)}
                  />
                ) : (
                  <Skeleton className="h-8 w-52" />
                )}
                <InputGroup
                  type="search"
                  aria-label="Search logs"
                  value={query}
                  placeholder={source === 'kubernetes' ? 'Search lines and pods…' : 'Search lines…'}
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
                  <Switch
                    id="logs-follow"
                    checked={follow && !unsupported}
                    disabled={unsupported}
                    onCheckedChange={setFollow}
                  />
                  <Label
                    htmlFor="logs-follow"
                    className="inline-flex items-center gap-1.5 text-foreground/85"
                  >
                    {polling ? (
                      <StatusDot
                        tone={requestError ? 'warning' : 'success'}
                        pulse={!requestError}
                      />
                    ) : null}
                    Follow
                  </Label>
                </span>
              </>
            }
          />
        </section>

        {inlineNotice ? (
          <div
            role={inlineNotice.tone === 'danger' ? 'alert' : 'status'}
            className={cn(
              'flex items-start gap-2 border-b border-border px-3 py-2 text-xs leading-5',
              inlineNotice.tone === 'danger'
                ? 'bg-destructive-soft/50 text-destructive-foreground'
                : 'bg-info-soft/50 text-foreground/85',
            )}
          >
            {inlineNotice.tone === 'danger' ? (
              <AlertTriangle aria-hidden="true" className="mt-[3px] size-3.5 shrink-0" />
            ) : (
              <Info aria-hidden="true" className="mt-[3px] size-3.5 shrink-0 text-info" />
            )}
            <span className="min-w-0">{inlineNotice.text}</span>
          </div>
        ) : null}

        <div className="relative min-h-0 flex-1">
          <div
            ref={bodyRef}
            onScroll={onScroll}
            className="absolute inset-0 overflow-y-auto bg-surface-sunken py-1.5 font-mono text-[12px] leading-5"
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
            ) : emptyNotice ? (
              <EmptyState
                compact
                icon={NoticeIcon}
                tone={noticeTone[emptyNotice.kind]}
                className="mx-auto h-full max-w-xl px-6 font-sans"
                title={emptyNotice.title}
                description={emptyNotice.description}
                actions={
                  emptyNotice.kind === 'empty' ? undefined : (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void logsRequest.run({ allowHidden: true })}
                    >
                      <RotateCw aria-hidden="true" />
                      Check again
                    </Button>
                  )
                }
              />
            ) : visible.length === 0 ? (
              <EmptyState
                compact
                icon={SearchX}
                tone="slate"
                className="h-full font-sans"
                title="No matching lines"
                description="Change the search or the level filter."
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
                  key={`${index}-${entryKey(entry)}`}
                  entry={entry}
                  index={index}
                  query={query}
                  wrap={wrap}
                  expanded={expanded.has(index)}
                  showTime={showTime}
                  sourceWidth={sourceWidth}
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
              ? `${logs.length.toLocaleString()} lines`
              : `${visible.length.toLocaleString()} of ${logs.length.toLocaleString()} lines`}
            {visibleErrors ? (
              <span className="text-destructive-foreground"> · {visibleErrors} errors</span>
            ) : null}
            {visibleWarnings ? (
              <span className="text-warning-foreground"> · {visibleWarnings} warnings</span>
            ) : null}
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
