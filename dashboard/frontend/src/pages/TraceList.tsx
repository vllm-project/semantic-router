import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { CircleAlert } from 'lucide-react'

import { Badge, DataTable, type Column } from '../ui'
import { formatAgo, formatMicros, serviceColor, type TraceSummary } from './jaegerTraces'
import { formatMoment } from './metricsFormat'

const SHOWN_SERVICES = 3

function useNow(intervalMs = 30_000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}

/** Search results, newest first: each trace's root span, services, size, duration and errors. */
export function TraceList({
  traces,
  palette,
  loading,
  activeTraceID,
  onOpen,
  empty,
}: {
  traces: TraceSummary[]
  palette: readonly string[]
  loading: boolean
  activeTraceID: string | null
  onOpen: (traceID: string) => void
  empty: ReactNode
}) {
  const now = useNow()
  const longest = traces.reduce((max, trace) => Math.max(max, trace.duration), 1)

  const columns = useMemo<Column<TraceSummary>[]>(
    () => [
      {
        id: 'trace',
        header: 'Trace',
        sortValue: (row) => row.rootOperation,
        cell: (row) => (
          <span className="flex min-w-0 max-w-[24rem] flex-col gap-0.5">
            <span className="flex min-w-0 items-center gap-1.5">
              {row.errorCount ? (
                <CircleAlert
                  className="size-3.5 shrink-0 text-destructive"
                  aria-label="Has errors"
                />
              ) : null}
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  onOpen(row.traceID)
                }}
                title={row.rootOperation}
                className="truncate rounded-sm text-left font-medium text-foreground underline-offset-4 outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/40"
              >
                {row.rootOperation}
              </button>
            </span>
            <span className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate">{row.rootService}</span>
              <span aria-hidden="true">·</span>
              <span className="font-mono" title={row.traceID}>
                {row.traceID.slice(0, 8)}
              </span>
            </span>
          </span>
        ),
      },
      {
        id: 'services',
        header: 'Services',
        hideBelow: 'lg',
        sortValue: (row) => row.services.length,
        cell: (row) => (
          <span className="flex items-center gap-1">
            {row.services.slice(0, SHOWN_SERVICES).map((service) => (
              <span
                key={service.name}
                className="inline-flex h-5 min-w-0 max-w-[8.5rem] items-center gap-1 rounded-md border border-border bg-surface-subtle px-1.5 text-[11.5px]"
              >
                <span
                  className="size-1.5 shrink-0 rounded-full"
                  style={{ background: serviceColor(palette, service.name) }}
                />
                <span className="truncate" title={service.name}>
                  {service.name}
                </span>
                <span className="text-muted-foreground tabular">{service.spans}</span>
              </span>
            ))}
            {row.services.length > SHOWN_SERVICES ? (
              <span className="text-xs text-muted-foreground">
                +{row.services.length - SHOWN_SERVICES}
              </span>
            ) : null}
          </span>
        ),
      },
      {
        id: 'spans',
        header: 'Spans',
        align: 'right',
        hideBelow: 'md',
        sortValue: (row) => row.spanCount,
        cell: (row) => <span className="tabular">{row.spanCount}</span>,
      },
      {
        id: 'duration',
        header: 'Duration',
        sortValue: (row) => row.duration,
        cell: (row) => (
          <span className="flex items-center gap-2.5">
            <span className="relative h-1.5 w-24 shrink-0 overflow-hidden rounded-full bg-muted">
              <span
                className="absolute inset-y-0 left-0 rounded-full"
                style={{
                  width: `${Math.max(2, (row.duration / longest) * 100)}%`,
                  background: row.errorCount
                    ? 'var(--destructive)'
                    : serviceColor(palette, row.rootService),
                }}
              />
            </span>
            <span className="w-16 text-right tabular">{formatMicros(row.duration)}</span>
          </span>
        ),
      },
      {
        id: 'errors',
        header: 'Errors',
        align: 'right',
        sortValue: (row) => row.errorCount,
        cell: (row) =>
          row.errorCount ? (
            <Badge tone="danger" size="sm">
              {row.errorCount}
            </Badge>
          ) : (
            <span className="text-subtle-foreground">—</span>
          ),
      },
      {
        id: 'started',
        header: 'Started',
        align: 'right',
        sortValue: (row) => row.start,
        cell: (row) => (
          <span className="text-muted-foreground tabular" title={formatMoment(row.start / 1_000)}>
            {formatAgo(row.start / 1_000, now)}
          </span>
        ),
      },
    ],
    [longest, now, onOpen, palette],
  )

  return (
    <DataTable
      data={traces}
      columns={columns}
      getRowId={(row) => row.traceID}
      onRowClick={(row) => onOpen(row.traceID)}
      activeRowId={activeTraceID ?? undefined}
      loading={loading}
      empty={empty}
    />
  )
}
