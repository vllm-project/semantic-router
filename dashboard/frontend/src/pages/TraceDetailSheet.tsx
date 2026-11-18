import { Fragment, useMemo, useState } from 'react'
import { ChartGantt } from 'lucide-react'

import ProductLoadingState from '../components/ProductLoadingState'
import {
  Alert,
  Badge,
  Button,
  CopyButton,
  DetailSheet,
  DetailSheetBody,
  DetailSheetHeader,
  IconTile,
  KeyValueGrid,
  SectionLabel,
  useRetainedValue,
} from '../ui'
import { TraceWaterfall } from './TraceWaterfall'
import {
  formatMicros,
  serviceColor,
  summarizeTrace,
  type Trace,
  type TraceSpan,
} from './jaegerTraces'
import { formatMoment } from './metricsFormat'
import type { ObservabilityError } from './observabilityProxy'
import { layoutWaterfall } from './traceWaterfall'

const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`

function SpanDetails({
  span,
  traceStart,
  color,
}: {
  span: TraceSpan
  traceStart: number
  color: string
}) {
  return (
    <section
      aria-label="Span details"
      className="max-h-[46%] shrink-0 overflow-y-auto border-t border-border bg-surface-subtle/40 px-5 py-4"
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="h-4 w-1 shrink-0 rounded-full" style={{ background: color }} />
        <h3 className="truncate text-sm font-semibold" title={span.operationName}>
          {span.operationName}
        </h3>
        {span.error ? (
          <Badge tone="danger" size="sm">
            Error
          </Badge>
        ) : null}
      </div>
      <KeyValueGrid
        columns={3}
        className="mt-3"
        items={[
          { label: 'Service', value: span.serviceName },
          { label: 'Duration', value: formatMicros(span.duration) },
          {
            label: 'Starts',
            value: `${formatMicros(span.startTime - traceStart)} into the trace`,
          },
          { label: 'Kind', value: span.kind ?? '—' },
          {
            label: 'Span ID',
            value: (
              <span className="flex min-w-0 items-center gap-1">
                <span className="truncate font-mono text-[12px]">{span.spanID}</span>
                <CopyButton value={span.spanID} label="Copy span ID" className="size-6" />
              </span>
            ),
          },
        ]}
      />
      <SectionLabel className="mt-5">Tags</SectionLabel>
      {span.tags.length ? (
        <dl className="mt-2 grid grid-cols-[minmax(8rem,32%)_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12px]">
          {span.tags.map((tag, index) => (
            <Fragment key={index}>
              <dt className="truncate font-mono text-muted-foreground" title={tag.key}>
                {tag.key}
              </dt>
              <dd className="min-w-0 whitespace-pre-wrap break-all font-mono">{tag.value}</dd>
            </Fragment>
          ))}
        </dl>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">This span has no tags.</p>
      )}
      <SectionLabel className="mt-5">Logs</SectionLabel>
      {span.logs.length ? (
        <ol className="mt-2 flex flex-col gap-2">
          {span.logs.map((log, index) => {
            const event = log.fields.find((field) => field.key === 'event')?.value
            const fields = log.fields.filter((field) => field.key !== 'event')
            return (
              <li key={index} className="rounded-lg border border-border bg-card px-3 py-2">
                <div className="flex items-baseline gap-2 text-[12px]">
                  <span className="text-muted-foreground tabular">
                    {formatMicros(log.timestamp - traceStart)}
                  </span>
                  <span className="min-w-0 truncate font-medium">{event || 'Log'}</span>
                </div>
                {fields.length ? (
                  <dl className="mt-1 grid grid-cols-[minmax(6rem,28%)_minmax(0,1fr)] gap-x-3 gap-y-1 font-mono text-[11.5px]">
                    {fields.map((field, fieldIndex) => (
                      <Fragment key={fieldIndex}>
                        <dt className="truncate text-muted-foreground" title={field.key}>
                          {field.key}
                        </dt>
                        <dd className="min-w-0 whitespace-pre-wrap break-all">{field.value}</dd>
                      </Fragment>
                    ))}
                  </dl>
                ) : null}
              </li>
            )
          })}
        </ol>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">This span has no logs.</p>
      )}
    </section>
  )
}

function TraceDetail({ trace, palette }: { trace: Trace; palette: readonly string[] }) {
  const summary = useMemo(() => summarizeTrace(trace), [trace])
  const waterfall = useMemo(() => layoutWaterfall(trace), [trace])
  const colors = useMemo(
    () => [
      ...palette,
      ...summary.services.map((service) => service.name).filter((name) => !palette.includes(name)),
    ],
    [palette, summary],
  )
  // By ID: the trace from the search is replaced by the one fetched on its own.
  const [selectedID, setSelectedID] = useState<string | null>(null)
  const selected =
    waterfall.rows.find((row) => row.span.spanID === selectedID)?.span ??
    waterfall.rows[0]?.span ??
    null

  return (
    <>
      <DetailSheetHeader
        icon={<IconTile icon={ChartGantt} tone="violet" />}
        title={summary.rootOperation || 'Trace'}
        badges={
          summary.errorCount ? (
            <Badge tone="danger" size="sm">
              {plural(summary.errorCount, 'error')}
            </Badge>
          ) : null
        }
        description={
          <>
            <span>{formatMicros(summary.duration)}</span>
            <span aria-hidden="true">·</span>
            <span>{plural(summary.spanCount, 'span')}</span>
            <span aria-hidden="true">·</span>
            <span>{formatMoment(summary.start / 1_000)}</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono">{summary.traceID}</span>
          </>
        }
        actions={<CopyButton value={summary.traceID} label="Copy trace ID" />}
      />
      <ul
        aria-label="Services"
        className="flex shrink-0 flex-wrap gap-1.5 border-b border-border px-5 py-2.5"
      >
        {summary.services.map((service) => (
          <li
            key={service.name}
            className="inline-flex h-6 max-w-[16rem] items-center gap-1.5 rounded-md border border-border bg-surface-subtle px-2 text-xs"
          >
            <span
              className="size-2 shrink-0 rounded-full"
              style={{ background: serviceColor(colors, service.name) }}
            />
            <span className="truncate">{service.name}</span>
            <span className="text-muted-foreground tabular">{service.spans}</span>
          </li>
        ))}
      </ul>
      <DetailSheetBody className="flex flex-col overflow-hidden">
        <TraceWaterfall
          waterfall={waterfall}
          palette={colors}
          selected={selected}
          onSelect={(span) => setSelectedID(span.spanID)}
          className="min-h-0 flex-1"
        />
        {selected ? (
          <SpanDetails
            span={selected}
            traceStart={summary.start}
            color={serviceColor(colors, selected.serviceName)}
          />
        ) : null}
      </DetailSheetBody>
    </>
  )
}

export function TraceDetailSheet({
  traceID,
  trace,
  error,
  palette,
  onClose,
  onRetry,
}: {
  traceID: string | null
  trace: Trace | null
  error: ObservabilityError | null
  palette: readonly string[]
  onClose: () => void
  onRetry: () => void
}) {
  // A closing sheet keeps showing its trace while it animates out.
  const retained = useRetainedValue(trace)
  const shown = retained && (!traceID || retained.traceID === traceID) ? retained : null
  return (
    <DetailSheet
      open={Boolean(traceID)}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="xl"
    >
      {shown ? (
        <TraceDetail key={shown.traceID} trace={shown} palette={palette} />
      ) : (
        <>
          <DetailSheetHeader
            icon={<IconTile icon={ChartGantt} tone="violet" />}
            title={error ? 'Trace could not load' : 'Loading trace'}
            description={<span className="font-mono">{traceID}</span>}
          />
          <DetailSheetBody className="p-5">
            {error ? (
              <Alert
                tone="danger"
                title="Jaeger did not return this trace"
                actions={
                  error.status === 404 ? null : (
                    <Button type="button" variant="outline" size="sm" onClick={onRetry}>
                      Try again
                    </Button>
                  )
                }
              >
                {error.status === 404
                  ? 'Jaeger has no trace with this ID. It may have expired from its storage.'
                  : error.message}
              </Alert>
            ) : (
              <ProductLoadingState label="Loading trace" compact />
            )}
          </DetailSheetBody>
        </>
      )}
    </DetailSheet>
  )
}
