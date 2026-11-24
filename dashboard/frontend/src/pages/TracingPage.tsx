import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ChartGantt, RefreshCw, SearchX, Timer } from 'lucide-react'

import ServiceNotConfigured, { type ServiceConfig } from '../components/ServiceNotConfigured'
import {
  Alert,
  Badge,
  Button,
  Card,
  DataToolbar,
  EmptyState,
  IconTile,
  InputGroup,
  MetaItem,
  PageContainer,
  PageHeader,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  cn,
} from '../ui'
import { DOCS_LINKS } from '../utils/docsLinks'
import { TraceDetailSheet } from './TraceDetailSheet'
import { TraceList } from './TraceList'
import {
  DEFAULT_TRACE_LIMIT,
  DEFAULT_TRACE_LOOKBACK,
  TRACE_LIMITS,
  TRACE_LOOKBACKS,
  jaegerOperationsUrl,
  jaegerServicesUrl,
  jaegerTraceUrl,
  normalizeTraceDuration,
  readJaegerNames,
  readJaegerTraces,
  readTraceLimit,
  readTraceLookback,
  readTraceSummaries,
  traceSummariesUrl,
} from './jaegerTraces'
import { useJaegerRead, useRouterTracingConfig } from './useJaegerTraces'

/** Longer than the 20 s the Dashboard gives Jaeger, so its own answer arrives first. */
const TRACE_SEARCH_TIMEOUT_MS = 25_000
/** One trace can carry thousands of spans with their tags and logs. */
const TRACE_TIMEOUT_MS = 45_000

const JAEGER_SERVICE: ServiceConfig = {
  name: 'Jaeger',
  envVar: 'TARGET_JAEGER_URL',
  description: 'Connect Jaeger to follow requests through the router and the model backends.',
  docsUrl: DOCS_LINKS.observability,
  exampleValue: 'http://localhost:16686',
}

/** Radix Select items cannot carry an empty value. */
const ALL_OPERATIONS = '__all__'

/** Jaeger reports its own query service, whose traces are not requests. */
const isJaegerService = (name: string) => /^jaeger/i.test(name)

function defaultService(services: readonly string[], routerService?: string): string {
  if (routerService && services.includes(routerService)) return routerService
  return services.find((name) => !isJaegerService(name)) ?? services[0] ?? ''
}

const withCurrent = (options: readonly string[], current: string) =>
  current && !options.includes(current) ? [current, ...options] : options

export default function TracingPage() {
  const [params, setParams] = useSearchParams()
  const tracing = useRouterTracingConfig()
  const routerService = tracing?.resource?.service_name?.trim() || undefined
  const [refreshKey, setRefreshKey] = useState(0)
  const refresh = useCallback(() => setRefreshKey((key) => key + 1), [])

  const services = useJaegerRead(jaegerServicesUrl(), readJaegerNames, refreshKey)
  const serviceNames = useMemo(() => services.data ?? [], [services.data])
  // Waiting for the router config keeps a first search from going to the wrong service.
  const fallbackService = tracing ? defaultService(serviceNames, routerService) : ''
  const service = params.get('service') || fallbackService
  const operation = params.get('operation') ?? ''
  const lookback = readTraceLookback(params.get('lookback'))
  const limit = readTraceLimit(params.get('limit'))
  const minDuration = normalizeTraceDuration(params.get('minDuration') ?? '') ?? ''
  const [durationDraft, setDurationDraft] = useState(minDuration)
  useEffect(() => setDurationDraft(minDuration), [minDuration])
  const draftDuration = normalizeTraceDuration(durationDraft)

  const operations = useJaegerRead(
    service ? jaegerOperationsUrl(service) : null,
    readJaegerNames,
    refreshKey,
  )
  const traces = useJaegerRead(
    service ? traceSummariesUrl({ service, operation, lookback, limit, minDuration }) : null,
    readTraceSummaries,
    refreshKey,
    TRACE_SEARCH_TIMEOUT_MS,
  )
  const summaries = useMemo(() => traces.data ?? [], [traces.data])

  const traceID = params.get('trace')
  const lookup = useJaegerRead(
    traceID ? jaegerTraceUrl(traceID) : null,
    readJaegerTraces,
    refreshKey,
    TRACE_TIMEOUT_MS,
  )
  const openTrace = lookup.data?.find((trace) => trace.traceID === traceID) ?? null

  const update = useCallback(
    (changes: Record<string, string | null>) =>
      setParams(
        (current) => {
          const next = new URLSearchParams(current)
          for (const [key, value] of Object.entries(changes)) {
            if (value) next.set(key, value)
            else next.delete(key)
          }
          return next
        },
        { replace: true },
      ),
    [setParams],
  )
  const openTraceID = useCallback((id: string) => update({ trace: id }), [update])
  const commitDuration = () => {
    if (draftDuration !== null) update({ minDuration: draftDuration || null })
  }

  const lookbackLabel =
    TRACE_LOOKBACKS.find((option) => option.value === lookback)?.label.toLowerCase() ?? ''
  const retry = (
    <Button type="button" variant="outline" size="sm" onClick={refresh}>
      Try again
    </Button>
  )

  let body
  if (services.error?.failure === 'unconfigured') {
    body = <ServiceNotConfigured service={JAEGER_SERVICE} onRetry={refresh} />
  } else if (services.error) {
    body = (
      <div role="alert">
        <Alert tone="danger" title="Traces could not load" actions={retry}>
          {services.error.message}
        </Alert>
      </div>
    )
  } else if (services.data?.length === 0) {
    body = (
      <Card className="border-dashed shadow-none">
        <EmptyState
          icon={ChartGantt}
          tone="violet"
          title="Jaeger has no traces yet"
          description={
            <>
              Jaeger is connected but has not received a span. Turn tracing on in the router config
              under{' '}
              <code className="font-mono text-[12px]">global.services.observability.tracing</code>,
              then send a request through the gateway.
            </>
          }
          actions={
            <Button variant="outline" onClick={refresh}>
              <RefreshCw />
              Check again
            </Button>
          }
        />
      </Card>
    )
  } else {
    body = (
      <>
        <section aria-label="Trace search">
          <DataToolbar
            filters={
              <>
                <Select
                  value={service}
                  onValueChange={(value) =>
                    update({ service: value === fallbackService ? null : value, operation: null })
                  }
                  disabled={!serviceNames.length}
                >
                  <SelectTrigger size="sm" className="w-[13rem]" aria-label="Service">
                    <SelectValue placeholder="Service" />
                  </SelectTrigger>
                  <SelectContent>
                    {withCurrent(serviceNames, service).map((name) => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select
                  value={operation || ALL_OPERATIONS}
                  onValueChange={(value) =>
                    update({ operation: value === ALL_OPERATIONS ? null : value })
                  }
                  disabled={!service}
                >
                  <SelectTrigger size="sm" className="w-[15rem]" aria-label="Operation">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={ALL_OPERATIONS}>All operations</SelectItem>
                    {withCurrent(operations.data ?? [], operation).map((name) => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select
                  value={lookback}
                  onValueChange={(value) =>
                    update({ lookback: value === DEFAULT_TRACE_LOOKBACK ? null : value })
                  }
                >
                  <SelectTrigger size="sm" className="w-[10.5rem]" aria-label="Time range">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TRACE_LOOKBACKS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <InputGroup
                  aria-label="Minimum duration"
                  placeholder="Min duration"
                  value={durationDraft}
                  leading={<Timer aria-hidden="true" />}
                  aria-invalid={draftDuration === null}
                  containerClassName="w-[9.5rem]"
                  className="h-8"
                  onChange={(event) => setDurationDraft(event.target.value)}
                  onBlur={commitDuration}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') commitDuration()
                  }}
                />
                <Select
                  value={String(limit)}
                  onValueChange={(value) =>
                    update({ limit: Number(value) === DEFAULT_TRACE_LIMIT ? null : value })
                  }
                >
                  <SelectTrigger size="sm" className="w-[8.5rem]" aria-label="Traces to show">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TRACE_LIMITS.map((option) => (
                      <SelectItem key={option} value={String(option)}>
                        {option} traces
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </>
            }
            actions={
              draftDuration === null ? (
                <span className="text-xs text-destructive-foreground" role="status">
                  Use a duration such as 250ms or 1.5s
                </span>
              ) : traces.data ? (
                <span className="text-xs text-muted-foreground tabular">
                  {summaries.length} {summaries.length === 1 ? 'trace' : 'traces'}
                </span>
              ) : null
            }
          />
        </section>
        {traces.error ? (
          <div role="alert">
            <Alert tone="danger" title="The trace search failed" actions={retry}>
              {traces.error.message}
            </Alert>
          </div>
        ) : (
          <TraceList
            traces={summaries}
            palette={serviceNames}
            loading={services.loading || !service || (traces.loading && !traces.data)}
            activeTraceID={traceID}
            onOpen={openTraceID}
            empty={
              <EmptyState
                compact
                icon={SearchX}
                tone="violet"
                title="No traces match"
                description={`Jaeger has no ${service} traces${operation ? ` for ${operation}` : ''} in the ${lookbackLabel}${minDuration ? ` longer than ${minDuration}` : ''}. Widen the time range or lower the minimum duration.`}
              />
            }
          />
        )}
      </>
    )
  }

  return (
    <PageContainer width="wide">
      <PageHeader
        icon={<IconTile icon={ChartGantt} tone="violet" size="lg" />}
        title="Traces"
        description="Follow each request through Envoy, the Router and the model backends, span by span, from Jaeger."
        meta={
          routerService || tracing?.enabled === false ? (
            <>
              {routerService ? (
                <MetaItem label="Router service">
                  <code className="font-mono">{routerService}</code>
                </MetaItem>
              ) : null}
              {tracing?.enabled === false ? (
                <Badge tone="warning" size="sm">
                  Tracing is off in the router config
                </Badge>
              ) : null}
            </>
          ) : null
        }
        actions={
          services.error?.failure === 'unconfigured' ? null : (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={refresh}
              disabled={services.loading || traces.loading}
            >
              <RefreshCw
                aria-hidden="true"
                className={cn((services.loading || traces.loading) && 'animate-spin')}
              />
              Refresh
            </Button>
          )
        }
      />
      {body}
      <TraceDetailSheet
        traceID={traceID}
        trace={openTrace}
        error={openTrace ? null : lookup.error}
        palette={serviceNames}
        onClose={() => update({ trace: null })}
        onRetry={refresh}
      />
    </PageContainer>
  )
}
