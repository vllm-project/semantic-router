import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ArrowUpRight, ChartLine, RefreshCw } from 'lucide-react'

import ServiceNotConfigured, { type ServiceConfig } from '../components/ServiceNotConfigured'
import { useTheme } from '../shell/theme'
import { Alert, Button, IconTile, PageContainer, PageHeader, Segmented, Skeleton, cn } from '../ui'
import { DOCS_LINKS } from '../utils/docsLinks'
import { DecisionBreakdown, ModelBreakdown } from './MetricsBreakdowns'
import { MetricsHealth } from './MetricsHealth'
import { MetricsOverview } from './MetricsOverview'
import { formatClock } from './metricsFormat'
import { scrapeTargets } from './metricsOverview'
import {
  DEFAULT_METRICS_RANGE,
  METRICS_RANGES,
  ROUTER_JOB,
  grafanaDashboardUrl,
  readMetricsRange,
  type MetricsRange,
} from './metricsQueries'
import { sampleTimes } from './metricsSeries'
import { dashboardFailure, useGrafanaAvailable, useMetricsDashboard } from './useMetricsDashboard'

const PROMETHEUS_SERVICE: ServiceConfig = {
  name: 'Prometheus',
  envVar: 'TARGET_PROMETHEUS_URL',
  description: "Connect Prometheus to chart the Router's traffic, latency, errors and spend.",
  docsUrl: DOCS_LINKS.observability,
  exampleValue: 'http://localhost:9090',
}

function MetricsSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-hidden="true">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton key={index} className="h-[132px] rounded-xl" />
        ))}
      </div>
      <Skeleton className="h-[356px] rounded-xl" />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Skeleton className="h-[316px] rounded-xl" />
        <Skeleton className="h-[316px] rounded-xl" />
      </div>
    </div>
  )
}

export default function MonitoringPage() {
  const { resolved } = useTheme()
  const [params, setParams] = useSearchParams()
  const range = readMetricsRange(params.get('range'))
  const dashboard = useMetricsDashboard(range)
  const grafana = useGrafanaAvailable()
  const { data, refresh } = dashboard
  const failure = data ? dashboardFailure(data) : null
  const times = useMemo(() => (data ? sampleTimes(data.window) : []), [data])
  const targets = useMemo(() => scrapeTargets(data?.series.up.data ?? []), [data])

  const setRange = (value: MetricsRange) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (value === DEFAULT_METRICS_RANGE) next.delete('range')
        else next.set('range', value)
        return next
      },
      { replace: true },
    )

  const retry = (
    <Button type="button" variant="outline" size="sm" onClick={refresh}>
      Try again
    </Button>
  )

  let body
  if (!data) {
    body = (
      <>
        <span role="status" className="sr-only">
          Loading metrics
        </span>
        <MetricsSkeleton />
      </>
    )
  } else if (failure?.failure === 'unconfigured') {
    body = <ServiceNotConfigured service={PROMETHEUS_SERVICE} onRetry={refresh} />
  } else if (failure) {
    body = (
      <div role="alert">
        <Alert tone="danger" title="Metrics could not load" actions={retry}>
          {failure.message}
        </Alert>
      </div>
    )
  } else {
    const down = targets.filter((target) => target.up === false)
    body = (
      <div className="flex flex-col gap-6">
        {dashboard.refreshError ? (
          <div role="alert">
            <Alert
              tone="warning"
              title={`Showing the metrics from ${dashboard.updatedAt ? formatClock(dashboard.updatedAt) : 'the last refresh'}`}
              actions={retry}
            >
              {dashboard.refreshError.message}
            </Alert>
          </div>
        ) : null}
        {data.series.up.data?.length === 0 ? (
          <Alert tone="warning" title="Prometheus does not scrape the Router">
            Prometheus answered, but it has no target in the{' '}
            <code className="font-mono">{ROUTER_JOB}</code> job, so there is nothing to chart. Add
            the Router&apos;s metrics endpoint (port 9190) to its scrape configuration under that
            job name.
          </Alert>
        ) : down.length > 0 && down.length === targets.length ? (
          <Alert tone="warning" title="Prometheus cannot read the Router's metrics">
            The last scrape of {down.map((target) => target.instance).join(', ')} failed, so the
            newest samples are missing. Check that the Router is running and its metrics port is
            reachable from Prometheus.
          </Alert>
        ) : null}
        <MetricsOverview data={data} times={times} range={range} />
        <ModelBreakdown data={data} times={times} range={range} />
        <DecisionBreakdown data={data} times={times} range={range} />
        <MetricsHealth data={data} times={times} range={range} />
      </div>
    )
  }

  return (
    <PageContainer width="wide">
      <PageHeader
        icon={<IconTile icon={ChartLine} tone="blue" size="lg" />}
        title="Metrics"
        description="Traffic, latency, errors and spend from the Router's metrics in Prometheus."
        meta={
          data && !failure && dashboard.updatedAt ? (
            <span className="tabular">Updated {formatClock(dashboard.updatedAt)}</span>
          ) : null
        }
        actions={
          <>
            {failure?.failure === 'unconfigured' ? null : (
              <Segmented
                aria-label="Time range"
                value={range}
                onValueChange={setRange}
                options={METRICS_RANGES.map((value) => ({ value, label: value }))}
              />
            )}
            {grafana ? (
              <Button variant="ghost" size="sm" asChild>
                <a
                  href={grafanaDashboardUrl(range, resolved)}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Open Grafana
                  <ArrowUpRight />
                </a>
              </Button>
            ) : null}
            {failure?.failure === 'unconfigured' ? null : (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={refresh}
                disabled={dashboard.loading}
              >
                <RefreshCw aria-hidden="true" className={cn(dashboard.loading && 'animate-spin')} />
                Refresh
              </Button>
            )}
          </>
        }
      />
      {body}
    </PageContainer>
  )
}
