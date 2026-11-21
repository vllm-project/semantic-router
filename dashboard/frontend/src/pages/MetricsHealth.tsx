import { useMemo } from 'react'
import { OctagonAlert, Unplug } from 'lucide-react'

import { Card, CardDescription, CardHeader, CardTitle, StatusDot, cn } from '../ui'
import { ChartNotice, MetricsChartCard, MetricsTimeChart } from './MetricsTimeChart'
import { formatClock, formatCount, formatMoment, formatPercent } from './metricsFormat'
import {
  availabilityBuckets,
  cacheRateSeries,
  cacheSummary,
  scrapeTargets,
  type AvailabilityState,
  type ScrapeTarget,
} from './metricsOverview'
import { METRICS_RANGE_LABEL, ROUTER_JOB, type MetricsRange } from './metricsQueries'
import { hasSamples } from './metricsSeries'
import type { MetricsDashboardData } from './useMetricsDashboard'

const STATE_LABEL: Record<AvailabilityState, string> = {
  up: 'every scrape succeeded',
  partial: 'some scrapes failed',
  down: 'every scrape failed',
  none: 'no scrape recorded',
}

const STATE_CLASS: Record<AvailabilityState, string> = {
  up: 'bg-success/80 hover:bg-success',
  partial: 'bg-warning/85 hover:bg-warning',
  down: 'bg-destructive/85 hover:bg-destructive',
  none: 'bg-muted hover:bg-border-strong',
}

function TargetRow({
  target,
  times,
  range,
}: {
  target: ScrapeTarget
  times: readonly number[]
  range: MetricsRange
}) {
  const buckets = useMemo(() => availabilityBuckets(target.values, times), [target.values, times])
  return (
    <li className="flex flex-col gap-2">
      <div className="flex min-w-0 items-center gap-2 text-[13px]">
        <StatusDot tone={target.up === null ? 'neutral' : target.up ? 'success' : 'danger'} />
        <span className="min-w-0 truncate font-mono text-[12.5px]" title={target.instance}>
          {target.instance}
        </span>
        <span className="shrink-0 text-muted-foreground">
          {target.up === null ? 'No recent scrape' : target.up ? 'Up' : 'Down'}
        </span>
        <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular">
          {formatPercent(target.availability)} of scrapes succeeded
        </span>
      </div>
      <div
        role="img"
        aria-label={`Scrapes of ${target.instance} over the ${METRICS_RANGE_LABEL[range]}`}
        className="flex h-6 items-stretch gap-[2px]"
      >
        {buckets.map((bucket) => (
          <span
            key={bucket.start}
            title={`${formatMoment(bucket.start)}–${formatClock(bucket.end)}: ${STATE_LABEL[bucket.state]}`}
            className={cn(
              'min-w-[2px] flex-1 rounded-[2px] transition-colors',
              STATE_CLASS[bucket.state],
            )}
          />
        ))}
      </div>
    </li>
  )
}

export function MetricsHealth({
  data,
  times,
  range,
}: {
  data: MetricsDashboardData
  times: readonly number[]
  range: MetricsRange
}) {
  const up = data.series.up
  const targets = useMemo(() => scrapeTargets(up.data ?? []), [up.data])
  const cache = cacheSummary(data.summary.cache.data ?? [])
  const cacheSeries = data.series.cache
  const hitRate = useMemo(
    () => cacheRateSeries(cacheSeries.data ?? [], times.length),
    [cacheSeries.data, times.length],
  )
  const hitRateChart = useMemo(
    () => [
      {
        label: 'Hit rate',
        color: 'var(--chart-4)',
        values: hitRate,
        format: formatPercent,
        shape: 'area' as const,
      },
    ],
    [hitRate],
  )

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <Card className="min-w-0">
        <section aria-label="Scrape targets" className="flex h-full flex-col">
          <CardHeader>
            <CardTitle>Scrape targets</CardTitle>
            <CardDescription className="text-xs">
              Whether Prometheus could read each Router&apos;s metrics, under the{' '}
              <code className="font-mono">{ROUTER_JOB}</code> job.
            </CardDescription>
          </CardHeader>
          <div className="flex flex-1 flex-col px-5 pb-5 pt-4">
            {up.error ? (
              <div className="h-[180px]">
                <ChartNotice icon={OctagonAlert} title="This query failed" tone="danger">
                  {up.error.message}
                </ChartNotice>
              </div>
            ) : targets.length === 0 ? (
              <div className="h-[180px]">
                <ChartNotice icon={Unplug} title="No Router target">
                  Prometheus has no target in the {ROUTER_JOB} job.
                </ChartNotice>
              </div>
            ) : (
              <ul className="flex flex-col gap-5" aria-label="Router instances">
                {targets.map((target) => (
                  <TargetRow key={target.instance} target={target} times={times} range={range} />
                ))}
              </ul>
            )}
          </div>
        </section>
      </Card>

      <MetricsChartCard
        title="Response cache"
        description="Share of response-cache lookups answered from the cache."
        legend={
          cache
            ? [
                {
                  label: 'Hit rate',
                  color: 'var(--chart-4)',
                  value: `${formatPercent(cache.hitRate)} of ${formatCount(cache.lookups)} lookups`,
                },
              ]
            : undefined
        }
        error={cacheSeries.error?.message}
        empty={!hasSamples(hitRate)}
        emptyHint={`No decision with the response cache plugin handled a request in the ${METRICS_RANGE_LABEL[range]}.`}
        plotClassName="h-[180px]"
      >
        <MetricsTimeChart
          times={times}
          series={hitRateChart}
          axisFormat={formatPercent}
          domain={[0, 1]}
        />
      </MetricsChartCard>
    </div>
  )
}
