import { useMemo } from 'react'

import { cn } from '../ui'
import {
  MetricsChartCard,
  MetricsKpi,
  MetricsTimeChart,
  type ChartSeries,
} from './MetricsTimeChart'
import {
  formatAxisNumber,
  formatCount,
  formatMoney,
  formatPercent,
  formatRate,
  formatSeconds,
  formatSecondsAxis,
} from './metricsFormat'
import {
  amountsByCurrency,
  quantileSeries,
  quantileSummary,
  requestSeries,
  requestSummary,
  tokenSeries,
  tokenSummary,
} from './metricsOverview'
import { METRICS_RANGE_LABEL, type MetricsRange } from './metricsQueries'
import { hasSamples, valuesWhere } from './metricsSeries'
import type { MetricsDashboardData } from './useMetricsDashboard'

const COLOR = {
  requests: 'var(--chart-1)',
  succeeded: 'var(--chart-6)',
  failed: 'var(--chart-5)',
  errorRate: 'var(--chart-7)',
  p50: 'var(--chart-1)',
  p95: 'var(--chart-3)',
  p99: 'var(--chart-7)',
  prompt: 'var(--chart-4)',
  completion: 'var(--chart-1)',
  spend: 'var(--chart-2)',
}

const SPEND_COLORS = ['var(--chart-2)', 'var(--chart-1)', 'var(--chart-4)', 'var(--chart-3)']
const QUANTILES = ['p50', 'p95', 'p99'] as const

export function MetricsOverview({
  data,
  times,
  range,
}: {
  data: MetricsDashboardData
  times: readonly number[]
  range: MetricsRange
}) {
  const { series, summary, window } = data
  const count = times.length
  const label = METRICS_RANGE_LABEL[range]

  const requests = useMemo(
    () => (series.outcomes.data ? requestSeries(series.outcomes.data, count) : null),
    [series.outcomes.data, count],
  )
  const latency = useMemo(
    () => (series.latency.data ? quantileSeries(series.latency.data, count) : null),
    [series.latency.data, count],
  )
  const tokens = useMemo(
    () => (series.tokens.data ? tokenSeries(series.tokens.data, count) : null),
    [series.tokens.data, count],
  )
  const requestTotals = requestSummary(summary.outcomes.data ?? [])
  const latencyTotals = quantileSummary(summary.latency.data ?? [])
  const tokenTotals = tokenSummary(summary.tokens.data ?? [])
  const spend = useMemo(() => amountsByCurrency(summary.cost.data ?? []), [summary.cost.data])
  const spendSeries = useMemo(
    () =>
      spend.map(({ currency }, index) => ({
        currency,
        color: SPEND_COLORS[index % SPEND_COLORS.length],
        values: valuesWhere(series.cost.data ?? [], 'currency', currency, count),
      })),
    [spend, series.cost.data, count],
  )
  const primarySpend = spend[0]
  const perSecond = (total: number | null) => (total === null ? null : total / window.seconds)

  const kpis = [
    {
      label: 'Requests',
      color: COLOR.requests,
      value: formatRate(perSecond(requestTotals.total)),
      hint:
        requestTotals.total === null
          ? 'No samples in this range'
          : `${formatCount(requestTotals.total)} in the ${label}`,
      help: 'Inference requests per second, averaged over the range.',
      spark: requests?.total,
    },
    {
      label: 'Error rate',
      color: COLOR.errorRate,
      value: formatPercent(requestTotals.errorRate),
      hint:
        requestTotals.total === null
          ? 'No samples in this range'
          : requestTotals.total === 0
            ? 'No requests in this range'
            : `${formatCount(requestTotals.failed)} of ${formatCount(requestTotals.total)} failed`,
      help: 'Share of inference requests that ended in a client error, server error, timeout, cancellation or incomplete response.',
      spark: requests?.errorRate,
    },
    {
      label: 'P95 latency',
      color: COLOR.p95,
      value: formatSeconds(latencyTotals.p95),
      hint: `p50 ${formatSeconds(latencyTotals.p50)} · p99 ${formatSeconds(latencyTotals.p99)}`,
      help: 'End-to-end duration of inference requests over the range, streaming included.',
      spark: latency?.p95,
    },
    {
      label: 'Tokens',
      color: COLOR.prompt,
      value: formatRate(perSecond(tokenTotals.total)),
      hint:
        tokenTotals.total === null
          ? 'No token usage reported'
          : `${formatCount(tokenTotals.total)} in the ${label}`,
      help: 'Prompt and completion tokens per second reported by the models, averaged over the range.',
      spark: tokens?.total,
    },
    ...(primarySpend
      ? [
          {
            label: 'Spend',
            color: COLOR.spend,
            value: formatMoney(primarySpend.amount, primarySpend.currency),
            hint:
              spend.length > 1
                ? `and ${spend
                    .slice(1)
                    .map((entry) => formatMoney(entry.amount, entry.currency))
                    .join(', ')}`
                : `${formatMoney(primarySpend.amount / (window.seconds / 3_600), primarySpend.currency)} per hour`,
            help: 'Spend over the range from the model prices in the router config.',
            spark: spendSeries[0]?.values,
          },
        ]
      : []),
  ]

  const requestChart: ChartSeries[] = requests
    ? [
        {
          label: 'Succeeded',
          color: COLOR.succeeded,
          values: requests.succeeded,
          format: formatRate,
          shape: 'area',
        },
        {
          label: 'Failed',
          color: COLOR.failed,
          values: requests.failed,
          format: formatRate,
          shape: 'area',
        },
        {
          label: 'Error rate',
          color: COLOR.errorRate,
          values: requests.errorRate,
          format: formatPercent,
          axis: 'right',
          dashed: true,
        },
      ]
    : []
  const latencyChart: ChartSeries[] = latency
    ? QUANTILES.map((key) => ({
        label: key.toUpperCase(),
        color: COLOR[key],
        values: latency[key],
        format: formatSeconds,
        dashed: key === 'p99',
      }))
    : []
  const tokenChart: ChartSeries[] = tokens
    ? [
        {
          label: 'Prompt',
          color: COLOR.prompt,
          values: tokens.prompt,
          format: formatRate,
          shape: 'area',
        },
        {
          label: 'Completion',
          color: COLOR.completion,
          values: tokens.completion,
          format: formatRate,
          shape: 'area',
        },
      ]
    : []
  const spendChart: ChartSeries[] = spendSeries.map((entry) => ({
    label: entry.currency,
    color: entry.color,
    values: entry.values,
    format: (value: number) => `${formatMoney(value, entry.currency)}/h`,
    shape: spendSeries.length === 1 ? 'area' : 'line',
  }))

  return (
    <div className="flex flex-col gap-4">
      <dl
        aria-label="Summary"
        className={cn(
          'grid grid-cols-1 gap-3 sm:grid-cols-2',
          kpis.length > 4 ? 'xl:grid-cols-5' : 'xl:grid-cols-4',
        )}
      >
        {kpis.map((kpi) => (
          <MetricsKpi key={kpi.label} {...kpi} />
        ))}
      </dl>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <MetricsChartCard
          className="xl:col-span-2"
          title="Requests"
          description="Inference requests per second and the share that failed."
          error={series.outcomes.error?.message}
          empty={!requests || !hasSamples(requests.total)}
          plotClassName="h-[260px]"
          legend={[
            {
              label: 'Succeeded',
              color: COLOR.succeeded,
              value:
                requestTotals.total === null
                  ? undefined
                  : formatCount(requestTotals.total - (requestTotals.failed ?? 0)),
            },
            { label: 'Failed', color: COLOR.failed, value: formatCount(requestTotals.failed) },
            {
              label: 'Error rate',
              color: COLOR.errorRate,
              value: formatPercent(requestTotals.errorRate),
            },
          ]}
        >
          <MetricsTimeChart
            times={times}
            series={requestChart}
            axisFormat={formatAxisNumber}
            rightAxisFormat={formatPercent}
            stacked
          />
        </MetricsChartCard>

        <MetricsChartCard
          title="Latency"
          description="End-to-end duration of inference requests, streaming included."
          error={series.latency.error?.message}
          empty={!latency || !QUANTILES.some((key) => hasSamples(latency[key]))}
          legend={QUANTILES.map((key) => ({
            label: key.toUpperCase(),
            color: COLOR[key],
            value: formatSeconds(latencyTotals[key]),
          }))}
        >
          <MetricsTimeChart times={times} series={latencyChart} axisFormat={formatSecondsAxis} />
        </MetricsChartCard>

        <MetricsChartCard
          title="Tokens"
          description="Prompt and completion tokens per second reported by the models."
          error={series.tokens.error?.message}
          empty={!tokens || !hasSamples(tokens.total)}
          emptyHint="The models have reported no token usage in this range."
          legend={[
            { label: 'Prompt', color: COLOR.prompt, value: formatCount(tokenTotals.prompt) },
            {
              label: 'Completion',
              color: COLOR.completion,
              value: formatCount(tokenTotals.completion),
            },
          ]}
        >
          <MetricsTimeChart
            times={times}
            series={tokenChart}
            axisFormat={formatAxisNumber}
            stacked
          />
        </MetricsChartCard>

        {primarySpend ? (
          <MetricsChartCard
            className="xl:col-span-2"
            title="Spend"
            description="Spend per hour from the model prices in the router config."
            error={series.cost.error?.message}
            empty={!spendSeries.some((entry) => hasSamples(entry.values))}
            legend={spend.map((entry, index) => ({
              label: entry.currency,
              color: spendSeries[index].color,
              value: formatMoney(entry.amount, entry.currency),
            }))}
          >
            <MetricsTimeChart
              times={times}
              series={spendChart}
              axisFormat={(value) =>
                spend.length === 1
                  ? formatMoney(value, primarySpend.currency)
                  : formatAxisNumber(value)
              }
            />
          </MetricsChartCard>
        ) : null}
      </div>
    </div>
  )
}
