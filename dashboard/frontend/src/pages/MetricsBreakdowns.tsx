import { useMemo, useState, type ReactNode } from 'react'

import {
  MiniBar,
  Segmented,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  entityTone,
} from '../ui'
import { MetricsChartCard, MetricsTimeChart, type ChartSeries } from './MetricsTimeChart'
import {
  formatAxisNumber,
  formatCount,
  formatMoney,
  formatPercent,
  formatRate,
  formatSeconds,
  formatSecondsAxis,
} from './metricsFormat'
import { decisionId, decisionRows, modelRows, type CurrencyAmount } from './metricsOverview'
import {
  METRICS_RANGE_LABEL,
  MODEL_METRICS,
  modelSeriesQuery,
  type MetricsRange,
  type ModelMetric,
} from './metricsQueries'
import { RANKED_COLORS, hasSamples, rankedColor, rankedSeries } from './metricsSeries'
import { usePrometheusRange, type MetricsDashboardData } from './useMetricsDashboard'

const MODEL_VIEWS: Record<
  ModelMetric,
  { label: string; description: string; additive: boolean; axis: (value: number) => string }
> = {
  requests: {
    label: 'Requests',
    description: 'Requests per second the Router sent to each model.',
    additive: true,
    axis: formatAxisNumber,
  },
  errors: {
    label: 'Errors',
    description: 'Errors per second the Router recorded for each model.',
    additive: true,
    axis: formatAxisNumber,
  },
  latency: {
    label: 'P95 latency',
    description: 'P95 completion latency of each model.',
    additive: false,
    axis: formatSecondsAxis,
  },
  tokens: {
    label: 'Tokens',
    description: 'Tokens per second each model reported.',
    additive: true,
    axis: formatAxisNumber,
  },
  cost: {
    label: 'Spend',
    description: 'Spend per hour of each model, from the model prices in the router config.',
    additive: true,
    axis: formatAxisNumber,
  },
}

/** The Router's decision label for a request that matched no decision. */
const NO_DECISION = 'unknown'

function decisionName(decision: string): string {
  return decision === NO_DECISION ? 'No decision matched' : decision
}

function NameCell({
  color,
  children,
  detail,
}: {
  color: string
  children: string
  detail?: string
}) {
  return (
    <span className="flex min-w-0 max-w-[22rem] items-center gap-2">
      <span className="size-2 shrink-0 rounded-[3px]" style={{ background: color }} />
      <span className="truncate font-medium" title={children}>
        {children}
      </span>
      {detail ? (
        <span className="truncate text-xs text-muted-foreground" title={detail}>
          {detail}
        </span>
      ) : null}
    </span>
  )
}

function ShareCell({ share, tone }: { share: number | null; tone: 'green' | 'violet' }) {
  return (
    <span className="flex items-center gap-2">
      <MiniBar value={(share ?? 0) * 100} tone={tone} className="w-16" />
      <span className="w-12 text-right text-xs text-muted-foreground tabular">
        {formatPercent(share)}
      </span>
    </span>
  )
}

function formatAmounts(amounts: CurrencyAmount[]): string {
  return amounts.length
    ? amounts.map((entry) => formatMoney(entry.amount, entry.currency)).join(' + ')
    : '—'
}

/** Lines the table's outer columns up with the card's 20px gutters. */
const CARD_TABLE =
  'border-t border-border [&_td:first-child]:pl-5 [&_td:last-child]:pr-5 [&_th:first-child]:pl-5 [&_th:last-child]:pr-5'

function EmptyRow({ columns, children }: { columns: number; children: ReactNode }) {
  return (
    <TableRow className="hover:bg-transparent">
      <TableCell colSpan={columns} className="h-20 text-center text-[13px] text-muted-foreground">
        {children}
      </TableCell>
    </TableRow>
  )
}

export function ModelBreakdown({
  data,
  times,
  range,
}: {
  data: MetricsDashboardData
  times: readonly number[]
  range: MetricsRange
}) {
  const [metric, setMetric] = useState<ModelMetric>('requests')
  const summary = data.summary.models
  const rows = useMemo(() => modelRows(summary.data ?? []), [summary.data])
  const spendExported = rows.some((row) => row.cost.length > 0)
  const shown: ModelMetric = metric === 'cost' && !spendExported ? 'requests' : metric
  const view = MODEL_VIEWS[shown]
  const query = useMemo(() => modelSeriesQuery(shown, data.window), [shown, data.window])
  const { result, loading } = usePrometheusRange(query, data.window)
  const order = useMemo(() => rows.map((row) => row.model), [rows])
  const totalRequests = rows.reduce((total, row) => total + (row.requests ?? 0), 0)
  const columns = spendExported ? 7 : 6
  const label = METRICS_RANGE_LABEL[range]

  const chart = useMemo(() => {
    const entries = (result?.data ?? []).flatMap((series) =>
      series.labels.model
        ? [
            {
              name: series.labels.model,
              currency: series.labels.currency ?? '',
              values: series.values,
            },
          ]
        : [],
    )
    const currencies = [...new Set(entries.map((entry) => entry.currency))]
    const mixed = shown === 'cost' && currencies.length > 1
    const format = (currency: string): ((value: number) => string) =>
      shown === 'latency'
        ? formatSeconds
        : shown === 'cost'
          ? (value) => `${formatMoney(value, currency)}/h`
          : formatRate
    const ranked = rankedSeries(
      entries.map((entry) => ({
        ...entry,
        label: mixed ? `${entry.name} · ${entry.currency}` : entry.name,
      })),
      order,
      times.length,
      MODEL_VIEWS[shown].additive && !mixed,
    )
    const charted = new Set(order.slice(0, RANKED_COLORS.length))
    return {
      series: ranked.map(
        (entry): ChartSeries => ({
          label: entry.label,
          color: entry.color,
          values: entry.values,
          format: format(entry.source?.currency || currencies[0] || 'USD'),
        }),
      ),
      axis:
        shown === 'cost' && currencies.length === 1
          ? (value: number) => formatMoney(value, currencies[0])
          : MODEL_VIEWS[shown].axis,
      dropped: !MODEL_VIEWS[shown].additive && entries.some((entry) => !charted.has(entry.name)),
    }
  }, [result, order, shown, times.length])

  return (
    <MetricsChartCard
      title="By model"
      description={
        chart.dropped
          ? `${view.description} Showing the ${RANKED_COLORS.length} busiest models.`
          : view.description
      }
      toolbar={
        <Segmented
          size="sm"
          aria-label="Model metric"
          value={shown}
          onValueChange={setMetric}
          options={MODEL_METRICS.filter((key) => key !== 'cost' || spendExported).map((key) => ({
            value: key,
            label: MODEL_VIEWS[key].label,
          }))}
        />
      }
      legend={chart.series.map((entry) => ({ label: entry.label, color: entry.color }))}
      loading={!result && loading}
      error={result?.error?.message}
      empty={!chart.series.some((entry) => hasSamples(entry.values))}
      emptyHint={`No model reported ${view.label.toLowerCase()} in the ${label}.`}
      plotClassName="h-[240px]"
      footer={
        <div className={CARD_TABLE}>
          <Table aria-label={`Models over the ${label}`}>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Requests</TableHead>
                <TableHead>Share</TableHead>
                <TableHead className="text-right">Errors</TableHead>
                <TableHead className="text-right">P95 latency</TableHead>
                <TableHead className="text-right">Tokens</TableHead>
                {spendExported ? <TableHead className="text-right">Spend</TableHead> : null}
              </TableRow>
            </TableHeader>
            <TableBody>
              {summary.error ? (
                <EmptyRow columns={columns}>{summary.error.message}</EmptyRow>
              ) : rows.length === 0 ? (
                <EmptyRow columns={columns}>No model received requests in the {label}.</EmptyRow>
              ) : (
                rows.map((row) => (
                  <TableRow key={row.model}>
                    <TableCell>
                      <NameCell color={rankedColor(order, row.model)}>{row.model}</NameCell>
                    </TableCell>
                    <TableCell className="text-right tabular">
                      {formatCount(row.requests)}
                    </TableCell>
                    <TableCell>
                      <ShareCell
                        share={totalRequests > 0 ? (row.requests ?? 0) / totalRequests : null}
                        tone={entityTone.model}
                      />
                    </TableCell>
                    <TableCell className="text-right tabular">
                      {/* The Router creates a model's error counter at its first error. */}
                      {formatCount(row.errors ?? (row.requests ? 0 : null))}
                      {row.errorRate !== null ? (
                        <span className="text-muted-foreground">
                          {' '}
                          · {formatPercent(row.errorRate)}
                        </span>
                      ) : null}
                    </TableCell>
                    <TableCell className="text-right tabular">
                      {formatSeconds(row.latency)}
                    </TableCell>
                    <TableCell className="text-right tabular">{formatCount(row.tokens)}</TableCell>
                    {spendExported ? (
                      <TableCell className="text-right tabular">
                        {formatAmounts(row.cost)}
                      </TableCell>
                    ) : null}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      }
    >
      <MetricsTimeChart times={times} series={chart.series} axisFormat={chart.axis} />
    </MetricsChartCard>
  )
}

export function DecisionBreakdown({
  data,
  times,
  range,
}: {
  data: MetricsDashboardData
  times: readonly number[]
  range: MetricsRange
}) {
  const selections = data.summary.decisions
  const cache = data.summary.decisionCache
  const rows = useMemo(
    () => decisionRows(selections.data ?? [], cache.data ?? []),
    [selections.data, cache.data],
  )
  const order = useMemo(() => rows.map((row) => row.id), [rows])
  const recipes = useMemo(() => new Set(rows.map((row) => row.recipe)).size > 1, [rows])
  const cacheExported = rows.some((row) => row.cache)
  const totalRequests = rows.reduce((total, row) => total + row.requests, 0)
  const columns = cacheExported ? 5 : 4
  const label = METRICS_RANGE_LABEL[range]
  const series = data.series.decisions

  const chart = useMemo(
    () =>
      rankedSeries(
        (series.data ?? []).map((entry) => {
          const recipe = entry.labels.recipe || 'default'
          const decision = entry.labels.decision || NO_DECISION
          return {
            name: decisionId(recipe, decision),
            label: recipes ? `${recipe} / ${decisionName(decision)}` : decisionName(decision),
            values: entry.values,
          }
        }),
        order,
        times.length,
        true,
      ).map(
        (entry): ChartSeries => ({
          label: entry.label,
          color: entry.color,
          values: entry.values,
          format: formatRate,
        }),
      ),
    [series.data, order, recipes, times.length],
  )

  return (
    <MetricsChartCard
      title="By decision"
      description="Model selections per second, by the decision that made them."
      legend={chart.map((entry) => ({ label: entry.label, color: entry.color }))}
      error={series.error?.message}
      empty={!chart.some((entry) => hasSamples(entry.values))}
      emptyHint={`No decision selected a model in the ${label}.`}
      plotClassName="h-[240px]"
      footer={
        <div className={CARD_TABLE}>
          <Table aria-label={`Decisions over the ${label}`}>
            <TableHeader>
              <TableRow>
                <TableHead>Decision</TableHead>
                <TableHead className="text-right">Selections</TableHead>
                <TableHead>Share</TableHead>
                <TableHead>Models chosen</TableHead>
                {cacheExported ? (
                  <TableHead className="text-right">Cache hit rate</TableHead>
                ) : null}
              </TableRow>
            </TableHeader>
            <TableBody>
              {selections.error ? (
                <EmptyRow columns={columns}>{selections.error.message}</EmptyRow>
              ) : rows.length === 0 ? (
                <EmptyRow columns={columns}>No decision selected a model in the {label}.</EmptyRow>
              ) : (
                rows.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell>
                      <NameCell
                        color={rankedColor(order, row.id)}
                        detail={recipes ? row.recipe : undefined}
                      >
                        {decisionName(row.decision)}
                      </NameCell>
                    </TableCell>
                    <TableCell className="text-right tabular">
                      {formatCount(row.requests)}
                    </TableCell>
                    <TableCell>
                      <ShareCell
                        share={totalRequests > 0 ? row.requests / totalRequests : null}
                        tone={entityTone.decision}
                      />
                    </TableCell>
                    <TableCell>
                      <span className="flex max-w-[26rem] items-center gap-1.5">
                        {row.models.length === 0 ? (
                          <span className="text-muted-foreground">—</span>
                        ) : null}
                        {row.models.slice(0, 2).map((model) => (
                          <span
                            key={model.model}
                            className="inline-flex min-w-0 items-center gap-1 rounded-md border border-border bg-surface-subtle px-1.5 py-0.5 text-[11.5px]"
                          >
                            <span className="truncate" title={model.model}>
                              {model.model}
                            </span>
                            <span className="text-muted-foreground tabular">
                              {formatPercent(
                                row.requests > 0 ? model.requests / row.requests : null,
                              )}
                            </span>
                          </span>
                        ))}
                        {row.models.length > 2 ? (
                          <span className="shrink-0 text-xs text-muted-foreground">
                            +{row.models.length - 2}
                          </span>
                        ) : null}
                      </span>
                    </TableCell>
                    {cacheExported ? (
                      <TableCell className="text-right tabular">
                        {row.cache ? formatPercent(row.cache.hitRate) : '—'}
                      </TableCell>
                    ) : null}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      }
    >
      <MetricsTimeChart times={times} series={chart} axisFormat={formatAxisNumber} />
    </MetricsChartCard>
  )
}
