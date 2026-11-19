import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ChartLegend, chartAxis, chartGrid } from '../../ui'
import { money, number, percent } from './model'
import { changeDirection, formatSignedChange } from './comparisonMetrics'
import { BenchBlock, EmptyNote } from './benchUi'
import { benchText } from './benchStyles'

export interface QualityCostPoint {
  name: string
  quality: number
  cost: number
  kind: 'single' | 'mom'
}

const colors = [
  'var(--chart-1)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-7)',
  'var(--chart-5)',
]
const changeColors = {
  positive: 'var(--success)',
  negative: 'var(--destructive)',
  neutral: 'var(--muted-foreground)',
  unknown: 'var(--muted-foreground)',
}
const tooltipClass =
  'min-w-40 rounded-lg border border-border bg-popover/95 px-3 py-2 text-xs text-popover-foreground shadow-lg backdrop-blur'
const cursor = { stroke: 'var(--border-strong)', strokeDasharray: '3 3' }

export function QualityCostChart({
  points,
  costBasis,
}: {
  points: QualityCostPoint[]
  costBasis: 'subject' | 'total'
}) {
  const costLabel = costBasis === 'total' ? 'Total cost' : 'Subject model cost'

  return (
    <BenchBlock className="my-4" aria-label="Quality and cost chart">
      <h3 className={benchText.h3}>Quality and {costLabel.toLowerCase()}</h3>
      <p className={`${benchText.muted} mt-0.5`}>
        Higher quality, lower cost.{' '}
        {costBasis === 'total'
          ? 'Includes model answers and evaluation calls at frozen prices.'
          : 'Subject model calls at frozen prices; evaluation calls are excluded.'}
      </p>
      {points.length ? (
        <div className="mt-3 h-[240px] min-w-0 sm:h-[270px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 20, bottom: 25, left: 0 }}>
              <CartesianGrid {...chartGrid} vertical />
              <XAxis
                {...chartAxis}
                type="number"
                dataKey="cost"
                name={costLabel}
                tickCount={4}
                minTickGap={22}
                tickFormatter={(v) => `$${number(v, v < 0.01 ? 4 : 2)}`}
                label={{
                  value: `${costLabel} (USD)`,
                  position: 'bottom',
                  offset: 5,
                  fill: 'var(--muted-foreground)',
                  fontSize: 11,
                }}
              />
              <YAxis
                {...chartAxis}
                type="number"
                dataKey="quality"
                name="Macro accuracy"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                width={48}
              />
              <Tooltip
                cursor={cursor}
                content={({ active, payload }) => {
                  const item = payload?.[0]?.payload as QualityCostPoint | undefined
                  return active && item ? (
                    <div className={tooltipClass}>
                      <strong className="mb-1 block font-medium text-foreground">
                        {item.name}
                      </strong>
                      <div className="text-muted-foreground">
                        {number(item.quality, 2)}% macro accuracy
                      </div>
                      <div className="text-muted-foreground">
                        {money(item.cost)} {costLabel.toLowerCase()}
                      </div>
                    </div>
                  ) : null
                }}
              />
              <Scatter data={points} isAnimationActive={false}>
                {points.map((point, index) => (
                  <Cell
                    key={point.name}
                    fill={colors[index % colors.length]}
                    stroke="var(--card)"
                    strokeWidth={1.5}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyNote className="my-3 py-6">
          Complete quality and cost evidence is needed for this chart.
        </EmptyNote>
      )}
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
        {points.map((point, index) => (
          <li key={point.name} className="inline-flex min-w-0 items-center gap-1.5">
            <span
              aria-hidden="true"
              className="size-2 shrink-0 rounded-full"
              style={{ background: colors[index % colors.length] }}
            />
            <span className="font-medium text-foreground/85">{point.name}</span>
            <span className="tabular-nums">
              {point.kind === 'mom' ? 'MoM' : 'Single'} · {number(point.quality, 2)}% ·{' '}
              {money(point.cost)}
            </span>
          </li>
        ))}
      </ul>
    </BenchBlock>
  )
}

export interface IterationPoint {
  stage: string
  quality: number | null
  saving: number | null
}

export function IterationChart({
  points,
  baselineQuality,
}: {
  points: IterationPoint[]
  baselineQuality: number | null
}) {
  return (
    <BenchBlock className="my-4" aria-label="Iteration progress chart">
      <h3 className={benchText.h3}>Iteration progress</h3>
      <p className={`${benchText.muted} mt-0.5`}>
        Same frozen cases and comparison protocol. Intervals are shown below.
      </p>
      <ChartLegend
        className="mt-3"
        items={[
          { label: 'Macro accuracy', color: 'var(--chart-1)' },
          { label: 'Total cost saving', color: 'var(--muted-foreground)' },
          ...(baselineQuality !== null
            ? [{ label: 'Best single', color: 'var(--subtle-foreground)' }]
            : []),
        ]}
      />
      <div className="mt-2 h-[240px] min-w-0 sm:h-[270px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 12, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid {...chartGrid} />
            <XAxis {...chartAxis} dataKey="stage" />
            <YAxis
              {...chartAxis}
              yAxisId="quality"
              domain={[0, 100]}
              width={48}
              tickFormatter={(v) => `${v}%`}
            />
            <YAxis
              {...chartAxis}
              yAxisId="saving"
              orientation="right"
              width={60}
              domain={([minimum, maximum]: [number, number]) => [
                Math.min(0, minimum),
                Math.max(0, maximum),
              ]}
              tickFormatter={(v) => `${formatSignedChange(v)}%`}
            />
            <Tooltip
              cursor={cursor}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className={tooltipClass}>
                    <div className="mb-1.5 font-medium text-foreground">{label}</div>
                    {payload.map((item) => {
                      const value = typeof item.value === 'number' ? item.value : null
                      const saving = item.name === 'Total cost saving'
                      return (
                        <div key={String(item.name)} className="flex items-center gap-2">
                          <span className="flex-1 text-muted-foreground">{item.name}</span>
                          <span
                            className="font-mono font-medium tabular-nums text-foreground"
                            style={
                              saving ? { color: changeColors[changeDirection(value)] } : undefined
                            }
                          >
                            {saving ? `${formatSignedChange(value)}%` : `${number(value, 2)}%`}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                ) : null
              }
            />
            <ReferenceLine
              yAxisId="saving"
              y={0}
              stroke="var(--border-strong)"
              strokeDasharray="3 3"
            />
            {baselineQuality !== null && (
              <ReferenceLine
                yAxisId="quality"
                y={baselineQuality}
                stroke="var(--subtle-foreground)"
                strokeDasharray="5 5"
                label={{
                  value: 'Best single',
                  fill: 'var(--muted-foreground)',
                  fontSize: 11,
                  position: 'insideTopRight',
                }}
              />
            )}
            <Line
              yAxisId="quality"
              type="linear"
              dataKey="quality"
              name="Macro accuracy"
              stroke="var(--chart-1)"
              strokeWidth={2}
              dot={{ r: 4, fill: 'var(--chart-1)', stroke: 'var(--card)', strokeWidth: 1.5 }}
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              yAxisId="saving"
              type="linear"
              dataKey="saving"
              name="Total cost saving"
              stroke="var(--muted-foreground)"
              strokeWidth={2}
              dot={({ cx, cy, value, key }) => (
                <circle
                  key={key}
                  cx={cx}
                  cy={cy}
                  r={4}
                  fill={changeColors[changeDirection(typeof value === 'number' ? value : null)]}
                  stroke="var(--card)"
                  strokeWidth={1.5}
                />
              )}
              isAnimationActive={false}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </BenchBlock>
  )
}

export function RoutingBars({
  title,
  entries,
}: {
  title: string
  entries: Array<[string, number]>
}) {
  const total = entries.reduce((sum, [, value]) => sum + value, 0)
  return (
    <BenchBlock aria-label={`${title} distribution`}>
      <h3 className={benchText.h3}>{title}</h3>
      {entries.length ? (
        <ul className="mt-3 grid gap-2.5 text-xs">
          {entries.map(([label, count], index) => (
            <li key={label} className="flex items-center gap-3">
              <span title={label} className="w-2/5 min-w-0 break-words text-foreground/85">
                {label}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(total ? count / total : 0) * 100}%`,
                    background: colors[index % colors.length],
                  }}
                />
              </div>
              <strong className="min-w-[4.5rem] text-right font-medium text-foreground tabular-nums">
                {percent(total ? count / total : 0)}
                <small className="block text-[10.5px] font-normal text-muted-foreground">
                  {number(count)} observations
                </small>
              </strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className={`${benchText.muted} mt-2`}>No routing trace recorded for these results.</p>
      )}
    </BenchBlock>
  )
}
