import {
  CartesianGrid,
  Cell,
  Legend,
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
import { money, number, percent } from './model'
import { changeDirection, formatSignedChange } from './comparisonMetrics'
import styles from './SrBench.module.css'

export interface QualityCostPoint {
  name: string
  quality: number
  cost: number
  kind: 'single' | 'mom'
}

const colors = ['#82b8ff', '#b59cff', '#67d8b4', '#f4ba72', '#f28f9d']
const changeColors = {
  positive: 'var(--color-success, #67b993)',
  negative: 'var(--color-danger, #f69494)',
  neutral: 'var(--text-secondary, #9c9ca8)',
  unknown: 'var(--text-secondary, #9c9ca8)',
}

export function QualityCostChart({ points }: { points: QualityCostPoint[] }) {
  return (
    <section className={styles.chartCard} aria-label="Quality and cost chart">
      <h3>Quality and cost</h3>
      <p className={styles.muted}>
        Higher quality, lower cost. Recorded usage at frozen model prices.
      </p>
      {points.length ? (
        <div className={styles.chartCanvas}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 20, bottom: 25, left: 0 }}>
              <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="cost"
                name="Model cost"
                tickCount={4}
                minTickGap={22}
                tickFormatter={(v) => `$${number(v, v < 0.01 ? 4 : 2)}`}
                label={{ value: 'Model cost (USD)', position: 'bottom', offset: 5 }}
              />
              <YAxis
                type="number"
                dataKey="quality"
                name="Macro accuracy"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                width={48}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  const item = payload?.[0]?.payload as QualityCostPoint | undefined
                  return active && item ? (
                    <div className={styles.chartTooltip}>
                      <strong>{item.name}</strong>
                      <div>{number(item.quality, 2)}% macro accuracy</div>
                      <div>{money(item.cost)} model cost</div>
                    </div>
                  ) : null
                }}
              />
              <Scatter data={points} isAnimationActive={false}>
                {points.map((point, index) => (
                  <Cell key={point.name} fill={colors[index % colors.length]} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className={styles.emptyState}>
          Complete quality and cost evidence is needed for this chart.
        </p>
      )}
      <ul className={styles.chartLegend}>
        {points.map((point, index) => (
          <li key={point.name}>
            <i style={{ background: colors[index % colors.length] }} />
            {point.name}
            <span>
              {point.kind === 'mom' ? 'MoM' : 'Single'} · {number(point.quality, 2)}% ·{' '}
              {money(point.cost)}
            </span>
          </li>
        ))}
      </ul>
    </section>
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
    <section className={styles.chartCard} aria-label="Iteration progress chart">
      <h3>Iteration progress</h3>
      <p className={styles.muted}>
        Same frozen cases and comparison protocol. Intervals are shown below.
      </p>
      <div className={styles.chartCanvas}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 12, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" />
            <XAxis dataKey="stage" />
            <YAxis yAxisId="quality" domain={[0, 100]} width={48} tickFormatter={(v) => `${v}%`} />
            <YAxis
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
              contentStyle={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: 8,
              }}
              formatter={(value, name) =>
                name === 'Cost saving' ? (
                  <span
                    style={{
                      color:
                        changeColors[changeDirection(typeof value === 'number' ? value : null)],
                    }}
                  >
                    {formatSignedChange(typeof value === 'number' ? value : null)}%
                  </span>
                ) : (
                  `${number(value, 2)}%`
                )
              }
            />
            <Legend />
            <ReferenceLine
              yAxisId="saving"
              y={0}
              stroke="var(--border-color)"
              strokeDasharray="3 3"
            />
            {baselineQuality !== null && (
              <ReferenceLine
                yAxisId="quality"
                y={baselineQuality}
                stroke="#9c9ca8"
                strokeDasharray="5 5"
                label={{ value: 'Best single', fill: '#9c9ca8', position: 'insideTopRight' }}
              />
            )}
            <Line
              yAxisId="quality"
              type="linear"
              dataKey="quality"
              name="Macro accuracy"
              stroke="#82b8ff"
              strokeWidth={2}
              dot={{ r: 4 }}
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              yAxisId="saving"
              type="linear"
              dataKey="saving"
              name="Cost saving"
              stroke="var(--text-secondary, #9c9ca8)"
              strokeWidth={2}
              dot={({ cx, cy, value, key }) => (
                <circle
                  key={key}
                  cx={cx}
                  cy={cy}
                  r={4}
                  fill={changeColors[changeDirection(typeof value === 'number' ? value : null)]}
                  stroke="none"
                />
              )}
              isAnimationActive={false}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
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
    <section className={styles.chartCard} aria-label={`${title} distribution`}>
      <h3>{title}</h3>
      {entries.length ? (
        <ul className={styles.distribution}>
          {entries.map(([label, count], index) => (
            <li key={label}>
              <span title={label}>{label}</span>
              <div className={styles.barTrack}>
                <div
                  style={{
                    width: `${(total ? count / total : 0) * 100}%`,
                    background: colors[index % colors.length],
                  }}
                />
              </div>
              <strong>
                {percent(total ? count / total : 0)}
                <small>{number(count)} observations</small>
              </strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className={styles.muted}>No routing trace recorded for these results.</p>
      )}
    </section>
  )
}
