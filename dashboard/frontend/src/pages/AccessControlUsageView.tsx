import { type CSSProperties, useState } from 'react'

import type {
  AccessAPIKey,
  AccessGroup,
  AccessTeam,
  AccessUser,
  UsagePoint,
  UsageSlice,
  UsageSummary,
} from '../utils/inferenceAccessApi'
import styles from './AccessControlPage.module.css'

export interface UsageScope {
  type: 'global' | 'user' | 'team' | 'key'
  id: string
  model: string
  range: '24h' | '7d' | '30d'
}

interface Props {
  usage: UsageSummary
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  usageScope: UsageScope
  onUsageScopeChange: (value: UsageScope) => void
  loading: boolean
}

const number = (value?: number) => new Intl.NumberFormat('en-US').format(value || 0)
const percent = (part: number, total: number) =>
  total ? `${((part / total) * 100).toFixed(1)}%` : '—'

export default function AccessControlUsageView(props: Props) {
  const [metric, setMetric] = useState<'tokens' | 'requests' | 'latency'>('tokens')
  const [dimension, setDimension] = useState<'model' | 'user' | 'team' | 'key'>('model')
  const tokensPerRequest = props.usage.requests
    ? Math.round(props.usage.totalTokens / props.usage.requests)
    : 0
  const outputShare = props.usage.totalTokens
    ? (props.usage.completionTokens / props.usage.totalTokens) * 100
    : 0
  const dimensionItems =
    dimension === 'model'
      ? props.usage.byModel
      : dimension === 'user'
        ? props.usage.byUser
        : dimension === 'team'
          ? props.usage.byTeam
          : props.usage.byKey
  const label = (id: string) => {
    if (dimension === 'user') return props.users.find((item) => item.id === id)?.name || id
    if (dimension === 'team') return props.teams.find((item) => item.id === id)?.name || id
    if (dimension === 'key') return props.keys.find((item) => item.id === id)?.name || id
    return id
  }

  return (
    <div className={styles.viewStack}>
      <UsageFilters {...props} />

      <div className={styles.usageMetricGrid}>
        <UsageMetric
          label="Requests"
          value={number(props.usage.requests)}
          detail={`${number(props.usage.failed)} failed`}
          tone="blue"
        />
        <UsageMetric
          label="Success rate"
          value={percent(props.usage.successful, props.usage.requests)}
          detail={`${number(props.usage.successful)} successful`}
          tone="green"
        />
        <UsageMetric
          label="Total tokens"
          value={number(props.usage.totalTokens)}
          detail={`${number(tokensPerRequest)} per request`}
          tone="violet"
        />
        <UsageMetric
          label="Output share"
          value={`${outputShare.toFixed(1)}%`}
          detail={`${number(props.usage.completionTokens)} output`}
          tone="violet"
        />
        <UsageMetric
          label="P95 latency"
          value={`${number(props.usage.p95LatencyMs)} ms`}
          detail={`${number(props.usage.averageLatencyMs)} ms average`}
          tone="amber"
        />
        <UsageMetric
          label="P95 first token"
          value={props.usage.p95TtftMs ? `${number(props.usage.p95TtftMs)} ms` : '—'}
          detail={
            props.usage.averageTtftMs
              ? `${number(props.usage.averageTtftMs)} ms average`
              : 'No TTFT samples'
          }
          tone="neutral"
        />
      </div>

      <section className={`${styles.panel} ${styles.usageTrendPanel}`}>
        <div className={styles.usagePanelHeader}>
          <div>
            <span>{props.usageScope.range === '24h' ? 'Hourly' : 'Daily'}</span>
            <h3>Traffic over time</h3>
          </div>
          <div className={styles.segmented}>
            {(['tokens', 'requests', 'latency'] as const).map((item) => (
              <button
                type="button"
                key={item}
                className={metric === item ? styles.segmentedActive : ''}
                onClick={() => setMetric(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
        <TrendChart points={props.usage.series} metric={metric} />
      </section>

      <div className={styles.usageInsightGrid}>
        <section className={styles.panel}>
          <div className={styles.usagePanelHeader}>
            <div>
              <span>Token mix</span>
              <h3>Input vs output</h3>
            </div>
            <strong>{number(props.usage.totalTokens)}</strong>
          </div>
          <div className={styles.tokenMix}>
            <div
              className={styles.tokenDonut}
              style={{ '--output-share': `${outputShare * 3.6}deg` } as CSSProperties}
            >
              <span>
                {outputShare.toFixed(0)}%<small>output</small>
              </span>
            </div>
            <div className={styles.tokenLegend}>
              <div>
                <i className={styles.tokenInput} />
                <span>Input</span>
                <strong>{number(props.usage.promptTokens)}</strong>
              </div>
              <div>
                <i className={styles.tokenOutput} />
                <span>Output</span>
                <strong>{number(props.usage.completionTokens)}</strong>
              </div>
            </div>
          </div>
        </section>
        <section className={styles.panel}>
          <div className={styles.usagePanelHeader}>
            <div>
              <span>Reliability</span>
              <h3>Successful requests</h3>
            </div>
            <strong>{percent(props.usage.successful, props.usage.requests)}</strong>
          </div>
          <div className={styles.reliabilityBar}>
            <i
              style={{
                width: `${props.usage.requests ? (props.usage.successful / props.usage.requests) * 100 : 0}%`,
              }}
            />
          </div>
          <div className={styles.reliabilityStats}>
            <div>
              <span>Successful</span>
              <strong>{number(props.usage.successful)}</strong>
            </div>
            <div>
              <span>Failed</span>
              <strong>{number(props.usage.failed)}</strong>
            </div>
            <div>
              <span>Active keys</span>
              <strong>{number(props.usage.activeKeys)}</strong>
            </div>
          </div>
        </section>
        <section className={styles.panel}>
          <div className={styles.usagePanelHeader}>
            <div>
              <span>Latency</span>
              <h3>Response timing</h3>
            </div>
          </div>
          <div className={styles.timingStack}>
            <Timing
              label="Average latency"
              value={props.usage.averageLatencyMs}
              max={Math.max(props.usage.p95LatencyMs, 1)}
            />
            <Timing
              label="P95 latency"
              value={props.usage.p95LatencyMs}
              max={Math.max(props.usage.p95LatencyMs, 1)}
            />
            <Timing
              label="Average first token"
              value={props.usage.averageTtftMs}
              max={Math.max(props.usage.p95LatencyMs, 1)}
            />
          </div>
        </section>
      </div>

      <section className={`${styles.panel} ${styles.usageRankingPanel}`}>
        <div className={styles.usagePanelHeader}>
          <div>
            <span>Breakdown</span>
            <h3>Usage leaders</h3>
          </div>
          <div className={styles.segmented}>
            {(['model', 'user', 'team', 'key'] as const).map((item) => (
              <button
                type="button"
                key={item}
                className={dimension === item ? styles.segmentedActive : ''}
                onClick={() => setDimension(item)}
              >
                {item === 'key' ? 'API keys' : `${item}s`}
              </button>
            ))}
          </div>
        </div>
        <UsageRanking items={dimensionItems} label={label} />
      </section>
    </div>
  )
}

function UsageFilters(props: Props) {
  const subjects =
    props.usageScope.type === 'user'
      ? props.users
      : props.usageScope.type === 'team'
        ? props.teams
        : props.usageScope.type === 'key'
          ? props.keys
          : []
  const models = [
    ...new Set([
      ...props.usage.byModel.map((item) => item.id),
      ...props.groups.flatMap((group) =>
        group.modelPatterns.filter((pattern) => !pattern.includes('*')),
      ),
    ]),
  ]
  return (
    <div className={styles.filterRail}>
      <div className={styles.segmented}>
        {(['24h', '7d', '30d'] as const).map((range) => (
          <button
            type="button"
            key={range}
            className={props.usageScope.range === range ? styles.segmentedActive : ''}
            onClick={() => props.onUsageScopeChange({ ...props.usageScope, range })}
          >
            {range}
          </button>
        ))}
      </div>
      <label>
        <span>Scope</span>
        <select
          value={props.usageScope.type}
          onChange={(event) =>
            props.onUsageScopeChange({
              ...props.usageScope,
              type: event.target.value as UsageScope['type'],
              id: '',
            })
          }
        >
          <option value="global">All traffic</option>
          <option value="user">User</option>
          <option value="team">Team</option>
          <option value="key">API key</option>
        </select>
      </label>
      {props.usageScope.type !== 'global' ? (
        <label>
          <span>{props.usageScope.type === 'key' ? 'API key' : props.usageScope.type}</span>
          <select
            value={props.usageScope.id}
            onChange={(event) =>
              props.onUsageScopeChange({ ...props.usageScope, id: event.target.value })
            }
          >
            <option value="">All</option>
            {subjects.map((subject) => (
              <option value={subject.id} key={subject.id}>
                {'prefix' in subject ? `${subject.name} · ${subject.prefix}` : subject.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <label>
        <span>Model</span>
        <select
          value={props.usageScope.model}
          onChange={(event) =>
            props.onUsageScopeChange({ ...props.usageScope, model: event.target.value })
          }
        >
          <option value="">All models</option>
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </label>
      {props.loading ? <span className={styles.filterLoading}>Updating…</span> : null}
    </div>
  )
}

function UsageMetric({
  label,
  value,
  detail,
  tone,
}: {
  label: string
  value: string
  detail: string
  tone: string
}) {
  return (
    <article className={`${styles.usageMetric} ${styles[`metric${tone}`]}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function TrendChart({
  points,
  metric,
}: {
  points: UsagePoint[]
  metric: 'tokens' | 'requests' | 'latency'
}) {
  if (!points.length)
    return (
      <div className={styles.usageEmpty}>
        <strong>No usage yet</strong>
        <span>Managed requests will appear here.</span>
      </div>
    )
  const width = 1000
  const height = 260
  const xPad = 28
  const yPad = 24
  const value = (point: UsagePoint) =>
    metric === 'tokens'
      ? point.totalTokens
      : metric === 'requests'
        ? point.requests
        : point.averageLatencyMs
  const max = Math.max(...points.map(value), 1)
  const step = (width - xPad * 2) / Math.max(points.length - 1, 1)
  const chartPoints = points.map((point, index) => ({
    point,
    value: value(point),
    x: xPad + index * step,
    y: height - yPad - (value(point) / max) * (height - yPad * 2),
  }))
  const line = chartPoints
    .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ')
  const area = `${line} L${chartPoints[chartPoints.length - 1].x},${height - yPad} L${chartPoints[0].x},${height - yPad} Z`
  const labelIndexes = new Set([
    0,
    points.length - 1,
    ...points
      .map((_, index) => index)
      .filter((index) => index % Math.max(1, Math.floor(points.length / 5)) === 0),
  ])
  return (
    <div className={styles.usageChartWrap}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${metric} over time`}
      >
        {[0.25, 0.5, 0.75, 1].map((ratio) => (
          <line
            key={ratio}
            x1={xPad}
            x2={width - xPad}
            y1={height - yPad - ratio * (height - yPad * 2)}
            y2={height - yPad - ratio * (height - yPad * 2)}
            className={styles.gridLine}
          />
        ))}
        <defs>
          <linearGradient id={`usage-${metric}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#70a5ff" stopOpacity=".32" />
            <stop offset="1" stopColor="#70a5ff" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={area} fill={`url(#usage-${metric})`} />
        <path d={line} className={styles.chartLine} />
        {chartPoints.map((item) => (
          <circle
            key={item.point.bucket}
            cx={item.x}
            cy={item.y}
            r="3"
            className={styles.chartPoint}
          >
            <title>
              {number(item.value)} {metric}
            </title>
          </circle>
        ))}
      </svg>
      <div className={styles.chartLabels}>
        {chartPoints
          .filter((_, index) => labelIndexes.has(index))
          .map(({ point }) => (
            <span key={point.bucket}>
              {new Intl.DateTimeFormat('en-US', {
                month: points.length > 48 ? 'short' : undefined,
                day: points.length > 48 ? 'numeric' : undefined,
                hour: points.length <= 48 ? 'numeric' : undefined,
              }).format(new Date(point.bucket))}
            </span>
          ))}
      </div>
    </div>
  )
}

function Timing({ label, value, max }: { label: string; value: number; max: number }) {
  return (
    <div>
      <div>
        <span>{label}</span>
        <strong>{value ? `${number(value)} ms` : '—'}</strong>
      </div>
      <div>
        <i style={{ width: `${value ? Math.max(3, (value / max) * 100) : 0}%` }} />
      </div>
    </div>
  )
}

function UsageRanking({ items, label }: { items: UsageSlice[]; label: (id: string) => string }) {
  return (
    <div className={styles.usageTableWrap}>
      <table className={styles.usageTable}>
        <thead>
          <tr>
            <th>Name</th>
            <th>Requests</th>
            <th>Success</th>
            <th>Input</th>
            <th>Output</th>
            <th>Total tokens</th>
            <th>P95</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id}>
              <td title={label(item.id)}>{label(item.id)}</td>
              <td>{number(item.requests)}</td>
              <td>{percent(item.successful, item.requests)}</td>
              <td>{number(item.promptTokens)}</td>
              <td>{number(item.completionTokens)}</td>
              <td>{number(item.totalTokens)}</td>
              <td>{number(item.p95LatencyMs)} ms</td>
            </tr>
          ))}
          {!items.length ? (
            <tr>
              <td colSpan={7}>
                <div className={styles.usageEmpty}>
                  <strong>No activity</strong>
                  <span>No usage in this window.</span>
                </div>
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  )
}
