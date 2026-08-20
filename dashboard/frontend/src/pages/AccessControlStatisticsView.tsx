import type { UsageSlice, UsageSummary } from '../utils/inferenceAccessApi'
import type { AccessControlViewProps as Props } from './AccessControlViewTypes'
import { Empty, Metric, PanelHeading } from './AccessControlViewPrimitives'
import { number, percent } from './AccessControlViewSupport'
import styles from './AccessControlPage.module.css'

export function Overview(props: Props) {
  const success = percent(props.overview.successfulToday, props.overview.requestsToday)
  return (
    <div className={styles.viewStack}>
      <div className={styles.metricStrip}>
        <Metric
          label="Requests today"
          value={number(props.overview.requestsToday)}
          detail={`${success} successful`}
          tone="blue"
        />
        <Metric
          label="Tokens today"
          value={number(props.overview.tokensToday)}
          detail={`${number(props.usage.completionTokens)} generated`}
          tone="violet"
        />
        <Metric
          label="P95 latency"
          value={`${number(props.overview.p95LatencyMs)} ms`}
          detail={`${number(props.usage.averageLatencyMs)} ms average`}
          tone="amber"
        />
        <Metric
          label="Active keys"
          value={number(props.overview.activeKeys)}
          detail={`${number(props.overview.expiringKeys)} expire soon`}
          tone="green"
        />
        <Metric
          label="Users & teams"
          value={`${number(props.overview.users)} / ${number(props.overview.teams)}`}
          detail="governed identities"
          tone="neutral"
        />
      </div>

      <div className={styles.overviewGrid}>
        <section className={`${styles.panel} ${styles.chartPanel}`}>
          <PanelHeading
            eyebrow="Last 24 hours"
            title="Traffic"
            aside={`${number(props.usage.totalTokens)} tokens`}
          />
          <UsageChart usage={props.usage} />
        </section>
        <section className={styles.panel}>
          <PanelHeading
            eyebrow="Distribution"
            title="Top models"
            aside={`${props.usage.byModel.length} active`}
          />
          <Breakdown items={props.usage.byModel} label={(id) => id} />
        </section>
      </div>

      <div className={styles.breakdownGrid}>
        <section className={styles.panel}>
          <PanelHeading eyebrow="Identity" title="Top users" />
          <Breakdown
            items={props.usage.byUser}
            label={(id) => props.users.find((item) => item.id === id)?.name || id}
          />
        </section>
        <section className={styles.panel}>
          <PanelHeading eyebrow="Ownership" title="Top teams" />
          <Breakdown
            items={props.usage.byTeam}
            label={(id) => props.teams.find((item) => item.id === id)?.name || id}
          />
        </section>
        <section className={styles.panel}>
          <PanelHeading eyebrow="Credentials" title="Top API keys" />
          <Breakdown
            items={props.usage.byKey}
            label={(id) => props.keys.find((item) => item.id === id)?.name || id}
          />
        </section>
      </div>
    </div>
  )
}

function UsageChart({ usage }: { usage: UsageSummary }) {
  if (!usage.series.length)
    return (
      <Empty
        title="No usage yet"
        detail="Traffic will appear here after the first managed request."
      />
    )
  const width = 1000
  const height = 250
  const padX = 28
  const padY = 22
  const max = Math.max(...usage.series.map((point) => point.totalTokens), 1)
  const step = (width - padX * 2) / Math.max(usage.series.length - 1, 1)
  const points = usage.series.map((point, index) => ({
    x: padX + index * step,
    y: height - padY - (point.totalTokens / max) * (height - padY * 2),
    point,
  }))
  const line = points
    .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ')
  const area = `${line} L${points[points.length - 1]?.x},${height - padY} L${points[0]?.x},${height - padY} Z`
  const labels = points.filter(
    (_, index) =>
      index === 0 ||
      index === points.length - 1 ||
      index % Math.max(1, Math.floor(points.length / 5)) === 0,
  )
  return (
    <div className={styles.chartWrap}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Token usage over time"
        preserveAspectRatio="none"
      >
        {[0.25, 0.5, 0.75, 1].map((ratio) => (
          <line
            key={ratio}
            x1={padX}
            x2={width - padX}
            y1={height - padY - ratio * (height - padY * 2)}
            y2={height - padY - ratio * (height - padY * 2)}
            className={styles.gridLine}
          />
        ))}
        <defs>
          <linearGradient id="usage-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#70a5ff" stopOpacity=".34" />
            <stop offset="1" stopColor="#70a5ff" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={area} fill="url(#usage-fill)" />
        <path d={line} className={styles.chartLine} />
        {points.map(({ x, y, point }) => (
          <circle key={point.bucket} cx={x} cy={y} r="3" className={styles.chartPoint}>
            <title>
              {number(point.totalTokens)} tokens · {number(point.requests)} requests
            </title>
          </circle>
        ))}
      </svg>
      <div className={styles.chartLabels}>
        {labels.map(({ point }) => (
          <span key={point.bucket}>
            {new Intl.DateTimeFormat('en-US', {
              month: usage.series.length > 48 ? 'short' : undefined,
              day: usage.series.length > 48 ? 'numeric' : undefined,
              hour: usage.series.length <= 48 ? 'numeric' : undefined,
            }).format(new Date(point.bucket))}
          </span>
        ))}
      </div>
      <div className={styles.chartLegend}>
        <span>
          <i className={styles.legendTotal} />
          Total tokens
        </span>
        <span>{number(usage.promptTokens)} input</span>
        <span>{number(usage.completionTokens)} output</span>
      </div>
    </div>
  )
}

function Breakdown({
  items,
  label,
  detailed = false,
}: {
  items: UsageSlice[]
  label: (id: string) => string
  detailed?: boolean
}) {
  const max = Math.max(...items.map((item) => item.totalTokens), 1)
  if (!items.length) return <Empty compact title="No activity" detail="No usage in this window." />
  return (
    <div className={styles.breakdownList}>
      {items.slice(0, 6).map((item) => (
        <div className={styles.breakdownItem} key={item.id}>
          <div>
            <strong title={label(item.id)}>{label(item.id)}</strong>
            <span>
              {number(item.totalTokens)} tokens
              {detailed
                ? ` · ${number(item.requests)} requests · ${number(item.p95LatencyMs)} ms P95`
                : ''}
            </span>
          </div>
          <div>
            <i style={{ width: `${Math.max(3, (item.totalTokens / max) * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}
