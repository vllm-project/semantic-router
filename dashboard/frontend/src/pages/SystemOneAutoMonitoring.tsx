import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useAuth } from '../contexts/AuthContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import { withRequestTimeout } from '../utils/boundedRequest'
import {
  DECISION_MODEL_TIME_WINDOWS,
  formatDecisionModelMetric,
  type DecisionModelTimeWindow,
} from './decisionModelMetrics'
import {
  AUTO_METRICS,
  loadSystemOneAutoMetrics,
  type AutoMetricSnapshot,
} from './systemOneAutoMetrics'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelPage.module.css'
import autoStyles from './SystemOneAutoMonitoring.module.css'

const empty: AutoMetricSnapshot = { values: {}, stages: [], unavailable: [] }
const outcomes = [
  { key: 'accepted', label: 'Accepted by gate', color: '#49bba5' },
  { key: 'rejected', label: 'Did not pass gate', color: '#dca965' },
  { key: 'failed', label: 'Errors or invalid results', color: '#ee8b9b' },
] as const

export default function SystemOneAutoMonitoring({ refreshedAt }: { refreshedAt: Date | null }) {
  const { user } = useAuth()
  const canReadMetrics = canAccessDashboardPath(user, '/monitoring')
  const [window, setWindow] = useState<DecisionModelTimeWindow>(3600)
  const [snapshot, setSnapshot] = useState(empty)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    setSnapshot(empty)
  }, [window])
  useEffect(() => {
    if (!canReadMetrics || document.visibilityState === 'hidden') return
    const controller = new AbortController()
    setLoading(true)
    void withRequestTimeout(
      (signal) => loadSystemOneAutoMetrics(window, signal),
      controller.signal,
      8000,
    )
      .then((next) => {
        if (!controller.signal.aborted) setSnapshot(next)
      })
      .catch(() => {
        if (!controller.signal.aborted) setSnapshot({ ...empty, unavailable: ['Auto metrics'] })
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [window, refreshedAt, canReadMetrics])
  return (
    <section className={styles.panel} aria-labelledby="auto-monitoring-title">
      <div className={styles.monitoringHeader}>
        <div>
          <h2 id="auto-monitoring-title">Automatic routing</h2>
          <p className={styles.muted}>
            All native auto routes on this instance. Acceptance measures the configured gate, not
            answer accuracy.
          </p>
        </div>
        <SystemOneSelect
          label="Stage observation window"
          value={String(window)}
          onChange={(value) => setWindow(Number(value) as DecisionModelTimeWindow)}
          options={DECISION_MODEL_TIME_WINDOWS.map((item) => ({
            value: String(item.seconds),
            label: item.label,
          }))}
        />
      </div>
      {!canReadMetrics ? (
        <p className={styles.muted}>Your role does not include monitoring access.</p>
      ) : (
        <>
          <dl
            className={`${styles.statistics} ${autoStyles.statistics}`}
            aria-label="Auto routing statistics"
          >
            {AUTO_METRICS.map((metric) => (
              <div key={metric.key}>
                <dt>{metric.label}</dt>
                <dd>{formatDecisionModelMetric(snapshot.values[metric.key], metric.unit)}</dd>
              </div>
            ))}
          </dl>
          <p className={styles.muted}>
            Statistics use a rolling 5-minute window. Execution time starts after recipe signals and
            excludes their latency.
          </p>
          {snapshot.unavailable.length > 0 && (
            <p className={styles.notice}>
              Unavailable observations: {snapshot.unavailable.join(', ')}
            </p>
          )}
          <div className={styles.chartCard}>
            <div className={styles.chartHeader}>
              <div>
                <h4>Stage activity</h4>
                <p>
                  Estimated stage attempts in the selected window, grouped by algorithm and model. A
                  request can visit several stages.
                </p>
              </div>
            </div>
            <div className={styles.chartLegend}>
              {outcomes.map((outcome) => (
                <span key={outcome.key}>
                  <i style={{ background: outcome.color }} />
                  {outcome.label}
                </span>
              ))}
            </div>
            <div
              className={styles.chartCanvas}
              style={{ height: Math.max(220, snapshot.stages.length * 52) }}
            >
              {snapshot.stages.length ? (
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <BarChart data={snapshot.stages} layout="vertical" accessibilityLayer>
                    <CartesianGrid
                      horizontal={false}
                      stroke="var(--color-border)"
                      strokeDasharray="3 5"
                    />
                    <XAxis
                      type="number"
                      tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    />
                    <YAxis
                      type="category"
                      dataKey="label"
                      width={160}
                      tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border)',
                        borderRadius: 10,
                      }}
                    />
                    {outcomes.map((outcome) => (
                      <Bar
                        key={outcome.key}
                        dataKey={outcome.key}
                        name={outcome.label}
                        stackId="attempts"
                        fill={outcome.color}
                        maxBarSize={32}
                        isAnimationActive={false}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className={styles.chartEmpty}>
                  {loading
                    ? 'Loading auto observations…'
                    : 'No native auto stage activity reported in this window.'}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  )
}
