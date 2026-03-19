import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ModelResearchCampaign } from '../types/modelResearch'
import {
  buildTrendPoints,
  formatImprovementPP,
  formatPercent,
  getCampaignElapsedLabel,
  getCompletedTrialCount,
} from './modelResearchPageSupport'
import styles from './ModelResearchTrendPanel.module.css'

interface ModelResearchTrendPanelProps {
  campaign: ModelResearchCampaign
}

function formatPercentTick(value: number) {
  return `${value.toFixed(0)}%`
}

export default function ModelResearchTrendPanel({ campaign }: ModelResearchTrendPanelProps) {
  const points = buildTrendPoints(campaign)
  const bestAccuracy = campaign.best_trial?.eval?.accuracy
  const latestPoint = points[points.length - 1]
  const baselineAccuracyPct = campaign.baseline_eval?.accuracy != null ? campaign.baseline_eval.accuracy * 100 : null

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <div>
          <h3 className={styles.title}>Progress trend</h3>
          <p className={styles.description}>
            Track how many rounds the campaign has completed, how much time it has spent, and
            whether best-so-far accuracy is still moving upward.
          </p>
        </div>
      </div>

      <div className={styles.summaryGrid}>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Rounds recorded</span>
          <strong className={styles.summaryValue}>{getCompletedTrialCount(campaign)}</strong>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Elapsed</span>
          <strong className={styles.summaryValue}>{getCampaignElapsedLabel(campaign)}</strong>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Current best</span>
          <strong className={styles.summaryValue}>
            {bestAccuracy != null ? formatPercent(bestAccuracy) : 'N/A'}
          </strong>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Last recorded lift</span>
          <strong className={styles.summaryValue}>
            {latestPoint ? formatImprovementPP(latestPoint.improvementPP) : 'N/A'}
          </strong>
        </article>
      </div>

      <div className={styles.chartShell}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={formatPercentTick}
              tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={42}
            />
            <Tooltip
              cursor={false}
              labelFormatter={(_label, payload) => {
                const point = payload?.[0]?.payload as
                  | { label?: string; elapsedLabel?: string; improvementPP?: number }
                  | undefined
                if (!point) {
                  return ''
                }
                return `${point.label || 'Point'} · ${point.elapsedLabel || '0m'} · ${formatImprovementPP(
                  point.improvementPP
                )}`
              }}
              formatter={(value: number | string, name: string) => {
                const metricLabel =
                  name === 'accuracyPct'
                    ? 'Trial accuracy'
                    : name === 'bestAccuracyPct'
                      ? 'Best so far'
                      : String(name)
                return [`${Number(value).toFixed(2)}%`, metricLabel]
              }}
              contentStyle={{
                background: 'var(--color-bg-secondary)',
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                color: 'var(--color-text)',
              }}
              itemStyle={{ color: 'var(--color-text)' }}
            />
            <Legend
              verticalAlign="top"
              height={28}
              wrapperStyle={{ color: 'var(--color-text-secondary)', fontSize: '12px' }}
            />
            {baselineAccuracyPct != null ? (
              <ReferenceLine
                y={baselineAccuracyPct}
                stroke="rgba(148, 163, 184, 0.45)"
                strokeDasharray="4 4"
              />
            ) : null}
            <Line
              type="monotone"
              dataKey="accuracyPct"
              name="Trial accuracy"
              stroke="#00d4ff"
              strokeWidth={2}
              dot={{ r: 4, strokeWidth: 0, fill: '#00d4ff' }}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="bestAccuracyPct"
              name="Best so far"
              stroke="#76b900"
              strokeWidth={2}
              dot={{ r: 4, strokeWidth: 0, fill: '#76b900' }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
