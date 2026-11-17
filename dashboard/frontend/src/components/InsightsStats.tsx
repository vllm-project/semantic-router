import styles from './InsightsCharts.module.css'
import { isInsightsNoCurrency } from '../utils/insightsCost'

export interface InsightsStat {
  label: string
  value: string
  /** The figure its block is about, marked with the accent. */
  lead?: boolean
  tone?: 'negative' | 'muted'
  caption?: string
}

const toneClassName = {
  negative: styles.summaryValueNegative,
  muted: styles.summaryValueNeutral,
}

export function InsightsStatGrid({ stats }: { stats: InsightsStat[] }) {
  return (
    <div className={styles.summaryGrid}>
      {stats.map((stat) => (
        <article
          key={stat.label}
          className={`${styles.summaryCard} ${stat.lead ? styles.summaryCardLead : ''}`.trim()}
        >
          <span className={styles.summaryLabel}>{stat.label}</span>
          <strong
            className={`${styles.summaryValue} ${stat.tone ? toneClassName[stat.tone] : ''}`.trim()}
          >
            {stat.value}
          </strong>
          {stat.caption ? <span className={styles.summaryCaption}>{stat.caption}</span> : null}
        </article>
      ))}
    </div>
  )
}

/** Marks amounts whose prices name no currency (ISO 4217 XXX): they are plain price units. */
export function InsightsUnitsBadge({ currencies }: { currencies: Array<string | undefined> }) {
  return currencies.some((currency) => isInsightsNoCurrency(currency)) ? (
    <span
      className={styles.unitsBadge}
      title="The configured prices name no currency (XXX), so amounts are in price units."
    >
      Price units
    </span>
  ) : null
}
