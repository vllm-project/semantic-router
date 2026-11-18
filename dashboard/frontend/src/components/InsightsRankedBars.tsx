import type { InsightsAggregateTokenEntry } from '../pages/insightsPageTypes'
import styles from './InsightsCharts.module.css'
import {
  formatInsightsCompactCount,
  formatInsightsCount,
  formatInsightsShare,
} from '../utils/insightsCost'

export interface InsightsRankedEntry {
  name: string
  value: number
}

interface InsightsRankedBarsProps {
  title: string
  entries: InsightsRankedEntry[]
  /** The whole each share is of: the entries' sum when the response lists every entry. */
  total?: number
  meta?: string
  /** Rows shown before the rest fold away. */
  visible?: number
}

const barWidth = (value: number, peak: number) => `${peak > 0 ? (value / peak) * 100 : 0}%`

/** A distribution as ranked rows, which stay legible however many entries a history holds. */
export default function InsightsRankedBars({
  title,
  entries,
  total,
  meta,
  visible = 8,
}: InsightsRankedBarsProps) {
  const ranked = [...entries].sort((a, b) => b.value - a.value || a.name.localeCompare(b.name))
  const whole = total ?? ranked.reduce((sum, entry) => sum + entry.value, 0)
  const peak = ranked[0]?.value ?? 0
  const row = (entry: InsightsRankedEntry, rank: number) => (
    <li key={entry.name} className={styles.rankedRow}>
      <span className={styles.rankedName} title={entry.name}>
        {entry.name}
      </span>
      <span className={styles.rankedValue}>{formatInsightsCount(entry.value)}</span>
      <span className={styles.rankedShare}>{formatInsightsShare(entry.value, whole)}</span>
      <span className={styles.rankedTrack} aria-hidden="true">
        <span
          className={rank === 0 ? styles.rankedFillLead : styles.rankedFill}
          style={{ width: barWidth(entry.value, peak) }}
        />
      </span>
    </li>
  )
  return (
    <section className={styles.panel} aria-label={title}>
      <div className={styles.panelHeader}>
        <h4 className={styles.panelTitle}>{title}</h4>
        {meta ? <span className={styles.panelMeta}>{meta}</span> : null}
      </div>
      {ranked.length === 0 ? (
        <p className={styles.summaryHint}>Nothing recorded yet.</p>
      ) : (
        <>
          <ol className={styles.rankedList}>
            {ranked.slice(0, visible).map((entry, rank) => row(entry, rank))}
          </ol>
          {ranked.length > visible ? (
            <details className={styles.rankedMore}>
              <summary>{ranked.length - visible} more</summary>
              <ol className={styles.rankedList} start={visible + 1}>
                {ranked.slice(visible).map((entry, rank) => row(entry, rank + visible))}
              </ol>
            </details>
          ) : null}
        </>
      )}
    </section>
  )
}

/** Each entry's tokens as one bar split into input and output, ranked by the total. */
export function InsightsTokenRows({
  title,
  entries,
}: {
  title: string
  entries: InsightsAggregateTokenEntry[]
}) {
  const peak = Math.max(0, ...entries.map((entry) => entry.total_tokens))
  return (
    <section className={styles.panel} aria-label={title}>
      <div className={styles.panelHeader}>
        <h4 className={styles.panelTitle}>{title}</h4>
        <span className={styles.legend}>
          <span className={styles.legendItem}>
            <span className={`${styles.legendSwatch} ${styles.tokenInput}`} />
            Input
          </span>
          <span className={styles.legendItem}>
            <span className={`${styles.legendSwatch} ${styles.tokenOutput}`} />
            Output
          </span>
        </span>
      </div>
      {entries.length === 0 ? (
        <p className={styles.summaryHint}>Nothing recorded yet.</p>
      ) : (
        <ol className={styles.rankedList}>
          {entries.map((entry) => (
            <li key={entry.name} className={styles.rankedRow}>
              <span className={styles.rankedName} title={entry.name}>
                {entry.name}
              </span>
              <span
                className={styles.rankedValue}
                title={`${formatInsightsCount(entry.total_tokens)} tokens`}
              >
                {formatInsightsCompactCount(entry.total_tokens)}
              </span>
              <span className={styles.rankedShare}>
                {formatInsightsShare(entry.output_tokens, entry.total_tokens)} out
              </span>
              <span className={styles.rankedTrack} aria-hidden="true">
                <span
                  className={styles.tokenInput}
                  style={{ width: barWidth(entry.input_tokens, peak) }}
                />
                <span
                  className={styles.tokenOutput}
                  style={{ width: barWidth(entry.output_tokens, peak) }}
                />
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
