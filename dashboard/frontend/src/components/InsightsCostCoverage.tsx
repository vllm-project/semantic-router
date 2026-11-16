import type { InsightsAggregateSummary } from '../pages/insightsPageTypes'
import {
  buildCostCoverageLines,
  describeMissingAggregateCost,
} from '../pages/insightsCostAccounting'
import styles from './InsightsCharts.module.css'

interface InsightsCostCoverageProps {
  summary: InsightsAggregateSummary
}

/** States beside the cost cards how many requests they cover and why the rest are left out. */
export default function InsightsCostCoverage({ summary }: InsightsCostCoverageProps) {
  const coverage = summary.coverage
  if (!coverage) {
    return null
  }
  const missing = describeMissingAggregateCost(coverage)
  return (
    <div className={styles.estimateContext} aria-label="Cost coverage">
      {missing ? <p className={styles.summaryHint}>{missing}</p> : null}
      {buildCostCoverageLines(coverage, summary.cost_record_count).map((line) => (
        <p key={line} className={styles.summaryHint}>
          {line}
        </p>
      ))}
    </div>
  )
}
