import { useMemo } from 'react'

import { DataTable } from '../components/DataTable'
import TableHeader from '../components/TableHeader'

import emptyStyles from './InsightsEmptyState.module.css'
import { modelText, useModelLabels } from './insightsModelLabelSupport'
import styles from './InsightsPage.module.css'
import type { InsightsRecord } from './insightsPageTypes'
import {
  createReviewTableColumns,
  REVIEW_RESULT_OPTIONS,
  REVIEW_RISK_OPTIONS,
  type ReviewFilters,
} from './insightsReviewSupport'

interface InsightsReviewsTableProps {
  records: InsightsRecord[]
  total: number
  filters: ReviewFilters
  sessionInput: string
  modelOptions: string[]
  hasReviews: boolean
  onSessionInput: (value: string) => void
  onFilterChange: (key: 'result' | 'risk' | 'model', value: string) => void
}

export function InsightsReviewsEmptyState() {
  return (
    <section className={emptyStyles.emptyState} aria-labelledby="insights-reviews-empty-title">
      <h3 id="insights-reviews-empty-title" className={emptyStyles.title}>
        No approval reviews yet
      </h3>
      <p className={emptyStyles.description}>
        Reviews appear here when Codex sends its Approve for me reviews through the gateway and
        Router Replay records the route that answers them.
      </p>
    </section>
  )
}

export default function InsightsReviewsTable({
  records,
  total,
  filters,
  sessionInput,
  modelOptions,
  hasReviews,
  onSessionInput,
  onFilterChange,
}: InsightsReviewsTableProps) {
  const columns = useMemo(() => createReviewTableColumns(), [])
  const labels = useModelLabels()
  return (
    <>
      <TableHeader
        title="Approval Reviews"
        count={total}
        searchPlaceholder="Filter by session ID..."
        searchValue={sessionInput}
        onSearchChange={onSessionInput}
        variant="embedded"
      />
      <div className={styles.filterRow}>
        <select
          className={styles.filterSelect}
          aria-label="Result"
          value={filters.result}
          onChange={(event) => onFilterChange('result', event.target.value)}
        >
          <option value="all">All results</option>
          {REVIEW_RESULT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <select
          className={styles.filterSelect}
          aria-label="Risk"
          value={filters.risk}
          onChange={(event) => onFilterChange('risk', event.target.value)}
        >
          <option value="all">All risk levels</option>
          {REVIEW_RISK_OPTIONS.map((risk) => (
            <option key={risk} value={risk}>
              {risk}
            </option>
          ))}
        </select>
        <select
          className={styles.filterSelect}
          aria-label="Review model"
          value={filters.model}
          onChange={(event) => onFilterChange('model', event.target.value)}
          disabled={modelOptions.length === 0}
        >
          <option value="all">All review models</option>
          {modelOptions.map((model) => (
            <option key={model} value={model}>
              {modelText(labels, model)}
            </option>
          ))}
        </select>
      </div>
      {hasReviews ? (
        <DataTable
          columns={columns}
          data={records}
          keyExtractor={(row) => row.id}
          emptyMessage="No reviews match your current filters"
          className={styles.insightsTable}
        />
      ) : (
        <InsightsReviewsEmptyState />
      )}
    </>
  )
}
