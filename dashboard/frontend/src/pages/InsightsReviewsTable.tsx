import { useMemo, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { ShieldCheck } from 'lucide-react'

import { DataTable, DataToolbar, EmptyState, FilterPill } from '../ui'

import { modelText, useModelLabels } from './insightsModelLabelSupport'
import { getInsightsRecordPath } from './insightsPageSupport'
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
  loading?: boolean
  actions?: ReactNode
  onSessionInput: (value: string) => void
  onFilterChange: (key: 'result' | 'risk' | 'model', value: string) => void
}

export function InsightsReviewsEmptyState() {
  return (
    <EmptyState
      compact
      icon={ShieldCheck}
      tone="blue"
      title="No approval reviews yet"
      description="Reviews appear here when Codex sends its Approve for me reviews through the gateway and Router Replay records the route that answers them."
      footer={
        <a
          className="underline-offset-4 hover:text-foreground hover:underline"
          href="https://vllm-sr.ai/docs/installation/agent-harness#approve-for-me-reviews"
          target="_blank"
          rel="noopener noreferrer"
        >
          Set up Approve for me reviews
        </a>
      }
    />
  )
}

// One choice at a time: the API reads a single value for each of these filters.
const single = (selected: string[], current: string) =>
  selected.filter((value) => value !== current).pop() ?? 'all'

export default function InsightsReviewsTable({
  records,
  total,
  filters,
  sessionInput,
  modelOptions,
  hasReviews,
  loading = false,
  actions,
  onSessionInput,
  onFilterChange,
}: InsightsReviewsTableProps) {
  const navigate = useNavigate()
  const columns = useMemo(() => createReviewTableColumns(), [])
  const labels = useModelLabels()
  const picked = (value: string) => (value === 'all' ? [] : [value])
  return (
    <>
      <DataToolbar
        search={sessionInput}
        onSearchChange={onSessionInput}
        placeholder="Filter by session ID…"
        filters={
          <>
            <FilterPill
              label="Result"
              options={REVIEW_RESULT_OPTIONS}
              selected={picked(filters.result)}
              onChange={(next) => onFilterChange('result', single(next, filters.result))}
            />
            <FilterPill
              label="Risk"
              options={REVIEW_RISK_OPTIONS.map((risk) => ({ value: risk, label: risk }))}
              selected={picked(filters.risk)}
              onChange={(next) => onFilterChange('risk', single(next, filters.risk))}
            />
            {modelOptions.length > 0 ? (
              <FilterPill
                label="Review model"
                options={modelOptions.map((model) => ({
                  value: model,
                  label: modelText(labels, model),
                }))}
                selected={picked(filters.model)}
                onChange={(next) => onFilterChange('model', single(next, filters.model))}
              />
            ) : null}
          </>
        }
        actions={actions}
      />
      <section aria-label="Approval reviews" data-total={total}>
        <DataTable
          data={hasReviews ? records : []}
          columns={columns}
          getRowId={(row) => row.id}
          loading={loading && records.length === 0}
          onRowClick={(row) => navigate(getInsightsRecordPath(row.id))}
          empty={
            hasReviews ? (
              <EmptyState
                compact
                icon={ShieldCheck}
                tone="blue"
                title="No reviews match"
                description="Clear the session or the filters to see every review."
              />
            ) : (
              <InsightsReviewsEmptyState />
            )
          }
        />
      </section>
    </>
  )
}
