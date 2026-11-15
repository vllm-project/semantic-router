import { Link } from 'react-router-dom'

import type { Column } from '../components/DataTable'
import type { ViewField, ViewSection } from '../components/ViewPanel'
import { formatDateTime, formatSeconds } from '../utils/dateTime'
import { formatInsightsCost } from '../utils/insightsCost'

import styles from './InsightsPage.module.css'
import type { InsightsRecord } from './insightsPageTypes'
import type { InsightsTrajectoryRoute } from './insightsPageRoutingTypes'
import type { InsightsReview } from './insightsReviewTypes'

export const REVIEW_REQUEST_KIND = 'codex_approval_review'

export interface ReviewFilters {
  session: string
  result: string
  risk: string
  model: string
}

export const EMPTY_REVIEW_FILTERS: ReviewFilters = {
  session: '',
  result: 'all',
  risk: 'all',
  model: 'all',
}

// A result is the verdict of a completed review, else the review's status.
export const REVIEW_RESULT_OPTIONS = [
  { value: 'allow', label: 'Allowed' },
  { value: 'deny', label: 'Denied' },
  { value: 'failed', label: 'Failed' },
  { value: 'invalid', label: 'Invalid answer' },
  { value: 'cancelled', label: 'Cancelled' },
  { value: 'tool_call', label: 'Tool check' },
  { value: 'pending', label: 'In progress' },
]

export const REVIEW_RISK_OPTIONS = ['low', 'medium', 'high', 'critical']

export function buildReviewQueryString(
  filters: ReviewFilters,
  pagination?: { limit: number; offset: number },
) {
  const params = new URLSearchParams({ request_kind: REVIEW_REQUEST_KIND })
  if (pagination) {
    params.set('limit', String(pagination.limit))
    params.set('offset', String(pagination.offset))
    params.set('showDetails', 'false')
  }
  const session = filters.session.trim()
  if (session) params.set('session_id', session)
  if (filters.result === 'allow' || filters.result === 'deny') {
    params.set('review_outcome', filters.result)
  } else if (filters.result !== 'all') {
    params.set('review_status', filters.result)
  }
  if (filters.risk !== 'all') params.set('review_risk', filters.risk)
  if (filters.model !== 'all') params.set('model', filters.model)
  return `?${params.toString()}`
}

export function getReviewsPath(session?: string) {
  const params = new URLSearchParams({ view: 'reviews' })
  if (session) params.set('session', session)
  return `/insights?${params.toString()}`
}

type ReviewTone = 'allow' | 'deny' | 'failed' | 'muted' | 'pending'

export function reviewResult(review?: InsightsReview): { label: string; tone: ReviewTone } {
  if (review?.status === 'completed' && review.assessment) {
    return review.assessment.outcome === 'allow'
      ? { label: 'Allowed', tone: 'allow' }
      : { label: 'Denied', tone: 'deny' }
  }
  switch (review?.status) {
    case 'failed':
      return { label: 'Failed', tone: 'failed' }
    case 'invalid':
      return { label: 'Invalid answer', tone: 'failed' }
    case 'cancelled':
      return { label: 'Cancelled', tone: 'muted' }
    case 'tool_call':
      return { label: 'Tool check', tone: 'muted' }
    case 'pending':
      return { label: 'In progress', tone: 'pending' }
    default:
      return { label: 'Not recorded', tone: 'muted' }
  }
}

export function ReviewResultBadge({ review }: { review?: InsightsReview }) {
  const result = reviewResult(review)
  return (
    <span className={styles.reviewResult} data-tone={result.tone}>
      {result.label}
    </span>
  )
}

const compactID = (value: string) =>
  value.length <= 20 ? value : `${value.slice(0, 8)}…${value.slice(-8)}`

function reviewLatency(record: InsightsRecord) {
  const latency = record.review?.latency_ms || record.duration_ms
  return typeof latency === 'number' && latency > 0 ? formatSeconds(latency) : '-'
}

function reviewTokens(record: InsightsRecord) {
  if (typeof record.prompt_tokens !== 'number' && typeof record.completion_tokens !== 'number') {
    return typeof record.total_tokens === 'number' ? String(record.total_tokens) : '-'
  }
  return `${record.prompt_tokens ?? 0} in · ${record.completion_tokens ?? 0} out`
}

function reviewCost(record: InsightsRecord) {
  return typeof record.actual_cost === 'number' && record.currency
    ? formatInsightsCost(record.actual_cost, record.currency)
    : 'Not priced'
}

// Tokens and cost share a cell, so the table fits beside the action.
function ReviewUsage({ record }: { record: InsightsRecord }) {
  return (
    <span className={styles.reviewUsage}>
      <span>{reviewCost(record)}</span>
      <small>{reviewTokens(record)}</small>
    </span>
  )
}

function ActionCell({ review }: { review?: InsightsReview }) {
  const action = review?.action
  return (
    <span className={styles.reviewAction} title={action?.summary || undefined}>
      {action?.tool ? <code>{action.tool}</code> : null}
      <span>{action?.summary || (review?.content_redacted ? 'Hidden from this viewer' : '')}</span>
    </span>
  )
}

export function createReviewTableColumns(): Column<InsightsRecord>[] {
  return [
    {
      key: 'timestamp',
      header: 'Time',
      width: '135px',
      render: (row) => (
        <Link className={styles.requestIdLink} to={`/insights/${encodeURIComponent(row.id)}`}>
          <span className={styles.timestamp}>{formatDateTime(row.timestamp)}</span>
        </Link>
      ),
    },
    {
      key: 'result',
      header: 'Result',
      width: '95px',
      render: (row) => <ReviewResultBadge review={row.review} />,
    },
    {
      key: 'risk',
      header: 'Risk',
      width: '70px',
      render: (row) => (
        <span className={styles.reviewRisk} data-risk={row.review?.assessment?.risk_level}>
          {row.review?.assessment?.risk_level || '-'}
        </span>
      ),
    },
    {
      key: 'action',
      header: 'Action',
      width: '240px',
      render: (row) => <ActionCell review={row.review} />,
    },
    {
      key: 'session_id',
      header: 'Session',
      width: '115px',
      render: (row) =>
        row.session_id ? (
          <span className={styles.requestId} title={row.session_id}>
            {compactID(row.session_id)}
          </span>
        ) : (
          '-'
        ),
    },
    {
      key: 'selected_model',
      header: 'Model',
      width: '135px',
      render: (row) => (
        <span className={styles.reviewModel} title={row.selected_model}>
          {row.selected_model || '-'}
        </span>
      ),
    },
    { key: 'latency', header: 'Latency', width: '80px', render: reviewLatency },
    {
      key: 'actual_cost',
      header: 'Cost',
      width: '120px',
      render: (record) => <ReviewUsage record={record} />,
    },
  ]
}

function hiddenOr(value: string | undefined, review: InsightsReview, missing: string) {
  if (value) return value
  return review.content_redacted ? 'Hidden from this viewer' : missing
}

function reviewActionFields(review: InsightsReview): ViewField[] {
  const action = review.action
  const fields: ViewField[] = [
    { label: 'Tool', value: action?.tool || 'Not recorded' },
    { label: 'Action', value: hiddenOr(action?.summary, review, 'Not recorded') },
    { label: 'Working directory', value: hiddenOr(action?.cwd, review, 'Not recorded') },
  ]
  if (action?.arguments) {
    fields.push({
      label: 'Planned action',
      fullWidth: true,
      value: (
        <pre className={styles.reviewArguments}>
          {action.arguments}
          {action.arguments_truncated ? '\n… cut at the record limit' : ''}
        </pre>
      ),
    })
  }
  return fields
}

/** The record detail section of an approval review, or null for other requests. */
export function buildReviewSection(record: InsightsRecord): ViewSection | null {
  const review = record.review
  if (!review) return null
  const assessment = review.assessment
  const session = review.reviewed_session_id
  return {
    title: 'Approval Review',
    fields: [
      { label: 'Result', value: <ReviewResultBadge review={review} /> },
      { label: 'Risk', value: assessment?.risk_level || 'Not reported' },
      { label: 'User authorization', value: assessment?.user_authorization || 'Not reported' },
      { label: 'Latency', value: reviewLatency(record) },
      {
        label: 'Rationale',
        fullWidth: true,
        value: assessment
          ? `${hiddenOr(assessment.rationale, review, 'The reviewer gave none.')}${assessment.rationale_truncated ? ' …' : ''}`
          : 'No verdict, so no rationale.',
      },
      ...reviewActionFields(review),
      {
        label: 'Reviewed session',
        value: session ? <Link to={getReviewsPath(session)}>{compactID(session)}</Link> : '-',
      },
      { label: 'Reviewer thread', value: review.thread_id || '-' },
      { label: 'Planned in turn', value: review.parent_turn_id || '-' },
      {
        label: 'Recognised by',
        value:
          review.identification_source === 'entrypoint'
            ? 'Review entrypoint'
            : "Codex's review request",
      },
    ],
  }
}

/** Counts a session's settled reviews for its summary strip. */
export function summarizeReviews(reviews: InsightsRecord[]) {
  const summary = { total: 0, allowed: 0, denied: 0, failed: 0 }
  for (const record of reviews) {
    const status = record.review?.status
    if (!status || status === 'pending' || status === 'tool_call') continue
    summary.total += 1
    const tone = reviewResult(record.review).tone
    if (tone === 'allow') summary.allowed += 1
    else if (tone === 'deny') summary.denied += 1
    else if (tone === 'failed') summary.failed += 1
  }
  return summary
}

/**
 * Places each review under the latest request of the session that started
 * before it: the turn whose planned action it judged. A review that is itself
 * one of the routes is marked as a review instead.
 */
export function assignReviewsToRoutes(
  routes: InsightsTrajectoryRoute[],
  reviews: InsightsRecord[],
): Map<string, InsightsRecord[]> {
  const reviewIDs = new Set(reviews.map((review) => review.id))
  const turns = routes
    .filter((route) => !reviewIDs.has(route.record_id))
    .map((route) => ({ id: route.record_id, start: Date.parse(route.timestamp) }))
    .filter((turn) => Number.isFinite(turn.start))
    .sort((left, right) => left.start - right.start)
  const marks = new Map<string, InsightsRecord[]>()
  for (const review of reviews) {
    const start = Date.parse(review.timestamp)
    const turn = turns.filter((candidate) => candidate.start <= start).pop()
    if (!turn) continue
    marks.set(turn.id, [...(marks.get(turn.id) ?? []), review])
  }
  return marks
}
