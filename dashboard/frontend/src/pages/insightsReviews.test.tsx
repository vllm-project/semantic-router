import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import InsightsReviewCharts from '../components/InsightsReviewCharts'
import InsightsReviewsTable from './InsightsReviewsTable'
import InsightsSessionRoutes from './InsightsSessionRoutes'
import { fetchInsightsSessionReviews } from './insightsPageApi'
import { buildInsightsRecordSections } from './insightsPageSupport'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'
import {
  assignReviewsToRoutes,
  buildReviewQueryString,
  buildReviewSection,
  EMPTY_REVIEW_FILTERS,
  summarizeReviews,
} from './insightsReviewSupport'
import type { InsightsReview, InsightsReviewAggregate } from './insightsReviewTypes'

function reviewRecord(
  id: string,
  timestamp: string,
  review: Partial<InsightsReview>,
): InsightsRecord {
  return {
    id,
    timestamp,
    turn_index: 0,
    decision_tier: 0,
    decision_priority: 0,
    signals: {},
    session_id: 'agent-session',
    selected_model: 'codex-auto-review',
    request_kind: 'codex_approval_review',
    review: {
      schema_version: 1,
      identification_source: 'entrypoint',
      status: 'completed',
      reviewed_session_id: 'agent-session',
      ...review,
    },
  }
}

const denied = reviewRecord('review-deny', '2026-11-15T09:00:30Z', {
  assessment: { outcome: 'deny', risk_level: 'high', rationale: 'Deletes build output.' },
  action: {
    tool: 'exec_command',
    summary: 'rm -rf build',
    cwd: '/work',
    arguments: '{"command":["rm"]}',
  },
  latency_ms: 1200,
})
const failed = reviewRecord('review-failed', '2026-11-15T09:05:00Z', {
  status: 'failed',
  action: { tool: 'apply_patch' },
})

const render = (node: React.ReactNode) => renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>)

afterEach(() => vi.unstubAllGlobals())

describe('Insights reviews', () => {
  it('maps the filters onto the replay query', () => {
    expect(buildReviewQueryString(EMPTY_REVIEW_FILTERS)).toBe('?request_kind=codex_approval_review')
    const params = new URLSearchParams(
      buildReviewQueryString(
        { session: ' agent-session ', result: 'deny', risk: 'high', model: 'codex-auto-review' },
        { limit: 25, offset: 50 },
      ),
    )
    expect(Object.fromEntries(params)).toEqual({
      request_kind: 'codex_approval_review',
      limit: '25',
      offset: '50',
      showDetails: 'false',
      session_id: 'agent-session',
      review_outcome: 'deny',
      review_risk: 'high',
      model: 'codex-auto-review',
    })
    const status = new URLSearchParams(
      buildReviewQueryString({ ...EMPTY_REVIEW_FILTERS, result: 'failed' }),
    )
    expect(status.get('review_status')).toBe('failed')
    expect(status.has('review_outcome')).toBe(false)
  })

  it('lists reviews with their result, risk, action and cost', () => {
    const html = render(
      <InsightsReviewsTable
        records={[denied, failed]}
        total={2}
        filters={EMPTY_REVIEW_FILTERS}
        sessionInput=""
        modelOptions={['codex-auto-review']}
        hasReviews
        onSessionInput={() => undefined}
        onFilterChange={() => undefined}
      />,
    )
    for (const text of [
      'Denied',
      'Failed',
      'high',
      'exec_command',
      'rm -rf build',
      'Not priced',
      '1.200s',
    ]) {
      expect(html).toContain(text)
    }
    expect(html).toContain('href="/insights/review-deny"')
    expect(html).toContain('<option value="tool_call">Tool check</option>')
    expect(html).toContain('<option value="codex-auto-review">codex-auto-review</option>')
    expect(html).not.toContain('No approval reviews yet')
  })

  it('shows an empty state when no review was recorded', () => {
    const html = render(
      <InsightsReviewsTable
        records={[]}
        total={0}
        filters={EMPTY_REVIEW_FILTERS}
        sessionInput=""
        modelOptions={[]}
        hasReviews={false}
        onSessionInput={() => undefined}
        onFilterChange={() => undefined}
      />,
    )
    expect(html).toContain('No approval reviews yet')
    expect(html).toMatch(/aria-label="Review model"[^>]*disabled/)
  })

  it('summarizes verdicts apart from failures and keeps unknown cost unpriced', () => {
    const reviews: InsightsReviewAggregate = {
      reviews: 6,
      allowed: 3,
      denied: 1,
      invalid: 1,
      failed: 1,
      cancelled: 0,
      pending: 0,
      tool_calls: 1,
      approval_rate: 0.75,
      latency_p50_ms: 200,
      latency_p95_ms: 1000,
      cost: [{ currency: 'USD', amount: 0.075, requests: 4 }],
      unpriced_requests: 3,
      bucket: 'minute',
      over_time: [],
      by_model: [],
    }
    const html = renderToStaticMarkup(<InsightsReviewCharts reviews={reviews} />)
    for (const text of [
      '75.0%',
      '1 tool check ·',
      '3 allowed of 4 verdicts',
      '1 invalid · 0 cancelled',
      '0.200s',
      'p95 1.000s',
      '$0.075',
      '4 of 7 requests priced',
    ]) {
      expect(html).toContain(text)
    }
    const none = renderToStaticMarkup(
      <InsightsReviewCharts
        reviews={{ ...reviews, approval_rate: undefined, cost: [], latency_p50_ms: undefined }}
      />,
    )
    expect(none).toContain('N/A')
    expect(none).toContain('Not priced')
  })

  it('shows the review section first in the record detail', () => {
    const sections = buildInsightsRecordSections(denied, { isReadonly: false })
    expect(sections[0].title).toBe('Approval Review')
    const html = render(
      <>
        {sections[0].fields.map((field) => (
          <div key={field.label}>{field.value}</div>
        ))}
      </>,
    )
    expect(html).toContain('Deletes build output.')
    expect(html).toContain('{&quot;command&quot;:[&quot;rm&quot;]}')
    expect(html).toContain('href="/insights?view=reviews&amp;session=agent-session"')

    const hidden = reviewRecord('hidden', denied.timestamp, {
      content_redacted: true,
      assessment: { outcome: 'deny', rationale: '' },
      action: { tool: 'exec_command', summary: '' },
    })
    const fields = buildReviewSection(hidden)!.fields
    expect(fields.find((field) => field.label === 'Rationale')?.value).toBe(
      'Hidden from this viewer',
    )
    expect(fields.find((field) => field.label === 'Action')?.value).toBe('Hidden from this viewer')
    expect(buildReviewSection({ ...denied, review: undefined })).toBeNull()
  })

  it('marks the turn each review judged in the session history', () => {
    const trajectory: InsightsTrajectory = {
      object: 'router_replay.trajectory',
      session_id: 'agent-session',
      record_count: 2,
      turn_count: 2,
      messages: [],
      routes: [
        {
          record_id: 'turn-1',
          timestamp: '2026-11-15T09:00:00Z',
          turn_index: 0,
          session_policy_applied: false,
          lifecycle_state: 'completed',
          response_status: 200,
          duration_ms: 900,
          selected_model: 'gpt-6-luna',
        },
        {
          record_id: 'turn-2',
          timestamp: '2026-11-15T09:01:00Z',
          turn_index: 1,
          session_policy_applied: false,
          lifecycle_state: 'completed',
          response_status: 200,
          duration_ms: 900,
          selected_model: 'gpt-6-luna',
        },
      ],
    }
    const marks = assignReviewsToRoutes(trajectory.routes!, [denied, failed])
    expect(marks.get('turn-1')?.map((review) => review.id)).toEqual(['review-deny'])
    expect(marks.get('turn-2')?.map((review) => review.id)).toEqual(['review-failed'])
    expect(summarizeReviews([denied, failed])).toEqual({
      total: 2,
      allowed: 0,
      denied: 1,
      failed: 1,
    })

    const html = render(
      <InsightsSessionRoutes trajectory={trajectory} reviews={[denied, failed]} />,
    )
    expect(html).toContain('<dt>Reviews</dt>')
    expect(html).toContain('href="/insights/review-deny"')
    expect(html).toContain('href="/insights/review-failed"')
    expect(render(<InsightsSessionRoutes trajectory={trajectory} />)).not.toContain(
      '<dt>Reviews</dt>',
    )
  })

  it('reads every page of a session reviews and says when it stops short', async () => {
    const ids = (from: number, count: number) => Array.from({ length: count }, (_, i) => from + i)
    const page = (pageIds: number[], hasMore: boolean) =>
      new Response(
        JSON.stringify({
          object: 'router_replay.list',
          count: pageIds.length,
          has_more: hasMore,
          data: pageIds.map((id) =>
            reviewRecord(`review-${id}`, '2026-11-15T09:00:30Z', {
              assessment: { outcome: 'allow' },
            }),
          ),
        }),
      )
    // 101 settled reviews; one recorded while paging pushes review-99 onto the second page.
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(page(ids(0, 100), true))
      .mockResolvedValueOnce(page([99, 100], false))
    vi.stubGlobal('fetch', fetch)
    const { reviews, truncated } = await fetchInsightsSessionReviews('agent-session')
    expect(
      fetch.mock.calls.map(([url]) => new URL(url, 'http://localhost').searchParams.get('offset')),
    ).toEqual(['0', '100'])
    expect(reviews).toHaveLength(101)
    expect(truncated).toBe(false)

    const trajectory: InsightsTrajectory = {
      object: 'router_replay.trajectory',
      session_id: 'agent-session',
      record_count: 1,
      turn_count: 1,
      messages: [],
      routes: [
        {
          record_id: 'turn-1',
          timestamp: '2026-11-15T09:00:00Z',
          turn_index: 0,
          session_policy_applied: false,
          lifecycle_state: 'completed',
          response_status: 200,
          duration_ms: 900,
        },
      ],
    }
    const html = render(<InsightsSessionRoutes trajectory={trajectory} reviews={reviews} />)
    expect(html).toContain('>101</a> · 101 allowed')
    expect(html).toContain('href="/insights/review-0"')
    expect(html).not.toContain('older reviews not shown')

    let offset = 0
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async () => page(ids((offset += 100) - 100, 100), true)),
    )
    const capped = await fetchInsightsSessionReviews('agent-session')
    expect(capped.reviews).toHaveLength(2000)
    expect(capped.truncated).toBe(true)
    expect(
      render(
        <InsightsSessionRoutes
          trajectory={trajectory}
          reviews={capped.reviews}
          reviewsTruncated={capped.truncated}
        />,
      ),
    ).toContain('>2000</a> · 2000 allowed · 0 denied · older reviews not shown')
  })
})
