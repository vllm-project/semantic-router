import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationReport, EvaluationTrackReport } from '../../types/evaluationReport'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import EvaluationReports from './EvaluationReports'
import EvaluationReportTracks from './EvaluationReportTracks'
import EvaluationRunLedger from './EvaluationRunLedger'
import type { EvaluationRunLedgerModel } from './useEvaluationRunLedger'

function expectTechnicalOnly(markup: string, rawText: string) {
  const rawIndex = markup.indexOf(rawText)
  const detailsIndex = markup.lastIndexOf('<details', rawIndex)
  const detailsEnd = markup.indexOf('</details>', rawIndex)

  expect(rawIndex).toBeGreaterThan(-1)
  expect(detailsIndex).toBeGreaterThan(-1)
  expect(detailsEnd).toBeGreaterThan(rawIndex)
  expect(markup.slice(detailsIndex, rawIndex)).toContain('data-evaluation-technical-details="true"')
  expect(markup.slice(0, detailsIndex)).not.toContain(rawText)
}

const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: '11111111-1111-4111-8111-111111111111',
  client_request_id: '11111111-1111-4111-8111-111111111111',
  name: 'Routing reliability baseline',
  description: '',
  status: 'failed',
  mode: 'replay',
  evidence_level: 'E0',
  track_evidence_levels: { routing: 'E0' },
  target_id: 'fixture',
  change_profile: 'recipe',
  suite_ids: ['evaluation-smoke'],
  track_ids: ['routing'],
  sample_limit: 4,
  concurrency: 1,
  seed: 42,
  progress: {
    percent: 50,
    completed: 1,
    total: 2,
    message: 'worker://progress E5 shard=private-internal',
  },
  created_at: '2026-08-30T00:00:00Z',
}

describe('evaluation product error boundary', () => {
  it('renders verbatim evidence in a closed reusable disclosure', () => {
    const rawText = 'decoder://contract G9 path=private.internal'
    const markup = renderToStaticMarkup(
      createElement(EvaluationIssueDetails, {
        issues: [
          { label: 'First decoder', message: rawText },
          { label: 'First decoder', message: rawText },
        ],
      }),
    )

    expect(markup).toContain('>Technical details</summary>')
    expect(markup.match(new RegExp(rawText.replace('.', '\\.'), 'g'))).toHaveLength(2)
    expect(markup).not.toContain('<details open')
  })

  it('keeps report request responses behind actionable product copy', () => {
    const backendMessage = 'backend://report-read E4 digest=private-chain'
    const markup = renderToStaticMarkup(
      createElement(EvaluationReports, {
        runs: [],
        selectedRunID: run.id,
        report: null,
        loading: false,
        runLedgerAvailable: true,
        totalRuns: 0,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        error: backendMessage,
        onSelect: () => undefined,
        onRetry: () => undefined,
        onLoadMoreRuns: () => undefined,
      }),
    )

    expect(markup).toContain('Retry to load the selected completed run and its saved evidence')
    expectTechnicalOnly(markup, backendMessage)
  })

  it('derives track summaries from structured evidence and retains backend text', () => {
    const recordedSummary = 'worker summary E5 internal-recipe-codename'
    const recordedError = 'backend://track G7 private-stack-chain'
    const track: EvaluationTrackReport = {
      track_id: 'routing',
      status: 'failed',
      evidence_level: 'E0',
      summary: recordedSummary,
      coverage: { evaluated: 2, total: 4, fraction: 0.5, unavailable: 2 },
      metrics: [],
      gates: [],
      error: recordedError,
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationReportTracks, {
        report: { tracks: [track] } as EvaluationReport,
      }),
    )

    expect(markup).toContain('This area stopped before a final result was published')
    expectTechnicalOnly(markup, recordedSummary)
    expectTechnicalOnly(markup, recordedError)
  })

  it('uses structured progress in the ledger and preserves worker progress text', () => {
    const model = {
      page: 1,
      pages: 1,
      visibleRuns: [run],
      filtersActive: false,
      resetFilters: () => undefined,
      setPage: () => undefined,
    } as unknown as EvaluationRunLedgerModel
    const markup = renderToStaticMarkup(
      createElement(EvaluationRunLedger, {
        runs: [run],
        selectedRunID: run.id,
        runLedgerAvailable: true,
        totalRuns: 1,
        hasMoreRuns: false,
        loadingMore: false,
        refreshing: false,
        model,
        onSelect: () => undefined,
        onLoadMore: () => undefined,
      }),
    )

    expect(markup).toContain('50% · Stopped before completion')
    expectTechnicalOnly(markup, run.progress.message || '')
  })
})
