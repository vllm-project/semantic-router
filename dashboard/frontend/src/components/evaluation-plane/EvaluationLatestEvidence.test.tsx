import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import EvaluationLatestEvidence from './EvaluationLatestEvidence'
import { buildEvaluationOverviewModel } from './evaluationOverview'

describe('EvaluationLatestEvidence', () => {
  it('keeps report service failures behind actionable product copy', () => {
    const backendError = 'decoder://latest-report G7 private-stack'
    const markup = renderToStaticMarkup(
      createElement(EvaluationLatestEvidence, {
        model: buildEvaluationOverviewModel({
          runs: [],
          latestReport: null,
          requestedReportRunID: null,
        }),
        latestReport: null,
        reportLoading: false,
        reportError: backendError,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        onRetryReport: () => undefined,
        onLoadMoreRuns: () => undefined,
        onOpenReport: () => undefined,
      }),
    )
    const detailsIndex = markup.indexOf('data-evaluation-technical-details="true"')

    expect(markup.slice(0, detailsIndex)).toContain(
      'Retry to load the newest verified headline results.',
    )
    expect(markup.slice(0, detailsIndex)).not.toContain(backendError)
    expect(markup.slice(detailsIndex)).toContain(backendError)
    expect(markup).not.toContain('<details open')
  })
})
