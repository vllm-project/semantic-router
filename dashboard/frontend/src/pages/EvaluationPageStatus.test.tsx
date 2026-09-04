import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import EvaluationPageStatus from './EvaluationPageStatus'

describe('EvaluationPageStatus', () => {
  it('keeps service contract errors behind an actionable product message', () => {
    const backendMessage = 'Evaluation catalog response is incomplete.'
    const markup = renderToStaticMarkup(
      createElement(EvaluationPageStatus, {
        readonlyLoading: false,
        serverReadonly: false,
        hasCatalog: true,
        catalogError: backendMessage,
        runsError: null,
        runsLoaded: true,
        refreshing: false,
        runLedgerComplete: true,
        runLedgerWarningCount: 0,
        runLedgerWarnings: [],
        mutationError: null,
        onRefresh: () => undefined,
        onClearMutationError: () => undefined,
      }),
    )
    const [productMessage, technicalDetails] = markup.split('<details')

    expect(productMessage).toContain(
      'Evaluation setup could not refresh. Showing the last loaded benchmark catalog.',
    )
    expect(productMessage).not.toContain(backendMessage)
    expect(technicalDetails).toContain('>Technical details</summary>')
    expect(technicalDetails).toContain(backendMessage)
  })

  it('keeps unreadable-record internals inside collapsed technical details', () => {
    const evidenceID = 'bundle-entry-7f9d2a'
    const evidenceFile = 'status.json'
    const backendMessage = 'Run status payload failed schema validation.'
    const markup = renderToStaticMarkup(
      createElement(EvaluationPageStatus, {
        readonlyLoading: false,
        serverReadonly: false,
        hasCatalog: true,
        catalogError: null,
        runsError: null,
        runsLoaded: true,
        refreshing: false,
        runLedgerComplete: false,
        runLedgerWarningCount: 1,
        runLedgerWarnings: [
          {
            code: 'corrupt_run_bundle',
            evidence_id: evidenceID,
            evidence_file: evidenceFile,
            message: backendMessage,
          },
        ],
        mutationError: null,
        onRefresh: () => undefined,
        onClearMutationError: () => undefined,
      }),
    )
    const [summary, technicalDetails] = markup.split('<details')

    expect(summary).toContain('Some saved runs could not be read')
    expect(summary).not.toContain(evidenceID)
    expect(summary).not.toContain(evidenceFile)
    expect(summary).not.toContain(backendMessage)
    expect(technicalDetails).toContain('>Technical details · 1</summary>')
    expect(technicalDetails).toContain(evidenceID)
    expect(technicalDetails).toContain(evidenceFile)
    expect(technicalDetails).toContain(backendMessage)
    expect(technicalDetails).not.toMatch(/^ open(?:=|>)/)
  })

  it('keeps mutation responses out of the default product message', () => {
    const backendMessage = 'worker://mutation E5 upstream=private.internal'
    const markup = renderToStaticMarkup(
      createElement(EvaluationPageStatus, {
        readonlyLoading: false,
        serverReadonly: false,
        hasCatalog: true,
        catalogError: null,
        runsError: null,
        runsLoaded: true,
        refreshing: false,
        runLedgerComplete: true,
        runLedgerWarningCount: 0,
        runLedgerWarnings: [],
        mutationError: backendMessage,
        onRefresh: () => undefined,
        onClearMutationError: () => undefined,
      }),
    )
    const boundaryIndex = markup.indexOf('data-evaluation-technical-details="true"')

    expect(markup.slice(0, boundaryIndex)).toContain(
      'The last evaluation action could not be completed',
    )
    expect(markup.slice(0, boundaryIndex)).not.toContain(backendMessage)
    expect(markup.slice(boundaryIndex)).toContain(backendMessage)
  })
})
