import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ConfirmDialog from './ConfirmDialog'
import EvaluationIssueDetails from './evaluation-plane/EvaluationIssueDetails'

describe('ConfirmDialog', () => {
  it('renders an accessible destructive confirmation', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfirmDialog, {
        isOpen: true,
        title: 'Delete route?',
        description: 'This change cannot be undone.',
        confirmLabel: 'Delete route',
        confirmationText: 'DELETE',
        onCancel: () => undefined,
        onConfirm: () => undefined,
      }),
    )

    expect(markup).toContain('role="alertdialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('Delete route?')
    expect(markup).toContain('Enter <strong>DELETE</strong> to confirm.')
    expect(markup).toMatch(/<input[^>]+data-dialog-initial-focus="true"/)
    expect(markup).toMatch(/<button[^>]+disabled=""[^>]*>Delete route<\/button>/)
  })

  it('marks pending work busy and disables both actions', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfirmDialog, {
        isOpen: true,
        title: 'Cancel run?',
        description: 'Execution will stop.',
        confirmLabel: 'Cancel run',
        pending: true,
        pendingLabel: 'Cancelling…',
        onCancel: () => undefined,
        onConfirm: () => undefined,
      }),
    )

    expect(markup).toContain('aria-busy="true"')
    expect(markup).toMatch(
      /<button[^>]+disabled=""[^>]+data-dialog-initial-focus="true"[^>]*>Cancel<\/button>/,
    )
    expect(markup).toMatch(/<button[^>]+disabled=""[^>]*>Cancelling…<\/button>/)
  })

  it('uses the same alert symbol for destructive and interrupting actions', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfirmDialog, {
        isOpen: true,
        title: 'Cancel run?',
        description: 'Execution will stop.',
        tone: 'warning',
        onCancel: () => undefined,
        onConfirm: () => undefined,
      }),
    )

    expect(markup).toContain('aria-hidden="true">!</div>')
    expect(markup).not.toContain('>•</div>')
  })

  it('does not render while closed', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfirmDialog, {
        isOpen: false,
        title: 'Hidden',
        description: 'Hidden',
        onCancel: () => undefined,
        onConfirm: () => undefined,
      }),
    )

    expect(markup).toBe('')
  })

  it('separates actionable error copy from closed technical details', () => {
    const rawError = 'backend://delete private-stack'
    const markup = renderToStaticMarkup(
      createElement(ConfirmDialog, {
        isOpen: true,
        title: 'Delete run?',
        description: 'This cannot be undone.',
        errorMessage: 'The run could not be deleted. Retry or close this dialog.',
        errorDetails: createElement(EvaluationIssueDetails, {
          issues: [{ label: 'Deletion request', message: rawError }],
        }),
        onCancel: () => undefined,
        onConfirm: () => undefined,
      }),
    )
    const detailsIndex = markup.indexOf('data-evaluation-technical-details="true"')

    expect(markup.slice(0, detailsIndex)).toContain('The run could not be deleted')
    expect(markup.slice(0, detailsIndex)).not.toContain(rawError)
    expect(markup.slice(detailsIndex)).toContain(rawError)
    expect(markup).not.toContain('<details open')
  })
})
