import { createElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfirmDialog from './ConfirmDialog'

// Static markup cannot mount portals; render dialog content in place instead.
vi.mock('radix-ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('radix-ui')>()
  return {
    ...actual,
    Dialog: { ...actual.Dialog, Portal: ({ children }: { children?: ReactNode }) => children },
  }
})

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
    expect(markup).toMatch(/Enter <strong[^>]*>DELETE<\/strong> to\s*confirm\./)
    expect(markup).toMatch(/<input[^>]+data-dialog-initial-focus="true"/)
    expect(markup).toMatch(/<button[^>]+disabled=""[^>]+type="submit"[^>]*>Delete route<\/button>/)
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
    expect(markup).toMatch(/<button[^>]+disabled=""[^>]+type="submit"[^>]*>.*Cancelling…<\/button>/)
  })

  it('uses the same alert symbol for destructive and interrupting actions', () => {
    const render = (tone: 'danger' | 'warning' | 'neutral') =>
      renderToStaticMarkup(
        createElement(ConfirmDialog, {
          isOpen: true,
          title: 'Cancel run?',
          description: 'Execution will stop.',
          tone,
          onCancel: () => undefined,
          onConfirm: () => undefined,
        }),
      )

    expect(render('danger')).toContain('data-tone-icon="alert"')
    expect(render('warning')).toContain('data-tone-icon="alert"')
    expect(render('neutral')).toContain('data-tone-icon="info"')
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
        errorDetails: createElement(
          'details',
          { 'data-technical-details': 'true' },
          createElement('summary', {}, 'Details'),
          rawError,
        ),
        onCancel: () => undefined,
        onConfirm: () => undefined,
      }),
    )
    const detailsIndex = markup.indexOf('data-technical-details="true"')

    expect(markup).toMatch(/role="alert"/)
    expect(markup.slice(0, detailsIndex)).toContain('The run could not be deleted')
    expect(markup.slice(0, detailsIndex)).not.toContain(rawError)
    expect(markup.slice(detailsIndex)).toContain(rawError)
    expect(markup).not.toContain('<details open')
  })
})
