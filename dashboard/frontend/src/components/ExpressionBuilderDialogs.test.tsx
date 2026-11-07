import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { AddChildPicker, EditSignalDialog } from './ExpressionBuilderDialogs'
import { predicateFromDraft } from './ExpressionBuilderSupport'

const availableSignals = [
  { signalType: 'domain', name: 'code' },
  { signalType: 'domain', name: 'math' },
]

describe('Expression builder dialog contracts', () => {
  it('renders the signal editor as a labelled modal with keyboard-operable choices', () => {
    const markup = renderToStaticMarkup(
      createElement(EditSignalDialog, {
        initial: { id: 'r1', signalType: 'domain', signalName: 'code' },
        availableSignals,
        onSave: vi.fn(),
        onCancel: vi.fn(),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toContain('aria-label="Close signal editor"')
    expect(markup).toContain('data-dialog-initial-focus="true"')
    expect(markup).toMatch(/<button[^>]*type="button"[^>]*>code<\/button>/)
    // label and on_error only apply to classifier and decision conditions.
    expect(markup).not.toContain('On error')
    expect(markup).toContain('Score predicate')
  })

  it('edits a classifier leaf label, predicate and on_error', () => {
    const markup = renderToStaticMarkup(
      createElement(EditSignalDialog, {
        initial: {
          id: 'r2',
          signalType: 'classifier',
          signalName: 'intent',
          label: 'billing',
          predicate: { gte: 0.7 },
          onError: 'match',
        },
        availableSignals: [{ signalType: 'classifier', name: 'intent', labels: ['billing', 'support'] }],
        onSave: vi.fn(),
        onCancel: vi.fn(),
      }),
    )

    expect(markup).toContain('value="billing"')
    expect(markup).toMatch(/<datalist[^>]*><option value="billing"><\/option><option value="support">/)
    expect(markup).toContain('value="0.7"')
    expect(markup).toContain('<option value="match" selected="">match</option>')
  })

  it('keeps predicate keys the dialog cannot edit and names them', () => {
    const initial = { gte: 0.7, eq: 1 } as unknown as Parameters<typeof predicateFromDraft>[1]
    expect(predicateFromDraft({ gt: '', gte: '0.8', lt: '', lte: '' }, initial)).toEqual({
      predicate: { eq: 1, gte: 0.8 },
      invalid: [],
    })
    expect(predicateFromDraft({ gt: '', gte: '', lt: '', lte: '' }, initial).predicate).toEqual({ eq: 1 })

    const markup = renderToStaticMarkup(
      createElement(EditSignalDialog, {
        initial: { id: 'r3', signalType: 'classifier', signalName: 'intent', label: 'billing', predicate: initial },
        availableSignals: [{ signalType: 'classifier', name: 'intent', labels: ['billing'] }],
        onSave: vi.fn(),
        onCancel: vi.fn(),
      }),
    )
    expect(markup).toContain('Unsupported predicate field eq is kept')
  })

  it('renders the child picker with a labelled dialog and explicit initial focus', () => {
    const markup = renderToStaticMarkup(
      createElement(AddChildPicker, {
        availableSignals,
        onPick: vi.fn(),
        onCancel: vi.fn(),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toContain('aria-label="Close child node picker"')
    expect(markup).toContain('aria-label="Search signals"')
    expect(markup).toContain('data-dialog-initial-focus="true"')
  })
})
