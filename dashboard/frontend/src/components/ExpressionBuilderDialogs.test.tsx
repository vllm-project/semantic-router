import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { AddChildPicker, EditSignalDialog } from './ExpressionBuilderDialogs'

const availableSignals = [
  { signalType: 'domain', name: 'code' },
  { signalType: 'domain', name: 'math' },
]

describe('Expression builder dialog contracts', () => {
  it('renders the signal editor as a labelled modal with keyboard-operable choices', () => {
    const markup = renderToStaticMarkup(
      createElement(EditSignalDialog, {
        signalType: 'domain',
        signalName: 'code',
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
