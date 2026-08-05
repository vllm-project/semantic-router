import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ConfirmDialog from './ConfirmDialog'

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
    expect(markup).toContain('Type <strong>DELETE</strong>')
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
})
