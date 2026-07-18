import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ViewModal, { transitionFromViewToEdit } from './ViewModal'

describe('ViewModal edit transition', () => {
  it('closes the view before opening edit', () => {
    const calls: string[] = []
    const onClose = vi.fn(() => calls.push('close'))
    const onEdit = vi.fn(() => calls.push('edit'))

    transitionFromViewToEdit(onClose, onEdit)

    expect(calls).toEqual(['close', 'edit'])
    expect(onClose).toHaveBeenCalledOnce()
    expect(onEdit).toHaveBeenCalledOnce()
  })

  it('exposes the detail drawer as a labelled modal dialog', () => {
    const markup = renderToStaticMarkup(
      createElement(ViewModal, {
        isOpen: true,
        onClose: vi.fn(),
        title: 'Model details',
        sections: [],
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('aria-label="Model details"')
    expect(markup).toContain('aria-label="Close"')
  })
})
