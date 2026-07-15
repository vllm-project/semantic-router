import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ModelDeleteDialog from './ModelDeleteDialog'

describe('ModelDeleteDialog', () => {
  it('renders an accessible typed destructive confirmation', () => {
    const markup = renderToStaticMarkup(
      createElement(ModelDeleteDialog, {
        modelNames: ['model-a', 'model-b'],
        pending: false,
        onCancel: vi.fn(),
        onConfirm: vi.fn(),
      }),
    )

    expect(markup).toContain('role="alertdialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('aria-busy="false"')
    expect(markup).toContain('Type <strong>DELETE 2</strong>')
    expect(markup).toContain('data-dialog-initial-focus="true"')
  })

  it('delegates focus management and blocks implicit dismissal after typed input or while pending', () => {
    const source = readFileSync(new URL('./ModelDeleteDialog.tsx', import.meta.url), 'utf8')

    expect(source).toContain('useAccessibleDialog<HTMLElement>')
    expect(source).toContain('const dismissible = !pending && confirmation.length === 0')
    expect(source).toContain('onMouseDown={dismissible ? onCancel : undefined}')
    expect(source).toContain('disabled={pending}')
  })
})
