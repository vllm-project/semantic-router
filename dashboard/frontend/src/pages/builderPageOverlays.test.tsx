import { createElement, createRef } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('react-dom', () => ({
  createPortal: (children: unknown) => children,
}))

import { BuilderGuideDrawer } from './builderPageGuideDrawer'
import { BuilderImportModal } from './builderPageImportModal'

beforeEach(() => {
  vi.stubGlobal('document', { body: {} })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('builder overlay accessibility contracts', () => {
  it('renders Recipe import as a labelled dialog with named inputs', () => {
    const markup = renderToStaticMarkup(
      createElement(BuilderImportModal, {
        open: true,
        importText: '',
        importError: null,
        importTextareaRef: createRef<HTMLTextAreaElement>(),
        onClose: vi.fn(),
        onImportTextChange: vi.fn(),
        onSelectFile: vi.fn(),
        onConfirm: vi.fn(),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('Import Recipe')
    expect(markup).toContain('aria-label="Recipe document"')
    expect(markup).not.toContain('Load from Router')
    expect(markup).toContain('aria-label="Close import dialog"')
  })

  it('renders the DSL guide as a labelled modal drawer', () => {
    const markup = renderToStaticMarkup(
      createElement(BuilderGuideDrawer, {
        open: true,
        width: 420,
        isDragging: false,
        onClose: vi.fn(),
        onDragStart: vi.fn(),
        onInsertSnippet: vi.fn(),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toContain('aria-label="Close DSL language guide"')
  })
})
