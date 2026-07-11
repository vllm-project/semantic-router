import { createElement, createRef } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

vi.mock('react-dom', () => ({
  createPortal: (children: unknown) => children,
}))

import { BuilderDeployConfirmModal } from './builderPageDeployOverlays'
import { BuilderGuideDrawer } from './builderPageGuideDrawer'
import { BuilderImportModal } from './builderPageImportModal'

beforeAll(() => {
  vi.stubGlobal('document', { body: {} })
})

afterAll(() => {
  vi.unstubAllGlobals()
})

describe('builder overlay accessibility contracts', () => {
  it('renders the deploy preview as a labelled modal dialog', () => {
    const markup = renderToStaticMarkup(
      createElement(BuilderDeployConfirmModal, {
        open: true,
        loading: true,
        error: null,
        currentYaml: '',
        mergedYaml: '',
        onClose: vi.fn(),
        onConfirm: vi.fn(),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toMatch(/aria-describedby="[^"]+"/)
    expect(markup).toContain('data-dialog-initial-focus="true"')
  })

  it('renders import as a labelled dialog with named config inputs', () => {
    const markup = renderToStaticMarkup(
      createElement(BuilderImportModal, {
        open: true,
        importUrl: '',
        importText: '',
        importError: null,
        importUrlLoading: false,
        loadingFromRouter: false,
        importTextareaRef: createRef<HTMLTextAreaElement>(),
        onClose: vi.fn(),
        onImportUrlChange: vi.fn(),
        onImportTextChange: vi.fn(),
        onImportUrl: vi.fn(),
        onSelectFile: vi.fn(),
        onLoadFromRouter: vi.fn(),
        onConfirm: vi.fn(),
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('aria-label="Router config URL"')
    expect(markup).toContain('aria-label="Router config YAML"')
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
