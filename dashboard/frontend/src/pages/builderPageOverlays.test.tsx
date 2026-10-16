import { createElement, createRef } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('react-dom', () => ({
  createPortal: (children: unknown) => children,
}))

import { BuilderDeployConfirmModal } from './builderPageDeployOverlays'
import { BuilderGuideDrawer } from './builderPageGuideDrawer'
import { BuilderImportModal } from './builderPageImportModal'

beforeEach(() => {
  vi.stubGlobal('document', { body: {} })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('builder overlay accessibility contracts', () => {
  it('renders the deploy preview as a labelled modal dialog', () => {
    const markup = renderToStaticMarkup(
      createElement(BuilderDeployConfirmModal, {
        open: true,
        loading: true,
        error: null,
        validationError: null,
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

  it('shows the Router verdict as an alert and blocks Deploy Now while keeping the diff', () => {
    const markup = renderToStaticMarkup(
      createElement(BuilderDeployConfirmModal, {
        open: true,
        loading: false,
        error: null,
        validationError: 'Merged config validation failed: complexity rule "r" sets both threshold and an explicit boundary pair; keep one',
        currentYaml: 'routing: {}',
        mergedYaml: 'routing: {}',
        onClose: vi.fn(),
        onConfirm: vi.fn(),
      }),
    )

    expect(markup).toContain('role="alert"')
    expect(markup).toContain('keep one')
    // The verdict blocks the deploy button but not the diff.
    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Deploy Now/s)
    expect(markup).not.toContain('Failed to load preview')
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
