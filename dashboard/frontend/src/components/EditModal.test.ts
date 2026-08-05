import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import EditModal from './EditModal'

describe('EditModal accessibility contract', () => {
  it('renders a labelled modal dialog with an explicit close control', () => {
    const markup = renderToStaticMarkup(
      createElement(EditModal, {
        isOpen: true,
        onClose: vi.fn(),
        onSave: vi.fn(async () => undefined),
        title: 'Edit model',
        data: null,
        fields: [],
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/aria-labelledby="[^"]+"/)
    expect(markup).toContain('aria-label="Close editor"')
    expect(markup).toContain('type="submit"')
  })

  it('uses the shared dialog behavior and guards unsaved changes', () => {
    const source = readFileSync(new URL('./EditModal.tsx', import.meta.url), 'utf8')

    expect(source).toContain('useAccessibleDialog<HTMLDivElement>')
    expect(source).toContain('if (isDirty)')
    expect(source).toContain('Discard unsaved changes?')
    expect(source).toContain('dismissible: !saving')
    expect(source).toContain('aria-busy={saving}')
  })
})
