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

  it('groups simple and complex fields into a shared responsive layout', () => {
    const markup = renderToStaticMarkup(
      createElement(EditModal, {
        isOpen: true,
        onClose: vi.fn(),
        onSave: vi.fn(async () => undefined),
        title: 'Add signal',
        data: { name: '', type: 'keyword', definition: {} },
        fields: [
          { name: 'name', label: 'Name', section: 'Identity', type: 'text' },
          {
            name: 'type',
            label: 'Type',
            section: 'Identity',
            type: 'select',
            options: ['keyword'],
          },
          { name: 'definition', label: 'Definition', section: 'Definition', type: 'textarea' },
        ],
      }),
    )

    expect(markup).toContain('>Identity</h3>')
    expect(markup).toContain('>Definition</h3>')
    expect(markup).toContain('<select')
    expect(markup).toContain('<textarea')
  })
})
