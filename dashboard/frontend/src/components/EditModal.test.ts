import { readFileSync } from 'node:fs'
import { createElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import EditModal, { type FieldConfig } from './EditModal'
import {
  convertPercentageFields,
  editDialogSize,
  selectFieldOptions,
  visibleFieldSections,
} from './editModalSupport'

// Static markup cannot mount portals; render dialog content in place instead.
vi.mock('radix-ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('radix-ui')>()
  return {
    ...actual,
    Dialog: { ...actual.Dialog, Portal: ({ children }: { children?: ReactNode }) => children },
  }
})

const renderModal = (props: Partial<Parameters<typeof EditModal>[0]>) =>
  renderToStaticMarkup(
    createElement(EditModal, {
      isOpen: true,
      onClose: vi.fn(),
      onSave: vi.fn(async () => undefined),
      title: 'Edit model',
      data: null,
      fields: [],
      ...props,
    }),
  )

describe('EditModal accessibility contract', () => {
  it('renders a labelled modal dialog with an explicit close control', () => {
    const markup = renderModal({})

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toMatch(/<h2 id="[^"]+"[^>]*>Edit model<\/h2>/)
    expect(markup).toContain('aria-label="Close editor"')
    expect(markup).toContain('type="submit"')
    expect(markup).toMatch(/type="submit"[^>]*>Save<\/button>/)
  })

  it('labels the primary action by mode and renders nothing while closed', () => {
    expect(renderModal({ mode: 'add' })).toMatch(/type="submit"[^>]*>Add<\/button>/)
    expect(renderModal({ isOpen: false })).toBe('')
  })

  it('uses the shared dialog behavior and guards unsaved changes', () => {
    const source = readFileSync(new URL('./EditModal.tsx', import.meta.url), 'utf8')

    expect(source).toContain('<DialogContent')
    expect(source).toContain('if (isDirty)')
    expect(source).toContain('Discard unsaved changes?')
    expect(source).toContain('if (saving) return')
    expect(source).toContain('aria-busy={saving}')
  })

  it('supplies current form state to custom dependent fields', () => {
    const render = vi.fn(() => createElement('span', null, 'Dependent value'))
    const markup = renderModal({
      fields: [{ name: 'api_format', label: 'API Format', type: 'custom', customRender: render }],
    })

    expect(render).toHaveBeenCalledWith(undefined, expect.any(Function), {})
    expect(markup).toContain('Dependent value')
    expect(markup).toMatch(/role="group" aria-labelledby="[^"]+-api_format-label"/)
  })

  it('groups simple and complex fields into a shared responsive layout', () => {
    const markup = renderModal({
      title: 'Add signal',
      data: { name: '', type: 'keyword', definition: {} },
      fields: [
        { name: 'name', label: 'Name', section: 'Identity', type: 'text', required: true },
        {
          name: 'type',
          label: 'Type',
          section: 'Identity',
          type: 'select',
          options: ['keyword'],
        },
        { name: 'definition', label: 'Definition', section: 'Definition', type: 'textarea' },
      ],
    })

    expect(markup).toContain('>Identity</h3>')
    expect(markup).toContain('>Definition</h3>')
    expect(markup).toMatch(/<select[^>]+data-slot="native-select"/)
    expect(markup).toContain('<textarea')
    expect(markup).toMatch(/<input[^>]+required=""/)
  })

  it('renders every field type with design-system controls', () => {
    const markup = renderModal({
      fields: [
        { name: 'enabled', label: 'Enabled', type: 'boolean' },
        { name: 'ratio', label: 'Ratio', type: 'percentage' },
        { name: 'weight', label: 'Weight', type: 'number', min: 0, max: 10 },
        { name: 'tags', label: 'Tags', type: 'multiselect', options: ['fast', 'cheap'] },
      ],
    })

    expect(markup).toContain('role="switch"')
    expect(markup).toContain('>%</span>')
    expect(markup).toMatch(/<input[^>]+type="number"[^>]+min="0"[^>]+max="10"/)
    expect(markup).toContain('role="checkbox"')
    expect(markup).toContain('0 of 2')
  })
})

describe('EditModal support', () => {
  const fields: FieldConfig[] = [
    { name: 'name', label: 'Name', type: 'text' },
    { name: 'threshold', label: 'Threshold', type: 'percentage' },
    {
      name: 'advanced',
      label: 'Advanced',
      type: 'text',
      section: 'Advanced',
      shouldHide: (data) => data.mode !== 'advanced',
    },
  ]

  it('converts percentages exactly between display and storage', () => {
    expect(convertPercentageFields({ threshold: 0.7 }, fields, 'toDisplay')).toEqual({
      threshold: 70,
    })
    expect(convertPercentageFields({ threshold: 70 }, fields, 'toStored')).toEqual({
      threshold: 0.7,
    })
    expect(convertPercentageFields({ threshold: '' }, fields, 'toStored')).toEqual({
      threshold: '',
    })
  })

  it('hides dependent fields and keeps section order', () => {
    expect(visibleFieldSections(fields, {}).map(([title]) => title)).toEqual([''])
    expect(visibleFieldSections(fields, { mode: 'advanced' }).map(([title]) => title)).toEqual([
      '',
      'Advanced',
    ])
  })

  it('sizes the dialog from its content', () => {
    expect(editDialogSize([{ name: 'name', label: 'Name', type: 'text' }])).toBe('md')
    expect(editDialogSize(fields)).toBe('md')
    expect(editDialogSize([...fields, { name: 'notes', label: 'Notes', type: 'textarea' }])).toBe(
      'lg',
    )
    expect(editDialogSize([{ name: 'rules', label: 'Rules', type: 'custom' }])).toBe('xl')
  })

  it('keeps an unlisted configured value selectable', () => {
    expect(selectFieldOptions(['a', 'b'], 'legacy')).toEqual(['legacy', 'a', 'b'])
    expect(selectFieldOptions(['', 'a'], '')).toEqual(['', 'a'])
  })
})
