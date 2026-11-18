import { createElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ViewModal from './ViewModal'
import ViewPanel from './ViewPanel'
import { transitionFromViewToEdit } from './viewModalSupport'

// Static markup cannot mount portals; render sheet content in place instead.
vi.mock('radix-ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('radix-ui')>()
  return {
    ...actual,
    Dialog: { ...actual.Dialog, Portal: ({ children }: { children?: ReactNode }) => children },
  }
})

const sections = [
  {
    title: 'Identity',
    fields: [
      { label: 'Name', value: 'guard_route' },
      { label: 'Description', value: '', fullWidth: true },
    ],
  },
]

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

  it('exposes the detail sheet as a titled dialog with sections and an edit action', () => {
    const markup = renderToStaticMarkup(
      createElement(ViewModal, {
        isOpen: true,
        onClose: vi.fn(),
        onEdit: vi.fn(),
        title: 'Model details',
        sections,
      }),
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toMatch(/<h2 id="[^"]+"[^>]*>Model details<\/h2>/)
    expect(markup).toContain('<span class="sr-only">Close</span>')
    expect(markup).toContain('>Identity</h3>')
    expect(markup).toContain('guard_route')
    expect(markup).toContain('—')
    expect(markup).toMatch(/<button[^>]*>.*Edit<\/button>/)
  })

  it('omits the footer when the view has no actions', () => {
    const markup = renderToStaticMarkup(
      createElement(ViewModal, { isOpen: true, onClose: vi.fn(), title: 'Read only', sections }),
    )

    expect(markup).not.toContain('>Edit</button>')
  })
})

describe('ViewPanel', () => {
  it('renders without a dialog context and labels its close control', () => {
    const markup = renderToStaticMarkup(
      createElement(ViewPanel, {
        title: 'Evidence',
        sections,
        onClose: vi.fn(),
        closeLabel: 'Close',
      }),
    )

    expect(markup).toContain('>Evidence</h2>')
    expect(markup).toContain('aria-label="Close"')
    expect(markup).toContain('guard_route')
  })
})
