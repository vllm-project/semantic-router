import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ExpressionBuilderContextMenu from './ExpressionBuilderContextMenu'

const props = {
  contextMenu: { x: 10, y: 20, path: [0] },
  tree: {
    operator: 'AND' as const,
    conditions: [{ signalType: 'domain', signalName: 'math' }],
  },
  onAddChild: vi.fn(),
  onChangeOp: vi.fn(),
  onClose: vi.fn(),
  onDeleteNode: vi.fn(),
  onEditSignal: vi.fn(),
  onInsertSibling: vi.fn(),
  onUnwrap: vi.fn(),
  onWrap: vi.fn(),
}

describe('ExpressionBuilderContextMenu accessibility', () => {
  it('renders actions as native menu-item buttons', () => {
    const markup = renderToStaticMarkup(createElement(ExpressionBuilderContextMenu, props))

    expect(markup).toContain('role="menu"')
    expect(markup).toContain('aria-label="Expression actions"')
    expect(markup).toContain('role="menuitem"')
    expect(markup).toContain('<button type="button" role="menuitem"')
    expect(markup).not.toMatch(/<div[^>]+ctxMenuItem/)
  })

  it('owns initial focus, cyclic arrow traversal, and Escape dismissal', () => {
    const source = readFileSync(
      new URL('./ExpressionBuilderContextMenu.tsx', import.meta.url),
      'utf8',
    )

    expect(source).toContain('querySelector<HTMLButtonElement>(\'[role="menuitem"]\')?.focus()')
    expect(source).toContain("['ArrowDown', 'ArrowUp', 'Home', 'End']")
    expect(source).toContain('(nextIndex + menuItems.length) % menuItems.length')
    expect(source).toContain("event.key === 'Escape'")
    expect(source).toContain('closeAndRestoreFocus()')
  })
})
