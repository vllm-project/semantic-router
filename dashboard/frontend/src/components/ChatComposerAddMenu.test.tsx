import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ChatComposerAddMenu from './ChatComposerAddMenu'

describe('ChatComposerAddMenu contract', () => {
  it('renders one compact, collapsed add trigger', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerAddMenu, {
        clawModeDisabled: false,
        clawModeEnabled: false,
        onAttachFiles: vi.fn(),
        onToggleClawMode: vi.fn(),
        onToggleWebSearch: vi.fn(),
        webSearchEnabled: true,
      }),
    )

    expect(markup).toContain('aria-label="Add to prompt"')
    expect(markup).toContain('aria-haspopup="menu"')
    expect(markup).toContain('aria-expanded="false"')
    expect(markup).not.toContain('role="menu"')
  })

  it('owns keyboard traversal, outside click, and focus restoration', () => {
    const source = readFileSync(new URL('./ChatComposerAddMenu.tsx', import.meta.url), 'utf8')

    expect(source).toContain("event.key === 'Escape'")
    expect(source).toContain("'ArrowDown', 'ArrowUp', 'Home', 'End'")
    expect(source).toContain("document.addEventListener('pointerdown'")
    expect(source).toContain('triggerRef.current?.focus()')
    expect(source).toContain("role={isToggle ? 'menuitemcheckbox' : 'menuitem'}")
  })
})
