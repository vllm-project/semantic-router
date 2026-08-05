import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import LayoutAccountControl from './LayoutAccountControl'

describe('LayoutAccountControl contract', () => {
  it('exposes a compact dialog trigger with the current account identity', () => {
    const markup = renderToStaticMarkup(
      createElement(LayoutAccountControl, {
        accountName: 'Ada Lovelace',
        accountEmail: 'ada@example.com',
        accountRole: 'platform_admin',
        accountPermissions: ['config.read'],
        isOpen: false,
        onToggle: vi.fn(),
        onClose: vi.fn(),
        onLogout: vi.fn(),
      }),
    )

    expect(markup).toContain('aria-haspopup="dialog"')
    expect(markup).toContain('aria-expanded="false"')
    expect(markup).toContain('aria-label="Open account menu for Ada Lovelace"')
    expect(markup).toContain('>AL<')
  })

  it('uses shared focus handling, stacked scroll locking, and grouped permissions', () => {
    const source = readFileSync(new URL('./LayoutAccountControl.tsx', import.meta.url), 'utf8')

    expect(source).toContain('useAccessibleDialog<HTMLDivElement>')
    expect(source).toContain('lockBodyScroll: true')
    expect(source).toContain('onMouseDown={onClose}')
    expect(source).toContain('groupAccountPermissions(accountPermissions)')
    expect(source).toContain('aria-modal="true"')
  })
})
