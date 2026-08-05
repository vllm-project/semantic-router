import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import LayoutMobileNavigation from './LayoutMobileNavigation'
import { BUILD_MENU_CATEGORIES } from './LayoutNavSupport'

describe('LayoutMobileNavigation contract', () => {
  it('keeps the active child and its workflow parent visible in the mobile hierarchy', () => {
    const markup = renderToStaticMarkup(
      createElement(
        MemoryRouter,
        { initialEntries: ['/config/models'] },
        createElement(LayoutMobileNavigation, {
          configSection: 'models',
          isConfigPage: true,
          openSection: 'build',
          pathname: '/config/models',
          sections: [{ key: 'build', label: 'Build', categories: BUILD_MENU_CATEGORIES }],
          onConfigSelect: vi.fn(),
          onNavigate: vi.fn(),
          onSectionToggle: vi.fn(),
        }),
      ),
    )

    expect(markup).toContain('<nav')
    expect(markup).toContain('aria-label="Mobile navigation"')
    expect(markup).toContain('aria-expanded="true"')
    expect(markup).toContain('aria-current="page"')
    expect(markup).toContain('Current')
    expect(markup).toContain('data-mobile-nav-control="true"')
    expect(markup).toContain('Routing')
    expect(markup).toContain('Integrations &amp; Policy')
  })

  it('supports roving keyboard focus and returns focus when dismissed', () => {
    const source = readFileSync(new URL('./LayoutMobileNavigation.tsx', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./Layout.module.css', import.meta.url), 'utf8')

    expect(source).toContain("event.key === 'Escape'")
    expect(source).toContain("['ArrowDown', 'ArrowUp', 'Home', 'End']")
    expect(source).toContain('[aria-controls="mobile-navigation"]')
    expect(styles).toMatch(/\.mobileNavLink\s*\{[\s\S]*?min-height:\s*46px/)
  })

  it('omits workflow controls whose permission-filtered categories are empty', () => {
    const markup = renderToStaticMarkup(
      createElement(
        MemoryRouter,
        { initialEntries: ['/dashboard'] },
        createElement(LayoutMobileNavigation, {
          isConfigPage: false,
          openSection: null,
          pathname: '/dashboard',
          sections: [
            { key: 'build', label: 'Build', categories: [] },
            { key: 'operate', label: 'Operate', categories: BUILD_MENU_CATEGORIES },
          ],
          onConfigSelect: vi.fn(),
          onNavigate: vi.fn(),
          onSectionToggle: vi.fn(),
        }),
      ),
    )

    expect(markup).not.toContain('>Build<')
    expect(markup).toContain('>Operate<')
  })
})
