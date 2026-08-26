import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import LayoutMobileNavigation from './LayoutMobileNavigation'
import { BUILD_MENU_CATEGORIES, PRIMARY_NAV_LINKS } from './LayoutNavSupport'

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
          primaryLinks: PRIMARY_NAV_LINKS,
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
    expect(markup).toContain('Integrations')
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
          primaryLinks: PRIMARY_NAV_LINKS,
          sections: [
            { key: 'build', label: 'Build', categories: [] },
            { key: 'system', label: 'System', categories: BUILD_MENU_CATEGORIES },
          ],
          onConfigSelect: vi.fn(),
          onNavigate: vi.fn(),
          onSectionToggle: vi.fn(),
        }),
      ),
    )

    expect(markup).not.toContain('>Build<')
    expect(markup).toContain('>System<')
  })

  it('renders only permission-filtered primary destinations', () => {
    const primaryLinks = PRIMARY_NAV_LINKS.filter((link) => link.label !== 'Playground')
    const markup = renderToStaticMarkup(
      createElement(
        MemoryRouter,
        { initialEntries: ['/dashboard'] },
        createElement(LayoutMobileNavigation, {
          isConfigPage: false,
          openSection: null,
          pathname: '/dashboard',
          primaryLinks,
          sections: [],
          onConfigSelect: vi.fn(),
          onNavigate: vi.fn(),
          onSectionToggle: vi.fn(),
        }),
      ),
    )

    expect(markup).toContain('>Dashboard<')
    expect(markup).toContain('>Access<')
    expect(markup).not.toContain('>Playground<')
  })
})
