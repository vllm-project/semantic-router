import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import LayoutMegaMenu from './LayoutMegaMenu'
import { BUILD_MENU_CATEGORIES } from './LayoutNavSupport'

describe('layout mega-menu accessibility contract', () => {
  it('renders a density-aware navigation popover instead of a modal dialog', () => {
    const markup = renderToStaticMarkup(
      createElement(
        MemoryRouter,
        null,
        createElement(LayoutMegaMenu, {
          id: 'build-navigation',
          triggerId: 'build-trigger',
          label: 'Build',
          categories: BUILD_MENU_CATEGORIES,
          activeCategoryKey: 'outcomes',
          isItemActive: () => false,
          onConfigSelect: vi.fn(),
          onItemIntent: vi.fn(),
          onNavigate: vi.fn(),
        }),
      ),
    )

    expect(markup).toContain('<nav')
    expect(markup).toContain('aria-labelledby="build-trigger"')
    expect(markup).toContain('data-density="compact"')
    expect(markup).toContain('role="tablist"')
    expect(markup).toContain('role="tab"')
    expect(markup).toContain('role="tabpanel"')
    expect(markup).not.toContain('role="dialog"')
    expect(markup).not.toContain('aria-modal="true"')
  })
})
