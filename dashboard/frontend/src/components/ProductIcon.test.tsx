import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ProductIcon, { type ProductIconName } from './ProductIcon'

describe('ProductIcon', () => {
  it('renders the shared action vocabulary as dependency-free accessible SVG', () => {
    const names: ProductIconName[] = [
      'activity',
      'alert',
      'arrow-left',
      'arrow-right',
      'audit',
      'budget',
      'chart',
      'check',
      'chevron-down',
      'chevron-left',
      'chevron-right',
      'chevron-up',
      'claw',
      'close',
      'code',
      'compute',
      'copy',
      'dashboard',
      'database',
      'decision',
      'download',
      'edit',
      'evaluation',
      'eye',
      'eye-off',
      'expand',
      'fleet',
      'fullscreen',
      'globe',
      'inbox',
      'info',
      'insight',
      'key',
      'label',
      'logs',
      'minus',
      'mixture',
      'model',
      'playground',
      'play',
      'plug',
      'plus',
      'power',
      'projection',
      'puzzle',
      'refresh',
      'search',
      'server',
      'settings',
      'shield',
      'signal',
      'status',
      'stop',
      'team',
      'topology',
      'tool',
      'trace',
      'trash',
      'undo',
      'redo',
      'user',
    ]

    for (const name of names) {
      const markup = renderToStaticMarkup(createElement(ProductIcon, { name }))
      expect(markup).toContain('<svg')
      expect(markup).toContain('aria-hidden="true"')
      expect(markup).not.toContain('💥')
    }
  })

  it('exposes an explicit accessible name without hiding the icon', () => {
    const markup = renderToStaticMarkup(
      createElement(ProductIcon, { name: 'info', 'aria-label': 'More information' }),
    )

    expect(markup).toContain('aria-label="More information"')
    expect(markup).toContain('role="img"')
    expect(markup).not.toContain('aria-hidden="true"')
  })
})
