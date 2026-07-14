import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import LayoutAccountSecurityLink from './LayoutAccountSecurityLink'

describe('LayoutAccountSecurityLink', () => {
  it('exposes password management from the account menu', () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <LayoutAccountSecurityLink onSelect={vi.fn()} />
      </MemoryRouter>,
    )

    expect(markup).toContain('href="/account/security"')
    expect(markup).toContain('Password &amp; security')
  })
})
