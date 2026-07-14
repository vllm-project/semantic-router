import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import AccountSecurityIdentityUnavailable from './AccountSecurityIdentityUnavailable'

describe('AccountSecurityIdentityUnavailable', () => {
  it('offers session recovery without mounting credential fields', () => {
    const markup = renderToStaticMarkup(<AccountSecurityIdentityUnavailable onRetry={vi.fn()} />)

    expect(markup).toContain('role="alert"')
    expect(markup).toContain('Retry account check')
    expect(markup).not.toContain('type="password"')
    expect(markup).not.toContain('autocomplete="current-password"')
    expect(markup).not.toContain('autocomplete="new-password"')
  })
})
