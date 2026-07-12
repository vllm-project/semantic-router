import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AuthProvider } from '../contexts/AuthContext'
import AccountSecurityPage from './AccountSecurityPage'

describe('AccountSecurityPage identity gate', () => {
  it('does not mount a password form when the authenticated user identity is unavailable', () => {
    const markup = renderToStaticMarkup(
      <AuthProvider>
        <AccountSecurityPage />
      </AuthProvider>,
    )

    expect(markup).toContain('Account identity unavailable')
    expect(markup).toContain('Retry account check')
    expect(markup).not.toContain('change-password-form')
    expect(markup).not.toContain('<input')
    expect(markup).not.toContain('autoComplete="current-password"')
    expect(markup).not.toContain('autoComplete="new-password"')
  })
})
