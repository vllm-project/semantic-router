import { createRef } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import AccountSecurityForm from './AccountSecurityForm'

const emptyFields = {
  currentPassword: '',
  newPassword: '',
}

describe('AccountSecurityForm password-manager contract', () => {
  it('exposes the standard username, current-password, and new-password field metadata', () => {
    const markup = renderToStaticMarkup(
      <AccountSecurityForm
        accountEmail="user@example.test"
        fields={emptyFields}
        error={null}
        pending={false}
        formRef={createRef<HTMLFormElement>()}
        onFieldChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(markup).toContain('id="change-password-form"')
    expect(markup).toContain('action="/api/auth/password"')
    expect(markup).toContain('method="post"')
    expect(markup).toContain('autoComplete="on"')
    expect(markup).toContain('name="username"')
    expect(markup).toContain('autoComplete="username"')
    expect(markup).toContain('id="current-password"')
    expect(markup).toContain('name="current-password"')
    expect(markup).toContain('autoComplete="current-password"')
    expect(markup).toContain('id="new-password"')
    expect(markup).toContain('name="new-password"')
    expect(markup.match(/autoComplete="new-password"/g)).toHaveLength(1)
    expect(markup).not.toContain('autoComplete="off"')
    expect(markup.match(/type="password"/g)).toHaveLength(2)
    expect(markup).toContain('aria-pressed="false"')
    expect(markup).toContain('aria-controls="current-password new-password"')
    expect(markup).toContain('Show passwords')
  })

  it('renders a policy error verbatim without introducing password values', () => {
    const markup = renderToStaticMarkup(
      <AccountSecurityForm
        accountEmail="user@example.test"
        fields={emptyFields}
        error="Password must contain a symbol."
        pending={false}
        formRef={createRef<HTMLFormElement>()}
        onFieldChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(markup).toContain('role="alert"')
    expect(markup).toContain('Password must contain a symbol.')
    expect(markup).not.toContain('current-value')
    expect(markup).not.toContain('new-value')
  })
})
