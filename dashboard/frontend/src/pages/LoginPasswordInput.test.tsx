import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { LoginPasswordInput } from './LoginPage'

const renderPasswordInput = (
  visible: boolean,
  autoComplete: 'current-password' | 'new-password' = 'current-password',
) =>
  renderToStaticMarkup(
    <LoginPasswordInput
      id={autoComplete}
      name="password"
      autoComplete={autoComplete}
      label="Password"
      value=""
      placeholder="Password"
      visible={visible}
      visibilityContext={
        autoComplete === 'new-password' ? 'first administrator' : 'sign-in'
      }
      onChange={vi.fn()}
      onToggleVisibility={vi.fn()}
    />,
  )

describe('LoginPasswordInput', () => {
  it('starts masked and exposes a non-submitting accessible reveal control', () => {
    const markup = renderPasswordInput(false)

    expect(markup).toContain('id="current-password"')
    expect(markup).toContain('type="password"')
    expect(markup).toContain('autoComplete="current-password"')
    expect(markup).toContain('type="button"')
    expect(markup).not.toContain('type="submit"')
    expect(markup).toContain('aria-label="Show sign-in password"')
    expect(markup).toContain('aria-controls="current-password"')
    expect(markup).toContain('aria-pressed="false"')
  })

  it('reveals only when explicitly requested while retaining new-password metadata', () => {
    const markup = renderPasswordInput(true, 'new-password')

    expect(markup).toContain('id="new-password"')
    expect(markup).toContain('type="text"')
    expect(markup).toContain('autoComplete="new-password"')
    expect(markup).toContain('aria-label="Hide first administrator password"')
    expect(markup).toContain('aria-controls="new-password"')
    expect(markup).toContain('aria-pressed="true"')
  })
})
