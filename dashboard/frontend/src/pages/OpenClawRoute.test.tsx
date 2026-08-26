import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

vi.mock('../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { permissions: ['openclaw.read', 'openclaw.manage'] },
    isLoading: false,
  }),
}))

vi.mock('../contexts/ReadonlyContext', () => ({
  useReadonly: () => ({ serverReadonly: false, isLoading: false }),
}))

import OpenClawPage from './OpenClawPage'

function renderRoute(path: string): string {
  return renderToStaticMarkup(
    createElement(
      MemoryRouter,
      { initialEntries: [path] },
      createElement(
        Routes,
        null,
        createElement(Route, { path: '/openclaw', element: createElement(OpenClawPage) }),
      ),
    ),
  )
}

describe('OpenClaw route view', () => {
  it('renders the architecture Overview for the OpenClaw entry route', () => {
    const markup = renderRoute('/openclaw')

    expect(markup).toContain('id="openclaw-panel-architecture"')
    expect(markup).toContain('Claw Operating System')
    expect(markup).not.toContain('id="openclaw-panel-dashboard"')
  })

  it('keeps the Console behind its explicit route state', () => {
    const markup = renderRoute('/openclaw?view=console')

    expect(markup).toContain('id="openclaw-panel-dashboard"')
    expect(markup).not.toContain('id="openclaw-panel-architecture"')
  })
})
