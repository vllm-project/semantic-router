import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8')

describe('official browser authentication contract', () => {
  it('does not persist or attach a dashboard bearer credential', () => {
    const transport = source('./authFetch.ts')
    const authContext = source('../contexts/AuthContext.tsx')
    const loginPage = source('../pages/LoginPage.tsx')
    expect(transport).not.toContain("headers.set('Authorization'")
    expect(transport).not.toContain('document.cookie =')
    expect(transport).not.toContain('searchParams.set')
    expect(transport).not.toContain('window.WebSocket =')
    expect(transport).not.toContain('window.EventSource =')
    expect(transport).not.toContain('HTMLIFrameElement')
    expect(authContext).not.toContain('payload.token')
    expect(loginPage).not.toContain('payload.token')
    expect(authContext).toContain('COOKIE_AUTH_RESPONSE_HEADERS')
    expect(loginPage).toContain('COOKIE_AUTH_RESPONSE_HEADERS')
  })

  it('keeps official embedded service URLs free of dashboard query credentials', () => {
    const embeddedSources = [
      source('../pages/MonitoringPage.tsx'),
      source('../pages/TracingPage.tsx'),
      source('../pages/KnowledgeMapPage.tsx'),
    ].join('\n')
    expect(embeddedSources).not.toContain('withAuthQuery')
    expect(embeddedSources).not.toContain('authToken=')
  })
})
