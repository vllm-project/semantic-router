import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('authenticated standalone routes', () => {
  it('guards the knowledge-map deep link with config access', () => {
    const source = readFileSync(new URL('./AuthenticatedAppRoutes.tsx', import.meta.url), 'utf8')
    const knowledgeMapRoute = source.slice(
      source.indexOf('path="/knowledge-bases/:name/map"'),
      source.indexOf('path="/knowledge-bases/:view"'),
    )

    expect(knowledgeMapRoute).toContain(
      "canAccessDashboardPath(user, '/knowledge-bases/map')",
    )
    expect(knowledgeMapRoute).toContain('<Navigate to="/dashboard" replace />')
  })
})
