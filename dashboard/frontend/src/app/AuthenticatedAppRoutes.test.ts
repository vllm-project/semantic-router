import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('authenticated standalone routes', () => {
  it('guards parameterized shell routes using the current browser path', () => {
    const source = readFileSync(new URL('./AuthenticatedAppRoutes.tsx', import.meta.url), 'utf8')

    expect(source).toContain('const { pathname } = useLocation()')
    expect(source).toContain('canAccessDashboardPath(user, pathname)')
    expect(source).not.toContain('canAccessDashboardPath(user, route.path)')
  })
})
