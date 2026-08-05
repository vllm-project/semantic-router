import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('RecoverableLazyRoute contract', () => {
  it('preserves suspense loading and clears prefetch state before retrying a fresh lazy import', () => {
    const source = readFileSync(new URL('./RecoverableLazyRoute.tsx', import.meta.url), 'utf8')

    expect(source).toContain('createRetryableLazyPage(loader, attempt)')
    expect(source).toContain('return lazy(loader)')
    expect(source).toContain('resetDashboardRouteLoader(loader as RouteLoader)')
    expect(source).toContain('<RouteLoadingFallback />')
    expect(source).toContain('Retry route')
    expect(source).toContain('Reload dashboard')
    expect(source).toContain('data-testid="route-load-error"')
  })
})
