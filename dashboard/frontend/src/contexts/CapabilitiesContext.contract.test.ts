import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = readFileSync(new URL('./CapabilitiesContext.tsx', import.meta.url), 'utf8')

describe('Dashboard capabilities fail closed', () => {
  it('clears capabilities whenever the caller or their access changes', () => {
    const resetEffect = source.indexOf('setCapabilities(null)')
    expect(resetEffect).toBeGreaterThan(-1)
    expect(source).toContain('}, [isAuthenticated, userID, accessSnapshot])')
    expect(source).toContain('}, [isAuthenticated, userID, accessSnapshot, revision])')
  })

  it('leaves capabilities closed when the request fails', () => {
    const catchBlock = source.slice(source.indexOf('} catch (err) {'))
    expect(catchBlock.indexOf('setCapabilities(null)')).toBeGreaterThan(-1)
    expect(catchBlock.indexOf('setCapabilities(null)')).toBeLessThan(catchBlock.indexOf('} finally {'))
    expect(source).toContain('decodeDashboardCapabilities(await response.json())')
    expect(source).toContain('return () => {\n      controller.abort()')
  })

  it('re-reads a pending probe a bounded number of times', () => {
    expect(source).toContain('pendingRetries < PROBE_PENDING_MAX_RETRIES')
    expect(source).toContain('if (retry) clearTimeout(retry)')
  })
})
