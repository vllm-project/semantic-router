import { describe, expect, it } from 'vitest'
import {
  CAPABILITY_IDS,
  decodeDashboardCapabilities,
  isCapabilityUsable,
  type DashboardCapability,
} from './dashboardCapabilities'

const entry = (id: string, overrides: Record<string, unknown> = {}) => ({
  id,
  enabled: true,
  configured: true,
  reachable: true,
  authorized: true,
  degraded: false,
  state: 'available',
  ...overrides,
})

const response = (overrides: Record<string, Record<string, unknown>> = {}) => ({
  capabilities: CAPABILITY_IDS.map((id) => entry(id, overrides[id])),
})

describe('decodeDashboardCapabilities', () => {
  it('decodes every capability, defaulting omitted reason, remediation, and checkedAt', () => {
    const decoded = decodeDashboardCapabilities(
      response({
        config: {
          reachable: false,
          state: 'unreachable',
          reason: 'router_unreachable',
          remediation: 'The Router management API did not report ready.',
          checkedAt: '2026-10-15T00:00:00Z',
        },
      }),
    )
    expect(Object.keys(decoded).sort()).toEqual([...CAPABILITY_IDS].sort())
    expect(decoded.config).toMatchObject({
      state: 'unreachable',
      reason: 'router_unreachable',
      checkedAt: '2026-10-15T00:00:00Z',
    })
    expect(decoded.mcp).toMatchObject({ reason: '', remediation: '', checkedAt: null })
  })

  it('ignores capabilities a newer server adds', () => {
    const payload = response()
    payload.capabilities.push(entry('futureFeature'))
    expect(() => decodeDashboardCapabilities(payload)).not.toThrow()
  })

  it.each([
    ['a non-object payload', []],
    ['a missing capabilities array', {}],
    ['a missing capability', { capabilities: response().capabilities.slice(1) }],
    ['an unknown state', response({ setup: { state: 'broken' } })],
    ['a non-boolean dimension', response({ evaluation: { reachable: 'yes' } })],
    ['a non-string remediation', response({ mlPipeline: { remediation: 7 } })],
  ])('rejects %s', (_label, payload) => {
    expect(() => decodeDashboardCapabilities(payload)).toThrow()
  })
})

describe('isCapabilityUsable', () => {
  const withState = (state: DashboardCapability['state']) =>
    ({ state }) as DashboardCapability

  it('allows available and degraded capabilities only', () => {
    expect(isCapabilityUsable(withState('available'))).toBe(true)
    expect(isCapabilityUsable(withState('degraded'))).toBe(true)
    for (const state of ['disabled', 'unconfigured', 'unauthorized', 'unreachable'] as const) {
      expect(isCapabilityUsable(withState(state))).toBe(false)
    }
    expect(isCapabilityUsable(undefined)).toBe(false)
  })
})
