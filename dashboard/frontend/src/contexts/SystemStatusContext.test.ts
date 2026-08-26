import { describe, expect, it } from 'vitest'

import type { SystemStatus } from '../utils/routerRuntime'
import { resolveRoutingAccessAvailability } from './SystemStatusContext'

const status = (routingHealthy: boolean): SystemStatus => ({
  overall: routingHealthy ? 'healthy' : 'degraded',
  services: [
    { name: 'Router', status: 'operational', healthy: true },
    {
      name: 'Routing access',
      status: routingHealthy ? 'operational' : 'unavailable',
      healthy: routingHealthy,
    },
  ],
  history: { windowHours: 90, through: '2026-08-26T12:00:00Z', services: [] },
})

describe('routing access availability', () => {
  it('uses the public Routing access service as the single readiness source', () => {
    expect(resolveRoutingAccessAvailability(status(true), null, false)).toBe('operational')
    expect(resolveRoutingAccessAvailability(status(false), null, false)).toBe('unavailable')
  })

  it('fails closed while status is unknown or cannot be refreshed', () => {
    expect(resolveRoutingAccessAvailability(null, null, true)).toBe('checking')
    expect(resolveRoutingAccessAvailability(null, null, false)).toBe('unavailable')
    expect(resolveRoutingAccessAvailability(status(true), 'Status request failed.', false)).toBe(
      'unavailable',
    )
  })
})
