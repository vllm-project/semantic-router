import { describe, expect, it } from 'vitest'

import { STATUS_HISTORY_HOURS, type StatusHistory } from '../utils/routerRuntime'
import { routingUnavailableStatus } from './statusPageSupport'

function history(): StatusHistory {
  const through = new Date('2026-08-26T14:00:00Z')
  return {
    windowHours: STATUS_HISTORY_HOURS,
    through: '2026-08-26T14:00:00Z',
    services: [
      {
        name: 'Dashboard',
        hours: Array.from({ length: STATUS_HISTORY_HOURS }, (_, index) => ({
          observedAt: new Date(
            through.getTime() - (STATUS_HISTORY_HOURS - 1 - index) * 3_600_000,
          )
            .toISOString()
            .replace('.000Z', 'Z'),
          status: 'operational' as const,
        })),
      },
    ],
  }
}

describe('status page availability projection', () => {
  it('projects the observed routing failure on today without fabricating prior history', () => {
    const projected = routingUnavailableStatus({
      overall: 'healthy',
      services: [{ name: 'Dashboard', status: 'operational', healthy: true }],
      history: history(),
    })
    expect(projected).toMatchObject({
      overall: 'degraded',
      services: [
        { name: 'Dashboard', status: 'operational', healthy: true },
        { name: 'Routing access', status: 'unavailable', healthy: false },
      ],
    })
    expect(projected.history.services).toHaveLength(2)
    const routingHours = projected.history.services[1].hours
    expect(routingHours).toHaveLength(STATUS_HISTORY_HOURS)
    expect(routingHours.slice(0, -1).every((hour) => hour.status === 'unknown')).toBe(true)
    expect(routingHours[routingHours.length - 1]).toEqual({
      observedAt: '2026-08-26T14:00:00Z',
      status: 'unavailable',
    })
  })
})
