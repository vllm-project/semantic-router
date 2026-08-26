import { describe, expect, it } from 'vitest'

import {
  assertSystemStatus,
  fetchSystemStatus,
  STATUS_HISTORY_HOURS,
  type StatusHistory,
} from './routerRuntime'

function response(status: number, body?: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response
}

function history(...serviceNames: string[]): StatusHistory {
  const through = new Date('2026-08-26T00:00:00Z')
  return {
    windowHours: STATUS_HISTORY_HOURS,
    through: through.toISOString().replace('.000Z', 'Z'),
    services: serviceNames.map((name) => ({
      name,
      hours: Array.from({ length: STATUS_HISTORY_HOURS }, (_, index) => ({
        observedAt: new Date(
          through.getTime() - (STATUS_HISTORY_HOURS - 1 - index) * 3_600_000,
        )
          .toISOString()
          .replace('.000Z', 'Z'),
        status:
          index === STATUS_HISTORY_HOURS - 1
            ? ('operational' as const)
            : ('unknown' as const),
      })),
    })),
  }
}

describe('public system status', () => {
  it('accepts the bounded public availability contract', () => {
    const payload = {
      overall: 'healthy',
      services: [{ name: 'Router', status: 'operational', healthy: true }],
      history: history('Router'),
    }
    expect(assertSystemStatus(payload)).toEqual(payload)
  })

  it('rejects malformed and failed responses instead of preserving stale health', async () => {
    expect(() => assertSystemStatus({ overall: 'healthy', services: [{ healthy: true }] })).toThrow(
      'invalid response',
    )
    expect(() =>
      assertSystemStatus({
        overall: 'healthy',
        services: [{ name: 'Router', status: 'operational', healthy: true }],
        history: { ...history('Router'), services: [] },
      }),
    ).toThrow('invalid response')
    expect(() =>
      assertSystemStatus({
        overall: 'healthy',
        services: [{ name: 'Router', status: 'operational', healthy: true }],
        history: {
          ...history('Router'),
          services: [
            {
              name: 'Router',
              hours: history('Router').services[0].hours.slice(1),
            },
          ],
        },
      }),
    ).toThrow('invalid response')
    expect(() =>
      assertSystemStatus({
        overall: 'healthy',
        services: [{ name: 'Router', status: 'operational', healthy: true }],
        history: history('Router', 'Router'),
      }),
    ).toThrow('invalid response')
    const unordered = history('Router')
    unordered.services[0].hours[0] = unordered.services[0].hours[1]
    expect(() =>
      assertSystemStatus({
        overall: 'healthy',
        services: [{ name: 'Router', status: 'operational', healthy: true }],
        history: unordered,
      }),
    ).toThrow('invalid response')
    await expect(fetchSystemStatus(async () => response(503))).rejects.toThrow('HTTP 503')
  })
})
