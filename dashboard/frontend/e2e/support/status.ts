const HISTORY_HOURS = 90
const HISTORY_THROUGH = Date.parse('2026-08-26T14:00:00Z')

interface StatusFixtureService {
  name: string
  status: 'operational' | 'starting' | 'unavailable'
  healthy?: boolean
}

export function withStatusHistory<T extends { services: StatusFixtureService[] }>(status: T) {
  return {
    ...status,
    history: {
      windowHours: HISTORY_HOURS,
      through: new Date(HISTORY_THROUGH).toISOString().replace('.000Z', 'Z'),
      services: status.services.map((service) => ({
        name: service.name,
        hours: Array.from({ length: HISTORY_HOURS }, (_, index) => ({
          observedAt: new Date(HISTORY_THROUGH - (HISTORY_HOURS - 1 - index) * 3_600_000)
            .toISOString()
            .replace('.000Z', 'Z'),
          status: index === HISTORY_HOURS - 1 ? service.status : ('unknown' as const),
        })),
      })),
    },
  }
}
