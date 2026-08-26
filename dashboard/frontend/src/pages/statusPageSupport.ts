import {
  STATUS_HISTORY_HOURS,
  type StatusHistory,
  type StatusHistoryHour,
  type SystemStatus,
} from '../utils/routerRuntime'

export const formatStatusLabel = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())

function unavailableCurrentHour(through: string): StatusHistoryHour[] {
  const throughTimestamp = Date.parse(through)
  return Array.from({ length: STATUS_HISTORY_HOURS }, (_, index) => ({
    observedAt: new Date(throughTimestamp - (STATUS_HISTORY_HOURS - 1 - index) * 3_600_000)
      .toISOString()
      .replace('.000Z', 'Z'),
    status: index === STATUS_HISTORY_HOURS - 1 ? ('unavailable' as const) : ('unknown' as const),
  }))
}

function routingHistory(status: SystemStatus | null): StatusHistory {
  const through = status?.history.through ?? `${new Date().toISOString().slice(0, 13)}:00:00Z`
  const services = (status?.history.services ?? []).filter(
    (service) => service.name !== 'Routing access',
  )
  return {
    windowHours: STATUS_HISTORY_HOURS,
    through,
    services: [...services, { name: 'Routing access', hours: unavailableCurrentHour(through) }],
  }
}

export function routingUnavailableStatus(status: SystemStatus | null): SystemStatus {
  const services = (status?.services ?? []).filter((service) => service.name !== 'Routing access')
  return {
    overall: 'degraded',
    services: [...services, { name: 'Routing access', status: 'unavailable', healthy: false }],
    history: routingHistory(status),
  }
}
