export interface ServiceStatus {
  name: string
  status: 'operational' | 'starting' | 'unavailable'
  healthy: boolean
}

export type StatusHistoryState = ServiceStatus['status'] | 'unknown'

export interface StatusHistoryHour {
  observedAt: string
  status: StatusHistoryState
}

export interface ServiceStatusHistory {
  name: string
  hours: StatusHistoryHour[]
}

export interface StatusHistory {
  windowHours: number
  through: string
  services: ServiceStatusHistory[]
}

export interface SystemStatus {
  overall: string
  services: ServiceStatus[]
  history: StatusHistory
}

export const STATUS_HISTORY_HOURS = 90

const ISO_UTC_HOUR_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:00:00Z$/

function isServiceStatus(value: unknown): value is ServiceStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const service = value as Partial<ServiceStatus>
  return (
    typeof service.name === 'string' &&
    Boolean(service.name.trim()) &&
    service.name.trim() === service.name &&
    (service.status === 'operational' ||
      service.status === 'starting' ||
      service.status === 'unavailable') &&
    typeof service.healthy === 'boolean'
  )
}

function isHistoryHour(value: unknown): value is StatusHistoryHour {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const hour = value as Partial<StatusHistoryHour>
  return (
    typeof hour.observedAt === 'string' &&
    ISO_UTC_HOUR_PATTERN.test(hour.observedAt) &&
    (hour.status === 'unknown' ||
      hour.status === 'operational' ||
      hour.status === 'starting' ||
      hour.status === 'unavailable')
  )
}

function isStatusHistory(value: unknown): value is StatusHistory {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const history = value as Partial<StatusHistory>
  if (
    history.windowHours !== STATUS_HISTORY_HOURS ||
    typeof history.through !== 'string' ||
    !ISO_UTC_HOUR_PATTERN.test(history.through) ||
    !Array.isArray(history.services)
  ) {
    return false
  }
  const throughTimestamp = Date.parse(history.through)
  if (!Number.isFinite(throughTimestamp)) return false
  const serviceNames = new Set<string>()
  return history.services.every((value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false
    const service = value as Partial<ServiceStatusHistory>
    if (
      typeof service.name !== 'string' ||
      !service.name.trim() ||
      service.name.trim() !== service.name ||
      serviceNames.has(service.name)
    ) {
      return false
    }
    serviceNames.add(service.name)
    return (
      Array.isArray(service.hours) &&
      service.hours.length === STATUS_HISTORY_HOURS &&
      service.hours.every((hour, index) => {
        if (!isHistoryHour(hour)) return false
        const expectedHour = new Date(
          throughTimestamp - (STATUS_HISTORY_HOURS - 1 - index) * 3_600_000,
        )
          .toISOString()
          .replace('.000Z', 'Z')
        return hour.observedAt === expectedHour
      })
    )
  })
}

export function assertSystemStatus(value: unknown): SystemStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('Status API returned an invalid response.')
  }
  const status = value as Partial<SystemStatus>
  const historyValid = isStatusHistory(status.history)
  if (
    typeof status.overall !== 'string' ||
    !status.overall.trim() ||
    !Array.isArray(status.services) ||
    !status.services.every(isServiceStatus) ||
    !historyValid
  ) {
    throw new TypeError('Status API returned an invalid response.')
  }
  const validStatus = status as SystemStatus
  const serviceNames = new Set(validStatus.services.map((service) => service.name))
  const historyNames = new Set(validStatus.history.services.map((service) => service.name))
  if (
    serviceNames.size !== validStatus.services.length ||
    !validStatus.services.every((service) => historyNames.has(service.name))
  ) {
    throw new TypeError('Status API returned an invalid response.')
  }
  return validStatus
}

export async function fetchSystemStatus(fetcher: typeof fetch = fetch): Promise<SystemStatus> {
  const response = await fetcher('/api/status', { cache: 'no-store' })
  if (!response.ok) throw new Error(`Status request failed (HTTP ${response.status}).`)
  return assertSystemStatus(await response.json())
}
