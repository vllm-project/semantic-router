import type { StatusTone } from '../ui'
import type {
  EmbeddingProviderRuntimeStatus,
  ServiceStatus,
  ServiceStatusHistory,
  StatusHistoryHour,
  StatusHistoryState,
  SystemStatus,
} from '../utils/routerRuntime'

export function formatStatusLabel(value?: string): string {
  if (!value) return 'Unavailable'
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}

export interface OverallState {
  tone: StatusTone
  title: string
  copy: string
  fullyOperational: boolean
}

const OFFLINE = new Set(['not_running', 'stopped'])

/** The headline for the whole deployment, from the live snapshot only. */
export function describeOverall(status: SystemStatus | null): OverallState {
  if (!status)
    return {
      tone: 'neutral',
      title: 'Status unavailable',
      copy: 'Live service status could not be verified.',
      fullyOperational: false,
    }
  const healthy = status.services.filter((service) => service.healthy).length
  if (
    status.services.length > 0 &&
    status.overall === 'healthy' &&
    healthy === status.services.length
  )
    return {
      tone: 'success',
      title: 'All systems operational',
      copy: 'Services are responding normally.',
      fullyOperational: true,
    }
  if (!status.services.length)
    return {
      tone: 'neutral',
      title: 'No running services detected',
      copy: 'Availability will appear when the Router starts.',
      fullyOperational: false,
    }
  if (OFFLINE.has(status.overall))
    return {
      tone: 'danger',
      title: 'Router offline',
      copy: 'The Router is not running, so requests cannot be routed. Start it with vllm-sr serve; this page keeps checking.',
      fullyOperational: false,
    }
  return {
    tone: status.overall === 'starting' ? 'info' : 'warning',
    title: formatStatusLabel(status.overall),
    copy: 'One or more components need attention.',
    fullyOperational: false,
  }
}

export interface ServiceState {
  label: string
  condition: string
  tone: StatusTone
}

export function describeService(service?: ServiceStatus): ServiceState {
  if (!service) return { label: 'Not reported', condition: 'Unknown', tone: 'neutral' }
  if (service.healthy) return { label: 'Operational', condition: 'Normal', tone: 'success' }
  if (service.status === 'starting')
    return { label: formatStatusLabel(service.status), condition: 'Starting', tone: 'info' }
  if (service.status === 'degraded')
    return { label: 'Degraded', condition: 'Needs attention', tone: 'warning' }
  return { label: formatStatusLabel(service.status), condition: 'Unavailable', tone: 'danger' }
}

/** Services from the live snapshot first, then any that only appear in history. */
export function orderedServiceNames(status: SystemStatus | null): string[] {
  const current = status?.services.map((service) => service.name) ?? []
  const historyOnly =
    status?.history?.services
      .map((service) => service.name)
      .filter((name) => !current.includes(name)) ?? []
  return [...current, ...historyOnly]
}

export const historyTone: Record<StatusHistoryState, StatusTone> = {
  operational: 'success',
  starting: 'info',
  unavailable: 'danger',
  unknown: 'neutral',
}

export function historyHourLabel(hour: StatusHistoryHour): string {
  return `${hour.observedAt.slice(0, 10)} ${hour.observedAt.slice(11, 16)} UTC: ${formatStatusLabel(hour.status)}`
}

export function historySummary(service: ServiceStatusHistory): string {
  const counts: Record<StatusHistoryState, number> = {
    operational: 0,
    starting: 0,
    unavailable: 0,
    unknown: 0,
  }
  for (const hour of service.hours) counts[hour.status] += 1
  return [
    `${service.name}, ${service.hours.length}-hour observed history:`,
    `${counts.operational} operational, ${counts.starting} starting,`,
    `${counts.unavailable} unavailable, ${counts.unknown} unknown.`,
  ].join(' ')
}

/** Share of observed hours that were operational; hours without an observation do not count. */
export function historyUptime(service: ServiceStatusHistory): number | null {
  const observed = service.hours.filter((hour) => hour.status !== 'unknown')
  if (!observed.length) return null
  return (observed.filter((hour) => hour.status === 'operational').length / observed.length) * 100
}

export function formatUptime(value: number | null): string {
  if (value === null) return 'No data'
  return `${value.toFixed(value === 0 || value === 100 ? 0 : 2)}%`
}

/** Mean uptime over services that have at least one observed hour. */
export function deploymentUptime(status: SystemStatus | null): number | null {
  const values = (status?.history?.services ?? [])
    .map(historyUptime)
    .filter((value): value is number => value !== null)
  if (!values.length) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

export function formatDeploymentType(type?: string): string {
  if (!type || type === 'none') return 'Not detected'
  return type
    .split(/[-_\s]+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

export function formatServingMode(mode?: string): string {
  if (mode === 'router') return 'Router'
  if (mode === 'engine') return 'Engine'
  return 'Not detected'
}

export type EmbeddingProviderTone = 'healthy' | 'unhealthy' | 'pending'

export function embeddingProviderTone(
  provider: EmbeddingProviderRuntimeStatus,
): EmbeddingProviderTone {
  if (provider.healthy === true) return 'healthy'
  if (provider.healthy === false) return 'unhealthy'
  return 'pending'
}

export function embeddingProviderHealthLabel(provider: EmbeddingProviderRuntimeStatus): string {
  const tone = embeddingProviderTone(provider)
  if (tone === 'healthy') return 'Healthy'
  if (tone === 'unhealthy') return 'Needs attention'
  return 'Not checked'
}

export function formatEmbeddingProviderBackend(backend?: string): string {
  if (!backend) return 'Not reported'
  if (backend === 'openai_compatible') return 'OpenAI compatible'
  return backend.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}

export function formatProviderCheckedAt(value?: string): string {
  if (!value) return 'Not checked'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

export function formatElapsed(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000))
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  return `${Math.round(minutes / 60)}h ago`
}
