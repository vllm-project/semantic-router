import type { EmbeddingProviderRuntimeStatus, ServiceStatus } from '../utils/routerRuntime'

export type ServiceHealthFilter = 'all' | 'healthy' | 'unhealthy'

export function filterServices(
  services: ServiceStatus[],
  query: string,
  health: ServiceHealthFilter,
): ServiceStatus[] {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  return services.filter((service) => {
    if (health === 'healthy' && !service.healthy) return false
    if (health === 'unhealthy' && service.healthy) return false
    if (!normalizedQuery) return true
    return [service.name, service.component, service.status, service.message]
      .filter(Boolean)
      .join(' ')
      .toLocaleLowerCase()
      .includes(normalizedQuery)
  })
}

export function clampPage(page: number, itemCount: number, pageSize: number): number {
  return Math.min(Math.max(page, 1), Math.max(1, Math.ceil(itemCount / pageSize)))
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
