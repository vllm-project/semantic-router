import type { ServiceStatus } from '../utils/routerRuntime'

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
