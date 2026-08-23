import type { RoutingEntrypoint } from '../utils/routingManagementApi'

export function assignedModelCount(entrypoint: RoutingEntrypoint): number {
  return entrypoint.assignedModelCount
}
