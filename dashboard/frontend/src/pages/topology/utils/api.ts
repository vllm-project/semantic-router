import { fetchManagedRoutingSnapshot } from '../../../utils/managedRoutingSnapshot'

/**
 * Fetch topology configuration
 */
export async function fetchTopologyConfig(scopeId?: string | null) {
  return fetchManagedRoutingSnapshot(scopeId)
}
