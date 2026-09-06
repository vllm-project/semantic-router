export interface TopologyCacheConfig {
  enabled: boolean
  backend_type?: string
  similarity_threshold?: number
  ttl_seconds?: number
}

export type TopologyOptionalCacheConfig = Omit<TopologyCacheConfig, 'enabled'> & {
  enabled?: boolean
}
