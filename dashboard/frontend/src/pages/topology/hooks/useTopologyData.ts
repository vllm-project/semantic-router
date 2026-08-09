// topology/hooks/useTopologyData.ts - Data fetching and parsing hook

import { useState, useEffect, useCallback, useRef } from 'react'
import { ParsedTopology, ConfigData } from '../types'
import { parseConfigToTopology } from '../utils/topologyParser'
import { fetchTopologyConfig } from '../utils/api'
import {
  listRoutingScopes,
  projectConfigForRoutingScope,
  type RoutingScope,
  type RoutingScopedConfigLike,
} from '../../../utils/routingScopes'

interface UseTopologyDataResult {
  data: ParsedTopology | null
  rawConfig: ConfigData | null
  routingScopes: RoutingScope[]
  selectedScopeId: string
  setSelectedScopeId: (scopeId: string) => void
  loading: boolean
  error: string | null
  refresh: () => void
}

export function useTopologyData(): UseTopologyDataResult {
  const [data, setData] = useState<ParsedTopology | null>(null)
  const [rawConfig, setRawConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedScopeId, setSelectedScopeId] = useState('')
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)

  const fetchData = useCallback(async () => {
    const requestId = ++requestIdRef.current
    try {
      setLoading(true)
      setError(null)
      const config = await fetchTopologyConfig()
      if (!mountedRef.current || requestId !== requestIdRef.current) return
      setRawConfig(config)
      const scopes = listRoutingScopes(config as RoutingScopedConfigLike)
      const nextScopeId = scopes[0]?.id ?? ''
      setSelectedScopeId((current) =>
        scopes.some((scope) => scope.id === current) ? current : nextScopeId,
      )
      const projected = projectConfigForRoutingScope(
        config as ConfigData & RoutingScopedConfigLike,
        nextScopeId,
      )
      const parsed = parseConfigToTopology(projected)
      setData(parsed)
    } catch (err) {
      if (!mountedRef.current || requestId !== requestIdRef.current) return
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
    } finally {
      if (mountedRef.current && requestId === requestIdRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!rawConfig) return
    const projected = projectConfigForRoutingScope(
      rawConfig as ConfigData & RoutingScopedConfigLike,
      selectedScopeId,
    )
    setData(parseConfigToTopology(projected))
  }, [rawConfig, selectedScopeId])

  useEffect(() => {
    mountedRef.current = true
    void fetchData()
    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
    }
  }, [fetchData])

  return {
    data,
    rawConfig,
    routingScopes: listRoutingScopes(rawConfig as RoutingScopedConfigLike | null),
    selectedScopeId,
    setSelectedScopeId,
    loading,
    error,
    refresh: fetchData,
  }
}
