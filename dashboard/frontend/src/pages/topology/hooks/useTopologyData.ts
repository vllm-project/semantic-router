// topology/hooks/useTopologyData.ts - Data fetching and parsing hook

import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  ManagedRoutingScope,
  ManagedRoutingSnapshot,
} from '../../../utils/managedRoutingSnapshot'
import type { ManagedTopologyConfig, ParsedTopology } from '../types'
import { parseConfigToTopology } from '../utils/topologyParser'
import { fetchTopologyConfig } from '../utils/api'

interface UseTopologyDataResult {
  data: ParsedTopology | null
  rawConfig: ManagedRoutingSnapshot | null
  routingScopes: ManagedRoutingScope[]
  selectedScopeId: string
  setSelectedScopeId: (scopeId: string) => void
  loading: boolean
  error: string | null
  refresh: () => void
}

export function useTopologyData(): UseTopologyDataResult {
  const [data, setData] = useState<ParsedTopology | null>(null)
  const [rawConfig, setRawConfig] = useState<ManagedRoutingSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedScopeId, setSelectedScopeId] = useState('')
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const initialScopeIdRef = useRef(new URLSearchParams(window.location.search).get('scope') ?? '')

  const fetchData = useCallback(async (requestedScopeId = '') => {
    const requestId = ++requestIdRef.current
    try {
      setLoading(true)
      setError(null)
      const config = await fetchTopologyConfig(requestedScopeId)
      if (!mountedRef.current || requestId !== requestIdRef.current) return
      setRawConfig(config)
      const scopes = config.routingScopes
      const nextScopeId =
        scopes.find((scope) => scope.id === requestedScopeId)?.id ??
        scopes.find((scope) => scope.id.startsWith(`${requestedScopeId}:`))?.id ??
        scopes[0]?.id ??
        ''
      setSelectedScopeId((current) =>
        scopes.some((scope) => scope.id === current) ? current : nextScopeId,
      )
      const selectedScope = scopes.find((scope) => scope.id === nextScopeId)
      const parsed = parseConfigToTopology({
        models: config.models,
        document: (selectedScope?.document ?? {
          decisions: [],
        }) as ManagedTopologyConfig['document'],
      })
      setData(parsed)
    } catch (err) {
      if (!mountedRef.current || requestId !== requestIdRef.current) return
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
    } finally {
      if (mountedRef.current && requestId === requestIdRef.current) setLoading(false)
    }
  }, [])

  const selectScope = useCallback(
    (scopeId: string) => {
      setSelectedScopeId(scopeId)
      const entrypointSuffix = scopeId.startsWith('entrypoint:')
        ? scopeId.slice('entrypoint:'.length)
        : ''
      if (entrypointSuffix && !entrypointSuffix.includes(':')) void fetchData(scopeId)
    },
    [fetchData],
  )

  useEffect(() => {
    if (!rawConfig) return
    const selectedScope = rawConfig.routingScopes.find((scope) => scope.id === selectedScopeId)
    setData(
      parseConfigToTopology({
        models: rawConfig.models,
        document: (selectedScope?.document ?? {
          decisions: [],
        }) as ManagedTopologyConfig['document'],
      }),
    )
  }, [rawConfig, selectedScopeId])

  useEffect(() => {
    mountedRef.current = true
    void fetchData(initialScopeIdRef.current)
    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
    }
  }, [fetchData])

  return {
    data,
    rawConfig,
    routingScopes: rawConfig?.routingScopes ?? [],
    selectedScopeId,
    setSelectedScopeId: selectScope,
    loading,
    error,
    refresh: () => {
      void fetchData(selectedScopeId)
    },
  }
}
