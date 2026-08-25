// topology/hooks/useTopologyData.ts - Data fetching and parsing hook

import { useState, useEffect, useCallback, useRef } from 'react'
import { useAuth } from '../../../contexts/AuthContext'
import { useInferenceRoutingAccess } from '../../../contexts/InferenceRoutingAccessContext'
import type {
  ManagedRoutingScope,
  ManagedRoutingSnapshot,
} from '../../../utils/managedRoutingSnapshot'
import type { ManagedTopologyConfig, ParsedTopology } from '../types'
import { parseConfigToTopology } from '../utils/topologyParser'
import { fetchTopologyConfig } from '../utils/api'
import { canReadRouting } from '../../../utils/accessControl'

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
  const { user } = useAuth()
  const { catalogError, catalogSnapshot, catalogStatus, retryCatalog, usesKeyScopedCatalog } =
    useInferenceRoutingAccess()
  const hasGlobalRoutingAccess = canReadRouting(user)
  const [data, setData] = useState<ParsedTopology | null>(null)
  const [rawConfig, setRawConfig] = useState<ManagedRoutingSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedScopeId, setSelectedScopeId] = useState('')
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const initialScopeIdRef = useRef(new URLSearchParams(window.location.search).get('scope') ?? '')

  const applyConfig = useCallback((config: ManagedRoutingSnapshot, requestedScopeId = '') => {
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
    setError(null)
    setLoading(false)
  }, [])

  const fetchData = useCallback(
    async (requestedScopeId = '') => {
      if (!hasGlobalRoutingAccess) return
      const requestId = ++requestIdRef.current
      try {
        setLoading(true)
        setError(null)
        const config = await fetchTopologyConfig(requestedScopeId)
        if (!mountedRef.current || requestId !== requestIdRef.current) return
        applyConfig(config, requestedScopeId)
      } catch (err) {
        if (!mountedRef.current || requestId !== requestIdRef.current) return
        setError(err instanceof Error ? err.message : 'Failed to load configuration')
      } finally {
        if (mountedRef.current && requestId === requestIdRef.current) setLoading(false)
      }
    },
    [applyConfig, hasGlobalRoutingAccess],
  )

  useEffect(() => {
    if (!usesKeyScopedCatalog) return
    requestIdRef.current += 1
    if (catalogStatus === 'ready' && catalogSnapshot) {
      applyConfig(catalogSnapshot, selectedScopeId || initialScopeIdRef.current)
      return
    }
    setRawConfig(null)
    setData(null)
    setLoading(catalogStatus === 'idle' || catalogStatus === 'loading')
    if (catalogStatus === 'unavailable') {
      setError('Create an API key to view your routing topology.')
    } else if (catalogStatus === 'error') {
      setError(catalogError ?? 'Routing topology is unavailable.')
    } else {
      setError(null)
    }
  }, [
    applyConfig,
    catalogError,
    catalogSnapshot,
    catalogStatus,
    selectedScopeId,
    usesKeyScopedCatalog,
  ])

  useEffect(() => {
    if (hasGlobalRoutingAccess || usesKeyScopedCatalog) return
    setLoading(false)
    setRawConfig(null)
    setData(null)
    setError('Routing access is required.')
  }, [hasGlobalRoutingAccess, usesKeyScopedCatalog])

  const selectScope = useCallback(
    (scopeId: string) => {
      setSelectedScopeId(scopeId)
      const entrypointSuffix = scopeId.startsWith('entrypoint:')
        ? scopeId.slice('entrypoint:'.length)
        : ''
      if (hasGlobalRoutingAccess && entrypointSuffix && !entrypointSuffix.includes(':')) {
        void fetchData(scopeId)
      }
    },
    [fetchData, hasGlobalRoutingAccess],
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
    if (hasGlobalRoutingAccess) void fetchData(initialScopeIdRef.current)
    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
    }
  }, [fetchData, hasGlobalRoutingAccess])

  const refresh = useCallback(() => {
    if (usesKeyScopedCatalog) {
      retryCatalog()
      return
    }
    void fetchData(selectedScopeId)
  }, [fetchData, retryCatalog, selectedScopeId, usesKeyScopedCatalog])

  return {
    data,
    rawConfig,
    routingScopes: rawConfig?.routingScopes ?? [],
    selectedScopeId,
    setSelectedScopeId: selectScope,
    loading,
    error,
    refresh,
  }
}
