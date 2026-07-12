// topology/hooks/useTopologyData.ts - Data fetching and parsing hook

import { useState, useEffect, useCallback, useRef } from 'react'
import { ParsedTopology, ConfigData } from '../types'
import { parseConfigToTopology } from '../utils/topologyParser'
import { fetchTopologyConfig } from '../utils/api'

interface UseTopologyDataResult {
  data: ParsedTopology | null
  rawConfig: ConfigData | null
  loading: boolean
  error: string | null
  refresh: () => void
}

export function useTopologyData(): UseTopologyDataResult {
  const [data, setData] = useState<ParsedTopology | null>(null)
  const [rawConfig, setRawConfig] = useState<ConfigData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
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
      const parsed = parseConfigToTopology(config)
      setData(parsed)
    } catch (err) {
      if (!mountedRef.current || requestId !== requestIdRef.current) return
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
    } finally {
      if (mountedRef.current && requestId === requestIdRef.current) setLoading(false)
    }
  }, [])

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
    loading,
    error,
    refresh: fetchData,
  }
}
