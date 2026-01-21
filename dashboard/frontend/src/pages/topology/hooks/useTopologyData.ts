// topology/hooks/useTopologyData.ts - Data fetching and parsing hook

import { useState, useEffect, useCallback } from 'react'
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

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const config = await fetchTopologyConfig()
      setRawConfig(config)
      const parsed = parseConfigToTopology(config)
      setData(parsed)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
      console.error('Error fetching topology config:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return {
    data,
    rawConfig,
    loading,
    error,
    refresh: fetchData,
  }
}
