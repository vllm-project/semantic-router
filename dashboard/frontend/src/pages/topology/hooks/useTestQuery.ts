// topology/hooks/useTestQuery.ts - Local, non-authoritative topology preview

import { useState, useCallback } from 'react'
import { TestQueryResult, ParsedTopology } from '../types'
import { simulateSignalMatching } from '../utils/signalMatcher'

interface UseTestQueryResult {
  testQuery: string
  setTestQuery: (query: string) => void
  testResult: TestQueryResult | null
  isLoading: boolean
  runTest: () => Promise<void>
  clearResult: () => void
}

export function useTestQuery(topologyData: ParsedTopology | null): UseTestQueryResult {
  const [testQuery, setTestQuery] = useState('')
  const [testResult, setTestResult] = useState<TestQueryResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const runTest = useCallback(async () => {
    if (!testQuery.trim() || !topologyData) return

    setIsLoading(true)
    try {
      setTestResult(await simulateSignalMatching(testQuery, topologyData))
    } finally {
      setIsLoading(false)
    }
  }, [testQuery, topologyData])

  const clearResult = useCallback(() => {
    setTestResult(null)
  }, [])

  return {
    testQuery,
    setTestQuery,
    testResult,
    isLoading,
    runTest,
    clearResult,
  }
}
