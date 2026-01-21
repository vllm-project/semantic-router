// topology/hooks/useTestQuery.ts - Test Query functionality (always uses backend verification)

import { useState, useCallback } from 'react'
import { TestQueryResult, ParsedTopology } from '../types'
import { simulateSignalMatching } from '../utils/signalMatcher'
import { testQueryDryRun } from '../utils/api'

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

  // Always use backend verification, with frontend fallback
  const runTest = useCallback(async () => {
    if (!testQuery.trim()) return

    setIsLoading(true)
    try {
      const result = await testQueryDryRun(testQuery)
      setTestResult({ ...result, mode: 'dry-run', isAccurate: true })
    } catch (error) {
      console.warn('Backend verification failed, falling back to simulation:', error)
      // Fallback to frontend simulation if backend unavailable
      if (topologyData) {
        const simResult = await simulateSignalMatching(testQuery, topologyData)
        setTestResult({
          ...simResult,
          mode: 'simulate',
          isAccurate: false,
          warning: 'Backend unavailable, showing simulated results',
        })
      }
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
