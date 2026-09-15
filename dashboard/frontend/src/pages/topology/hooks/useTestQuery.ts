// topology/hooks/useTestQuery.ts - Test Query functionality (always uses backend verification)

import { useState, useCallback } from 'react'
import { TestQueryResult, ParsedTopology } from '../types'
import { testQueryDryRun } from '../utils/api'

interface UseTestQueryResult {
  testQuery: string
  setTestQuery: (query: string) => void
  testResult: TestQueryResult | null
  isLoading: boolean
  runTest: () => Promise<void>
  clearResult: () => void
}

export function useTestQuery(
  _topologyData: ParsedTopology | null,
  routingModel?: string,
): UseTestQueryResult {
  const [testQuery, setTestQuery] = useState('')
  const [testResult, setTestResult] = useState<TestQueryResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Keep backend diagnostics and accuracy; never simulate a failed live preview.
  const runTest = useCallback(async () => {
    if (!testQuery.trim()) return

    setIsLoading(true)
    try {
      const result = await testQueryDryRun(testQuery, routingModel)
      setTestResult(result)
    } catch (error) {
      setTestResult({
        query: testQuery,
        mode: 'dry-run',
        matchedSignals: [],
        matchedDecision: null,
        matchedModels: [],
        highlightedPath: [],
        isAccurate: false,
        warning: error instanceof Error ? error.message : 'Live router preview failed',
      })
    } finally {
      setIsLoading(false)
    }
  }, [testQuery, routingModel])

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
