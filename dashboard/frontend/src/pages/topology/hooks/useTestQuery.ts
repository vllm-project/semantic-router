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

  // A failed Preview must not be presented as a simulated runtime result.
  const runTest = useCallback(async () => {
    if (!testQuery.trim()) return

    setIsLoading(true)
    try {
      setTestResult(await runTestQueryPreview(testQuery, routingModel))
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

export async function runTestQueryPreview(query: string, model?: string): Promise<TestQueryResult> {
  try {
    const result = await testQueryDryRun(query, model)
    return { ...result, mode: 'dry-run' }
  } catch (error) {
    return {
      query,
      mode: 'dry-run',
      matchedSignals: [],
      matchedDecision: null,
      matchedModels: [],
      highlightedPath: ['client'],
      isAccurate: false,
      warning: error instanceof Error ? `Preview unavailable: ${error.message}` : 'Preview unavailable',
    }
  }
}
